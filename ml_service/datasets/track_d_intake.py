"""
OsteoGuard AI — Stage 33 Track D Clinical Dataset Intake, Quality Validation,
Quarantine, Deterministic Leakage-Safe Splitting, and Readiness Gate.

Operates cleanly with pure Python 3.10+ stdlib inside AI Studio while
automatically utilizing Pillow / pydicom when installed on an ML workstation.

Strict Clinical & Scientific Safeguards:
- Separates imaging modalities strictly (`xray`, `bone_scan`, `ct`).
- Never fabricates synthetic samples or converts uncertain diagnoses.
- Never stores direct PHI (patient names, MRNs, emails, SSNs, addresses, unmasked DOBs).
- Protects BTXRD Center 2 (N=549) & Center 3 (N=259) external holdout partitions
  and FracAtlas burned test splits from entering Track D training/validation.
- Does not invent minimum clinical sample thresholds without documented justification.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from math import prod
from pathlib import Path
import random
import re
import shutil
import struct
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import zlib

from jsonschema import Draft202012Validator, FormatChecker

try:
    from PIL import Image as _PILImage  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _PILImage = None

try:
    import pydicom as _pydicom  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _pydicom = None

try:
    import nibabel as _nibabel  # type: ignore
except Exception:  # pragma: no cover - parser is required for NIfTI intake
    _nibabel = None


from ml_service.datasets.clinical_ontology import (
    AdjudicationStatus,
    ONTOLOGY_TO_SHORT_CODE,
    ONTOLOGY_VERSION,
    OntologyTargetClass,
    TARGET_SHORT_TO_ONTOLOGY,
    adjudicate_diagnostic_label,
    check_label_provenance_completeness,
    normalize_diagnosis_term,
)
from ml_service.datasets.holdout_policy import (
    ProtectedHoldoutRegistry,
    load_protected_holdouts,
    normalize_holdout_sites,
)

TARGET_CLASSES: Tuple[str, ...] = ("metastasis", "paget", "oi", "healthy")
SUPPORTED_MODALITIES: Tuple[str, ...] = ("xray", "bone_scan", "ct")
SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
    ".dcm": "DICOM",
    ".dicom": "DICOM",
    ".nii": "NIFTI",
    ".nii.gz": "NIFTI",
}

REQUIRED_MANIFEST_FIELDS: Tuple[str, ...] = (
    "sample_id",
    "patient_id",
    "diagnostic_class",
    "imaging_modality",
    "image_path",
    "image_format",
    "image_dimensions",
    "image_bit_depth",
    "anatomical_region",
    "institution_id",
    "dataset_source",
    "license_permissions",
    "ethics_approval_ref",
    "deidentification_status",
    "qc_status",
    "split_assignment",
)

MODALITY_ALIASES: Dict[str, str] = {
    "xray": "xray",
    "x-ray": "xray",
    "x_ray": "xray",
    "radiograph": "xray",
    "plain_radiography": "xray",
    "bone_scan": "bone_scan",
    "bonescan": "bone_scan",
    "scintigraphy": "bone_scan",
    "bone_scintigraphy": "bone_scan",
    "ct": "ct",
    "computed_tomography": "ct",
}

BANNED_PHI_KEYS: Set[str] = {
    "patient_name",
    "full_name",
    "first_name",
    "last_name",
    "mrn",
    "medical_record_number",
    "address",
    "street_address",
    "phone",
    "phone_number",
    "email",
    "ssn",
    "social_security_number",
    "dob",
    "date_of_birth",
    "birth_date",
}

PHI_VALUE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL_PATTERN", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("SSN_PATTERN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE_PATTERN", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("MRN_PREFIX_PATTERN", re.compile(r"\b(?:MRN|HOSP|REG)[:_-]?\d{4,}\b", re.IGNORECASE)),
    ("DOB_PATTERN", re.compile(r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b")),
)

PSEUDO_PATIENT_ID_RE = re.compile(r"^(?:PT|SUBJ)_[A-Za-z0-9_-]{2,}$")

MIN_IMAGE_DIMENSION = 32
DEFAULT_NEAR_DUPLICATE_HAMMING = 3
MAX_NIFTI_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
TRACK_D_MANIFEST_SCHEMA = (
    Path(__file__).resolve().parents[1] / "data" / "manifests" / "track_d_manifest.schema.json"
)

INVALID_PLACEHOLDER_TOKENS: Set[str] = {
    "",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "missing",
    "unverified",
    "pending",
    "tbd",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def _compute_dhash_from_grayscale(gray: List[List[int]]) -> str:
    """Computes 64-bit dHash from a 2D grayscale integer matrix without external deps."""
    if not gray or not gray[0]:
        return ""
    src_h = len(gray)
    src_w = len(gray[0])
    # Downsample via nearest-neighbor to 9 columns x 8 rows
    bits = 0
    for y in range(8):
        sy = min(src_h - 1, int(y * src_h / 8))
        row_vals = [gray[sy][min(src_w - 1, int(x * src_w / 9))] for x in range(9)]
        for x in range(8):
            bits = (bits << 1) | (1 if row_vals[x] < row_vals[x + 1] else 0)
    return f"{bits:016x}"


def compute_hamming_distance(hex_a: str, hex_b: str) -> Optional[int]:
    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        return None
    try:
        return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()
    except ValueError:
        return None


def _decode_png_grayscale(raw_bytes: bytes) -> Tuple[int, int, int, Optional[List[List[int]]]]:
    """
    Parses PNG IHDR and validates zlib IDAT stream in pure Python stdlib.
    Returns (width, height, bit_depth, optional_grayscale_grid).
    """
    if len(raw_bytes) < 33 or raw_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")

    pos = 8
    width = height = bit_depth = color_type = 0
    ihdr_seen = False
    iend_seen = False
    idat_chunks: List[bytes] = []

    while pos + 8 <= len(raw_bytes):
        length = struct.unpack(">I", raw_bytes[pos : pos + 4])[0]
        chunk_type = raw_bytes[pos + 4 : pos + 8]
        data_start = pos + 8
        data_end = data_start + length
        if data_end + 4 > len(raw_bytes):
            raise ValueError("Truncated PNG chunk")
        chunk_data = raw_bytes[data_start:data_end]
        if chunk_type == b"IHDR":
            if len(chunk_data) < 13:
                raise ValueError("Corrupt PNG IHDR")
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
            ihdr_seen = True
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            iend_seen = True
            break
        pos = data_end + 4

    if not ihdr_seen or not iend_seen or not idat_chunks:
        raise ValueError("Incomplete PNG structure (missing IHDR/IDAT/IEND)")

    # Validate compressed IDAT stream integrity
    decompressed = zlib.decompress(b"".join(idat_chunks))
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if channels is None or bit_depth not in (1, 2, 4, 8, 16):
        raise ValueError("Unsupported PNG color_type or bit_depth")

    # Reconstruct 8-bit grayscale matrix for pure-stdlib dHash when 8-bit/16-bit non-palette
    gray: Optional[List[List[int]]] = None
    if bit_depth in (8, 16) and color_type in (0, 2, 4, 6) and width <= 2048 and height <= 2048:
        bpp = max(1, (channels * bit_depth) // 8)
        stride = width * bpp
        expected_len = height * (1 + stride)
        if len(decompressed) >= expected_len:
            prev = bytearray(stride)
            gray = []
            idx = 0
            for _row in range(height):
                ftype = decompressed[idx]
                idx += 1
                scan = bytearray(decompressed[idx : idx + stride])
                idx += stride
                if ftype == 1:
                    for i in range(bpp, stride):
                        scan[i] = (scan[i] + scan[i - bpp]) & 0xFF
                elif ftype == 2:
                    for i in range(stride):
                        scan[i] = (scan[i] + prev[i]) & 0xFF
                elif ftype == 3:
                    for i in range(stride):
                        left = scan[i - bpp] if i >= bpp else 0
                        scan[i] = (scan[i] + ((left + prev[i]) >> 1)) & 0xFF
                elif ftype == 4:
                    for i in range(stride):
                        a = scan[i - bpp] if i >= bpp else 0
                        b = prev[i]
                        c = prev[i - bpp] if i >= bpp else 0
                        p = a + b - c
                        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                        pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                        scan[i] = (scan[i] + pr) & 0xFF
                elif ftype != 0:
                    gray = None
                    break
                prev = scan

                row_gray: List[int] = []
                step = bpp
                for px in range(0, stride, step):
                    if color_type in (0, 4):
                        row_gray.append(scan[px])
                    else:
                        cstep = bit_depth // 8
                        r = scan[px]
                        g = scan[px + cstep]
                        b_ch = scan[px + 2 * cstep]
                        row_gray.append((77 * r + 150 * g + 29 * b_ch) >> 8)
                gray.append(row_gray)

    effective_depth = bit_depth * (3 if color_type in (2, 6) and bit_depth == 8 else 1)
    if effective_depth not in (8, 10, 12, 16, 24, 32):
        effective_depth = 8 if bit_depth <= 8 else 16
    return width, height, effective_depth, gray


def _parse_jpeg_header(raw_bytes: bytes) -> Tuple[int, int, int]:
    if len(raw_bytes) < 4 or raw_bytes[:2] != b"\xff\xd8":
        raise ValueError("Invalid JPEG SOI marker")
    pos = 2
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB}
    while pos + 4 <= len(raw_bytes):
        if raw_bytes[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(raw_bytes) and raw_bytes[pos] == 0xFF:
            pos += 1
        if pos >= len(raw_bytes):
            break
        marker = raw_bytes[pos]
        pos += 1
        if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
            continue
        if pos + 2 > len(raw_bytes):
            raise ValueError("Truncated JPEG segment")
        seg_len = struct.unpack(">H", raw_bytes[pos : pos + 2])[0]
        if seg_len < 2 or pos + seg_len > len(raw_bytes):
            raise ValueError("Corrupt JPEG segment length")
        if marker in sof_markers:
            if seg_len < 8:
                raise ValueError("Corrupt JPEG SOF frame")
            precision, height, width, components = struct.unpack(">BHHB", raw_bytes[pos + 2 : pos + 8])
            bit_depth = precision * components
            if bit_depth not in (8, 10, 12, 16, 24, 32):
                bit_depth = 24 if components == 3 else 8
            return width, height, bit_depth
        pos += seg_len
    raise ValueError("JPEG SOF frame not found")


def _parse_bmp_header(raw_bytes: bytes) -> Tuple[int, int, int, Optional[List[List[int]]]]:
    if len(raw_bytes) < 26 or raw_bytes[:2] != b"BM":
        raise ValueError("Invalid BMP signature")
    offset = struct.unpack("<I", raw_bytes[10:14])[0]
    dib_size = struct.unpack("<I", raw_bytes[14:18])[0]
    if dib_size < 12:
        raise ValueError("Unsupported BMP DIB header")
    width, height = struct.unpack("<ii", raw_bytes[18:26])
    bpp = struct.unpack("<H", raw_bytes[28:30])[0] if len(raw_bytes) >= 30 else 24
    width = abs(width)
    abs_h = abs(height)
    gray: Optional[List[List[int]]] = None
    if bpp in (8, 24) and width > 0 and abs_h > 0 and width <= 2048 and abs_h <= 2048:
        row_bytes = ((bpp * width + 31) // 32) * 4
        if offset + row_bytes * abs_h <= len(raw_bytes):
            gray = []
            for r in range(abs_h):
                src_r = (abs_h - 1 - r) if height > 0 else r
                base = offset + src_r * row_bytes
                row_gray: List[int] = []
                if bpp == 8:
                    row_gray = list(raw_bytes[base : base + width])
                else:
                    for c in range(width):
                        b_ch, g_ch, r_ch = raw_bytes[base + c * 3 : base + c * 3 + 3]
                        row_gray.append((77 * r_ch + 150 * g_ch + 29 * b_ch) >> 8)
                gray.append(row_gray)
    bit_depth = bpp if bpp in (8, 16, 24, 32) else 24
    return width, abs_h, bit_depth, gray


def _parse_tiff_header(raw_bytes: bytes) -> Tuple[int, int, int]:
    if len(raw_bytes) < 8:
        raise ValueError("Truncated TIFF header")
    if raw_bytes[:4] == b"II*\x00":
        endian = "<"
    elif raw_bytes[:4] == b"MM\x00*":
        endian = ">"
    else:
        raise ValueError("Invalid TIFF endian magic")
    ifd_offset = struct.unpack(endian + "I", raw_bytes[4:8])[0]
    if ifd_offset + 2 > len(raw_bytes):
        raise ValueError("Invalid TIFF IFD offset")
    num_entries = struct.unpack(endian + "H", raw_bytes[ifd_offset : ifd_offset + 2])[0]
    width = height = 0
    bit_depth = 8
    for i in range(num_entries):
        entry_off = ifd_offset + 2 + i * 12
        if entry_off + 12 > len(raw_bytes):
            break
        tag, dtype, count, value_raw = struct.unpack(endian + "HHII", raw_bytes[entry_off : entry_off + 12])
        if dtype == 3:  # SHORT
            val = struct.unpack(endian + "H", raw_bytes[entry_off + 8 : entry_off + 10])[0]
        else:
            val = value_raw
        if tag == 256:
            width = val
        elif tag == 257:
            height = val
        elif tag == 258 and count == 1:
            bit_depth = val
    if width <= 0 or height <= 0:
        raise ValueError("Missing TIFF ImageWidth/ImageLength tags")
    if bit_depth not in (8, 10, 12, 16, 24, 32):
        bit_depth = 16 if bit_depth > 8 else 8
    return width, height, bit_depth


def _parse_dicom_header(path: Path, raw_bytes: bytes) -> Tuple[int, int, int, Optional[str]]:
    if _pydicom is None:
        raise ValueError("DICOM_METADATA_SCREENING_UNAVAILABLE")
    ds = _pydicom.dcmread(str(path), stop_before_pixels=True)
    _screen_dicom_metadata(ds)
    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)
    bits = int(getattr(ds, "BitsAllocated", 16) or 16)
    mod = getattr(ds, "Modality", None)
    if rows <= 0 or cols <= 0:
        raise ValueError("DICOM missing Rows/Columns")
    return cols, rows, bits, str(mod) if mod else None

def _screen_dicom_metadata(dataset: Any) -> None:
    """Reject direct identifiers, unpseudonymized PatientID, and private tags.

    Only reason codes may leave this function; DICOM values must never appear in
    raised errors, logs, or intake reports.
    """
    direct_identifier_keywords = {
        "PatientName", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
        "OtherPatientNames", "OtherPatientIDs", "MedicalRecordLocator", "AccessionNumber",
        "PatientComments", "InstitutionName", "ReferringPhysicianName",
        "PerformingPhysicianName", "OperatorsName",
    }
    for element in dataset.iterall():
        keyword = str(getattr(element, "keyword", "") or "")
        value = getattr(element, "value", None)
        if getattr(getattr(element, "tag", None), "is_private", False) and value not in (None, "", b""):
            raise ValueError("DICOM_PRIVATE_TAG_PRESENT")
        if keyword in direct_identifier_keywords and value not in (None, "", b""):
            raise ValueError("DICOM_DIRECT_IDENTIFIER_PRESENT")
        if keyword == "PatientID" and value not in (None, "", b""):
            if not PSEUDO_PATIENT_ID_RE.match(str(value).strip()):
                raise ValueError("DICOM_PATIENT_ID_NOT_PSEUDONYMIZED")
        if keyword == "BurnedInAnnotation" and str(value).strip().upper() == "YES":
            raise ValueError("DICOM_BURNED_IN_ANNOTATION_PRESENT")
        if isinstance(value, (str, bytes)):
            text_value = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
            if any(pattern.search(text_value) for _, pattern in PHI_VALUE_PATTERNS):
                raise ValueError("DICOM_PHI_PATTERN_DETECTED")


def _parse_nifti_header(path: Path, raw_bytes: bytes) -> Tuple[int, int, int, int]:
    if _nibabel is not None:
        try:
            image = _nibabel.load(str(path), mmap=False)
            shape = tuple(int(dim) for dim in image.shape)
            if not 2 <= len(shape) <= 3 or any(dim <= 0 for dim in shape):
                raise ValueError("NIFTI_INVALID_DIMENSIONALITY")
            if len(shape) == 2:
                shape = (*shape, 1)
            dtype = image.get_data_dtype()
            bit_depth = int(dtype.itemsize * 8)
            if bit_depth not in (8, 16, 24, 32, 64):
                raise ValueError("NIFTI_UNSUPPORTED_BIT_DEPTH")
            header = image.header
            if bytes(header["magic"]).rstrip(b"\x00") != b"n+1":
                raise ValueError("NIFTI_SINGLE_FILE_MAGIC_INVALID")
            offset = int(getattr(image.dataobj, "offset", 0))
            if offset < 348:
                raise ValueError("NIFTI_INVALID_VOXEL_OFFSET")
            expected_size = offset + prod(shape) * int(header.get_data_dtype().itemsize)
            if expected_size > MAX_NIFTI_UNCOMPRESSED_BYTES:
                raise ValueError("NIFTI_VOLUME_EXCEEDS_SAFE_SIZE_LIMIT")
            if path.name.lower().endswith(".nii.gz"):
                bytes_read = 0
                try:
                    with gzip.open(path, "rb") as stream:
                        while bytes_read < expected_size:
                            chunk = stream.read(min(1024 * 1024, expected_size - bytes_read))
                            if not chunk:
                                break
                            bytes_read += len(chunk)
                except (OSError, EOFError) as exc:
                    raise ValueError("NIFTI_GZIP_INVALID") from exc
                if bytes_read < expected_size:
                    raise ValueError("NIFTI_VOXEL_DATA_TRUNCATED")
            elif path.stat().st_size < expected_size:
                raise ValueError("NIFTI_VOXEL_DATA_TRUNCATED")
            # Accessing proxy shape validates header geometry without loading voxels into memory.
            proxy_shape = tuple(int(dim) for dim in image.dataobj.shape)
            if proxy_shape != tuple(image.shape):
                raise ValueError("NIFTI_HEADER_DATA_SHAPE_MISMATCH")
            return shape[0], shape[1], shape[2], bit_depth
        except Exception as exc:
            reason = str(exc)
            if reason.startswith("NIFTI_"):
                raise ValueError(reason) from exc
            raise ValueError("NIFTI_HEADER_INVALID") from exc

    # Strict NIfTI-1 fallback for minimal environments; NIfTI-2 requires nibabel.
    data = raw_bytes
    if path.name.lower().endswith(".nii.gz"):
        try:
            data = gzip.decompress(raw_bytes)
        except (OSError, EOFError) as exc:
            raise ValueError("NIFTI_GZIP_INVALID") from exc
    if len(data) < 352:
        raise ValueError("NIFTI_HEADER_TRUNCATED")
    sizeof_hdr = struct.unpack("<i", data[:4])[0]
    endian = "<" if sizeof_hdr == 348 else (">" if struct.unpack(">i", data[:4])[0] == 348 else "")
    if not endian or data[344:348] not in (b"n+1\x00", b"ni1\x00"):
        raise ValueError("NIFTI_HEADER_INVALID")
    dims = struct.unpack(endian + "8h", data[40:56])
    bitpix = struct.unpack(endian + "h", data[72:74])[0]
    ndim = int(dims[0])
    if ndim not in (2, 3) or any(int(dim) <= 0 for dim in dims[1 : ndim + 1]):
        raise ValueError("NIFTI_DIMENSIONS_INVALID")
    width, height = int(dims[1]), int(dims[2])
    depth = int(dims[3]) if ndim == 3 else 1
    if bitpix not in (8, 16, 24, 32, 64):
        raise ValueError("NIFTI_UNSUPPORTED_BIT_DEPTH")
    return width, height, depth, bitpix


def inspect_image_file(file_path: Path) -> Dict[str, Any]:
    """
    Validates image binary headers, extracts format/dimensions/bit-depth,
    computes SHA-256 and perceptual dHash (when raster decodable).
    """
    lower_name = file_path.name.lower()
    if lower_name.endswith(".nii.gz"):
        ext = ".nii.gz"
    else:
        ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "valid": False,
            "error": f"UNSUPPORTED_FORMAT:{ext or 'none'}",
            "image_format": "UNKNOWN",
            "width": 0,
            "height": 0,
            "depth": 1,
            "bit_depth": 0,
            "sha256": "",
            "dhash": "",
            "dicom_modality": None,
        }

    if not file_path.exists() or not file_path.is_file():
        return {
            "valid": False,
            "error": "MISSING_FILE",
            "image_format": SUPPORTED_EXTENSIONS[ext],
            "width": 0,
            "height": 0,
            "depth": 1,
            "bit_depth": 0,
            "sha256": "",
            "dhash": "",
            "dicom_modality": None,
        }

    size_bytes = file_path.stat().st_size
    if size_bytes == 0:
        return {
            "valid": False,
            "error": "EMPTY_FILE",
            "image_format": SUPPORTED_EXTENSIONS[ext],
            "width": 0,
            "height": 0,
            "depth": 1,
            "bit_depth": 0,
            "sha256": "",
            "dhash": "",
            "dicom_modality": None,
        }

    raw_bytes = file_path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest().upper()
    fmt = SUPPORTED_EXTENSIONS[ext]
    width = height = bit_depth = 0
    depth = 1
    dhash = ""
    dicom_mod: Optional[str] = None

    try:
        if fmt == "PNG":
            width, height, bit_depth, gray = _decode_png_grayscale(raw_bytes)
            if gray is not None:
                dhash = _compute_dhash_from_grayscale(gray)
        elif fmt == "JPEG":
            width, height, bit_depth = _parse_jpeg_header(raw_bytes)
        elif fmt == "BMP":
            width, height, bit_depth, gray = _parse_bmp_header(raw_bytes)
            if gray is not None:
                dhash = _compute_dhash_from_grayscale(gray)
        elif fmt == "TIFF":
            width, height, bit_depth = _parse_tiff_header(raw_bytes)
        elif fmt == "DICOM":
            width, height, bit_depth, dicom_mod = _parse_dicom_header(file_path, raw_bytes)
        elif fmt == "NIFTI":
            width, height, depth, bit_depth = _parse_nifti_header(file_path, raw_bytes)

        # Optional Pillow enhancement for JPEG/TIFF perceptual dHash when PIL is available
        if not dhash and _PILImage is not None and fmt in ("PNG", "JPEG", "TIFF", "BMP"):
            with _PILImage.open(file_path) as im:
                im.verify()
            with _PILImage.open(file_path) as im2:
                gray_im = im2.convert("L").resize((9, 8))
                pixels = list(gray_im.getdata())
                grid = [pixels[r * 9 : (r + 1) * 9] for r in range(8)]
                dhash = _compute_dhash_from_grayscale(grid)
    except Exception as exc:
        safe_error = "DICOM_HEADER_OR_PHI_VALIDATION_FAILED" if fmt == "DICOM" else str(exc)
        return {
            "valid": False,
            "error": f"CORRUPT_IMAGE_HEADER:{safe_error}",
            "image_format": fmt,
            "width": 0,
            "height": 0,
            "depth": 1,
            "bit_depth": 0,
            "sha256": sha256,
            "dhash": "",
            "dicom_modality": dicom_mod,
        }

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        return {
            "valid": False,
            "error": f"INVALID_DIMENSIONS:{width}x{height}",
            "image_format": fmt,
            "width": width,
            "height": height,
            "depth": depth,
            "bit_depth": bit_depth,
            "sha256": sha256,
            "dhash": dhash,
            "dicom_modality": dicom_mod,
        }

    return {
        "valid": True,
        "error": None,
        "image_format": fmt,
        "width": width,
        "height": height,
        "depth": depth,
        "bit_depth": bit_depth,
        "sha256": sha256,
        "dhash": dhash,
        "dicom_modality": dicom_mod,
    }


def detect_phi_violations(record: Dict[str, Any]) -> List[str]:
    """Returns reason codes if direct PHI or banned identifier keys/values exist."""
    reasons: List[str] = []
    for key in record.keys():
        if key.lower() in BANNED_PHI_KEYS and str(record.get(key) or "").strip():
            reasons.append(f"PHI_DISALLOWED_FIELD:{key}")

    pid = str(record.get("patient_id") or "").strip()
    if pid and not PSEUDO_PATIENT_ID_RE.match(pid):
        reasons.append("PHI_NON_PSEUDONYMIZED_PATIENT_ID")

    for field in ("sample_id", "patient_id", "anatomical_region", "dataset_source", "diagnostic_evidence"):
        val = str(record.get(field) or "").strip()
        if not val:
            continue
        for pat_name, pat in PHI_VALUE_PATTERNS:
            if pat.search(val):
                reasons.append(f"PHI_PATTERN_DETECTED:{field}:{pat_name}")

    return reasons


def load_protected_holdout_registry(root_dir: Path) -> Dict[str, Set[str]]:
    """
    Loads protected external holdout SHA-256 hashes & sample IDs for:
    - BTXRD Center 2 (N=549) & Center 3 (N=259)
    - FracAtlas burned holdout test split
    """
    registry = load_protected_holdouts(root_dir)
    return {"hashes": registry.hashes, "sample_ids": registry.sample_ids}


def _check_modality_compatibility(
    declared_mod: str,
    image_format: str,
    image_path: str,
    dicom_modality: Optional[str],
) -> Optional[str]:
    norm_path = image_path.replace("\\", "/").lower()
    for mod in SUPPORTED_MODALITIES:
        if f"/{mod}/" in norm_path and mod != declared_mod:
            return f"MODALITY_PATH_MISMATCH:declared={declared_mod}:path={mod}"

    if image_format == "NIFTI" and declared_mod == "xray":
        return "MODALITY_FORMAT_MISMATCH:nifti_in_xray"

    if dicom_modality:
        if declared_mod == "xray" and dicom_modality in ("CT", "NM", "PT", "MR"):
            return f"MODALITY_DICOM_MISMATCH:declared=xray:dicom={dicom_modality}"
        if declared_mod == "ct" and dicom_modality in ("CR", "DX", "RG", "NM"):
            return f"MODALITY_DICOM_MISMATCH:declared=ct:dicom={dicom_modality}"
        if declared_mod == "bone_scan" and dicom_modality in ("CR", "DX", "CT"):
            return f"MODALITY_DICOM_MISMATCH:declared=bone_scan:dicom={dicom_modality}"
    return None


def discover_track_d_records(root_dir: Path, manifest_path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Loads records from an explicit live Track D manifest (`track_d_manifest.json` or `.csv`)
    OR discovers physical imaging files in `ml_service/data/raw/{xray,bone_scan,ct}/{metastasis,paget,oi,healthy}/`.
    Never ingests `.example.json` / `.example.csv` schema templates or `.gitkeep`.
    """
    manifests_dir = root_dir / "ml_service" / "data" / "manifests"
    target_manifest = manifest_path
    if target_manifest is None:
        json_candidate = manifests_dir / "track_d_manifest.json"
        csv_candidate = manifests_dir / "track_d_manifest.csv"
        if json_candidate.exists():
            target_manifest = json_candidate
        elif csv_candidate.exists():
            target_manifest = csv_candidate

    if target_manifest is not None and not target_manifest.exists():
        raise ValueError("The explicitly selected Track D manifest does not exist; raw-directory fallback is disabled.")

    if target_manifest is not None and target_manifest.exists():
        if target_manifest.suffix.lower() == ".json":
            try:
                payload = json.loads(target_manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Track D JSON manifest could not be read as valid UTF-8 JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Track D manifest must be a JSON object.")
            if payload.get("example_notice"):
                return [], "EXAMPLE_MANIFEST_IGNORED"
            _validate_track_d_manifest_schema(payload)
            return list(payload.get("records") or []), "LIVE_JSON_MANIFEST"
        if target_manifest.suffix.lower() == ".csv":
            rows: List[Dict[str, Any]] = []
            try:
                f = open(target_manifest, "r", encoding="utf-8-sig", newline="")
            except OSError as exc:
                raise ValueError("Track D CSV manifest could not be opened.") from exc
            with f:
                for row in csv.DictReader(f):
                    if str(row.get("sample_id") or "").startswith("EX_"):
                        continue
                    dims = {
                        "width": int(row["image_width"]) if str(row.get("image_width") or "").isdigit() else 0,
                        "height": int(row["image_height"]) if str(row.get("image_height") or "").isdigit() else 0,
                    }
                    conflict_raw = str(row.get("conflicting_diagnosis_indicator") or "").strip().lower()
                    rows.append({
                        **row,
                        "image_dimensions": dims,
                        "image_bit_depth": int(row["image_bit_depth"]) if str(row.get("image_bit_depth") or "").isdigit() else 0,
                        "conflicting_diagnosis_indicator": conflict_raw in ("true", "1", "yes"),
                    })
            csv_payload = {
                "manifest_version": "2.0.0",
                "track": "track_d",
                "ontology_version": ONTOLOGY_VERSION,
                "records": rows,
            }
            _validate_track_d_manifest_schema(csv_payload)
            return rows, "LIVE_CSV_MANIFEST"
        raise ValueError("Track D manifest must use .json or .csv format.")

    # Physical directory discovery under ml_service/data/raw/{modality}/{class}/
    raw_dir = root_dir / "ml_service" / "data" / "raw"
    discovered: List[Dict[str, Any]] = []
    if raw_dir.exists():
        for mod in SUPPORTED_MODALITIES:
            for cls in TARGET_CLASSES:
                folder = raw_dir / mod / cls
                if not folder.exists():
                    continue
                for file_path in sorted(folder.rglob("*")):
                    if not file_path.is_file() or file_path.name.startswith("."):
                        continue
                    rel = file_path.relative_to(root_dir).as_posix()
                    stem = file_path.stem
                    pid = stem.rsplit("_", 1)[0] if stem.startswith(("PT_", "SUBJ_")) and "_" in stem[3:] else ""
                    ont_cls = TARGET_SHORT_TO_ONTOLOGY[cls].value
                    discovered.append({
                        "sample_id": f"{mod.upper()}_{cls.upper()}_{stem[:24]}",
                        "patient_id": pid,
                        "source_diagnosis": ont_cls,
                        "diagnostic_class": cls,
                        "imaging_modality": mod,
                        "image_path": rel,
                        "image_format": "",
                        "image_dimensions": {"width": 0, "height": 0},
                        "image_bit_depth": 0,
                        "anatomical_region": "",
                        "institution_id": "",
                        "dataset_source": "",
                        "license_permissions": "",
                        "ethics_approval_ref": "",
                        "deidentification_status": "PENDING",
                        "qc_status": "PENDING",
                        "split_assignment": "unassigned",
                        "source_report_ref": "",
                        "diagnostic_evidence_category": "",
                        "histopathology_confirmation_status": "PENDING",
                        "radiology_report_confirmation_status": "PENDING",
                        "clinical_adjudication_status": "PENDING_ADJUDICATION",
                        "reviewer_pseudo_id": "",
                        "review_date": "",
                        "conflicting_diagnosis_indicator": False,
                        "label_provenance": "",
                        "label_ontology_version": ONTOLOGY_VERSION,
                    })
        # Also detect files placed in legacy raw/{class}/ folders without modality separation
        for cls in TARGET_CLASSES:
            legacy_folder = raw_dir / cls
            if not legacy_folder.exists():
                continue
            for file_path in sorted(legacy_folder.rglob("*")):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue
                rel = file_path.relative_to(root_dir).as_posix()
                stem = file_path.stem
                pid = stem.rsplit("_", 1)[0] if stem.startswith(("PT_", "SUBJ_")) and "_" in stem[3:] else ""
                ont_cls = TARGET_SHORT_TO_ONTOLOGY[cls].value
                discovered.append({
                    "sample_id": f"UNMOD_{cls.upper()}_{stem[:24]}",
                    "patient_id": pid,
                    "source_diagnosis": ont_cls,
                    "diagnostic_class": cls,
                    "imaging_modality": "",
                    "image_path": rel,
                    "image_format": "",
                    "image_dimensions": {"width": 0, "height": 0},
                    "image_bit_depth": 0,
                    "anatomical_region": "",
                    "institution_id": "",
                    "dataset_source": "",
                    "license_permissions": "",
                    "ethics_approval_ref": "",
                    "deidentification_status": "PENDING",
                    "qc_status": "PENDING",
                    "split_assignment": "unassigned",
                    "source_report_ref": "",
                    "diagnostic_evidence_category": "",
                    "histopathology_confirmation_status": "PENDING",
                    "radiology_report_confirmation_status": "PENDING",
                    "clinical_adjudication_status": "PENDING_ADJUDICATION",
                    "reviewer_pseudo_id": "",
                    "review_date": "",
                    "conflicting_diagnosis_indicator": False,
                    "label_provenance": "",
                    "label_ontology_version": ONTOLOGY_VERSION,
                })
    return discovered, "DISK_RAW_SCAN"


def _validate_track_d_manifest_schema(payload: Dict[str, Any]) -> None:
    """Validate a live manifest against the checked-in Draft 2020-12 contract.

    Error details intentionally include only JSON paths and schema keywords. They
    never include invalid values, which may contain PHI or other sensitive data.
    """
    try:
        schema = json.loads(TRACK_D_MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Track D manifest schema is unavailable or invalid; intake is blocked.") from exc

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: (tuple(str(item) for item in error.absolute_path), error.validator or ""),
    )
    if errors:
        safe_errors = []
        for error in errors[:50]:
            json_path = "$" + "".join(
                f"[{item}]" if isinstance(item, int) else f".{item}"
                for item in error.absolute_path
            )
            safe_errors.append(f"{json_path} ({error.validator or 'schema'})")
        remaining = len(errors) - len(safe_errors)
        suffix = f"; and {remaining} additional schema violations" if remaining else ""
        raise ValueError(
            f"Track D manifest failed Draft 2020-12 validation at {len(errors)} location(s): "
            + ", ".join(safe_errors)
            + suffix
        )


def validate_track_d_dataset(
    root_dir: Path,
    records: Optional[List[Dict[str, Any]]] = None,
    manifest_path: Optional[Path] = None,
    near_duplicate_threshold: int = DEFAULT_NEAR_DUPLICATE_HAMMING,
    copy_quarantined_files: bool = False,
    protected_holdout_registry: Optional[ProtectedHoldoutRegistry] = None,
) -> Dict[str, Any]:
    """
    Runs full Stage 33/34 data quality, PHI, centralized clinical ontology adjudication,
    healthy-control eligibility, provenance, duplicate, annotation, and external-holdout validation.
    """
    if records is None:
        records, discovery_mode = discover_track_d_records(root_dir, manifest_path=manifest_path)
        manifest_schema_validation = (
            "VALID_DRAFT_2020_12" if discovery_mode in ("LIVE_JSON_MANIFEST", "LIVE_CSV_MANIFEST")
            else "NOT_APPLICABLE"
        )
    else:
        discovery_mode = "EXPLICIT_RECORDS"
        manifest_schema_validation = "NOT_APPLICABLE_EXPLICIT_RECORDS"

    holdout_reg = protected_holdout_registry or load_protected_holdouts(root_dir)
    evaluated: List[Dict[str, Any]] = []
    quarantined: List[Dict[str, Any]] = []
    passed: List[Dict[str, Any]] = []

    sha_groups: Dict[str, List[int]] = defaultdict(list)
    dhash_items: List[Tuple[int, str]] = []
    patient_classes: Dict[str, Set[str]] = defaultdict(set)
    missing_metadata_fields_count = 0

    # Pre-pass: collect all patient-level source/target diagnoses for cross-record conflict adjudication
    for raw_rec in records:
        pid_pre = str(raw_rec.get("patient_id") or "").strip()
        if not pid_pre:
            continue
        diag_token = normalize_diagnosis_term(
            str(raw_rec.get("source_diagnosis") or raw_rec.get("diagnostic_class") or "")
        )
        if diag_token:
            patient_classes[pid_pre].add(diag_token)

    adjudication_counts: Dict[str, int] = {s.value: 0 for s in AdjudicationStatus}
    accepted_by_class: Dict[str, int] = {c.value: 0 for c in OntologyTargetClass}
    healthy_control_eligibility_failures = 0
    provenance_complete_count = 0

    for raw_rec in records:
        rec = dict(raw_rec)
        reasons: List[str] = []

        # 1. Check required 16 manifest fields presence
        missing_fields: List[str] = []
        for field in REQUIRED_MANIFEST_FIELDS:
            val = rec.get(field)
            if val is None:
                missing_fields.append(field)
            elif isinstance(val, str):
                sval = val.strip().lower()
                if field == "qc_status":
                    if sval not in ("pending", "passed", "quarantined"):
                        missing_fields.append(field)
                elif field == "split_assignment":
                    if sval not in ("unassigned", "train", "val", "test", "external_test", "quarantined"):
                        missing_fields.append(field)
                elif sval in INVALID_PLACEHOLDER_TOKENS:
                    missing_fields.append(field)
            elif isinstance(val, dict) and (not val.get("width") or not val.get("height")):
                missing_fields.append(field)
            elif isinstance(val, (int, float)) and val <= 0 and field == "image_bit_depth":
                missing_fields.append(field)

        metadata_only_missing = [
            f for f in missing_fields if f not in ("image_format", "image_dimensions", "image_bit_depth")
        ]
        if metadata_only_missing:
            missing_metadata_fields_count += len(metadata_only_missing)
            reasons.extend(f"MISSING_METADATA:{f}" for f in metadata_only_missing)

        # 2. PHI & Pseudonymization Guard
        reasons.extend(detect_phi_violations(rec))

        # 3. Stage 34 Centralized Ontology & Diagnostic Label Adjudication
        pid_str = str(rec.get("patient_id") or "").strip()
        adj = adjudicate_diagnostic_label(
            rec,
            patient_conflicting_diagnoses=patient_classes.get(pid_str) if pid_str else None,
        )
        rec["adjudication_record"] = adj
        adj_status = adj["adjudication_status"]
        adjudication_counts[adj_status] = adjudication_counts.get(adj_status, 0) + 1

        prov_ok, _prov_missing = check_label_provenance_completeness(rec)
        if prov_ok:
            provenance_complete_count += 1

        if adj["healthy_eligibility_failures"]:
            healthy_control_eligibility_failures += 1
            reasons.extend(adj["healthy_eligibility_failures"])

        if adj["provenance_missing_fields"]:
            missing_metadata_fields_count += len(adj["provenance_missing_fields"])
            reasons.extend(adj["provenance_missing_fields"])

        canonical_cls = adj["short_class_code"] or ""
        if adj_status == AdjudicationStatus.ACCEPTED.value and canonical_cls:
            rec["diagnostic_class"] = canonical_cls
        elif adj_status == AdjudicationStatus.UNCERTAIN.value:
            reasons.append("UNCERTAIN_DIAGNOSTIC_LABEL")
            reasons.append("ADJUDICATION_UNCERTAIN")
        elif adj_status == AdjudicationStatus.OUT_OF_SCOPE.value:
            reasons.append("ADJUDICATION_OUT_OF_SCOPE")
            oos_prefix = str(adj.get("reason") or "").split(":", 1)[0].strip()
            if oos_prefix:
                reasons.append(oos_prefix)
        elif adj_status == AdjudicationStatus.REQUIRES_ADJUDICATION.value:
            reasons.append("ADJUDICATION_REQUIRES_HUMAN_REVIEW")
            if bool(rec.get("conflicting_diagnosis_indicator")) or (
                pid_str and len(patient_classes.get(pid_str, set())) > 1
            ):
                reasons.append("CONFLICTING_PATIENT_DIAGNOSIS")
        elif adj_status == AdjudicationStatus.REJECTED.value:
            reasons.append("ADJUDICATION_REJECTED")
            rej_prefix = str(adj.get("reason") or "").split(":", 1)[0].strip()
            if rej_prefix and rej_prefix.isupper():
                reasons.append(rej_prefix)

        # 4. Modality Validation (never mix modalities)
        raw_mod = str(rec.get("imaging_modality") or "").strip().lower()
        canonical_mod = MODALITY_ALIASES.get(raw_mod, "")
        if not canonical_mod:
            reasons.append(f"INVALID_MODALITY:{raw_mod or 'empty'}")
        else:
            rec["imaging_modality"] = canonical_mod

        # 5. Governance / Permissions / Ethics / De-identification Verification
        deid_status = str(rec.get("deidentification_status") or "").strip().upper()
        if deid_status != "VERIFIED_PSEUDONYMIZED":
            reasons.append(f"DEIDENTIFICATION_UNVERIFIED:{deid_status or 'MISSING'}")

        perms = str(rec.get("license_permissions") or "").strip().lower()
        if perms in INVALID_PLACEHOLDER_TOKENS or "prohibit" in perms or "denied" in perms:
            reasons.append("MISSING_OR_INVALID_PERMISSIONS")

        ethics_ref = str(rec.get("ethics_approval_ref") or "").strip().lower()
        if ethics_ref in INVALID_PLACEHOLDER_TOKENS:
            reasons.append("MISSING_ETHICS_APPROVAL_REF")

        # 6. Protected Holdout Site / Sample ID Guard (BTXRD Center 2 & 3, FracAtlas Test)
        reasons.extend(holdout_reg.record_violations(rec))

        # 7. Physical Image Quality Inspection
        rel_or_abs = Path(str(rec.get("image_path") or ""))
        img_file = rel_or_abs if rel_or_abs.is_absolute() else (root_dir / rel_or_abs)
        img_info = inspect_image_file(img_file)
        if not img_info["valid"]:
            reasons.append(str(img_info["error"] or "INVALID_IMAGE_FILE"))
        else:
            declared_fmt = str(rec.get("image_format") or "").strip().upper()
            rec["image_format"] = declared_fmt or img_info["image_format"]
            if rec["image_format"] != img_info["image_format"]:
                reasons.append(f"FORMAT_HEADER_MISMATCH:declared={rec['image_format']}:actual={img_info['image_format']}")
            rec["image_dimensions"] = {
                "width": img_info["width"],
                "height": img_info["height"],
                "depth": img_info["depth"],
            }
            if not rec.get("image_bit_depth"):
                rec["image_bit_depth"] = img_info["bit_depth"]
            rec["sha256"] = img_info["sha256"]
            if img_info["dhash"]:
                rec["perceptual_hash"] = img_info["dhash"]

            if img_info["sha256"] in holdout_reg.hashes:
                reasons.append("PROTECTED_EXTERNAL_HOLDOUT_HASH_MATCH")

            if canonical_mod:
                mod_err = _check_modality_compatibility(
                    canonical_mod,
                    img_info["image_format"],
                    str(rec.get("image_path") or ""),
                    img_info["dicom_modality"],
                )
                if mod_err:
                    reasons.append(mod_err)

        # 8. Optional / Required Annotation Completeness Check
        ann_status = str(rec.get("annotation_status") or "").strip().upper()
        ann_path = str(rec.get("annotation_path") or "").strip()
        if ann_status == "REQUIRED" and not ann_path:
            reasons.append("INCOMPLETE_ANNOTATION:missing_path")
        elif ann_path:
            ann_file = Path(ann_path) if Path(ann_path).is_absolute() else (root_dir / ann_path)
            if not ann_file.exists() or ann_file.stat().st_size == 0:
                reasons.append("INCOMPLETE_ANNOTATION:missing_or_empty_file")

        idx = len(evaluated)
        rec["rejection_reasons"] = sorted(set(reasons))
        evaluated.append(rec)

        if rec.get("sha256"):
            sha_groups[str(rec["sha256"])].append(idx)
        if rec.get("perceptual_hash"):
            dhash_items.append((idx, str(rec["perceptual_hash"])))

    # 9. Cross-Record Conflict, Exact Duplicate, and Near-Duplicate Detection
    exact_duplicate_count = 0
    near_duplicate_count = 0
    duplicate_patient_conflicts = sum(1 for classes in patient_classes.values() if len(classes) > 1)
    conflicting_diagnoses_count = sum(
        1
        for rec in evaluated
        if "CONFLICTING_PATIENT_DIAGNOSIS" in rec["rejection_reasons"]
        or bool(rec.get("conflicting_diagnosis_indicator"))
    )

    for sha, indices in sha_groups.items():
        if len(indices) > 1:
            unique_pids = {str(evaluated[i].get("patient_id") or "") for i in indices}
            exact_duplicate_count += len(indices) - 1
            if len(unique_pids) > 1:
                for i in indices:
                    evaluated[i]["rejection_reasons"] = sorted(
                        set(evaluated[i]["rejection_reasons"] + ["DUPLICATE_EXACT_SHA256_CROSS_PATIENT"])
                    )
            else:
                for i in indices[1:]:
                    evaluated[i]["rejection_reasons"] = sorted(
                        set(evaluated[i]["rejection_reasons"] + ["DUPLICATE_EXACT_SHA256"])
                    )

    for a_pos in range(len(dhash_items)):
        idx_a, hash_a = dhash_items[a_pos]
        for b_pos in range(a_pos + 1, len(dhash_items)):
            idx_b, hash_b = dhash_items[b_pos]
            if evaluated[idx_a].get("sha256") and evaluated[idx_a].get("sha256") == evaluated[idx_b].get("sha256"):
                continue
            dist = compute_hamming_distance(hash_a, hash_b)
            if dist is not None and dist <= near_duplicate_threshold:
                near_duplicate_count += 1
                pid_a = str(evaluated[idx_a].get("patient_id") or "")
                pid_b = str(evaluated[idx_b].get("patient_id") or "")
                code = (
                    "DUPLICATE_NEAR_PERCEPTUAL_CROSS_PATIENT"
                    if pid_a != pid_b
                    else "DUPLICATE_NEAR_PERCEPTUAL"
                )
                evaluated[idx_b]["rejection_reasons"] = sorted(set(evaluated[idx_b]["rejection_reasons"] + [code]))

    # Finalize QC status & optionally copy quarantined files safely (never deleting originals)
    for rec in evaluated:
        if rec["rejection_reasons"]:
            rec["qc_status"] = "QUARANTINED"
            rec["split_assignment"] = "quarantined"
            quarantined.append(rec)
            if copy_quarantined_files and rec.get("image_path"):
                src = Path(rec["image_path"])
                src_abs = src if src.is_absolute() else (root_dir / src)
                if src_abs.exists() and src_abs.is_file():
                    mod_dir = rec.get("imaging_modality") if rec.get("imaging_modality") in SUPPORTED_MODALITIES else "xray"
                    cls_dir = rec.get("diagnostic_class") if rec.get("diagnostic_class") in TARGET_CLASSES else "healthy"
                    dest_dir = root_dir / "ml_service" / "data" / "quarantine" / mod_dir / cls_dir
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_abs, dest_dir / src_abs.name)
        else:
            rec["qc_status"] = "PASSED"
            passed.append(rec)
            cand_cls = rec["adjudication_record"].get("candidate_target_class")
            if cand_cls in accepted_by_class:
                accepted_by_class[cand_cls] += 1

    total_eval = len(evaluated)
    provenance_pct = round(100.0 * provenance_complete_count / total_eval, 2) if total_eval > 0 else 0.0
    rejection_reason_counts: Dict[str, int] = defaultdict(int)
    for record in quarantined:
        for reason in record.get("rejection_reasons", []):
            rejection_reason_counts[reason] += 1

    return {
        "discovery_mode": discovery_mode,
        "manifest_schema_validation": manifest_schema_validation,
        "ontology_version": ONTOLOGY_VERSION,
        "total_inspected": total_eval,
        "passed_count": len(passed),
        "quarantined_count": len(quarantined),
        "missing_metadata_fields_count": missing_metadata_fields_count,
        "exact_duplicate_count": exact_duplicate_count,
        "near_duplicate_count": near_duplicate_count,
        "conflicting_patient_count": duplicate_patient_conflicts,
        "conflicting_diagnoses_count": conflicting_diagnoses_count,
        "healthy_control_eligibility_failures": healthy_control_eligibility_failures,
        "label_provenance_completeness_pct": provenance_pct,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "protected_holdout_registry": {
            "sample_id_count": len(holdout_reg.sample_ids),
            "sha256_count": len(holdout_reg.hashes),
            "source_count": len(holdout_reg.sources),
            "sources": sorted(holdout_reg.sources),
        },
        "adjudication_counts": adjudication_counts,
        "accepted_by_class": accepted_by_class,
        "records": evaluated,
        "passed_records": passed,
        "quarantined_records": quarantined,
    }


def generate_leakage_safe_splits(
    passed_records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    holdout_institutions: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Deterministic patient-level split generator keeping modalities strictly separated
    (`xray`, `bone_scan`, `ct` are partitioned independently).
    Never synthesizes samples or permits patient/hash overlap across splits.
    """
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    holdout_insts = normalize_holdout_sites(holdout_institutions or ())
    rng = random.Random(seed)

    # First group globally by patient_id to guarantee a patient with multi-modality studies
    # never lands in train for one modality and test for another.
    # Simultaneously track modality pools strictly separately in reports.
    patient_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    patient_institutions: Dict[str, Set[str]] = defaultdict(set)
    patient_strata: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    for rec in passed_records:
        pid = str(rec["patient_id"])
        patient_records[pid].append(rec)
        institution = str(rec.get("institution_id") or "").strip()
        if institution:
            patient_institutions[pid].add(institution)
        # Use an order-independent patient stratum when a patient has multiple modalities.
        patient_strata[pid].add((str(rec["imaging_modality"]), str(rec["diagnostic_class"])))

    patient_split: Dict[str, str] = {}
    strata_patients: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for pid in sorted(patient_records.keys()):
        patient_sites = patient_institutions[pid]
        if any(
            site in holdout_insts or "".join(ch.lower() for ch in site if ch.isalnum()) in holdout_insts
            for site in patient_sites
        ):
            patient_split[pid] = "external_test"
        else:
            strata_patients[min(patient_strata[pid])].append(pid)

    for stratum_key in sorted(strata_patients.keys()):
        pids = sorted(strata_patients[stratum_key])
        rng.shuffle(pids)
        n = len(pids)
        if n == 1:
            patient_split[pids[0]] = "train"
        elif n == 2:
            patient_split[pids[0]] = "train"
            patient_split[pids[1]] = "val"
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            n_test = n - n_train - n_val
            if n_test <= 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(1, n_val - 1)
            for p in pids[:n_train]:
                patient_split[p] = "train"
            for p in pids[n_train : n_train + n_val]:
                patient_split[p] = "val"
            for p in pids[n_train + n_val :]:
                patient_split[p] = "test"

    assigned: List[Dict[str, Any]] = []
    for rec in passed_records:
        updated = dict(rec)
        updated["split_assignment"] = patient_split[str(rec["patient_id"])]
        assigned.append(updated)

    leakage_audit = verify_cross_split_leakage(assigned)
    if not leakage_audit["is_clean"]:
        raise RuntimeError(f"Cross-split leakage detected: {leakage_audit['violations']}")

    distribution: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "external_test": 0}
    by_modality: Dict[str, Dict[str, int]] = {
        m: {"train": 0, "val": 0, "test": 0, "external_test": 0} for m in SUPPORTED_MODALITIES
    }
    for rec in assigned:
        sp = rec["split_assignment"]
        mod = rec["imaging_modality"]
        distribution[sp] = distribution.get(sp, 0) + 1
        if mod in by_modality:
            by_modality[mod][sp] = by_modality[mod].get(sp, 0) + 1

    return {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_assigned": len(assigned),
        "distribution": distribution,
        "by_modality": by_modality,
        "leakage_audit": leakage_audit,
        "records": assigned,
    }


def verify_cross_split_leakage(
    records: List[Dict[str, Any]],
    near_duplicate_threshold: int = DEFAULT_NEAR_DUPLICATE_HAMMING,
) -> Dict[str, Any]:
    """
    Audits assigned splits for:
    1) Patient ID overlap across train/val/test/external_test.
    2) Exact SHA-256 overlap across different splits.
    3) Near-duplicate perceptual hash collisions across different splits.
    """
    split_patients: Dict[str, Set[str]] = defaultdict(set)
    split_hashes: Dict[str, Set[str]] = defaultdict(set)
    split_dhashes: List[Tuple[str, str, str]] = []

    for rec in records:
        sp = str(rec.get("split_assignment") or rec.get("split") or "").strip()
        if not sp or sp in ("unassigned", "quarantined"):
            continue
        pid = str(rec.get("patient_id") or "").strip()
        sha = str(rec.get("sha256") or "").strip().upper()
        dh = str(rec.get("perceptual_hash") or "").strip()
        sid = str(rec.get("sample_id") or "")
        if pid:
            split_patients[sp].add(pid)
        if sha:
            split_hashes[sp].add(sha)
        if dh:
            split_dhashes.append((sp, sid, dh))

    violations: List[str] = []
    splits = sorted(split_patients.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            p_overlap = split_patients[s1].intersection(split_patients[s2])
            if p_overlap:
                violations.append(f"PATIENT_OVERLAP:{s1}<->{s2}:{len(p_overlap)}")
            h_overlap = split_hashes[s1].intersection(split_hashes[s2])
            if h_overlap:
                violations.append(f"EXACT_HASH_CROSS_SPLIT:{s1}<->{s2}:{len(h_overlap)}")

    for i in range(len(split_dhashes)):
        sp1, sid1, dh1 = split_dhashes[i]
        for j in range(i + 1, len(split_dhashes)):
            sp2, sid2, dh2 = split_dhashes[j]
            if sp1 == sp2:
                continue
            dist = compute_hamming_distance(dh1, dh2)
            if dist is not None and dist <= near_duplicate_threshold:
                violations.append(f"NEAR_DUPLICATE_CROSS_SPLIT:{sp1}({sid1})<->{sp2}({sid2}):dist={dist}")

    return {"is_clean": len(violations) == 0, "violations": violations}


def evaluate_track_d_readiness(
    validation_result: Dict[str, Any],
    split_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates the 7 machine-readable readiness gates for Track D.
    Returns explicit `NOT_AVAILABLE` when zero valid samples exist.
    Never invents minimum clinical sample counts.
    """
    passed: List[Dict[str, Any]] = validation_result["passed_records"]
    quarantined: List[Dict[str, Any]] = validation_result["quarantined_records"]
    total_inspected = validation_result["total_inspected"]
    assigned_records: List[Dict[str, Any]] = split_result.get("records", [])

    class_counts = {c: 0 for c in TARGET_CLASSES}
    modality_counts = {m: 0 for m in SUPPORTED_MODALITIES}
    class_modality_matrix = {m: {c: 0 for c in TARGET_CLASSES} for m in SUPPORTED_MODALITIES}
    patients: Set[str] = set()
    institutions: Set[str] = set()
    sources: Set[str] = set()
    ethics_refs: Set[str] = set()
    permissions_list: Set[str] = set()

    for rec in passed:
        cls = rec["diagnostic_class"]
        mod = rec["imaging_modality"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
        if mod in class_modality_matrix and cls in class_modality_matrix[mod]:
            class_modality_matrix[mod][cls] += 1
        patients.add(str(rec["patient_id"]))
        institutions.add(str(rec["institution_id"]))
        sources.add(str(rec["dataset_source"]))
        ethics_refs.add(str(rec["ethics_approval_ref"]))
        permissions_list.add(str(rec["license_permissions"]))

    has_records = len(passed) > 0
    provenance_ok = has_records and all(bool(r.get("dataset_source")) and bool(r.get("institution_id")) for r in passed)
    permissions_ok = has_records and all(
        str(r.get("license_permissions") or "").strip().lower() not in INVALID_PLACEHOLDER_TOKENS for r in passed
    )
    deid_ok = has_records and all(
        str(r.get("deidentification_status") or "").strip().upper() == "VERIFIED_PSEUDONYMIZED"
        and not detect_phi_violations(r)
        for r in passed
    )
    adj_counts = validation_result.get("adjudication_counts") or {s.value: 0 for s in AdjudicationStatus}
    accepted_by_class = validation_result.get("accepted_by_class") or {c.value: 0 for c in OntologyTargetClass}
    labels_ok = (
        has_records
        and validation_result["conflicting_patient_count"] == 0
        and adj_counts.get(AdjudicationStatus.UNCERTAIN.value, 0) == 0
        and adj_counts.get(AdjudicationStatus.REQUIRES_ADJUDICATION.value, 0) == 0
        and adj_counts.get(AdjudicationStatus.REJECTED.value, 0) == 0
        and all(
            (r.get("adjudication_record") or {}).get("adjudication_status") == AdjudicationStatus.ACCEPTED.value
            and str(r.get("ethics_approval_ref") or "").strip().lower() not in INVALID_PLACEHOLDER_TOKENS
            and check_label_provenance_completeness(r)[0]
            for r in passed
        )
    )
    dist = split_result.get("distribution", {})
    leakage_clean = bool(split_result.get("leakage_audit", {}).get("is_clean", False))
    split_ok = (
        has_records
        and leakage_clean
        and dist.get("train", 0) > 0
        and dist.get("val", 0) > 0
        and dist.get("test", 0) > 0
    )
    quality_ok = has_records and len(quarantined) == 0
    coverage_ok = all(class_counts[c] > 0 for c in TARGET_CLASSES)

    gates = {
        "provenance_documented": provenance_ok,
        "permissions_verified": permissions_ok,
        "deidentification_satisfied": deid_ok,
        "diagnostic_labels_documented": labels_ok,
        "patient_level_split_integrity": split_ok,
        "image_quality_verified": quality_ok,
        "class_and_modality_coverage": coverage_ok,
    }

    missing_classes = [c for c in TARGET_CLASSES if class_counts[c] == 0]
    blockers: List[str] = []
    if total_inspected == 0:
        blockers.append("Zero real Track D clinical dataset records or files found in ml_service/data/raw/ or manifests.")
    if missing_classes:
        blockers.append(f"Missing verified diagnostic classes: {', '.join(missing_classes)}.")
    if not provenance_ok:
        blockers.append("Documented institutional provenance (dataset_source & institution_id) is incomplete or absent.")
    if not permissions_ok:
        blockers.append("Verified dataset license / Data Use Agreement (DUA) permissions are absent.")
    if not ethics_refs:
        blockers.append("Institutional Ethics Committee (IEC/IRB) approval references are missing.")
    if has_records and not labels_ok:
        blockers.append("Diagnostic label ontology adjudication or clinical provenance verification is incomplete.")
    if len(quarantined) > 0:
        blockers.append(f"{len(quarantined)} sample(s) quarantined for quality, PHI, duplicate, or label/adjudication issues.")
    if has_records and not split_ok:
        blockers.append("Non-empty leakage-safe train/val/test patient splits are not yet satisfied.")

    if total_inspected == 0:
        readiness_status = "NOT_AVAILABLE"
    elif all(gates.values()):
        readiness_status = "READY_FOR_TRAINING"
    else:
        readiness_status = "BLOCKED"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track": "track_d",
        "ontology_version": validation_result.get("ontology_version", ONTOLOGY_VERSION),
        "readiness_status": readiness_status,
        "ready_for_training": readiness_status == "READY_FOR_TRAINING",
        "total_inspected_samples": total_inspected,
        "valid_samples": len(passed),
        "quarantined_samples": len(quarantined),
        "unique_patients": len(patients),
        "institution_count": len(institutions),
        "institutions": sorted(institutions),
        "dataset_sources": sorted(sources),
        "permissions_verified": sorted(permissions_list),
        "ethics_references": sorted(ethics_refs),
        "class_counts": class_counts,
        "modality_counts": modality_counts,
        "class_modality_matrix": class_modality_matrix,
        "accepted_by_class": accepted_by_class,
        "adjudication_summary": adj_counts,
        "healthy_control_eligibility_failures": validation_result.get("healthy_control_eligibility_failures", 0),
        "conflicting_diagnoses_count": validation_result.get("conflicting_diagnoses_count", 0),
        "label_provenance_completeness_pct": validation_result.get("label_provenance_completeness_pct", 0.0),
        "missing_metadata_count": validation_result["missing_metadata_fields_count"],
        "duplicate_samples_count": validation_result["exact_duplicate_count"] + validation_result["near_duplicate_count"],
        "exact_duplicate_count": validation_result["exact_duplicate_count"],
        "near_duplicate_count": validation_result["near_duplicate_count"],
        "conflicting_patient_count": validation_result["conflicting_patient_count"],
        "split_distribution": {
            "train": dist.get("train", 0),
            "val": dist.get("val", 0),
            "test": dist.get("test", 0),
            "external_test": dist.get("external_test", 0),
        },
        "split_by_modality": split_result.get("by_modality", {}),
        "leakage_audit": split_result.get("leakage_audit", {"is_clean": True, "violations": []}),
        "gates": gates,
        "blockers": blockers,
        "clinical_sample_size_justification_note": (
            "No arbitrary minimum clinical sample count is fabricated in software; "
            "final statistical power and institutional cohort sizes require documented biostatistical and IEC approval."
        ),
        "assigned_records_count": len(assigned_records),
    }


def write_track_d_artifacts(
    root_dir: Path,
    validation_result: Dict[str, Any],
    split_result: Dict[str, Any],
    readiness_report: Dict[str, Any],
) -> Dict[str, str]:
    """Writes standardized JSON and CSV Track D intake, quarantine, split, and readiness artifacts."""
    reports_dir = root_dir / "ml_service" / "data" / "reports"
    quarantine_dir = root_dir / "ml_service" / "data" / "quarantine"
    splits_dir = root_dir / "ml_service" / "data" / "splits"
    for d in (reports_dir, quarantine_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    quality_path = reports_dir / "track_d_quality_report.json"
    quarantine_report_path = reports_dir / "track_d_quarantine_report.json"
    readiness_path = reports_dir / "track_d_readiness_report.json"
    quarantine_json_path = quarantine_dir / "quarantine_manifest.json"
    quarantine_csv_path = quarantine_dir / "quarantine_manifest.csv"
    split_csv_path = splits_dir / "track_d_split_manifest.csv"
    split_summary_path = splits_dir / "track_d_split_summary.json"

    quarantined_records = validation_result["quarantined_records"]
    reason_histogram: Dict[str, int] = defaultdict(int)
    for rec in quarantined_records:
        for reason in rec.get("rejection_reasons", []):
            reason_histogram[reason] += 1

    quality_payload = {
        "generated_at": readiness_report["generated_at"],
        "discovery_mode": validation_result["discovery_mode"],
        "manifest_schema_validation": validation_result.get("manifest_schema_validation", "NOT_TESTED"),
        "protected_holdout_registry": validation_result.get("protected_holdout_registry", {}),
        "total_inspected": validation_result["total_inspected"],
        "passed_count": validation_result["passed_count"],
        "quarantined_count": validation_result["quarantined_count"],
        "missing_metadata_fields_count": validation_result["missing_metadata_fields_count"],
        "exact_duplicate_count": validation_result["exact_duplicate_count"],
        "near_duplicate_count": validation_result["near_duplicate_count"],
        "rejection_reason_counts": validation_result.get("rejection_reason_counts", {}),
        "conflicting_patient_count": validation_result["conflicting_patient_count"],
        "rejection_reason_counts": dict(sorted(reason_histogram.items())),
    }
    quality_path.write_text(json.dumps(quality_payload, indent=2), encoding="utf-8")

    quarantine_payload = {
        "generated_at": readiness_report["generated_at"],
        "quarantined_count": len(quarantined_records),
        "reason_counts": dict(sorted(reason_histogram.items())),
        "records": quarantined_records,
    }
    quarantine_report_path.write_text(json.dumps(quarantine_payload, indent=2), encoding="utf-8")
    quarantine_json_path.write_text(json.dumps(quarantine_payload, indent=2), encoding="utf-8")

    with open(quarantine_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "patient_id", "diagnostic_class", "imaging_modality", "image_path", "rejection_reasons"])
        for rec in quarantined_records:
            writer.writerow([
                rec.get("sample_id", ""),
                rec.get("patient_id", ""),
                rec.get("diagnostic_class", ""),
                rec.get("imaging_modality", ""),
                rec.get("image_path", ""),
                ";".join(rec.get("rejection_reasons", [])),
            ])

    assigned = split_result.get("records", [])
    with open(split_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "patient_id",
            "diagnostic_class",
            "imaging_modality",
            "institution_id",
            "dataset_source",
            "image_path",
            "sha256",
            "split_assignment",
        ])
        for rec in assigned:
            writer.writerow([
                rec.get("sample_id", ""),
                rec.get("patient_id", ""),
                rec.get("diagnostic_class", ""),
                rec.get("imaging_modality", ""),
                rec.get("institution_id", ""),
                rec.get("dataset_source", ""),
                rec.get("image_path", ""),
                rec.get("sha256", ""),
                rec.get("split_assignment", ""),
            ])

    split_summary_payload = {
        "generated_at": readiness_report["generated_at"],
        "seed": split_result.get("seed", 42),
        "ratios": split_result.get("ratios", {"train": 0.70, "val": 0.15, "test": 0.15}),
        "total_assigned": split_result.get("total_assigned", 0),
        "distribution": split_result.get("distribution", {}),
        "by_modality": split_result.get("by_modality", {}),
        "leakage_audit": split_result.get("leakage_audit", {"is_clean": True, "violations": []}),
    }
    split_summary_path.write_text(json.dumps(split_summary_payload, indent=2), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness_report, indent=2), encoding="utf-8")

    adjudication_audit_path = reports_dir / "track_d_adjudication_audit.json"
    adjudication_payload = {
        "generated_at": readiness_report["generated_at"],
        "ontology_version": validation_result.get("ontology_version", ONTOLOGY_VERSION),
        "total_inspected": validation_result["total_inspected"],
        "accepted_by_class": validation_result.get("accepted_by_class", {}),
        "adjudication_summary": validation_result.get("adjudication_counts", {}),
        "healthy_control_eligibility_failures": validation_result.get("healthy_control_eligibility_failures", 0),
        "conflicting_diagnoses_count": validation_result.get("conflicting_diagnoses_count", 0),
        "label_provenance_completeness_pct": validation_result.get("label_provenance_completeness_pct", 0.0),
        "records": [
            {
                "sample_id": r.get("sample_id", ""),
                "patient_id": r.get("patient_id", ""),
                "qc_status": r.get("qc_status", ""),
                "adjudication_record": r.get("adjudication_record", {}),
                "rejection_reasons": r.get("rejection_reasons", []),
            }
            for r in validation_result.get("records", [])
        ],
    }
    adjudication_audit_path.write_text(json.dumps(adjudication_payload, indent=2), encoding="utf-8")

    return {
        "quality_report": str(quality_path),
        "quarantine_report": str(quarantine_report_path),
        "quarantine_manifest_json": str(quarantine_json_path),
        "quarantine_manifest_csv": str(quarantine_csv_path),
        "split_manifest_csv": str(split_csv_path),
        "split_summary_json": str(split_summary_path),
        "readiness_report": str(readiness_path),
        "adjudication_audit_report": str(adjudication_audit_path),
    }


def run_track_d_intake_pipeline(
    root_dir: Path,
    manifest_path: Optional[Path] = None,
    seed: int = 42,
    holdout_institutions: Optional[Iterable[str]] = None,
    copy_quarantined_files: bool = False,
    write_outputs: bool = True,
) -> Dict[str, Any]:
    """Executes end-to-end Track D intake validation, splitting, and readiness gating."""
    validation = validate_track_d_dataset(
        root_dir=root_dir,
        manifest_path=manifest_path,
        copy_quarantined_files=copy_quarantined_files,
    )
    splits = generate_leakage_safe_splits(
        passed_records=validation["passed_records"],
        seed=seed,
        holdout_institutions=holdout_institutions,
    )
    readiness = evaluate_track_d_readiness(validation, splits)
    outputs = write_track_d_artifacts(root_dir, validation, splits, readiness) if write_outputs else {}
    return {
        "validation": validation,
        "splits": splits,
        "readiness": readiness,
        "outputs": outputs,
    }
