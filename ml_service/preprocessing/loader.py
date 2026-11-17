"""
OsteoGuard AI - Unified Medical Image Loader & Quality Engine.
Supports DICOM, PNG, JPEG, TIFF, and NIfTI with spatial metadata preservation,
orientation validation, corrupt-file rejection, and non-fatal metadata warnings.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import pydicom
from pydicom.errors import InvalidDicomError
import nibabel as nib

from ml_service.preprocessing.deidentification import sanitize_dicom_metadata
from ml_service.utils.logging import logger


class CorruptImageError(ValueError):
    """Raised when an image file is corrupt, unreadable, or invalid."""
    pass


class UnsupportedFormatError(ValueError):
    """Raised when file extension or signature is not recognized."""
    pass


@dataclass
class SpatialMetadata:
    """Preserves physical and anatomical coordinate geometry."""
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (row_spacing, col_spacing, slice_spacing)
    slice_thickness: float = 1.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction_cosines: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    original_shape: Tuple[int, ...] = (0, 0)
    original_dtype: str = "float32"


@dataclass
class LoadedMedicalImage:
    """Container holding raw loaded image, spatial metadata, and diagnostic warnings."""
    pixel_array: np.ndarray
    modality: str
    format: str
    metadata: Dict[str, Any]
    spatial: SpatialMetadata
    source_path: str
    checksum_sha256: str
    warnings: List[str] = field(default_factory=list)


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 hash of image file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_orientation(direction_cosines: Tuple[float, ...]) -> Tuple[bool, Optional[str]]:
    """
    Validates Image Orientation Patient vector [Fx, Fy, Fz, Cx, Cy, Cz].
    Verifies row and column direction cosines are approximately orthogonal.
    """
    if len(direction_cosines) < 6:
        return False, f"Orientation vector incomplete: expected 6 elements, got {len(direction_cosines)}"

    row_vec = np.array(direction_cosines[:3], dtype=np.float64)
    col_vec = np.array(direction_cosines[3:6], dtype=np.float64)

    dot_prod = np.dot(row_vec, col_vec)
    if abs(dot_prod) > 0.05:
        return False, f"Row and column orientation vectors are non-orthogonal: dot_product={dot_prod:.4f}"

    return True, None


def validate_image_quality(arr: np.ndarray, min_dim: int = 32) -> Tuple[bool, List[str]]:
    """
    Audits array dimensions, dynamic range, and dead pixel saturation.
    """
    issues: List[str] = []

    if arr.size == 0:
        return False, ["Image array is empty (0 pixels)."]

    if np.isnan(arr).any():
        issues.append("NaN values detected in image array.")
    if np.isinf(arr).any():
        issues.append("Infinite values detected in image array.")

    # Dimension checks (at least 2D)
    if arr.ndim < 2:
        return False, [f"Invalid dimensionality: expected at least 2D, got {arr.ndim}D"]

    h, w = arr.shape[:2]
    if h < min_dim or w < min_dim:
        return False, [f"Image dimensions ({w}x{h}) below minimum threshold ({min_dim}x{min_dim})."]

    # Dynamic range check
    dynamic_range = float(np.nanmax(arr) - np.nanmin(arr))
    if dynamic_range <= 0.0:
        issues.append("Dead image detected: dynamic range is zero (all pixels identical).")

    is_valid = len([i for i in issues if "Dead" in i or "empty" in i or "below" in i]) == 0
    return is_valid, issues


def detect_modality_from_dicom(ds: pydicom.Dataset) -> str:
    """Infers modality from DICOM Modality (0008,0060) tag."""
    raw_mod = getattr(ds, "Modality", "").upper()
    if raw_mod in ["CT"]:
        return "ct"
    elif raw_mod in ["CR", "DX", "RG", "XR"]:
        return "xray"
    elif raw_mod in ["NM", "PT"]:
        return "bone_scan"
    return "unknown"


def load_dicom_file(path: Path) -> LoadedMedicalImage:
    """Reads DICOM file, handles PhotometricInterpretation, and extracts spatial info."""
    warnings: List[str] = []
    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception as e:
        raise CorruptImageError(f"Failed to parse DICOM file {path.name}: {str(e)}") from e

    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise CorruptImageError(f"Unable to decode DICOM pixel data in {path.name}: {str(e)}") from e

    # Check Photometric Interpretation
    photo_interp = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photo_interp == "MONOCHROME1":
        # Invert so 0 = black, max = white
        max_val = np.max(pixel_array)
        pixel_array = max_val - pixel_array
        warnings.append("Inverted MONOCHROME1 to standard MONOCHROME2 photometric representation.")

    # Extract metadata safely
    raw_metadata: Dict[str, Any] = {}
    for elem in ds:
        if elem.keyword and elem.VR != "OB" and elem.VR != "OW":
            try:
                raw_metadata[elem.keyword] = str(elem.value)
            except Exception:
                pass

    sanitized_metadata, _ = sanitize_dicom_metadata(raw_metadata)

    # Extract spatial parameters
    spacing = (1.0, 1.0, 1.0)
    if "PixelSpacing" in ds:
        try:
            ps = [float(x) for x in ds.PixelSpacing]
            st = float(getattr(ds, "SliceThickness", 1.0))
            spacing = (ps[0], ps[1], st)
        except Exception as e:
            warnings.append(f"Incomplete PixelSpacing metadata: {e}. Defaulting spacing to (1.0, 1.0, 1.0).")
    else:
        warnings.append("Missing PixelSpacing tag (0028,0030). Defaulting spacing to (1.0, 1.0, 1.0).")

    slice_thick = float(getattr(ds, "SliceThickness", 1.0))
    origin = (0.0, 0.0, 0.0)
    if "ImagePositionPatient" in ds:
        try:
            origin = tuple(float(x) for x in ds.ImagePositionPatient)
        except Exception:
            pass

    orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    if "ImageOrientationPatient" in ds:
        try:
            orientation = tuple(float(x) for x in ds.ImageOrientationPatient)
            valid, err = validate_orientation(orientation)
            if not valid:
                warnings.append(f"Orientation anomaly: {err}")
        except Exception as e:
            warnings.append(f"ImageOrientationPatient parsing error: {e}")

    spatial = SpatialMetadata(
        voxel_spacing=spacing,
        slice_thickness=slice_thick,
        origin=origin,
        direction_cosines=orientation,
        original_shape=pixel_array.shape,
        original_dtype=str(ds.pixel_array.dtype),
    )

    modality = detect_modality_from_dicom(ds)
    if modality == "unknown":
        warnings.append("DICOM Modality tag (0008,0060) is missing or unmapped; inferred as 'unknown'.")

    # Pass through quality validator
    valid_q, q_issues = validate_image_quality(pixel_array)
    if not valid_q:
        raise CorruptImageError(f"DICOM quality rejection for {path.name}: {q_issues}")
    if q_issues:
        warnings.extend(q_issues)

    return LoadedMedicalImage(
        pixel_array=pixel_array,
        modality=modality,
        format="dicom",
        metadata=sanitized_metadata,
        spatial=spatial,
        source_path=str(path),
        checksum_sha256=compute_sha256(path),
        warnings=warnings,
    )


def load_nifti_file(path: Path) -> LoadedMedicalImage:
    """Reads NIfTI (.nii, .nii.gz) volume and extracts spatial affine metadata."""
    warnings: List[str] = []
    try:
        nii = nib.load(str(path))
        data = nii.get_fdata(dtype=np.float32)
    except Exception as e:
        raise CorruptImageError(f"Failed to parse NIfTI volume {path.name}: {str(e)}") from e

    zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])
    if len(zooms) < 3:
        zooms = (zooms[0], zooms[1], 1.0)

    affine = nii.affine
    origin = tuple(float(x) for x in affine[:3, 3])

    spatial = SpatialMetadata(
        voxel_spacing=zooms,
        slice_thickness=zooms[2],
        origin=origin,
        direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        original_shape=data.shape,
        original_dtype=str(data.dtype),
    )

    valid_q, q_issues = validate_image_quality(data)
    if not valid_q:
        raise CorruptImageError(f"NIfTI quality rejection for {path.name}: {q_issues}")
    if q_issues:
        warnings.extend(q_issues)

    return LoadedMedicalImage(
        pixel_array=data,
        modality="ct" if data.ndim >= 3 else "xray",
        format="nifti",
        metadata={"affine": affine.tolist(), "dim": list(nii.header["dim"])},
        spatial=spatial,
        source_path=str(path),
        checksum_sha256=compute_sha256(path),
        warnings=warnings,
    )


def load_standard_image(path: Path) -> LoadedMedicalImage:
    """Reads standard 2D image (PNG, JPG, TIFF) via Pillow."""
    warnings: List[str] = []
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            # Convert to grayscale if RGB
            if img.mode in ["RGB", "RGBA"]:
                gray = ImageOps.grayscale(img)
                arr = np.array(gray, dtype=np.float32)
                warnings.append(f"Converted multi-channel {img.mode} to single-channel grayscale.")
            else:
                arr = np.array(img, dtype=np.float32)
    except Exception as e:
        raise CorruptImageError(f"Failed to read image file {path.name}: {str(e)}") from e

    valid_q, q_issues = validate_image_quality(arr)
    if not valid_q:
        raise CorruptImageError(f"Image quality check failed for {path.name}: {q_issues}")
    if q_issues:
        warnings.extend(q_issues)

    spatial = SpatialMetadata(
        voxel_spacing=(1.0, 1.0, 1.0),
        slice_thickness=1.0,
        origin=(0.0, 0.0, 0.0),
        direction_cosines=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        original_shape=arr.shape,
        original_dtype=str(arr.dtype),
    )

    ext = path.suffix.lower().lstrip(".")
    fmt = "png" if ext == "png" else ("jpeg" if ext in ["jpg", "jpeg"] else ("tiff" if ext in ["tif", "tiff"] else ext))

    return LoadedMedicalImage(
        pixel_array=arr,
        modality="xray",  # Planar formats default to X-ray pipeline
        format=fmt,
        metadata={"filename": path.name, "format": fmt},
        spatial=spatial,
        source_path=str(path),
        checksum_sha256=compute_sha256(path),
        warnings=warnings,
    )


def load_medical_image(file_path: Union[str, Path]) -> LoadedMedicalImage:
    """
    Unified entrypoint for loading medical image files.
    Dispatches to format-specific loaders based on extension and byte headers.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Medical image file not found: {path}")
    if path.stat().st_size == 0:
        raise CorruptImageError(f"Zero-byte file detected: {path.name}")

    ext = path.suffix.lower()
    double_ext = "".join(path.suffixes[-2:]).lower() if len(path.suffixes) >= 2 else ""

    if double_ext in [".nii.gz"] or ext in [".nii"]:
        return load_nifti_file(path)
    elif ext in [".dcm", ".dicom"]:
        return load_dicom_file(path)
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        return load_standard_image(path)
    else:
        # Inspect magic bytes for header identification
        with open(path, "rb") as f:
            header = f.read(132)
            if len(header) >= 132 and header[128:132] == b"DICM":
                return load_dicom_file(path)

        raise UnsupportedFormatError(
            f"Unsupported file format for {path.name}. Permitted: .dcm, .png, .jpg, .jpeg, .tiff, .nii, .nii.gz"
        )

