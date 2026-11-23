"""Stage 39 reproducible dataset inventory, QC, harmonization and split audit.

This service operates only on Stage 38 verified staged Track D packages. It does
not train models, change source files, establish clinical validity, or grant
readiness. Reports contain pseudonymous group references rather than raw IDs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from functools import wraps
from pathlib import Path, PurePosixPath
import secrets
import tempfile
from copy import deepcopy
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml_service.app.services.cohort_onboarding import CohortOnboardingService
from ml_service.datasets.holdout_policy import load_protected_holdouts, is_protected_site
from ml_service.datasets.track_d_intake import generate_leakage_safe_splits, verify_cross_split_leakage


class PreparationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config_version: Literal["1.0.0"] = "1.0.0"
    xray: dict[str, Any] = Field(default_factory=lambda: {"size": [1024, 1024], "lower_percentile": 0.5, "upper_percentile": 99.5, "preserve_aspect": True, "padding": 0})
    bone_scan: dict[str, Any] = Field(default_factory=lambda: {"size": [256, 1024], "clip_max": 25000, "preserve_aspect": True, "padding": 0})
    ct: dict[str, Any] = Field(default_factory=lambda: {"window_center": 400, "window_width": 1500, "require_validated_hu": True,
        "nifti_hu_evidence_reference": None})
    split: dict[str, Any] = Field(default_factory=lambda: {"train": 0.70, "validation": 0.15, "internal_test": 0.15, "seed": 42})

    @model_validator(mode="after")
    def validate_values(self):
        allowed = {
            "xray": {"size", "lower_percentile", "upper_percentile", "preserve_aspect", "padding"},
            "bone_scan": {"size", "clip_max", "preserve_aspect", "padding"},
            "ct": {"window_center", "window_width", "require_validated_hu", "nifti_hu_evidence_reference"},
            "split": {"train", "validation", "internal_test", "seed"},
        }
        for section, keys in allowed.items():
            unknown=set(getattr(self,section))-keys
            if unknown: raise ValueError(f"Unsupported {section} configuration keys: {', '.join(sorted(unknown))}")
        for modality in ("xray", "bone_scan"):
            cfg = self.model_dump()[modality]
            size = cfg.get("size")
            if not isinstance(size, list) or len(size) != 2 or any(not isinstance(v, int) or v < 16 or v > 4096 for v in size):
                raise ValueError(f"{modality}.size must contain two integers in [16, 4096]")
        split = self.split
        ratios = [split.get("train"), split.get("validation"), split.get("internal_test")]
        if any(not isinstance(x, (int, float)) or not math.isfinite(float(x)) or x <= 0 or x >= 1 for x in ratios) or abs(sum(ratios) - 1) > 1e-6:
            raise ValueError("split ratios must be positive and sum to one")
        if not isinstance(split.get("seed"), int) or not 0 <= split["seed"] <= 2**32 - 1:
            raise ValueError("split.seed must be an unsigned 32-bit integer")
        if any(not isinstance(self.ct.get(key), (int, float)) or not math.isfinite(float(self.ct[key])) for key in ("window_center", "window_width")) or self.ct.get("window_width", 0) <= 0:
            raise ValueError("ct.window_width must be positive")
        if not (0 <= float(self.xray.get("lower_percentile", -1)) < float(self.xray.get("upper_percentile", 101)) <= 100):
            raise ValueError("xray percentile clipping must satisfy 0 <= lower < upper <= 100")
        if not isinstance(self.bone_scan.get("clip_max"), (int, float)) or not math.isfinite(float(self.bone_scan["clip_max"])) or self.bone_scan["clip_max"] <= 0:
            raise ValueError("bone_scan.clip_max must be positive and finite")
        if not isinstance(self.bone_scan.get("padding"), int) or not 0 <= self.bone_scan["padding"] <= 255:
            raise ValueError("bone_scan.padding must be an 8-bit grayscale value")
        return self


class PreparationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$")
    config: PreparationConfig = Field(default_factory=PreparationConfig)


class PreparationError(ValueError):
    pass


def _block_pending_failure(method):
    @wraps(method)
    def wrapped(self, run_id, actor, *args, **kwargs):
        try:
            return method(self, run_id, actor, *args, **kwargs)
        except Exception:
            try:
                run = self._run(run_id)
                if run["state"].endswith("_PENDING"):
                    self._block(run_id, actor, "Preparation operation failed; review input and configuration before creating a new run.")
            except Exception:
                pass
            raise
    return wrapped


class DatasetPreparationService:
    SCHEMA_VERSION = "1.0.0"
    TERMINAL = {"PREPARATION_COMPLETED", "BLOCKED", "REJECTED"}

    def __init__(self, registry_file: Path | None = None, staged_root: Path | None = None,
                 output_root: Path | None = None, holdout_root: Path | None = None,
                 onboarding: CohortOnboardingService | None = None):
        self.registry_file = registry_file or Path(os.environ.get("OSTEO_PREPARATION_REGISTRY_FILE", "ml_service/data/preparation/registry.json"))
        self.staged_root = staged_root or Path(os.environ.get("OSTEO_STAGING_ROOT", "ml_service/data/onboarding/staged"))
        self.output_root = output_root or Path(os.environ.get("OSTEO_PREPARATION_ROOT", "ml_service/data/preparation/runs"))
        self.holdout_root = holdout_root or Path(os.environ.get("OSTEO_BASE_DIR", str(Path(__file__).resolve().parents[3])))
        self.onboarding = onboarding or CohortOnboardingService()
        self.key = os.environ.get("OSTEO_PREPARATION_HMAC_KEY", "").encode()
        if os.environ.get("OSTEO_ENV", "development").lower() == "production" and len(self.key) < 32:
            raise PreparationError("Production dataset preparation requires OSTEO_PREPARATION_HMAC_KEY (32+ bytes).")

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    def _signature(self, value: Any) -> str:
        raw = self._canonical(value)
        return hmac.new(self.key, raw, hashlib.sha256).hexdigest() if self.key else hashlib.sha256(raw).hexdigest()

    def _pseudonym(self, kind: str, source_id: str, value: str) -> str:
        raw=f"{kind}\0{source_id}\0{value}".encode()
        digest=hmac.new(self.key,raw,hashlib.sha256).hexdigest() if self.key else hashlib.sha256(raw).hexdigest()
        return digest[:20]

    def _load(self) -> dict[str, Any]:
        if not self.registry_file.exists():
            return {"schema_version": self.SCHEMA_VERSION, "runs": {}, "integrity": self._signature({"runs": {}})}
        try:
            data = json.loads(self.registry_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreparationError("Preparation registry is unavailable or malformed.") from exc
        if data.get("schema_version") != self.SCHEMA_VERSION or not isinstance(data.get("runs"), dict):
            raise PreparationError("Preparation registry schema is unsupported.")
        expected = self._signature({"runs": data["runs"]})
        if not hmac.compare_digest(str(data.get("integrity", "")), expected):
            raise PreparationError("Preparation registry integrity check failed.")
        return data

    def _save(self, data: dict[str, Any]) -> None:
        data["integrity"] = self._signature({"runs": data["runs"]})
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".preparation-", dir=self.registry_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, sort_keys=True, indent=2)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(name, self.registry_file)
        finally:
            if os.path.exists(name): os.unlink(name)

    def _run(self, run_id: str) -> dict[str, Any]:
        run = self._load()["runs"].get(run_id)
        if run is None: raise KeyError("Preparation run not found.")
        return run

    def _package_root(self, run: dict[str, Any]) -> Path:
        base = self.staged_root.resolve()
        raw = self.staged_root / run["source_id"] / run["source_version"]
        if any(part.is_symlink() for part in (self.staged_root, self.staged_root / run["source_id"], raw)):
            raise PreparationError("Staged package path contains a symbolic link.")
        candidate = raw.resolve()
        if not candidate.is_dir() or not candidate.is_relative_to(base):
            raise PreparationError("Staged package is missing or has an unsafe path.")
        return candidate.resolve()

    def _check_staged_integrity(self, root: Path, run: dict[str, Any] | None = None) -> None:
        """Compare staged files to Stage 38's recorded inventory before use."""
        provenance_path = root / "provenance.json"
        if provenance_path.is_symlink() or not provenance_path.is_file():
            raise PreparationError("Stage 38 provenance record is missing from the staged package.")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            expected = {row["path"]: (row["size_bytes"], row["sha256"]) for row in provenance["file_inventory"]}
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PreparationError("Stage 38 provenance inventory is invalid.") from exc
        provenance_sha256 = self._sha(provenance_path)
        # Stage 38's registry is the trust anchor. Pin both its verified inventory
        # fingerprint and the staged provenance bytes into this signed run record.
        if run is not None and run.get("provenance_sha256") and not hmac.compare_digest(
            str(run["provenance_sha256"]), provenance_sha256
        ):
            raise PreparationError("Stage 38 provenance changed after preparation was created.")
        if run is not None and run.get("stage38_inventory_sha256"):
            if provenance.get("inventory_sha256") != run["stage38_inventory_sha256"]:
                raise PreparationError("Staged provenance does not match Stage 38's verified inventory fingerprint.")
        observed = {}
        for path in root.rglob("*"):
            if path.is_symlink(): raise PreparationError("Symlinks are forbidden in staged packages.")
            if path.is_file() and path.name != "provenance.json":
                rel = path.relative_to(root).as_posix()
                observed[rel] = (path.stat().st_size, self._sha(path))
        if observed != expected:
            raise PreparationError("Staged package no longer matches its Stage 38 verified file inventory.")

    @staticmethod
    def _safe_file(root: Path, rel: str) -> Path:
        pure = PurePosixPath(rel.replace("\\", "/"))
        if pure.is_absolute() or not pure.parts or any(p in {"", ".", ".."} for p in pure.parts):
            raise PreparationError("Manifest contains an unsafe image path.")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise PreparationError("Manifest image is missing or escapes the staged package.")
        return path

    @staticmethod
    def _sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_manifest(root: Path) -> list[dict[str, Any]]:
        path = root / "manifest.json"
        if path.is_symlink() or not path.is_file(): raise PreparationError("Staged Track D manifest is missing.")
        try: manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc: raise PreparationError("Staged manifest is invalid.") from exc
        if manifest.get("track") != "track_d" or not isinstance(manifest.get("records"), list):
            raise PreparationError("Stage 39 only accepts a Stage 33 Track D manifest.")
        if len(manifest["records"]) > 50_000: raise PreparationError("Stage 39 is bounded to 50,000 manifest records per run.")
        return manifest["records"]

    def create(self, source_id: str, operator: str, config: PreparationConfig) -> dict[str, Any]:
        source = self.onboarding.source(source_id)
        if source.get("track", "track_d").lower() not in {"track_d", "d"}:
            raise PreparationError("Stage 39 excludes Track A, B and C sources.")
        if source.get("verification_status") != "VERIFIED" or not source.get("package", {}).get("valid") or source.get("staging_status") != "STAGED_FOR_STAGE33_INTAKE":
            raise PreparationError("Preparation requires a verified Stage 38 package staged for Stage 33 intake.")
        run_id = "prep_" + secrets.token_hex(12)
        raw_root = self.staged_root / source_id / source["version"]
        if any(part.is_symlink() for part in (self.staged_root, self.staged_root / source_id, raw_root)):
            raise PreparationError("Staged package path contains a symbolic link.")
        root = raw_root.resolve()
        if not root.is_relative_to(self.staged_root.resolve()): raise PreparationError("Staged package path is unsafe.")
        self._check_staged_integrity(root)
        provenance_path = root / "provenance.json"
        provenance_sha256 = self._sha(provenance_path)
        stage38_inventory_sha256 = source.get("package", {}).get("inventory_sha256")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if stage38_inventory_sha256 and provenance.get("inventory_sha256") != stage38_inventory_sha256:
            raise PreparationError("Staged provenance does not match Stage 38's verified inventory fingerprint.")
        manifest = self._read_manifest(root)
        if not manifest: raise PreparationError("Stage 39 requires a non-empty Track D manifest.")
        record = {"run_id": run_id, "source_id": source_id, "source_version": source["version"],
            "staging_reference": f"{source_id}/{source['version']}", "manifest_version": "2.0.0",
            "governance_state": "APPROVED" if source.get("governance_approved") is True else "NOT_APPROVED_FOR_TRAINING",
            "access_restriction": source.get("access_restriction", "RESTRICTED"),
            "source_restrictions": list(source.get("restrictions") or []),
            "config": config.model_dump(mode="json"), "config_version": config.config_version,
            "configuration_sha256": hashlib.sha256(self._canonical(config.model_dump(mode="json"))).hexdigest(),
            "state": "SOURCE_STAGED", "operator_pseudo_id": operator,
            "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(),
            "validation_status": "PENDING", "blockers": [], "inventory": None, "qc": None,
            "stage38_inventory_sha256": stage38_inventory_sha256, "provenance_sha256": provenance_sha256,
            "harmonization": None, "splits": None, "report_sha256": None,
            "lifecycle_events": [{"state": "SOURCE_STAGED", "at": datetime.now(timezone.utc).isoformat(), "operator_pseudo_id": operator}]}
        data = self._load(); data["runs"][run_id] = record; self._save(data)
        return self.public_run(record)

    @staticmethod
    def public_run(run: dict[str, Any]) -> dict[str, Any]:
        public = {k: deepcopy(v) for k, v in run.items() if k not in {"operator_pseudo_id"}}
        config=public.get("config")
        if isinstance(config,dict) and isinstance(config.get("ct"),dict):
            evidence=str(config["ct"].pop("nifti_hu_evidence_reference", "") or "")
            if evidence:
                config["ct"]["nifti_hu_evidence_reference_sha256"]=hashlib.sha256(evidence.encode()).hexdigest()
                config["ct"]["nifti_hu_evidence_reference"]="RECORDED_PRIVATELY"
        return public

    def list_runs(self) -> list[dict[str, Any]]:
        return [self.public_run(row) for row in self._load()["runs"].values()]

    def get(self, run_id: str) -> dict[str, Any]: return self.public_run(self._run(run_id))

    def _advance(self, run_id: str, expected: str, state: str, actor: str, field: str, result: Any) -> dict[str, Any]:
        data = self._load(); run = data["runs"].get(run_id)
        if not run: raise KeyError("Preparation run not found.")
        if run["state"] != expected: raise PreparationError(f"Invalid preparation transition from {run['state']} to {state}.")
        run[field] = result; run["state"] = state; run["validation_status"] = "PASSED" if state != "BLOCKED" else "BLOCKED"
        run["updated_at"] = datetime.now(timezone.utc).isoformat(); run["last_operator_pseudo_id"] = actor
        run.setdefault("lifecycle_events", []).append({"state": state, "at": run["updated_at"], "operator_pseudo_id": actor})
        self._save(data); return self.public_run(run)

    def _begin(self, run_id: str, expected: str, pending: str, actor: str) -> None:
        data=self._load(); run=data["runs"].get(run_id)
        if not run: raise KeyError("Preparation run not found.")
        if run["state"] != expected: raise PreparationError(f"Invalid preparation transition from {run['state']} to {pending}.")
        run["state"]=pending; run["validation_status"]="PENDING"; run["updated_at"]=datetime.now(timezone.utc).isoformat()
        run.setdefault("lifecycle_events", []).append({"state":pending,"at":run["updated_at"],"operator_pseudo_id":actor})
        self._save(data)

    def _block(self, run_id: str, actor: str, summary: str) -> None:
        data=self._load(); run=data["runs"].get(run_id)
        if run:
            run["state"]="BLOCKED"; run["validation_status"]="BLOCKED"; run["blockers"]=[summary]
            run["updated_at"]=datetime.now(timezone.utc).isoformat()
            run.setdefault("lifecycle_events", []).append({"state":"BLOCKED","at":run["updated_at"],"operator_pseudo_id":actor,"reason":"OPERATION_BLOCKED"})
            self._save(data)

    @_block_pending_failure
    def inventory(self, run_id: str, actor: str) -> dict[str, Any]:
        run = self._run(run_id)
        if run["state"] != "SOURCE_STAGED": raise PreparationError("Inventory can only run from SOURCE_STAGED.")
        self._begin(run_id,"SOURCE_STAGED","INVENTORY_PENDING",actor)
        try:
            root = self._package_root(run); self._check_staged_integrity(root, run); records = self._read_manifest(root); files = []; labels = {}; modalities = {}; missing = set(); seen_hashes = {}; group_rows=[]
        except Exception as exc:
            self._block(run_id,actor,"Inventory could not safely process the staged package."); raise
        for index, rec in enumerate(records):
            if not isinstance(rec, dict): raise PreparationError("Manifest contains a malformed record.")
            path = self._safe_file(root, str(rec.get("image_path", "")))
            digest = self._sha(path); mod = str(rec.get("imaging_modality", "")).lower()
            pid = str(rec.get("patient_id", "")).strip(); cls = str(rec.get("diagnostic_class", ""))
            if not pid: missing.add("patient_id")
            if not rec.get("institution_id"): missing.add("institution_id")
            group_ref = self._pseudonym("patient",run["source_id"],pid) if pid else "UNASSIGNED"
            perceptual_hash = None
            study_ref = series_ref = None; spacing = orientation = None
            try:
                with Image.open(path) as im:
                    dims = list(im.size); fmt = im.format or path.suffix.lstrip(".").upper()
                    try:
                        import imagehash
                        perceptual_hash = str(imagehash.phash(im.convert("L")))
                    except Exception: pass
            except Exception: dims = None; fmt = "DICOM" if path.suffix.lower() in {".dcm", ".dicom"} else "NIFTI" if path.name.lower().endswith((".nii", ".nii.gz")) else path.suffix.lstrip(".").upper()
            try:
                if path.suffix.lower() in {".dcm", ".dicom"}:
                    import pydicom
                    ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
                    if getattr(ds,"StudyInstanceUID",None): study_ref=self._pseudonym("study",run["source_id"],str(ds.StudyInstanceUID))
                    if getattr(ds,"SeriesInstanceUID",None): series_ref=self._pseudonym("series",run["source_id"],str(ds.SeriesInstanceUID))
                    spacing=[float(v) for v in getattr(ds,"PixelSpacing",[]) if np.isfinite(float(v))]
                    orientation=[float(v) for v in getattr(ds,"ImageOrientationPatient",[]) if np.isfinite(float(v))]
                elif path.name.lower().endswith((".nii", ".nii.gz")):
                    import nibabel as nib
                    nii=nib.load(str(path)); spacing=[float(v) for v in nii.header.get_zooms()]
                    orientation=list(nib.aff2axcodes(nii.affine))
            except Exception: pass
            duplicate = seen_hashes.get(digest)
            seen_hashes.setdefault(digest, str(index))
            files.append({"file_ref": hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()[:20],
                "size_bytes": path.stat().st_size, "sha256": digest, "file_type": fmt, "modality": mod,
                "dimensions": dims, "spacing": spacing, "orientation": orientation,
                "study_group_ref": study_ref, "series_group_ref": series_ref,
                "duplicate_of_record_index": duplicate, "perceptual_hash": perceptual_hash})
            labels[cls] = labels.get(cls, 0) + 1; modalities[mod] = modalities.get(mod, 0) + 1
            group_rows.append({"patient_group_ref": group_ref,
                "institution_ref": self._pseudonym("institution",run["source_id"],str(rec.get("institution_id"))) if rec.get("institution_id") else None,
                "class": cls, "modality": mod,
                "sample_ref": self._pseudonym("sample",run["source_id"],str(rec.get("sample_id") or index)),
                "file_ref": hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()[:20], "sha256": digest})
        inventory = {"source_id": run["source_id"], "source_version": run["source_version"], "config_version": run["config_version"],
            "generated_at": datetime.now(timezone.utc).isoformat(), "file_count": len(files), "files": files,
            "labels": dict(sorted(labels.items())), "modalities": dict(sorted(modalities.items())),
            "missing_metadata": sorted(missing), "duplicate_file_count": sum(1 for f in files if f["duplicate_of_record_index"] is not None),
            "groups": group_rows}
        inventory["inventory_sha256"] = hashlib.sha256(self._canonical({k:v for k,v in inventory.items() if k != "generated_at"})).hexdigest()
        self._write_report(run_id, "inventory.json", inventory)
        return self._advance(run_id, "INVENTORY_PENDING", "INVENTORY_COMPLETED", actor, "inventory", inventory)

    @_block_pending_failure
    def qc(self, run_id: str, actor: str) -> dict[str, Any]:
        run = self._run(run_id)
        if run["state"] != "INVENTORY_COMPLETED": raise PreparationError("QC requires a completed inventory.")
        self._begin(run_id,"INVENTORY_COMPLETED","QC_PENDING",actor)
        try: root = self._package_root(run); self._check_staged_integrity(root, run); records = self._read_manifest(root); findings=[]
        except Exception:
            self._block(run_id,actor,"QC could not safely process the staged package."); raise
        for index, rec in enumerate(records):
            path = self._safe_file(root, str(rec.get("image_path", ""))); modality = str(rec.get("imaging_modality", "")).lower()
            file_ref = hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()[:20]
            def add(rule, severity, status, finding, evidence=None, blocks=False):
                findings.append({"file_ref": file_ref, "rule_id": rule, "severity": severity, "status": status,
                    "finding": finding, "evidence": evidence or {}, "recommended_action": "Review source image and manifest before use.", "blocks_preparation": blocks})
            if modality not in {"xray", "bone_scan", "ct"}:
                add("MODALITY_SUPPORTED", "BLOCKER", "FAIL", "Unsupported modality.", blocks=True); continue
            ext = path.suffix.lower()
            if ext in {".dcm", ".dicom"}:
                try:
                    import pydicom
                    ds = pydicom.dcmread(path, force=False)
                    patient_tags = ["PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientAddress",
                        "OtherPatientIDs", "OtherPatientIDsSequence", "AccessionNumber", "StudyID", "InstitutionName",
                        "InstitutionAddress", "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName"]
                    found = [tag for tag in patient_tags if getattr(ds, tag, None)]
                    add("DICOM_PHYSICAL_PRIVACY", "BLOCKER" if found else "INFO", "FAIL" if found else "PASS", "Potential direct identifiers in DICOM metadata." if found else "No screened direct identifier tags found.", {"tag_names": found}, bool(found))
                    stored_modality = str(getattr(ds, "Modality", "")).upper()
                    allowed_dicom = {"ct": {"CT"}, "xray": {"DX", "CR"}, "bone_scan": {"NM"}}[modality]
                    modality_ok = stored_modality in allowed_dicom
                    add("DICOM_MODALITY_CONSISTENCY", "INFO" if modality_ok else "BLOCKER", "PASS" if modality_ok else "FAIL",
                        "DICOM modality agrees with the manifest." if modality_ok else "DICOM modality is absent or conflicts with the manifest.",
                        {"dicom_modality": stored_modality or "MISSING"}, not modality_ok)
                    pixels = ds.pixel_array
                    if pixels.size == 0 or not np.isfinite(pixels).all(): add("PIXEL_INTEGRITY", "BLOCKER", "FAIL", "DICOM pixel data is empty or non-finite.", blocks=True)
                    else: add("PIXEL_INTEGRITY", "INFO", "PASS", "DICOM pixels decoded.", {"dimensions": list(pixels.shape)})
                    if modality == "ct": add("CT_SERIES_RECONSTRUCTION", "WARNING", "REVIEW_REQUIRED", "This run validates a DICOM object; CT series ordering and volume reconstruction are not performed.", {"series_instance_uid_present": bool(getattr(ds,"SeriesInstanceUID",None))})
                    if modality == "ct":
                        slope = getattr(ds, "RescaleSlope", None); intercept = getattr(ds, "RescaleIntercept", None)
                        valid = slope is not None and intercept is not None and np.isfinite(float(slope)) and float(slope) != 0 and np.isfinite(float(intercept))
                        add("CT_HU_TRANSFORM", "INFO" if valid else "BLOCKER", "PASS" if valid else "FAIL", "DICOM rescale values are present; HU semantics still require source confirmation." if valid else "HU conversion cannot be confirmed without finite RescaleSlope and RescaleIntercept.", {"rescale_present": bool(valid)}, not valid)
                except Exception as exc: add("DICOM_READABLE", "BLOCKER", "FAIL", "DICOM could not be safely decoded.", {"error_type": type(exc).__name__}, True)
            elif path.name.lower().endswith((".nii", ".nii.gz")):
                try:
                    import nibabel as nib
                    image = nib.load(str(path)); shape = tuple(int(x) for x in image.shape); zooms = tuple(float(x) for x in image.header.get_zooms())
                    valid = 2 <= len(shape) <= 4 and all(0 < n <= 8192 for n in shape) and all(np.isfinite(z) and z > 0 for z in zooms)
                    add("NIFTI_HEADER", "INFO" if valid else "BLOCKER", "PASS" if valid else "FAIL", "NIfTI header dimensions and spacing are valid." if valid else "NIfTI shape or spacing is outside supported bounds.", {"shape": shape, "spacing": zooms}, not valid)
                    if image.header.extensions:
                        add("NIFTI_EXTENSION_REVIEW", "BLOCKER", "REVIEW_REQUIRED", "NIfTI extensions may contain unreviewed identifying metadata.", {"extension_count": len(image.header.extensions)}, True)
                    if valid:
                        arr = np.asanyarray(image.dataobj)
                        add("PIXEL_INTEGRITY", "INFO" if np.isfinite(arr).all() else "BLOCKER", "PASS" if arr.size and np.isfinite(arr).all() else "FAIL", "NIfTI volume decoded." if arr.size and np.isfinite(arr).all() else "NIfTI volume is empty or non-finite.", blocks=not arr.size or not np.isfinite(arr).all())
                except Exception as exc: add("NIFTI_READABLE", "BLOCKER", "FAIL", "NIfTI could not be safely decoded.", {"error_type": type(exc).__name__}, True)
            else:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    with Image.open(path) as img:
                        arr = np.asarray(ImageOps.grayscale(img))
                        exif_names = []
                        try:
                            from PIL.ExifTags import TAGS
                            exif_names = [TAGS.get(tag, str(tag)) for tag, value in img.getexif().items() if value and TAGS.get(tag, str(tag)) in {
                                "Artist", "Copyright", "ImageDescription", "DocumentName", "HostComputer", "DateTimeOriginal",
                                "XPAuthor", "XPComment", "XPTitle", "XPSubject", "CameraOwnerName", "BodySerialNumber"}]
                        except Exception: pass
                    add("RASTER_METADATA_PRIVACY", "BLOCKER" if exif_names else "INFO", "FAIL" if exif_names else "PASS",
                        "Potentially identifying raster metadata requires removal." if exif_names else "No screened identifying EXIF fields found.",
                        {"tag_names": sorted(set(exif_names))}, bool(exif_names))
                    add("BURNED_IN_IDENTIFIER_REVIEW", "WARNING", "REVIEW_REQUIRED", "Pixels require human privacy review for burned-in identifiers; automated QC cannot certify de-identification.")
                    if arr.ndim != 2 or not arr.size: add("IMAGE_DIMENSIONS", "BLOCKER", "FAIL", "Image has invalid dimensions.", blocks=True)
                    elif max(arr.shape) > 8192: add("IMAGE_DIMENSIONS", "BLOCKER", "FAIL", "Image exceeds the supported 8192-pixel side bound.", {"dimensions": list(arr.shape)}, True)
                    else:
                        span = int(arr.max()) - int(arr.min())
                        if span == 0: add("IMAGE_DYNAMIC_RANGE", "BLOCKER", "FAIL", "Image is empty or has no intensity variation.", {"range": span}, True)
                        else: add("IMAGE_DYNAMIC_RANGE", "INFO", "PASS", "Image is readable with nonzero dynamic range.", {"dimensions": list(arr.shape), "range": span})
                except Exception as exc: add("IMAGE_READABLE", "BLOCKER", "FAIL", "Image is corrupt or unsupported.", {"error_type": type(exc).__name__}, True)
        blockers = sum(f["blocks_preparation"] for f in findings)
        report = {"finding_count": len(findings), "blocker_count": blockers, "findings": findings, "status": "BLOCKED" if blockers else "PASSED", "clinical_assessment": False}
        self._write_report(run_id, "qc.json", report)
        return self._advance(run_id, "QC_PENDING", "BLOCKED" if blockers else "QC_COMPLETED", actor, "qc", report)

    @_block_pending_failure
    def harmonize(self, run_id: str, actor: str) -> dict[str, Any]:
        run = self._run(run_id)
        if run["state"] != "QC_COMPLETED": raise PreparationError("Harmonization requires QC_COMPLETED with no blocking findings.")
        self._begin(run_id,"QC_COMPLETED","HARMONIZATION_PENDING",actor)
        try: root = self._package_root(run); self._check_staged_integrity(root, run); records = self._read_manifest(root); output = self.output_root.resolve() / run_id / "prepared"
        except Exception:
            self._block(run_id,actor,"Harmonization could not safely access the staged package."); raise
        if output.exists(): raise FileExistsError("Prepared output already exists; create a new run/config version.")
        output.parent.mkdir(parents=True, exist_ok=True); tmp = Path(tempfile.mkdtemp(prefix=".prepared-", dir=output.parent)); results=[]
        try:
            for index, rec in enumerate(records):
                source = self._safe_file(root, str(rec["image_path"])); modality = str(rec["imaging_modality"]).lower()
                if modality == "ct":
                    cfg = run["config"]["ct"]
                    center, width = float(cfg["window_center"]), float(cfg["window_width"])
                    lower, upper = center - width / 2, center + width / 2
                    geometry: dict[str, Any] = {}
                    if source.name.lower().endswith((".nii", ".nii.gz")):
                        import nibabel as nib
                        evidence = str(cfg.get("nifti_hu_evidence_reference") or "").strip()
                        if not evidence: raise PreparationError("NIfTI CT harmonization requires a source-documented HU evidence reference in the run configuration.")
                        nii = nib.load(str(source)); raw = np.asarray(nii.get_fdata(dtype=np.float32))
                        if raw.ndim not in (2, 3) or not raw.size or not np.isfinite(raw).all(): raise PreparationError("NIfTI CT dimensions or pixel values are invalid.")
                        geometry = {"affine": np.asarray(nii.affine).round(7).tolist(), "spacing_mm": [float(x) for x in nii.header.get_zooms()[:raw.ndim]], "orientation": nib.aff2axcodes(nii.affine), "hu_evidence_reference_sha256": hashlib.sha256(evidence.encode()).hexdigest()}
                    elif source.suffix.lower() in {".dcm", ".dicom"}:
                        import pydicom
                        ds = pydicom.dcmread(source, force=False)
                        slope, intercept = getattr(ds, "RescaleSlope", None), getattr(ds, "RescaleIntercept", None)
                        if str(getattr(ds, "Modality", "")).upper() != "CT" or str(getattr(ds, "RescaleType", "")).strip().upper() != "HU":
                            raise PreparationError("DICOM CT conversion requires Modality=CT and RescaleType=HU; arbitrary pixel values are not converted.")
                        if slope is None or intercept is None or not np.isfinite(float(slope)) or float(slope) == 0 or not np.isfinite(float(intercept)):
                            raise PreparationError("DICOM CT requires finite nonzero RescaleSlope and finite RescaleIntercept.")
                        raw = np.asarray(ds.pixel_array, dtype=np.float32) * float(slope) + float(intercept)
                        if not np.isfinite(raw).all(): raise PreparationError("DICOM CT pixel data contains non-finite values.")
                        geometry = {"pixel_spacing_mm": [float(x) for x in getattr(ds, "PixelSpacing", [])],
                            "orientation": [float(x) for x in getattr(ds, "ImageOrientationPatient", [])],
                            "rescale_slope": float(slope), "rescale_intercept": float(intercept), "rescale_type": "HU"}
                    else:
                        raise PreparationError("CT harmonization supports HU-confirmed DICOM slices and source-documented HU NIfTI volumes only.")
                    normalized = (np.clip(raw, lower, upper) - lower) / width
                    group = self._pseudonym("patient",run["source_id"],str(rec.get("patient_id", "")))
                    name = f"{group}_{index:06d}.npy"; dest = tmp / name
                    np.save(dest, normalized.astype("<f4"), allow_pickle=False)
                    (tmp / f"{group}_{index:06d}.geometry.json").write_text(json.dumps(geometry, sort_keys=True), encoding="utf-8")
                    results.append({"file_ref": name, "sha256": self._sha(dest), "input_sha256": self._sha(source), "modality": modality,
                        "output_shape": list(normalized.shape), "window_hu": {"center": center, "width": width}, "geometry_preserved": True})
                    continue
                cfg = run["config"][modality]; target_w, target_h = cfg["size"]
                with Image.open(source) as im:
                    im.load(); im = ImageOps.exif_transpose(im); original_size = im.size
                    if modality == "xray":
                        arr = np.asarray(ImageOps.grayscale(im), dtype=np.float32); lo, hi = np.percentile(arr, [cfg["lower_percentile"], cfg["upper_percentile"]])
                        if not np.isfinite([lo, hi]).all() or hi <= lo: raise PreparationError("X-ray intensity range is degenerate.")
                        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
                    else:
                        ceiling = cfg["clip_max"]
                        if ceiling <= 0: raise PreparationError("Bone scan clip_max must be positive.")
                        arr = np.asarray(im, dtype=np.float32)
                        if arr.ndim == 3: arr = np.asarray(ImageOps.grayscale(im), dtype=np.float32)
                        arr = np.clip(arr, 0, ceiling) / ceiling
                    normalized = Image.fromarray(np.uint8(np.rint(arr * 255)), mode="L")
                    if cfg.get("preserve_aspect", True):
                        normalized.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
                        canvas = Image.new("L", (target_w, target_h), color=int(cfg.get("padding", 0)))
                        canvas.paste(normalized, ((target_w-normalized.width)//2, (target_h-normalized.height)//2)); normalized = canvas
                    else: normalized = normalized.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    group = self._pseudonym("patient",run["source_id"],str(rec.get("patient_id", "")))
                    name = f"{group}_{index:06d}.png"; dest = tmp / name; normalized.save(dest, format="PNG", optimize=False)
                results.append({"file_ref": name, "sha256": self._sha(dest), "input_sha256": self._sha(source), "modality": modality,
                    "original_dimensions": original_size, "output_dimensions": [target_w, target_h]})
            os.replace(tmp, output)
        except Exception as exc:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            self._block(run_id,actor,"Harmonization was blocked; review the configuration and modality support.")
            raise
        summary = {"status": "COMPLETED", "output_reference": run_id + "/prepared", "config_version": run["config_version"],
            "file_count": len(results), "files": results, "augmentation": False, "source_modified": False,
            "sha256": hashlib.sha256(self._canonical(results)).hexdigest()}
        self._write_report(run_id, "harmonization.json", summary)
        return self._advance(run_id, "HARMONIZATION_PENDING", "HARMONIZATION_COMPLETED", actor, "harmonization", summary)

    @_block_pending_failure
    def prepare_splits(self, run_id: str, actor: str) -> dict[str, Any]:
        run = self._run(run_id)
        if run["state"] not in {"QC_COMPLETED", "HARMONIZATION_COMPLETED"}: raise PreparationError("Split audit requires completed QC (and any requested harmonization).")
        start_state=run["state"]; self._begin(run_id,start_state,"SPLIT_AUDIT_PENDING",actor)
        root = self._package_root(run); self._check_staged_integrity(root, run); records = self._read_manifest(root); registry = load_protected_holdouts(self.holdout_root)
        included=[]; excluded=0; site_patient={}; file_hash_by_sample={}
        for rec in records:
            path = self._safe_file(root, str(rec.get("image_path", ""))); item = dict(rec); item["sha256"] = self._sha(path)
            reasons = registry.record_violations(item)
            if reasons or is_protected_site(rec.get("institution_id")) or is_protected_site(rec.get("dataset_source")):
                excluded += 1; continue
            if not rec.get("patient_id") or not rec.get("diagnostic_class") or not rec.get("imaging_modality"):
                continue
            try:
                import imagehash
                with Image.open(path) as image: item["perceptual_hash"] = str(imagehash.phash(image.convert("L")))
            except Exception:
                item["perceptual_hash"] = ""
            pid = str(rec["patient_id"]); site = str(rec.get("institution_id") or "")
            site_patient.setdefault(pid, set()).add(site)
            item["dataset_source"] = run["source_id"]
            included.append(item)
        if not included: raise PreparationError("No eligible non-holdout patient records are available for split preparation.")
        split_cfg=run["config"]["split"]
        split = generate_leakage_safe_splits(included, split_cfg["train"], split_cfg["validation"], split_cfg["internal_test"], split_cfg["seed"])
        assigned=split["records"]
        # Institution overlap is reported for interpretation; patient grouping is the mandatory split unit.
        by_split={name:set() for name in ("train","val","test")}; patient_split={}
        for rec in assigned:
            pid=str(rec["patient_id"]); sp=rec["split_assignment"]; patient_split[pid]=sp
            if rec.get("institution_id"): by_split[sp].add(str(rec["institution_id"]))
        institution_overlap=[]
        keys=list(by_split)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                overlap=by_split[a]&by_split[b]
                if overlap: institution_overlap.append({"splits":[a,b],"institution_count":len(overlap)})
        leakage=verify_cross_split_leakage(assigned)
        if not leakage["is_clean"]: raise PreparationError("Cross-split patient or hash leakage detected.")
        leakage["violations"] = [str(item).split(":", 1)[0] for item in leakage.get("violations", [])]
        distribution={"patient_counts":{k:sum(1 for s in patient_split.values() if s==k) for k in ("train","val","test")},
            "image_counts":dict(split["distribution"]), "class_counts":{}, "modality_counts":{}}
        for rec in assigned:
            sp=rec["split_assignment"]; cls=rec["diagnostic_class"]; mod=rec["imaging_modality"]
            distribution["class_counts"].setdefault(cls,{}).setdefault(sp,0); distribution["class_counts"][cls][sp]+=1
            distribution["modality_counts"].setdefault(mod,{}).setdefault(sp,0); distribution["modality_counts"][mod][sp]+=1
        safe_assignments=[{"patient_group_ref":self._pseudonym("patient",run["source_id"],str(r["patient_id"])),
            "split":r["split_assignment"],"class":r["diagnostic_class"],"modality":r["imaging_modality"],"sha256":r["sha256"]} for r in assigned]
        result={"seed":split_cfg["seed"],"ratios":{"train":split_cfg["train"],"validation":split_cfg["validation"],"internal_test":split_cfg["internal_test"]},
            "excluded_protected_holdout_records":excluded,"protected_holdouts_used_for_split":False,"leakage_audit":leakage,
            "source_level_leakage":{"source_count":1,"source_cross_split":False},
            "protected_holdout_audit":{"registry_sources_loaded_count":len(registry.sources),"excluded_record_count":excluded},
            "institution_overlap":institution_overlap,"distribution":distribution,"missing_classes":sorted({"metastasis","paget","oi","healthy"}-{r["diagnostic_class"] for r in assigned}),
            "assignments":safe_assignments,"status":"AUDITED","training_authorized":False}
        self._write_report(run_id,"splits.json",result)
        self._advance(run_id,"SPLIT_AUDIT_PENDING","SPLIT_AUDIT_COMPLETED",actor,"splits",result)
        self._advance(run_id,"SPLIT_AUDIT_COMPLETED","PREPARATION_COMPLETED",actor,"splits",result)
        report=self.report(run_id); report_hash=report["report_sha256"]
        data=self._load(); data["runs"][run_id]["report_sha256"]=report_hash; self._save(data)
        report=self.report(run_id)
        self._write_report(run_id,"report.json",report)
        self._write_report(run_id,"report.md",{"content":report["markdown"]})
        return self.get(run_id)

    def _write_report(self, run_id: str, filename: str, payload: dict[str, Any]) -> None:
        directory=self.output_root.resolve()/run_id
        if not directory.is_relative_to(self.output_root.resolve()): raise PreparationError("Report path is unsafe.")
        directory.mkdir(parents=True,exist_ok=True); target=directory/filename
        if target.exists() or target.is_symlink(): raise FileExistsError("Report artifact already exists.")
        fd,name=tempfile.mkstemp(prefix=".report-",dir=directory)
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as stream:
                if filename.endswith(".md"): stream.write(str(payload.get("content", "")))
                else: json.dump(payload,stream,sort_keys=True,indent=2)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(name,target)
        finally:
            if os.path.exists(name): os.unlink(name)

    def report(self, run_id: str) -> dict[str, Any]:
        run=self._run(run_id); public=self.public_run(run); public.pop("report_sha256",None)
        report={"schema_version":self.SCHEMA_VERSION,"run":public,
            "inventory":run.get("inventory"),"qc":run.get("qc"),"harmonization":run.get("harmonization"),"splits":run.get("splits"),
            "clinical_validity":"NOT_ESTABLISHED","clinical_readiness":"NOT_ESTABLISHED","training_authorized":False}
        markdown=[f"# Stage 39 preparation {run_id}","",f"- Source: `{run['source_id']}` version `{run['source_version']}`",f"- State: **{run['state']}**","",
            "This report describes technical data preparation only. It does not establish clinical validity, governance approval, model readiness, or training authorization.",""]
        if run.get("inventory"): markdown += [f"- Files inventoried: {run['inventory']['file_count']}",f"- Inventory SHA-256: `{run['inventory']['inventory_sha256']}`"]
        if run.get("qc"): markdown += [f"- QC: {run['qc']['status']} ({run['qc']['finding_count']} findings, {run['qc']['blocker_count']} blockers)"]
        if run.get("splits"): markdown += [f"- Protected records excluded: {run['splits']['excluded_protected_holdout_records']}",f"- Split leakage clean: {run['splits']['leakage_audit']['is_clean']}"]
        report["markdown"]="\n".join(markdown)+"\n"
        report["report_sha256"]=hashlib.sha256(self._canonical(report)).hexdigest()
        expected=run.get("report_sha256")
        if expected and not hmac.compare_digest(str(expected), report["report_sha256"]):
            raise PreparationError("Preparation report integrity verification failed.")
        json_path=self.output_root.resolve()/run_id/"report.json"
        markdown_path=self.output_root.resolve()/run_id/"report.md"
        if json_path.exists():
            try:
                stored=json.loads(json_path.read_text(encoding="utf-8")); stored_digest=stored.pop("report_sha256", None)
                actual=hashlib.sha256(self._canonical(stored)).hexdigest()
            except (OSError,UnicodeError,json.JSONDecodeError) as exc: raise PreparationError("Persisted preparation report is invalid.") from exc
            if not hmac.compare_digest(str(stored_digest or ""),report["report_sha256"]) or not hmac.compare_digest(actual,report["report_sha256"]):
                raise PreparationError("Persisted preparation report integrity verification failed.")
            if markdown_path.is_file() and markdown_path.read_text(encoding="utf-8") != report["markdown"]:
                raise PreparationError("Persisted Markdown report integrity verification failed.")
        return report
