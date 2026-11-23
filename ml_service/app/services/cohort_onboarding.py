"""Governance-aware dataset source, package verification and staging service.

Hashes establish integrity relative to this record; they do not establish source
authenticity, clinical validity, permission, ethics approval, or training readiness.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from ml_service.datasets.holdout_policy import load_protected_holdouts


class SourceRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$")
    name: str = Field(min_length=2, max_length=160)
    source_type: Literal["INSTITUTIONAL", "PUBLIC_REPOSITORY", "RESEARCH_COLLABORATION", "OTHER"]
    source_reference: str = Field(min_length=3, max_length=500)
    description: str = Field(min_length=5, max_length=2000)
    modalities: list[Literal["xray", "bone_scan", "ct"]] = Field(min_length=1)
    target_categories: list[Literal["SECONDARY_BONE_METASTASIS", "PAGETS_DISEASE", "OSTEOGENESIS_IMPERFECTA", "HEALTHY_SKELETAL_CONTROL"]] = Field(min_length=1)
    owner_institution: str | None = Field(default=None, max_length=200)
    acquisition_method: str = Field(min_length=3, max_length=300)
    license_or_terms: str | None = Field(default=None, max_length=1000)
    access_restriction: Literal["RESTRICTED", "CONTROLLED", "OPEN_LICENSE_REVIEW_REQUIRED"] = "RESTRICTED"
    ethics_reference: str | None = Field(default=None, max_length=300)
    data_use_reference: str | None = Field(default=None, max_length=300)
    acquired_at: datetime | None = None
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    fingerprint_sha256: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{64}$")
    contact: str | None = Field(default=None, max_length=200)
    intended_use: str = Field(min_length=3, max_length=500)
    restrictions: list[str] = Field(default_factory=list, max_length=30)
    notes: str | None = Field(default=None, max_length=2000)


class AcquisitionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ACCESS_PENDING", "ACCESS_APPROVED", "ACQUIRED", "VERIFICATION_PENDING", "VERIFIED", "REJECTED", "RESTRICTED", "ARCHIVED"]
    reason: str = Field(min_length=3, max_length=1000)
    evidence_reference: str | None = Field(default=None, max_length=300)
    authorization_reference: str | None = Field(default=None, max_length=300)


class OnboardingStateError(ValueError):
    pass


class CohortOnboardingService:
    SCHEMA_VERSION = "1.0.0"
    _lock = RLock()

    def __init__(self, state_file: Path | None = None, package_root: Path | None = None, staging_root: Path | None = None, holdout_registry_root: Path | None = None):
        self.state_file = state_file or Path(os.environ.get("OSTEO_ONBOARDING_REGISTRY_FILE", "ml_service/data/onboarding/registry.json"))
        self.package_root = package_root or Path(os.environ.get("OSTEO_ONBOARDING_ROOT", "ml_service/data/onboarding/packages"))
        self.staging_root = staging_root or Path(os.environ.get("OSTEO_STAGING_ROOT", "ml_service/data/onboarding/staged"))
        self.holdout_registry_root = holdout_registry_root or Path(os.environ.get("OSTEO_BASE_DIR", str(Path(__file__).resolve().parents[3])))
        self.hmac_key = os.environ.get("OSTEO_ONBOARDING_LEDGER_HMAC_KEY", "").encode()
        if os.environ.get("OSTEO_ENV", "development").lower() == "production" and len(self.hmac_key) < 32:
            raise OnboardingStateError("Production onboarding requires an external ledger HMAC key.")

    @staticmethod
    def _canonical(data: Any) -> bytes:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _signature(self, data: Any) -> str:
        raw = self._canonical(data)
        return hmac.new(self.hmac_key, raw, hashlib.sha256).hexdigest() if self.hmac_key else hashlib.sha256(raw).hexdigest()

    def _empty(self) -> dict[str, Any]:
        sources: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        return {"schema_version": self.SCHEMA_VERSION, "sources": sources, "events": events,
                "ledger_hmac": self._signature({"sources": sources, "events": events})}

    def _load(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return self._empty()
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise OnboardingStateError("Onboarding registry is unavailable or invalid.") from exc
        if not isinstance(state, dict) or state.get("schema_version") != self.SCHEMA_VERSION or not isinstance(state.get("sources"), dict) or not isinstance(state.get("events"), list):
            raise OnboardingStateError("Onboarding registry schema is unsupported or malformed.")
        expected = self._signature({"sources": state["sources"], "events": state["events"]})
        if not hmac.compare_digest(str(state.get("ledger_hmac", "")), expected):
            raise OnboardingStateError("Onboarding provenance ledger integrity verification failed.")
        previous = "0" * 64
        for stored in state["events"]:
            event = dict(stored)
            digest = event.pop("event_hash", None)
            if event.get("previous_hash") != previous or not isinstance(digest, str) or not hmac.compare_digest(digest, hashlib.sha256(self._canonical(event)).hexdigest()):
                raise OnboardingStateError("Onboarding provenance event chain verification failed.")
            previous = digest
        return state

    def _save(self, state: dict[str, Any]) -> None:
        state["ledger_hmac"] = self._signature({"sources": state["sources"], "events": state["events"]})
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".onboarding-", dir=self.state_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, sort_keys=True, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.state_file)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _event(self, state: dict[str, Any], source_id: str, action: str, actor: str, details: dict[str, Any]) -> None:
        previous = state["events"][-1]["event_hash"] if state["events"] else "0" * 64
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), "source_id": source_id, "action": action,
                 "actor_pseudo_id": actor, "details": details, "previous_hash": previous}
        event["event_hash"] = hashlib.sha256(self._canonical(event)).hexdigest()
        state["events"].append(event)

    def list_sources(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for saved in self._load()["sources"].values():
                source = dict(saved)
                if source.get("package"):
                    source["package"] = {key: value for key, value in source["package"].items() if key != "inventory"}
                rows.append(source)
            return rows

    def source(self, source_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return self._load()["sources"][source_id]
            except KeyError as exc:
                raise KeyError("Dataset source not found.") from exc

    def register(self, payload: SourceRegistration, actor: str) -> dict[str, Any]:
        with self._lock:
            state = self._load()
            if payload.source_id in state["sources"]:
                raise FileExistsError("Dataset source identifier already exists.")
            source = payload.model_dump(mode="json")
            source.update({"acquisition_status": "REGISTERED", "verification_status": "NOT_VERIFIED",
                           "integrity_status": "NOT_CHECKED", "staging_status": "NOT_STAGED",
                           "registered_at": datetime.now(timezone.utc).isoformat(), "package": None,
                           "governance_approved": False})
            state["sources"][payload.source_id] = source
            self._event(state, payload.source_id, "SOURCE_REGISTERED", actor, {"version": payload.version})
            self._save(state)
            return source

    def transition(self, source_id: str, payload: AcquisitionEvent, actor: str) -> dict[str, Any]:
        transitions = {"REGISTERED": {"ACCESS_PENDING", "REJECTED", "RESTRICTED"},
            "ACCESS_PENDING": {"ACCESS_APPROVED", "REJECTED", "RESTRICTED"},
            "ACCESS_APPROVED": {"ACQUIRED", "REJECTED", "RESTRICTED"},
            "ACQUIRED": {"VERIFICATION_PENDING", "REJECTED", "RESTRICTED"},
            "VERIFICATION_PENDING": {"VERIFIED", "REJECTED", "RESTRICTED"},
            "VERIFIED": {"ARCHIVED", "RESTRICTED"}, "RESTRICTED": {"ARCHIVED"}, "REJECTED": {"ARCHIVED"}, "ARCHIVED": set()}
        with self._lock:
            state = self._load()
            source = state["sources"].get(source_id)
            if not source:
                raise KeyError("Dataset source not found.")
            current = source["acquisition_status"]
            if payload.status not in transitions[current]:
                raise OnboardingStateError(f"Invalid acquisition transition from {current} to {payload.status}.")
            if payload.status == "ACCESS_APPROVED" and not payload.authorization_reference:
                raise OnboardingStateError("Access approval requires an externally verifiable authorization reference.")
            source["acquisition_status"] = payload.status
            if payload.status == "ACQUIRED":
                source["acquired_at"] = datetime.now(timezone.utc).isoformat()
            if payload.status == "ACCESS_APPROVED":
                source["governance_approved"] = False  # Acquisition authorization is not training approval.
            if payload.status == "VERIFIED":
                package = self.verify_package(source_id)
                if not package["valid"]:
                    raise OnboardingStateError("Package verification failed; source cannot enter VERIFIED state.")
                source["package"] = package
                source["verification_status"] = "VERIFIED"
                source["integrity_status"] = "INTEGRITY_VERIFIED_ONLY"
            source["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._event(state, source_id, "STATUS_CHANGED", actor, {"from": current, "to": payload.status,
                "reason": payload.reason, "evidence_reference": payload.evidence_reference,
                "authorization_reference": payload.authorization_reference})
            self._save(state)
            return source

    def events(self, source_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._load()["events"]
            return [e for e in rows if source_id is None or e["source_id"] == source_id]

    def verify_package(self, source_id: str) -> dict[str, Any]:
        state = self._load()
        source = state["sources"].get(source_id)
        if not source or source["acquisition_status"] not in {"ACQUIRED", "VERIFICATION_PENDING", "VERIFIED"}:
            raise OnboardingStateError("Only acquired packages may be verified.")
        if not source.get("license_or_terms") or not source.get("data_use_reference") or not source.get("ethics_reference"):
            return {"valid": False, "errors": ["Required license/data-use and ethics references are not documented."], "integrity_status": "NOT_VERIFIED"}
        root_base = self.package_root.resolve()
        package_root = (root_base / source_id / source["version"]).resolve()
        if not package_root.is_relative_to(root_base) or not package_root.is_dir() or package_root.is_symlink():
            return {"valid": False, "errors": ["Authorized package directory is missing or unsafe."], "integrity_status": "NOT_VERIFIED"}
        inventory = []
        errors = []
        intake_report: dict[str, Any] | None = None
        allowed_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".nii", ".gz", ".json", ".csv"}
        direct_identifier = re.compile(r"(?:\bMRN[_-]?\d{4,}\b|\bHOSP[_-]?\d{4,}\b|\b\d{3}-\d{2}-\d{4}\b|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|\b(?:19|20)\d{2}[-_]\d{2}[-_]\d{2}\b)", re.IGNORECASE)
        for current, dirs, files in os.walk(package_root, followlinks=False):
            current_path = Path(current)
            if any((current_path / d).is_symlink() for d in dirs):
                errors.append("Symbolic-link directories are not permitted.")
                dirs[:] = []
                continue
            for name in files:
                path = current_path / name
                if path.is_symlink():
                    errors.append("Symbolic-link files are not permitted.")
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(package_root):
                    errors.append("Package contains a path escaping its authorized root.")
                    continue
                rel = PurePosixPath(path.relative_to(package_root).as_posix())
                if str(rel).startswith(".") or any(part in {"..", "."} for part in rel.parts):
                    errors.append("Package contains an unsafe relative path.")
                    continue
                if direct_identifier.search(path.name):
                    errors.append("Potential direct identifier found in a filename; de-identify filenames before verification.")
                    continue
                if path.suffix.lower() not in allowed_ext:
                    errors.append(f"Unsupported file extension: {path.suffix or '(none)'}")
                    continue
                size = path.stat().st_size
                if size == 0:
                    errors.append(f"Empty file: {rel}")
                    continue
                if size > 10 * 1024**3:
                    errors.append(f"File exceeds 10 GiB safety limit: {rel}")
                    continue
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                inventory.append({"path": str(rel), "size_bytes": size, "sha256": digest.hexdigest()})
        inventory.sort(key=lambda row: row["path"])
        package_manifest = package_root / "manifest.json"
        if errors:
            intake_report = {"status": "NOT_RUN", "reason": "Package-level path or file checks failed."}
        elif package_manifest.is_symlink() or not package_manifest.is_file():
            errors.append("manifest.json is required for Stage 33 intake handoff.")
        else:
            try:
                from ml_service.datasets.track_d_intake import validate_track_d_dataset
                intake = validate_track_d_dataset(package_root, manifest_path=package_manifest,
                    protected_holdout_registry=load_protected_holdouts(self.holdout_registry_root))
                if intake.get("manifest_schema_validation") != "VALID_DRAFT_2020_12":
                    errors.append("Stage 33 manifest validation did not pass.")
                if intake.get("passed_count", 0) == 0:
                    errors.append("Stage 33 intake accepted no records.")
                intake_report = {"discovery_mode": intake.get("discovery_mode"),
                    "manifest_schema_validation": intake.get("manifest_schema_validation"),
                    "total_inspected": intake.get("total_inspected", 0),
                    "passed_count": intake.get("passed_count", 0),
                    "quarantined_count": intake.get("quarantined_count", 0),
                    "rejection_reason_counts": intake.get("rejection_reason_counts", {}),
                    "protected_holdout_registry": intake.get("protected_holdout_registry", {})}
            except Exception as exc:
                errors.append(f"Stage 33 intake validation failed: {type(exc).__name__}")
        inventory_payload = {"source_id": source_id, "version": source["version"], "files": inventory}
        inventory_hash = hashlib.sha256(self._canonical(inventory_payload)).hexdigest()
        if source.get("fingerprint_sha256") and not hmac.compare_digest(source["fingerprint_sha256"].lower(), inventory_hash):
            errors.append("Computed inventory does not match the externally supplied expected fingerprint.")
        verified = {"valid": not errors, "errors": errors, "inventory": inventory,
                "inventory_sha256": inventory_hash,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "manifest_version": "track_d_manifest.schema.json",
                "intake_validation": intake_report,
                "integrity_status": "INTEGRITY_VERIFIED_ONLY" if not errors else "FAILED",
                "clinical_validity": "NOT_ESTABLISHED", "training_authorized": False}
        prior = source.get("package") or {}
        if source.get("acquisition_status") == "VERIFIED" and (
            not verified["valid"] or not hmac.compare_digest(str(prior.get("inventory_sha256", "")), inventory_hash)
        ):
            self._mark_integrity_failure(state, source_id, source, "Reverification detected package modification or validation failure.")
        return verified

    def _mark_integrity_failure(self, state: dict[str, Any], source_id: str, source: dict[str, Any], reason: str) -> None:
        previous = source.get("acquisition_status")
        source["acquisition_status"] = "RESTRICTED"
        source["verification_status"] = "FAILED"
        source["integrity_status"] = "FAILED"
        source["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._event(state, source_id, "POST_VERIFICATION_INTEGRITY_FAILURE", "AUTOMATED_INTEGRITY_CHECK", {"from": previous, "to": "RESTRICTED", "reason": reason})
        self._save(state)

    def stage(self, source_id: str, actor: str) -> dict[str, Any]:
        with self._lock:
            state = self._load()
            source = state["sources"].get(source_id)
            if not source or source.get("verification_status") != "VERIFIED" or not source.get("package", {}).get("valid"):
                raise OnboardingStateError("Only verified packages can be staged.")
            src = (self.package_root.resolve() / source_id / source["version"]).resolve()
            staging_root = self.staging_root.resolve()
            source_stage_dir = staging_root / source_id
            if source_stage_dir.is_symlink() or (source_stage_dir.exists() and not source_stage_dir.resolve().is_relative_to(staging_root)):
                raise OnboardingStateError("Staging destination escapes the configured staging root.")
            dest = source_stage_dir / source["version"]
            if dest.exists():
                raise FileExistsError("This dataset version is already staged.")
            if not src.is_relative_to(self.package_root.resolve()) or not src.is_dir():
                raise OnboardingStateError("Package root is unavailable or unsafe.")
            source_stage_dir.mkdir(parents=True, exist_ok=True)
            temp = Path(tempfile.mkdtemp(prefix=".stage-", dir=dest.parent))
            source["staging_status"] = "STAGED_FOR_STAGE33_INTAKE"
            source["staged_at"] = datetime.now(timezone.utc).isoformat()
            self._event(state, source_id, "PACKAGE_STAGED_FOR_INTAKE", actor, {"version": source["version"], "training_authorized": False})
            try:
                shutil.copytree(src, temp / "package", symlinks=True)
                if self._inventory_hash(temp / "package", source_id, source["version"]) != source["package"]["inventory_sha256"]:
                    self._mark_integrity_failure(state, source_id, source, "Package changed after verification; staging denied.")
                    raise OnboardingStateError("Package changed after verification; staging rolled back.")
                provenance = {"schema_version": self.SCHEMA_VERSION, "source_id": source_id,
                    "source_reference": source["source_reference"], "source_version": source["version"],
                    "registered_at": source["registered_at"], "acquired_at": source.get("acquired_at"),
                    "verified_at": source["package"].get("verified_at"),
                    "inventory_sha256": source["package"]["inventory_sha256"],
                    "file_inventory": source["package"]["inventory"],
                    "event_chain": [row for row in state["events"] if row["source_id"] == source_id], "integrity_status": "INTEGRITY_VERIFIED_ONLY",
                    "clinical_validity": "NOT_ESTABLISHED", "governance_approved": False,
                    "training_authorized": False, "restrictions": source["restrictions"]}
                (temp / "package" / "provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2), encoding="utf-8")
                os.replace(temp / "package", dest)
            finally:
                shutil.rmtree(temp, ignore_errors=True)
            try:
                self._save(state)
            except Exception:
                shutil.rmtree(dest, ignore_errors=True)
                raise
            return {"source_id": source_id, "version": source["version"], "status": source["staging_status"], "training_authorized": False}

    def _inventory_hash(self, root: Path, source_id: str, version: str) -> str:
        files = []
        for path in root.rglob("*"):
            if path.is_symlink():
                raise OnboardingStateError("Staged package contains a symlink.")
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                files.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        files.sort(key=lambda row: row["path"])
        return hashlib.sha256(self._canonical({"source_id": source_id, "version": version, "files": files})).hexdigest()
