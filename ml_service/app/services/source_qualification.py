"""Evidence-backed qualification and acquisition planning for candidate cohorts.

This registry records research leads and reviewer decisions. It does not grant
access, certify ethics, verify data, or authorize model development.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CLASSES = ("SECONDARY_BONE_METASTASIS", "PAGETS_DISEASE", "OSTEOGENESIS_IMPERFECTA", "HEALTHY_SKELETAL_CONTROL")
MODALITIES = ("xray", "bone_scan", "ct")
QUALIFICATION_STATES = ("DISCOVERED", "INFORMATION_VERIFIED", "LICENSE_REVIEW_PENDING", "ACCESS_RESTRICTED", "ACCESS_REQUEST_REQUIRED", "GOVERNANCE_REVIEW_REQUIRED", "QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION", "REJECTED", "ARCHIVED")
EVIDENCE_KINDS = ("VERIFIED_FACT", "SOURCE_CLAIM", "PUBLISHED_CLAIM", "REVIEWER_ASSESSMENT", "UNVERIFIED_ASSUMPTION")
PLAN_STATES = ("DRAFT", "REVIEW_REQUIRED", "READY_FOR_EXTERNAL_REQUEST", "REQUEST_SUBMITTED", "ACCESS_DECISION_PENDING", "APPROVED_EXTERNALLY", "HANDED_OFF_TO_STAGE38", "BLOCKED", "REJECTED", "ARCHIVED")


class QualificationError(ValueError):
    pass


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid4().hex[:16]}", pattern=r"^[A-Za-z0-9_.-]{3,80}$")
    kind: Literal["VERIFIED_FACT", "SOURCE_CLAIM", "PUBLISHED_CLAIM", "REVIEWER_ASSESSMENT", "UNVERIFIED_ASSUMPTION"]
    statement: str = Field(min_length=5, max_length=1200)
    reference_url: str = Field(min_length=8, max_length=1000)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("reference_url")
    @classmethod
    def evidence_url_is_web_link(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("Evidence reference must be an HTTP(S) URL.")
        return value


class SourceQualification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$")
    source_name: str = Field(min_length=2, max_length=200)
    original_url: str = Field(min_length=8, max_length=1000)
    repository_or_institution: str = Field(min_length=2, max_length=200)
    publication_references: list[str] = Field(default_factory=list, max_length=20)
    publication_dois: list[str] = Field(default_factory=list, max_length=20)
    license: str | None = Field(default=None, max_length=300)
    license_verification_status: Literal["UNVERIFIED", "VERIFIED", "PENDING", "NOT_APPLICABLE"] = "UNVERIFIED"
    access_requirements: list[str] = Field(default_factory=list, max_length=30)
    governance_requirements_reviewed: bool = False
    access_request_status: Literal["NOT_STARTED", "REQUIRED", "SUBMITTED", "APPROVED", "DENIED", "UNKNOWN"] = "UNKNOWN"
    data_use_agreement_status: Literal["UNKNOWN", "NOT_REQUIRED_VERIFIED", "REQUIRED", "PENDING", "EXECUTED"] = "UNKNOWN"
    ethics_requirements: list[str] = Field(default_factory=list, max_length=30)
    modalities: list[Literal["xray", "bone_scan", "ct"]] = Field(default_factory=list, max_length=3)
    available_imaging_types: list[str] = Field(default_factory=list, max_length=30)
    available_labels: list[str] = Field(default_factory=list, max_length=100)
    target_class_interest: list[Literal["SECONDARY_BONE_METASTASIS", "PAGETS_DISEASE", "OSTEOGENESIS_IMPERFECTA", "HEALTHY_SKELETAL_CONTROL"]] = Field(default_factory=list, max_length=4)
    label_mapping: dict[str, list[str]] = Field(default_factory=dict)
    patient_id_available: Literal["YES", "NO", "UNKNOWN"] = "UNKNOWN"
    patient_grouping_feasibility: Literal["FEASIBLE", "PARTIAL", "NOT_FEASIBLE", "UNKNOWN"] = "UNKNOWN"
    institution_metadata_available: Literal["YES", "NO", "UNKNOWN"] = "UNKNOWN"
    acquisition_metadata_available: Literal["YES", "NO", "UNKNOWN"] = "UNKNOWN"
    imaging_format: list[str] = Field(default_factory=list, max_length=30)
    verified_dataset_size: str | None = Field(default=None, max_length=100)
    verified_patient_count: int | None = Field(default=None, ge=0)
    verified_image_count: int | None = Field(default=None, ge=0)
    source_reported_dataset_size: str | None = Field(default=None, max_length=100)
    source_reported_patient_count: int | None = Field(default=None, ge=0)
    source_reported_image_count: int | None = Field(default=None, ge=0)
    known_limitations: list[str] = Field(default_factory=list, max_length=50)
    potential_overlap: list[str] = Field(default_factory=list, max_length=30)
    protected_holdout_risk: Literal["NONE_IDENTIFIED", "POSSIBLE", "CONFIRMED", "UNKNOWN"] = "UNKNOWN"
    privacy_governance_concerns: list[str] = Field(default_factory=list, max_length=30)
    acquisition_feasibility: Literal["UNKNOWN", "LOW", "MODERATE", "HIGH"] = "UNKNOWN"
    qualification_status: Literal["DISCOVERED", "INFORMATION_VERIFIED", "LICENSE_REVIEW_PENDING", "ACCESS_RESTRICTED", "ACCESS_REQUEST_REQUIRED", "GOVERNANCE_REVIEW_REQUIRED", "QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION", "REJECTED", "ARCHIVED"] = "DISCOVERED"
    evidence: list[Evidence] = Field(default_factory=list, max_length=100)
    reviewer_notes: str | None = Field(default=None, max_length=3000)
    reviewed_at: datetime | None = None
    responsible_reviewer_reference: str | None = Field(default=None, max_length=160)
    decision_evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    assessment_references: dict[str, str | None] = Field(default_factory=dict)
    assessment_evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    assessment_reviewer_reference: str | None = Field(default=None, max_length=160)
    assessment_reviewed_at: datetime | None = None
    # REVIEWED claims/counters are evidence-qualified only when tied to evidence IDs.
    count_evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage38_source_id: str | None = None

    @field_validator("original_url")
    @classmethod
    def original_url_is_web_link(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("Original source URL must be an HTTP(S) URL.")
        return value

    @model_validator(mode="after")
    def valid_mapping(self):
        if any(key not in CLASSES for key in self.label_mapping):
            raise ValueError("Label mapping keys must use the fixed Track D ontology.")
        evidence_ids = {item.evidence_id for item in self.evidence}
        if any(item not in evidence_ids for item in self.decision_evidence_ids + self.assessment_evidence_ids):
            raise ValueError("Reviewer decisions and assessments must reference evidence on this record.")
        if any(item not in evidence_ids for item in self.count_evidence_ids):
            raise ValueError("Reported dataset counts must reference evidence on this record.")
        if any(value is not None for value in (self.verified_patient_count, self.verified_image_count, self.verified_dataset_size,
                                                self.source_reported_patient_count, self.source_reported_image_count, self.source_reported_dataset_size)) and not self.count_evidence_ids:
            raise ValueError("Dataset size and count claims must reference evidence.")
        if self.qualification_status in {"QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION"}:
            if not self.responsible_reviewer_reference or not self.reviewed_at:
                raise ValueError("Qualification requires an explicit reviewer reference and review timestamp.")
        return self


class QualificationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: SourceQualification


class EvidenceAdd(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence: Evidence


class QualificationReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["INFORMATION_VERIFIED", "LICENSE_REVIEW_PENDING", "ACCESS_RESTRICTED", "ACCESS_REQUEST_REQUIRED", "GOVERNANCE_REVIEW_REQUIRED", "QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION", "REJECTED", "ARCHIVED"]
    evidence_ids: list[str] = Field(min_length=1, max_length=30)
    explicit_confirmation: bool
    notes: str = Field(min_length=3, max_length=2000)


class SourceAssessmentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str] = Field(min_length=1, max_length=30)
    explicit_confirmation: bool
    notes: str = Field(min_length=3, max_length=2000)
    license_verification_status: Literal["UNVERIFIED", "VERIFIED", "PENDING", "NOT_APPLICABLE"] | None = None
    access_request_status: Literal["NOT_STARTED", "REQUIRED", "SUBMITTED", "APPROVED", "DENIED", "UNKNOWN"] | None = None
    data_use_agreement_status: Literal["UNKNOWN", "NOT_REQUIRED_VERIFIED", "REQUIRED", "PENDING", "EXECUTED"] | None = None
    access_authorization_reference: str | None = Field(default=None, max_length=300)
    data_use_agreement_reference: str | None = Field(default=None, max_length=300)
    access_requirements: list[str] | None = Field(default=None, max_length=30)
    ethics_requirements: list[str] | None = Field(default=None, max_length=30)
    governance_requirements_reviewed: bool | None = None
    patient_id_available: Literal["YES", "NO", "UNKNOWN"] | None = None
    patient_grouping_feasibility: Literal["FEASIBLE", "PARTIAL", "NOT_FEASIBLE", "UNKNOWN"] | None = None
    institution_metadata_available: Literal["YES", "NO", "UNKNOWN"] | None = None
    acquisition_metadata_available: Literal["YES", "NO", "UNKNOWN"] | None = None
    target_class_interest: list[Literal["SECONDARY_BONE_METASTASIS", "PAGETS_DISEASE", "OSTEOGENESIS_IMPERFECTA", "HEALTHY_SKELETAL_CONTROL"]] | None = None
    label_mapping: dict[str, list[str]] | None = None
    source_reported_patient_count: int | None = Field(default=None, ge=0)
    source_reported_image_count: int | None = Field(default=None, ge=0)
    source_reported_dataset_size: str | None = Field(default=None, max_length=100)


class AcquisitionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str = Field(default_factory=lambda: f"acq-{uuid4().hex[:16]}")
    source_id: str
    method: str = Field(min_length=3, max_length=500)
    access_request_instructions: str = Field(min_length=3, max_length=2000)
    required_application: str | None = Field(default=None, max_length=1000)
    required_institutional_approval: list[str] = Field(default_factory=list, max_length=30)
    license_requirements: list[str] = Field(default_factory=list, max_length=30)
    data_use_agreement: str | None = Field(default=None, max_length=1000)
    transfer_method: str = Field(min_length=3, max_length=500)
    storage_location_reference: str = Field(min_length=3, max_length=300)
    expected_package_structure: str = Field(min_length=3, max_length=1000)
    expected_modalities: list[Literal["xray", "bone_scan", "ct"]] = Field(min_length=1)
    expected_labels: list[str] = Field(min_length=1, max_length=100)
    expected_manifest: str = Field(min_length=3, max_length=500)
    expected_provenance: list[str] = Field(min_length=1, max_length=30)
    verification_requirements: list[str] = Field(min_length=1, max_length=30)
    responsible_researcher_reference: str = Field(min_length=3, max_length=160)
    status: Literal["DRAFT", "REVIEW_REQUIRED", "READY_FOR_EXTERNAL_REQUEST", "REQUEST_SUBMITTED", "ACCESS_DECISION_PENDING", "APPROVED_EXTERNALLY", "HANDED_OFF_TO_STAGE38", "BLOCKED", "REJECTED", "ARCHIVED"] = "DRAFT"
    blockers: list[str] = Field(default_factory=list, max_length=50)
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage38_source_id: str | None = None


class Stage38Handoff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage38_source_id: str = Field(min_length=3, max_length=80)
    explicit_confirmation: bool
    notes: str = Field(min_length=3, max_length=1000)


class SourceQualificationService:
    SCHEMA_VERSION = "1.0.0"
    _lock = RLock()
    _transitions = {
        "DISCOVERED": {"INFORMATION_VERIFIED", "REJECTED", "ARCHIVED"},
        "INFORMATION_VERIFIED": {"LICENSE_REVIEW_PENDING", "ACCESS_RESTRICTED", "ACCESS_REQUEST_REQUIRED", "GOVERNANCE_REVIEW_REQUIRED", "QUALIFIED_FOR_REQUEST", "REJECTED", "ARCHIVED"},
        "LICENSE_REVIEW_PENDING": {"INFORMATION_VERIFIED", "ACCESS_RESTRICTED", "ACCESS_REQUEST_REQUIRED", "GOVERNANCE_REVIEW_REQUIRED", "QUALIFIED_FOR_REQUEST", "REJECTED", "ARCHIVED"},
        "ACCESS_RESTRICTED": {"GOVERNANCE_REVIEW_REQUIRED", "REJECTED", "ARCHIVED"},
        "ACCESS_REQUEST_REQUIRED": {"QUALIFIED_FOR_REQUEST", "GOVERNANCE_REVIEW_REQUIRED", "ACCESS_RESTRICTED", "REJECTED", "ARCHIVED"},
        "GOVERNANCE_REVIEW_REQUIRED": {"QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION", "ACCESS_RESTRICTED", "REJECTED", "ARCHIVED"},
        "QUALIFIED_FOR_REQUEST": {"GOVERNANCE_REVIEW_REQUIRED", "QUALIFIED_FOR_ACQUISITION", "ACCESS_RESTRICTED", "REJECTED", "ARCHIVED"},
        "QUALIFIED_FOR_ACQUISITION": {"GOVERNANCE_REVIEW_REQUIRED", "ACCESS_RESTRICTED", "REJECTED", "ARCHIVED"},
        "REJECTED": {"ARCHIVED"}, "ARCHIVED": set(),
    }
    _plan_transitions = {"DRAFT": {"REVIEW_REQUIRED", "BLOCKED", "REJECTED"}, "REVIEW_REQUIRED": {"READY_FOR_EXTERNAL_REQUEST", "BLOCKED", "REJECTED"}, "READY_FOR_EXTERNAL_REQUEST": {"REQUEST_SUBMITTED", "BLOCKED", "REJECTED"}, "REQUEST_SUBMITTED": {"ACCESS_DECISION_PENDING", "APPROVED_EXTERNALLY", "BLOCKED", "REJECTED"}, "ACCESS_DECISION_PENDING": {"APPROVED_EXTERNALLY", "BLOCKED", "REJECTED"}, "APPROVED_EXTERNALLY": {"HANDED_OFF_TO_STAGE38", "BLOCKED", "ARCHIVED"}, "HANDED_OFF_TO_STAGE38": {"ARCHIVED"}, "BLOCKED": {"REVIEW_REQUIRED", "REJECTED", "ARCHIVED"}, "REJECTED": {"ARCHIVED"}, "ARCHIVED": set()}

    def __init__(self, state_file: Path | None = None):
        self.state_file = state_file or Path(os.environ.get("OSTEO_SOURCE_QUALIFICATION_FILE", "ml_service/data/qualification/registry.json"))
        self.hmac_key = os.environ.get("OSTEO_SOURCE_QUALIFICATION_HMAC_KEY", "").encode()
        if os.environ.get("OSTEO_ENV", "development").lower() == "production" and len(self.hmac_key) < 32:
            raise QualificationError("Production qualification registry requires an external HMAC key of at least 32 characters.")

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

    def _digest(self, value: Any) -> str:
        raw = self._canonical(value)
        return hmac.new(self.hmac_key, raw, hashlib.sha256).hexdigest() if self.hmac_key else hashlib.sha256(raw).hexdigest()

    def _empty(self):
        state = {"schema_version": self.SCHEMA_VERSION, "sources": self._initial_catalog(), "plans": {}, "events": []}
        for source_id in state["sources"]:
            self._event(state, source_id, "CURATED_LEAD_LOADED", "STATIC_CATALOG", {"qualification_status": "DISCOVERED", "reviewed": False})
        state["integrity"] = self._digest(state)
        return state

    @staticmethod
    def _initial_catalog() -> dict[str, dict[str, Any]]:
        """Curated source leads only; all are unreviewed DISCOVERED records."""
        leads = [
            ("TCIA_SPINE_METS_CT_SEG", "TCIA Spine-Mets-CT-SEG", "https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/", "The Cancer Imaging Archive", ["ct"], ["DICOM CT", "DICOM SEG"], "Official page lists CC BY 4.0; lesion morphology labels are not the Track D disease ontology.", "CC BY 4.0", "VERIFIED", "NOT_STARTED", ["TCIA Data Retriever download"], ["Not a direct four-class diagnosis dataset.", "Review repeated/pre/post treatment scans and grouping metadata."]),
            ("TCIA_HEALTHY_TOTAL_BODY_CT", "TCIA Healthy-Total-Body-CTs", "https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/", "The Cancer Imaging Archive", ["ct"], ["DICOM CT", "NIfTI segmentation"], "Official page reports 30 subjects and NIH Controlled Data Access Policy for CT images; image access is controlled.", "NIH Controlled Data Access Policy", "VERIFIED", "REQUIRED", ["NIH controlled-data request", "Review face reconstruction risk"], ["Controlled CT access; separate component licenses.", "Confirm healthy-control definition and repeated timepoint grouping."]),
            ("IICS_UNA_BONE_SCAN", "IICS-UNA Bone Scan Images", "https://zenodo.org/records/13900966", "IICS-UNA / Zenodo", ["bone_scan"], ["Planar bone scintigraphy PNG", "Excel labels"], "Repository/paper reports 582 images from 291 adult breast cancer patients; article license is CC BY-NC 4.0, image package terms need direct review.", "Article CC BY-NC 4.0; image terms pending", "PENDING", "UNKNOWN", ["Verify Zenodo component license and source terms"], ["No Paget/OI/healthy cohorts.", "No-metastasis labels in a breast-cancer cohort are not healthy controls."]),
            ("BS80K_BONE_SCINTIGRAPHY", "BS-80K Bone Scan Dataset", "https://doi.org/10.1016/j.compbiomed.2022.106221", "West China Hospital (paper-reported source)", ["bone_scan"], ["Whole-body and regional bone scintigraphy images", "Specialist annotations"], "Publication reports 82,544 images associated with 3,247 patients and links a Google Drive package; image license was not verified.", None, "PENDING", "UNKNOWN", ["Request/verify image license and package documentation"], ["Publication counts are not acquired local cohort counts.", "No four-class label mapping.", "Confirm patient identifiers and exact label definitions."]),
            ("EURREB_OI_REGISTRY", "EuRREB Osteogenesis Imperfecta Registry Module", "https://eurreb.eu/osteogenesis-imperfecta-oi/", "European Registries for Rare Endocrine and Bone Conditions", [], ["Clinical registry records; imaging not documented"], "Official module lists clinical variables; its public description does not identify an imaging dataset.", None, "UNVERIFIED", "REQUIRED", ["Registry data request and governance review"], ["Registry lead only; do not claim it contains images.", "Data request and agreements required."]),
            ("ROTTERDAM_PAGET_RADIOLOGY", "Rotterdam Study Paget Radiological Survey", "https://pubmed.ncbi.nlm.nih.gov/15005843/", "Rotterdam Study / academic publication", ["xray"], ["Radiographs described in publication"], "Publication reports a radiological survey; no reusable public image package or image license was identified.", None, "UNVERIFIED", "REQUIRED", ["Contact study owners; institutional access and governance review"], ["Publication is not dataset access.", "No public image package identified."]),
            ("NEW_ENGLAND_PAGET_REGISTRY", "New England Registry for Paget Disease study", "https://pubmed.ncbi.nlm.nih.gov/20858558/", "Academic publication / registry", ["xray"], ["Radiographic records described in publication"], "Publication describes a registry linked to radiographic records; public image package was not identified.", None, "UNVERIFIED", "REQUIRED", ["Contact registry/study owners; institutional access review"], ["Publication is not dataset access.", "No public image package identified."]),
        ]
        rows = {}
        for source_id, name, url, repository, modalities, imaging, statement, license, license_state, access_state, access, limitations in leads:
            evidence_id = f"ev-catalog-{source_id.lower()}"
            evidence = Evidence(evidence_id=evidence_id, kind="SOURCE_CLAIM", statement=statement, reference_url=url, notes="Curated source lead; not independently reviewed by an authorized institutional reviewer.")
            row = SourceQualification(source_id=source_id, source_name=name, original_url=url, repository_or_institution=repository,
                modalities=modalities, available_imaging_types=imaging, license=license, license_verification_status=license_state,
                access_request_status=access_state, access_requirements=access, known_limitations=limitations, evidence=[evidence],
                qualification_status="DISCOVERED")
            interests = {
                "TCIA_SPINE_METS_CT_SEG": ["SECONDARY_BONE_METASTASIS"],
                "TCIA_HEALTHY_TOTAL_BODY_CT": ["HEALTHY_SKELETAL_CONTROL"],
                "IICS_UNA_BONE_SCAN": ["SECONDARY_BONE_METASTASIS"],
                "BS80K_BONE_SCINTIGRAPHY": ["SECONDARY_BONE_METASTASIS"],
                "EURREB_OI_REGISTRY": ["OSTEOGENESIS_IMPERFECTA"],
                "ROTTERDAM_PAGET_RADIOLOGY": ["PAGETS_DISEASE"],
                "NEW_ENGLAND_PAGET_REGISTRY": ["PAGETS_DISEASE"],
            }
            row = row.model_copy(update={"target_class_interest": interests[source_id]})
            reported_counts = {
                "TCIA_SPINE_METS_CT_SEG": {"source_reported_dataset_size": "20.36 GB", "source_reported_patient_count": 55, "source_reported_image_count": 35582},
                "TCIA_HEALTHY_TOTAL_BODY_CT": {"source_reported_dataset_size": "58.95 GB", "source_reported_patient_count": 30, "source_reported_image_count": 111778},
                "IICS_UNA_BONE_SCAN": {"source_reported_patient_count": 291, "source_reported_image_count": 582},
                "BS80K_BONE_SCINTIGRAPHY": {"source_reported_patient_count": 3247, "source_reported_image_count": 82544},
            }
            if source_id in reported_counts:
                row = row.model_copy(update={**reported_counts[source_id], "count_evidence_ids": [evidence_id]})
            rows[source_id] = row.model_dump(mode="json")
        return rows

    def _load(self):
        if not self.state_file.exists():
            return self._empty()
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QualificationError("Qualification registry is unavailable or malformed.") from exc
        integrity = state.pop("integrity", None) if isinstance(state, dict) else None
        if not isinstance(state, dict) or state.get("schema_version") != self.SCHEMA_VERSION or not hmac.compare_digest(str(integrity or ""), self._digest(state)):
            raise QualificationError("Qualification registry integrity check failed.")
        state["integrity"] = integrity
        previous = "0" * 64
        for stored in state["events"]:
            event = dict(stored); event_hash = event.pop("event_hash", None)
            if event.get("previous_hash") != previous or not isinstance(event_hash, str) or not hmac.compare_digest(event_hash, hashlib.sha256(self._canonical(event)).hexdigest()):
                raise QualificationError("Qualification audit event chain verification failed.")
            previous = event_hash
        return state

    def _save(self, state):
        state.pop("integrity", None); state["integrity"] = self._digest(state)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.state_file.is_symlink():
            raise QualificationError("Registry symlink paths are not permitted.")
        fd, tmp = tempfile.mkstemp(prefix=".qualification-", dir=self.state_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(state, out, sort_keys=True, indent=2, ensure_ascii=False, default=lambda x: x.isoformat() if isinstance(x, datetime) else str(x)); out.flush(); os.fsync(out.fileno())
            os.replace(tmp, self.state_file)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def _event(self, state, entity_id, action, actor, detail):
        prev = state["events"][-1]["event_hash"] if state["events"] else "0" * 64
        event = {"at": datetime.now(timezone.utc).isoformat(), "entity_id": entity_id, "action": action, "actor_reference": actor, "detail": detail, "previous_hash": prev}
        event["event_hash"] = hashlib.sha256(self._canonical(event)).hexdigest(); state["events"].append(event)

    def create_source(self, source: SourceQualification, actor: str):
        with self._lock:
            state = self._load()
            if source.source_id in state["sources"]: raise FileExistsError("Qualification source ID already exists.")
            if source.qualification_status != "DISCOVERED": raise QualificationError("New source records must start in DISCOVERED state.")
            if (source.responsible_reviewer_reference or source.reviewed_at or source.stage38_source_id
                    or source.assessment_reviewer_reference or source.assessment_reviewed_at or source.decision_evidence_ids
                    or source.assessment_evidence_ids or source.assessment_references):
                raise QualificationError("New source records cannot claim reviewer review or a Stage38 handoff.")
            if source.license_verification_status == "VERIFIED" or source.access_request_status in {"SUBMITTED", "APPROVED", "DENIED"} or source.data_use_agreement_status == "EXECUTED":
                raise QualificationError("Verified license/access/agreement states must be recorded by an evidence-linked reviewer update.")
            if source.verified_patient_count is not None or source.verified_image_count is not None or source.verified_dataset_size is not None:
                raise QualificationError("Verified counts are not accepted during candidate registration.")
            row = source.model_dump(mode="json"); row["created_by_reference"] = actor
            state["sources"][source.source_id] = row; self._event(state, source.source_id, "SOURCE_DISCOVERED", actor, {})
            self._save(state); return row

    def list_sources(self):
        with self._lock: return list(self._load()["sources"].values())

    def get_source(self, source_id):
        with self._lock:
            try: return self._load()["sources"][source_id]
            except KeyError as exc: raise KeyError("Qualification source not found.") from exc

    def add_evidence(self, source_id: str, item: Evidence, actor: str):
        with self._lock:
            state = self._load(); row = state["sources"].get(source_id)
            if not row: raise KeyError("Qualification source not found.")
            if any(ev["evidence_id"] == item.evidence_id for ev in row["evidence"]): raise FileExistsError("Evidence identifier already exists.")
            row["evidence"].append(item.model_dump(mode="json")); row["version"] += 1; row["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._event(state, source_id, "EVIDENCE_ADDED", actor, {"evidence_id": item.evidence_id, "kind": item.kind})
            self._save(state); return row

    def review(self, source_id: str, decision: QualificationReview, actor: str):
        with self._lock:
            state = self._load(); row = state["sources"].get(source_id)
            if not row: raise KeyError("Qualification source not found.")
            current = row["qualification_status"]
            if decision.status not in self._transitions[current]: raise QualificationError(f"Invalid qualification transition {current} -> {decision.status}.")
            known = {ev["evidence_id"]: ev for ev in row["evidence"]}
            if not decision.explicit_confirmation: raise QualificationError("Reviewer must explicitly confirm this evidence-backed decision.")
            if any(eid not in known for eid in decision.evidence_ids): raise QualificationError("Every qualification decision must reference evidence on this source record.")
            if any(known[eid]["kind"] == "UNVERIFIED_ASSUMPTION" for eid in decision.evidence_ids): raise QualificationError("Unverified assumptions cannot support a qualification decision.")
            if decision.status in {"QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION"}:
                if not any(known[eid]["kind"] in {"VERIFIED_FACT", "SOURCE_CLAIM", "PUBLISHED_CLAIM", "REVIEWER_ASSESSMENT"} for eid in decision.evidence_ids): raise QualificationError("Qualification requires evidence-backed findings.")
            if decision.status == "QUALIFIED_FOR_REQUEST" and (row["access_request_status"] == "UNKNOWN" or row["license_verification_status"] == "UNVERIFIED"):
                raise QualificationError("Request qualification requires reviewed access-route and license status.")
            if decision.status == "QUALIFIED_FOR_ACQUISITION" and (
                row["license_verification_status"] not in {"VERIFIED", "NOT_APPLICABLE"}
                or row["access_request_status"] == "UNKNOWN"
                or row["data_use_agreement_status"] == "UNKNOWN"
                or row["governance_requirements_reviewed"] is not True
            ):
                raise QualificationError("Acquisition qualification requires reviewed license, access, data-use, and governance requirements; this does not grant permission.")
            row.update({"qualification_status": decision.status, "reviewer_notes": decision.notes, "reviewed_at": datetime.now(timezone.utc).isoformat(), "responsible_reviewer_reference": actor, "decision_evidence_ids": decision.evidence_ids, "version": row["version"] + 1, "updated_at": datetime.now(timezone.utc).isoformat()})
            # Pydantic model revalidation catches reviewer/status omissions.
            SourceQualification.model_validate({k: v for k, v in row.items() if k != "created_by_reference"})
            self._event(state, source_id, "QUALIFICATION_REVIEWED", actor, {"from": current, "to": decision.status, "evidence_ids": decision.evidence_ids, "notes": decision.notes})
            self._save(state); return row

    def update_assessment(self, source_id: str, update: SourceAssessmentUpdate, actor: str):
        with self._lock:
            state = self._load(); row = state["sources"].get(source_id)
            if not row: raise KeyError("Qualification source not found.")
            if not update.explicit_confirmation: raise QualificationError("An explicit reviewer confirmation is required.")
            evidence = {ev["evidence_id"]: ev for ev in row["evidence"]}
            if any(eid not in evidence or evidence[eid]["kind"] == "UNVERIFIED_ASSUMPTION" for eid in update.evidence_ids):
                raise QualificationError("Assessment updates must reference existing non-assumption evidence.")
            values = update.model_dump(exclude={"evidence_ids", "explicit_confirmation", "notes"}, exclude_none=True)
            if values.get("access_request_status") == "APPROVED" and not values.get("access_authorization_reference"):
                raise QualificationError("Approved access status requires an external authorization reference.")
            if values.get("data_use_agreement_status") == "EXECUTED" and not values.get("data_use_agreement_reference"):
                raise QualificationError("Executed data-use status requires an external agreement reference.")
            if "label_mapping" in values and any(key not in CLASSES for key in values["label_mapping"]):
                raise QualificationError("Label mapping keys must use the fixed Track D ontology.")
            if any(k in values for k in ("source_reported_patient_count", "source_reported_image_count", "source_reported_dataset_size")):
                values["count_evidence_ids"] = update.evidence_ids
            values.pop("access_authorization_reference", None); values.pop("data_use_agreement_reference", None)
            values["assessment_references"] = {"access_authorization_reference": update.access_authorization_reference,
                "data_use_agreement_reference": update.data_use_agreement_reference}
            row.update(values); row["assessment_evidence_ids"] = update.evidence_ids
            row["assessment_reviewer_reference"] = actor; row["assessment_reviewed_at"] = datetime.now(timezone.utc).isoformat()
            row["reviewer_notes"] = update.notes; row["version"] += 1; row["updated_at"] = datetime.now(timezone.utc).isoformat()
            SourceQualification.model_validate({k: v for k, v in row.items() if k not in {"created_by_reference", "decision_evidence_ids", "assessment_evidence_ids", "assessment_reviewer_reference", "assessment_reviewed_at", "assessment_references"}})
            self._event(state, source_id, "SOURCE_ASSESSMENT_UPDATED", actor, {"evidence_ids": update.evidence_ids, "fields": sorted(values.keys()), "notes": update.notes})
            self._save(state); return row

    def create_plan(self, plan: AcquisitionPlan, actor: str):
        with self._lock:
            state = self._load(); source = state["sources"].get(plan.source_id)
            if not source: raise KeyError("Qualification source not found.")
            if plan.plan_id in state["plans"]: raise FileExistsError("Acquisition plan identifier already exists.")
            if plan.status != "DRAFT" or plan.stage38_source_id or plan.audit_events or plan.blockers:
                raise QualificationError("New acquisition plans must be unadvanced, unlinked, and contain no user-supplied audit history.")
            if source["qualification_status"] not in {"QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION", "GOVERNANCE_REVIEW_REQUIRED", "ACCESS_REQUEST_REQUIRED"}:
                plan.blockers.append("Source qualification review is required before an acquisition request.")
            if source["qualification_status"] != "QUALIFIED_FOR_ACQUISITION": plan.blockers.append("Acquisition qualification is not complete; this plan cannot imply access approval.")
            row = plan.model_dump(mode="json"); row["created_by_reference"] = actor
            row["audit_events"] = [{"at": datetime.now(timezone.utc).isoformat(), "action": "PLAN_CREATED", "actor_reference": actor}]
            state["plans"][plan.plan_id] = row; self._event(state, plan.plan_id, "ACQUISITION_PLAN_CREATED", actor, {"source_id": plan.source_id})
            self._save(state); return row

    def list_plans(self):
        with self._lock: return list(self._load()["plans"].values())

    def get_plan(self, plan_id):
        with self._lock:
            try: return self._load()["plans"][plan_id]
            except KeyError as exc: raise KeyError("Acquisition plan not found.") from exc

    def transition_plan(self, plan_id, new_status, actor, note, external_authorization_reference=None):
        with self._lock:
            state = self._load(); row = state["plans"].get(plan_id)
            if not row: raise KeyError("Acquisition plan not found.")
            old = row["status"]
            if new_status not in self._plan_transitions[old]: raise QualificationError(f"Invalid acquisition plan transition {old} -> {new_status}.")
            if new_status == "READY_FOR_EXTERNAL_REQUEST":
                source = state["sources"].get(row["source_id"])
                if not source or source["qualification_status"] not in {"QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION"}:
                    raise QualificationError("Source must be explicitly qualified for a request before the plan can be marked ready.")
                row["blockers"] = [item for item in row["blockers"] if "Source qualification review" not in item and "Acquisition qualification is not complete" not in item]
                if row["blockers"] or not note: raise QualificationError("Resolve documented blockers and provide a transition note before external requests.")
            if new_status == "APPROVED_EXTERNALLY" and not external_authorization_reference: raise QualificationError("External access approval requires an explicit externally verifiable authorization reference.")
            if new_status == "APPROVED_EXTERNALLY": row["external_authorization_reference"] = external_authorization_reference
            row["status"] = new_status; row["updated_at"] = datetime.now(timezone.utc).isoformat(); row["audit_events"].append({"at": row["updated_at"], "action": "STATUS_CHANGED", "from": old, "to": new_status, "actor_reference": actor, "note": note})
            self._event(state, plan_id, "ACQUISITION_PLAN_STATUS_CHANGED", actor, {"from": old, "to": new_status})
            self._save(state); return row

    def handoff_stage38(self, source_id, handoff: Stage38Handoff, actor, onboarding_service):
        """Link an existing Stage38 registered source; never fabricates or registers one."""
        with self._lock:
            state = self._load(); row = state["sources"].get(source_id)
            if not row: raise KeyError("Qualification source not found.")
            if row["qualification_status"] != "QUALIFIED_FOR_ACQUISITION": raise QualificationError("Only an explicitly reviewed acquisition-qualified source can be handed to Stage38.")
            if row.get("stage38_source_id"): raise FileExistsError("Source already has a Stage38 handoff.")
            if not handoff.explicit_confirmation: raise QualificationError("An explicit reviewer confirmation is required for the Stage38 handoff.")
            stage38 = next((item for item in onboarding_service.list_sources() if item["source_id"] == handoff.stage38_source_id), None)
            if not stage38: raise KeyError("Stage38 source registration must be created separately by an authorized operator.")
            if stage38.get("source_reference", "").rstrip("/") != row["original_url"].rstrip("/"):
                raise QualificationError("Stage38 source reference does not match the qualified original source URL.")
            if not set(stage38.get("modalities", [])).intersection(row["modalities"]):
                raise QualificationError("Stage38 source modality does not match the qualified source record.")
            row["stage38_source_id"] = handoff.stage38_source_id; row["version"] += 1; row["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._event(state, source_id, "STAGE38_HANDOFF_LINKED", actor, {"stage38_source_id": handoff.stage38_source_id, "notes": handoff.notes})
            self._save(state); return row

    def coverage(self, onboarding_sources=None, prepared_source_ids=None):
        sources = self.list_sources(); onboarding_sources = onboarding_sources or []; prepared = set(prepared_source_ids or [])
        coverage = []
        for disease in CLASSES:
            for modality in MODALITIES:
                rows = [s for s in sources if modality in s["modalities"] and (disease in s["target_class_interest"] or disease in s["label_mapping"])
                        and any(e["kind"] != "UNVERIFIED_ASSUMPTION" for e in s["evidence"])]
                ids = {s["source_id"] for s in rows}
                mapped = [s for s in rows if disease in s["label_mapping"]]
                registered = [s for s in onboarding_sources if s["source_id"] in {x.get("stage38_source_id") for x in rows if x.get("stage38_source_id")}]
                coverage.append({"diagnostic_class": disease, "modality": modality,
                    "candidate_source_ids": sorted(ids), "ontology_mapping_source_ids": sorted(s["source_id"] for s in mapped),
                    "qualification_statuses": sorted({s["qualification_status"] for s in rows}),
                    "evidence_backed_candidate": any(any(e["kind"] != "UNVERIFIED_ASSUMPTION" for e in s["evidence"]) for s in rows),
                    "access_requirements": sorted({a for s in rows for a in s["access_requirements"]}),
                    "label_availability": sorted({lab for s in rows for lab in s["available_labels"]}),
                    "patient_grouping_feasibility": sorted({s["patient_grouping_feasibility"] for s in rows}),
                    "source_reported_counts": [{"source_id": s["source_id"], "patients": s["source_reported_patient_count"], "images": s["source_reported_image_count"], "evidence_ids": s["count_evidence_ids"]} for s in rows if s["source_reported_patient_count"] is not None or s["source_reported_image_count"] is not None],
                    "stage38_registered": bool(registered), "acquired": any(x.get("acquisition_status") == "ACQUIRED" for x in registered),
                    "verified": any(x.get("verification_status") == "VERIFIED" for x in registered),
                    "staged": any(x.get("staging_status") in {"STAGED", "STAGED_FOR_STAGE33_INTAKE"} for x in registered),
                    "prepared": any(x.get("source_id") in prepared for x in registered),
                    "blockers": sorted({b for s in rows for b in s["known_limitations"]})})
        return {"basis": "CANDIDATE_SOURCE_RECORDS; NOT A PATIENT COHORT", "rows": coverage,
                "missing_class_modality_pairs": [{"diagnostic_class": r["diagnostic_class"], "modality": r["modality"]} for r in coverage if not r["ontology_mapping_source_ids"]],
                "actual_patient_counts": None}

    def feasibility(self, onboarding_sources=None, prepared_source_ids=None):
        matrix = self.coverage(onboarding_sources, prepared_source_ids); sources = self.list_sources()
        return {"basis": "THEORETICAL_CANDIDATE_FEASIBILITY; NOT AN ACQUIRED COHORT", "coverage": matrix,
                "candidate_source_count": len(sources), "acquired_source_count": sum(1 for x in (onboarding_sources or []) if x.get("acquisition_status") in {"ACQUIRED", "VERIFICATION_PENDING", "VERIFIED"}),
                "actual_patient_count": None, "missing_classes": sorted({r["diagnostic_class"] for r in matrix["rows"] if not any(s["label_mapping"].get(r["diagnostic_class"]) and r["modality"] in s["modalities"] and s["qualification_status"] in {"QUALIFIED_FOR_REQUEST", "QUALIFIED_FOR_ACQUISITION"} for s in sources)}),
                "blockers": ["No candidate record grants access or proves acquisition.", "Class mapping requires clinical validation and reviewer confirmation.", "Stage38 verification and Stage39 preparation remain mandatory.", "No clinical cohort counts are inferred from publications or repository claims."],
                "sources": [{"source_id": s["source_id"], "status": s["qualification_status"], "patient_grouping_feasibility": s["patient_grouping_feasibility"], "license_status": s["license_verification_status"], "protected_holdout_risk": s["protected_holdout_risk"], "source_reported_patient_count": s["source_reported_patient_count"], "source_reported_image_count": s["source_reported_image_count"], "count_evidence_ids": s["count_evidence_ids"], "limitations": s["known_limitations"]} for s in sources]}
