"""
OsteoGuard AI — Track D Clinical Adjudication & Label Provenance Service.

Manages sample diagnostic statuses (`ACCEPTED`, `REJECTED`, `OUT_OF_SCOPE`,
`UNCERTAIN`, `REQUIRES_ADJUDICATION`) and end-to-end label provenance tracking
for Track D intake by delegating canonical ontology & eligibility enforcement to
`ml_service.datasets.clinical_ontology` without duplicating rules.

Operates cleanly in pure Python 3.10+ stdlib and supports auditable reviewer
decisions, conflict resolution tracking, and JSON ledger persistence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import hashlib
import hmac
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from ml_service.datasets.clinical_ontology import (
    ONTOLOGY_TO_SHORT_CODE,
    ONTOLOGY_VERSION,
    PSEUDO_REVIEWER_ID_RE,
    AdjudicationStatus,
    OntologyTargetClass,
    PLACEHOLDER_VALUES,
    VALID_EVIDENCE_CATEGORIES,
    adjudicate_diagnostic_label,
    check_label_provenance_completeness,
    normalize_diagnosis_term,
    validate_healthy_control_eligibility,
)
from ml_service.app.services.reviewer_authorization import (
    ReviewerAuthorizationError,
    ReviewerAuthorizationService,
)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Structured diagnostic label provenance snapshot (PHI-free)."""

    sample_id: str
    patient_id: str
    source_diagnosis: str
    candidate_target_class: Optional[str]
    source_report_ref: str
    diagnostic_evidence_category: str
    histopathology_confirmation_status: str
    radiology_report_confirmation_status: str
    clinical_adjudication_status: str
    reviewer_pseudo_id: str
    review_date: str
    label_provenance: str
    label_ontology_version: str
    conflicting_diagnosis_indicator: bool
    provenance_complete: bool
    missing_provenance_fields: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProvenanceTransitionEntry:
    """Append-only audit trail entry recording status & provenance changes."""

    timestamp: str
    sample_id: str
    previous_status: Optional[str]
    new_status: str
    actor_pseudo_id: str
    action: str
    rationale: str
    ontology_version: str
    provenance_snapshot: Dict[str, Any]


class AdjudicationEngine:
    """
    Stateful Track D service managing sample adjudication statuses and
    chronological label provenance tracking.
    """

    def __init__(self, ledger_path: Optional[Path] = None) -> None:
        self.ledger_path = ledger_path
        self._records: Dict[str, Dict[str, Any]] = {}
        self._statuses: Dict[str, Dict[str, Any]] = {}
        self._provenance: Dict[str, ProvenanceRecord] = {}
        self._history: Dict[str, List[ProvenanceTransitionEntry]] = defaultdict(list)
        self._ledger_key = os.environ.get("OSTEO_AUDIT_LEDGER_HMAC_KEY", "").encode("utf-8")
        self._reviewer_authorization = ReviewerAuthorizationService()

        if self.ledger_path is not None and self.ledger_path.exists():
            self.load_ledger(self.ledger_path)

    @staticmethod
    def build_provenance_record(record: Dict[str, Any], adj_result: Dict[str, Any]) -> ProvenanceRecord:
        prov_ok, missing = check_label_provenance_completeness(record)
        return ProvenanceRecord(
            sample_id=str(record.get("sample_id") or "").strip(),
            patient_id=str(record.get("patient_id") or "").strip(),
            source_diagnosis=str(adj_result.get("original_source_diagnosis") or "MISSING"),
            candidate_target_class=adj_result.get("candidate_target_class"),
            source_report_ref=str(record.get("source_report_ref") or "").strip() or "MISSING",
            diagnostic_evidence_category=str(record.get("diagnostic_evidence_category") or "").strip() or "MISSING",
            histopathology_confirmation_status=str(record.get("histopathology_confirmation_status") or "").strip() or "ABSENT",
            radiology_report_confirmation_status=str(record.get("radiology_report_confirmation_status") or "").strip() or "ABSENT",
            clinical_adjudication_status=str(record.get("clinical_adjudication_status") or "").strip() or "PENDING_ADJUDICATION",
            reviewer_pseudo_id=str(record.get("reviewer_pseudo_id") or "").strip() or "UNASSIGNED",
            review_date=str(record.get("review_date") or "").strip() or "UNREVIEWED",
            label_provenance=str(record.get("label_provenance") or "").strip() or "MISSING",
            label_ontology_version=str(record.get("label_ontology_version") or ONTOLOGY_VERSION).strip(),
            conflicting_diagnosis_indicator=bool(record.get("conflicting_diagnosis_indicator")),
            provenance_complete=prov_ok,
            missing_provenance_fields=list(missing),
        )

    def adjudicate_sample(
        self,
        record: Dict[str, Any],
        patient_conflicting_diagnoses: Optional[Set[str]] = None,
        actor_pseudo_id: str = "SYS_ONTOLOGY_V1",
    ) -> Dict[str, Any]:
        """
        Evaluates a single Track D intake record, assigns its canonical adjudication
        status (`ACCEPTED`, `REJECTED`, `OUT_OF_SCOPE`, `UNCERTAIN`, `REQUIRES_ADJUDICATION`),
        records its provenance snapshot, and appends an audit transition entry.
        """
        sample_id = str(record.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("Sample record must define a non-empty 'sample_id'.")

        adj = adjudicate_diagnostic_label(
            record,
            patient_conflicting_diagnoses=patient_conflicting_diagnoses,
        )
        prev_status = self._statuses[sample_id]["adjudication_status"] if sample_id in self._statuses else None
        prov = self.build_provenance_record(record, adj)

        rec_copy = dict(record)
        rec_copy["adjudication_record"] = adj

        self._records[sample_id] = rec_copy
        self._statuses[sample_id] = adj
        self._provenance[sample_id] = prov

        self._history[sample_id].append(
            ProvenanceTransitionEntry(
                timestamp=adj["audit_timestamp"],
                sample_id=sample_id,
                previous_status=prev_status,
                new_status=adj["adjudication_status"],
                actor_pseudo_id=actor_pseudo_id,
                action="INTAKE_ADJUDICATION",
                rationale=adj["reason"],
                ontology_version=ONTOLOGY_VERSION,
                provenance_snapshot=asdict(prov),
            )
        )
        return {
            "sample_id": sample_id,
            "adjudication": adj,
            "provenance": asdict(prov),
        }

    def adjudicate_batch(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a batch of Track D records with two-pass patient conflict detection
        and computes aggregate status and provenance completeness metrics.
        """
        record_list = [dict(r) for r in records]
        patient_classes: Dict[str, Set[str]] = defaultdict(set)
        for rec in record_list:
            pid = str(rec.get("patient_id") or "").strip()
            if not pid:
                continue
            diag = normalize_diagnosis_term(
                str(rec.get("source_diagnosis") or rec.get("diagnostic_class") or "")
            )
            if diag:
                patient_classes[pid].add(diag)

        results: List[Dict[str, Any]] = []
        for rec in record_list:
            pid = str(rec.get("patient_id") or "").strip()
            res = self.adjudicate_sample(
                rec,
                patient_conflicting_diagnoses=patient_classes.get(pid) if pid else None,
            )
            results.append(res)

        summary = self.get_provenance_summary(sample_ids=[r["sample_id"] for r in results])
        return {
            "ontology_version": ONTOLOGY_VERSION,
            "total_evaluated": len(results),
            "results": results,
            "summary": summary,
        }

    def submit_reviewer_decision(
        self,
        sample_id: str,
        decision: AdjudicationStatus | str,
        reviewer_pseudo_id: str,
        rationale: str,
        reviewer_token: Optional[str] = None,
        resolved_target_class: Optional[OntologyTargetClass | str] = None,
        source_report_ref: Optional[str] = None,
        diagnostic_evidence_category: Optional[str] = None,
        diagnostic_evidence: Optional[str] = None,
        radiology_report_confirmation_status: Optional[str] = None,
        histopathology_confirmation_status: Optional[str] = None,
        review_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Applies an authorized human clinician/reviewer adjudication decision to a tracked
        sample while enforcing pseudonymous reviewer IDs, strict ontology bounds,
        provenance completeness, and healthy-control eligibility safeguards.
        """
        if sample_id not in self._records:
            raise KeyError(f"Unknown sample_id '{sample_id}' in AdjudicationEngine.")

        principal = self._reviewer_authorization.authorize_bearer(
            reviewer_token, {"CLINICAL_ADJUDICATOR", "RADIOLOGIST"}
        )
        reviewer_clean = reviewer_pseudo_id.strip()
        if reviewer_clean != principal["reviewer_pseudo_id"]:
            raise ReviewerAuthorizationError("Reviewer identity does not match authenticated credentials.")
        if not PSEUDO_REVIEWER_ID_RE.match(reviewer_clean):
            raise ValueError(
                "reviewer_pseudo_id must be pseudonymized (REV_*, ADJ_*, RAD_*, or CLIN_*)."
            )

        if isinstance(decision, str):
            target_decision = AdjudicationStatus(decision.strip().upper())
        else:
            target_decision = decision

        if not rationale.strip():
            raise ValueError("Clinical rationale is required for reviewer adjudication transitions.")

        rec = dict(self._records[sample_id])
        prev_status = self._statuses[sample_id]["adjudication_status"]
        today_str = review_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Update documented provenance attributes supplied during review
        rec["reviewer_pseudo_id"] = reviewer_clean
        rec["review_date"] = today_str
        if source_report_ref is not None:
            rec["source_report_ref"] = source_report_ref.strip()
        if diagnostic_evidence_category is not None:
            cat_up = diagnostic_evidence_category.strip().upper()
            if cat_up not in VALID_EVIDENCE_CATEGORIES:
                raise ValueError(f"Unsupported diagnostic_evidence_category: {cat_up}")
            rec["diagnostic_evidence_category"] = cat_up
        if diagnostic_evidence is not None:
            rec["diagnostic_evidence"] = diagnostic_evidence.strip()
        if radiology_report_confirmation_status is not None:
            rec["radiology_report_confirmation_status"] = radiology_report_confirmation_status.strip().upper()
        if histopathology_confirmation_status is not None:
            rec["histopathology_confirmation_status"] = histopathology_confirmation_status.strip().upper()

        if target_decision == AdjudicationStatus.ACCEPTED:
            # Resolve target class strictly within the 4 canonical classes
            if resolved_target_class is None:
                cand = self._statuses[sample_id].get("candidate_target_class")
                if not cand:
                    raise ValueError(
                        "Cannot ACCEPT without a canonical target class (OUT_OF_SCOPE/UNCERTAIN cannot be forced)."
                    )
                target_enum = OntologyTargetClass(cand)
            else:
                target_enum = (
                    resolved_target_class
                    if isinstance(resolved_target_class, OntologyTargetClass)
                    else OntologyTargetClass(resolved_target_class.strip().upper())
                )

            rec["source_diagnosis"] = target_enum.value
            rec["diagnostic_class"] = ONTOLOGY_TO_SHORT_CODE[target_enum]
            rec["clinical_adjudication_status"] = "ADJUDICATED_CONFIRMED"
            rec["conflicting_diagnosis_indicator"] = False
            rec.setdefault("label_ontology_version", ONTOLOGY_VERSION)

            # Re-run full automated clinical ontology + healthy eligibility + provenance audit
            re_adj = adjudicate_diagnostic_label(rec)
            if re_adj["adjudication_status"] != AdjudicationStatus.ACCEPTED.value:
                raise ValueError(
                    f"Cannot transition {sample_id} to ACCEPTED: {re_adj['reason']}"
                )
            if target_enum == OntologyTargetClass.HEALTHY_SKELETAL_CONTROL:
                h_ok, h_fail = validate_healthy_control_eligibility(rec)
                if not h_ok:
                    raise ValueError(f"Healthy-control eligibility check failed: {'; '.join(h_fail)}")
            adj = re_adj
        else:
            status_map = {
                AdjudicationStatus.REJECTED: "REJECTED",
                AdjudicationStatus.OUT_OF_SCOPE: "OUT_OF_SCOPE",
                AdjudicationStatus.UNCERTAIN: "UNCERTAIN",
                AdjudicationStatus.REQUIRES_ADJUDICATION: "PENDING_ADJUDICATION",
            }
            rec["clinical_adjudication_status"] = status_map[target_decision]
            now_iso = datetime.now(timezone.utc).isoformat()
            cand_cls = (
                None
                if target_decision in (AdjudicationStatus.OUT_OF_SCOPE, AdjudicationStatus.UNCERTAIN)
                else self._statuses[sample_id].get("candidate_target_class")
            )
            adj = {
                **self._statuses[sample_id],
                "candidate_target_class": cand_cls,
                "short_class_code": (
                    ONTOLOGY_TO_SHORT_CODE[OntologyTargetClass(cand_cls)]
                    if cand_cls in {e.value for e in OntologyTargetClass}
                    else None
                ),
                "adjudication_status": target_decision.value,
                "reason": rationale.strip(),
                "reviewer_status": "VERIFIED_REVIEWER",
                "reviewer_pseudo_id": reviewer_clean,
                "review_date": today_str,
                "audit_timestamp": now_iso,
                "ontology_version": ONTOLOGY_VERSION,
            }

        rec["adjudication_record"] = adj
        prov = self.build_provenance_record(rec, adj)
        self._records[sample_id] = rec
        self._statuses[sample_id] = adj
        self._provenance[sample_id] = prov

        self._history[sample_id].append(
            ProvenanceTransitionEntry(
                timestamp=adj["audit_timestamp"],
                sample_id=sample_id,
                previous_status=prev_status,
                new_status=adj["adjudication_status"],
                actor_pseudo_id=reviewer_clean,
                action="REVIEWER_DECISION",
                rationale=rationale.strip(),
                ontology_version=ONTOLOGY_VERSION,
                provenance_snapshot=asdict(prov),
            )
        )
        return {
            "sample_id": sample_id,
            "adjudication": adj,
            "provenance": asdict(prov),
        }

    def get_sample_status(self, sample_id: str) -> Dict[str, Any]:
        if sample_id not in self._statuses:
            raise KeyError(f"Unknown sample_id '{sample_id}' in AdjudicationEngine.")
        return dict(self._statuses[sample_id])

    def get_sample_provenance_history(self, sample_id: str) -> Dict[str, Any]:
        if sample_id not in self._provenance:
            raise KeyError(f"Unknown sample_id '{sample_id}' in AdjudicationEngine.")
        return {
            "sample_id": sample_id,
            "current_status": dict(self._statuses[sample_id]),
            "current_provenance": asdict(self._provenance[sample_id]),
            "transition_history": [asdict(e) for e in self._history[sample_id]],
        }

    def get_provenance_summary(self, sample_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        ids = list(sample_ids) if sample_ids is not None else list(self._statuses.keys())
        status_counts: Dict[str, int] = {s.value: 0 for s in AdjudicationStatus}
        accepted_by_class: Dict[str, int] = {c.value: 0 for c in OntologyTargetClass}
        provenance_complete = 0
        healthy_failures = 0
        conflicts = 0
        missing_field_histogram: Dict[str, int] = defaultdict(int)

        for sid in ids:
            adj = self._statuses.get(sid)
            prov = self._provenance.get(sid)
            if not adj or not prov:
                continue
            st = adj["adjudication_status"]
            status_counts[st] = status_counts.get(st, 0) + 1
            if st == AdjudicationStatus.ACCEPTED.value and adj.get("candidate_target_class") in accepted_by_class:
                accepted_by_class[adj["candidate_target_class"]] += 1
            if prov.provenance_complete:
                provenance_complete += 1
            for mf in prov.missing_provenance_fields:
                missing_field_histogram[mf] += 1
            if adj.get("healthy_eligibility_failures"):
                healthy_failures += 1
            if prov.conflicting_diagnosis_indicator or "Conflicting" in adj.get("reason", ""):
                conflicts += 1

        total = len(ids)
        pct = round(100.0 * provenance_complete / total, 2) if total > 0 else 0.0
        return {
            "ontology_version": ONTOLOGY_VERSION,
            "total_tracked_samples": total,
            "status_counts": status_counts,
            "accepted_by_class": accepted_by_class,
            "provenance_complete_count": provenance_complete,
            "label_provenance_completeness_pct": pct,
            "healthy_control_eligibility_failures": healthy_failures,
            "conflicting_diagnoses_count": conflicts,
            "missing_provenance_field_counts": dict(sorted(missing_field_histogram.items())),
        }

    def save_ledger(self, path: Optional[Path] = None) -> Path:
        target = path or self.ledger_path
        if target is None:
            raise ValueError("No ledger_path configured for AdjudicationEngine.")
        target.parent.mkdir(parents=True, exist_ok=True)
        if len(self._ledger_key) < 32:
            raise RuntimeError("A configured ledger integrity key of at least 32 bytes is required.")
        payload = {
            "ledger_version": "2.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ontology_version": ONTOLOGY_VERSION,
            "summary": self.get_provenance_summary(),
            "samples": {
                sid: self.get_sample_provenance_history(sid)
                for sid in sorted(self._records.keys())
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        payload["integrity"] = {"algorithm": "HMAC-SHA256", "signature": hmac.new(self._ledger_key, canonical, hashlib.sha256).hexdigest()}
        fd, temp_name = tempfile.mkstemp(prefix=".ledger-", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name): os.unlink(temp_name)
        return target

    def load_ledger(self, path: Optional[Path] = None) -> int:
        target = path or self.ledger_path
        if target is None or not target.exists():
            return 0
        payload = json.loads(target.read_text(encoding="utf-8"))
        if len(self._ledger_key) < 32:
            raise RuntimeError("A configured ledger integrity key of at least 32 bytes is required.")
        if payload.get("ledger_version") != "2.0" or payload.get("ontology_version") != ONTOLOGY_VERSION:
            raise ValueError("Adjudication ledger version or ontology version is unsupported.")
        integrity = payload.pop("integrity", {})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        expected = hmac.new(self._ledger_key, canonical, hashlib.sha256).hexdigest()
        if integrity.get("algorithm") != "HMAC-SHA256" or not hmac.compare_digest(str(integrity.get("signature", "")), expected):
            raise ValueError("Adjudication ledger integrity verification failed.")
        samples = payload.get("samples") or {}
        if not isinstance(samples, dict):
            raise ValueError("Adjudication ledger sample structure is invalid.")
        # Validate the entire signed document before mutating engine state.
        for entry in samples.values():
            if not isinstance(entry, dict):
                raise ValueError("Adjudication ledger sample structure is invalid.")
            cur_status = entry.get("current_status")
            cur_prov = entry.get("current_provenance")
            history = entry.get("transition_history")
            if not isinstance(cur_status, dict) or not isinstance(cur_prov, dict) or not isinstance(history, list):
                raise ValueError("Adjudication ledger sample structure is invalid.")
            if cur_status.get("ontology_version") != ONTOLOGY_VERSION or cur_prov.get("label_ontology_version") != ONTOLOGY_VERSION:
                raise ValueError("Adjudication ledger contains an unsupported ontology version.")
            for transition in history:
                if not isinstance(transition, dict) or transition.get("ontology_version") != ONTOLOGY_VERSION:
                    raise ValueError("Adjudication ledger history contains an unsupported ontology version.")
                if not all(key in transition for key in ("timestamp", "sample_id", "new_status", "actor_pseudo_id", "action", "rationale")):
                    raise ValueError("Adjudication ledger transition structure is invalid.")
        loaded = 0
        for sid, entry in samples.items():
            cur_status = entry.get("current_status") or {}
            cur_prov = entry.get("current_provenance") or {}
            history = entry.get("transition_history") or []
            if not cur_status or not cur_prov:
                continue
            self._statuses[sid] = cur_status
            self._provenance[sid] = ProvenanceRecord(
                sample_id=cur_prov.get("sample_id", sid),
                patient_id=cur_prov.get("patient_id", ""),
                source_diagnosis=cur_prov.get("source_diagnosis", "MISSING"),
                candidate_target_class=cur_prov.get("candidate_target_class"),
                source_report_ref=cur_prov.get("source_report_ref", "MISSING"),
                diagnostic_evidence_category=cur_prov.get("diagnostic_evidence_category", "MISSING"),
                histopathology_confirmation_status=cur_prov.get("histopathology_confirmation_status", "ABSENT"),
                radiology_report_confirmation_status=cur_prov.get("radiology_report_confirmation_status", "ABSENT"),
                clinical_adjudication_status=cur_prov.get("clinical_adjudication_status", "PENDING_ADJUDICATION"),
                reviewer_pseudo_id=cur_prov.get("reviewer_pseudo_id", "UNASSIGNED"),
                review_date=cur_prov.get("review_date", "UNREVIEWED"),
                label_provenance=cur_prov.get("label_provenance", "MISSING"),
                label_ontology_version=cur_prov.get("label_ontology_version", ONTOLOGY_VERSION),
                conflicting_diagnosis_indicator=bool(cur_prov.get("conflicting_diagnosis_indicator", False)),
                provenance_complete=bool(cur_prov.get("provenance_complete", False)),
                missing_provenance_fields=list(cur_prov.get("missing_provenance_fields") or []),
            )
            self._records[sid] = {
                "sample_id": self._provenance[sid].sample_id,
                "patient_id": self._provenance[sid].patient_id,
                "source_diagnosis": self._provenance[sid].source_diagnosis,
                "diagnostic_class": cur_status.get("short_class_code") or self._provenance[sid].source_diagnosis,
                "source_report_ref": "" if self._provenance[sid].source_report_ref in PLACEHOLDER_VALUES else self._provenance[sid].source_report_ref,
                "diagnostic_evidence_category": "" if self._provenance[sid].diagnostic_evidence_category in PLACEHOLDER_VALUES else self._provenance[sid].diagnostic_evidence_category,
                "histopathology_confirmation_status": self._provenance[sid].histopathology_confirmation_status,
                "radiology_report_confirmation_status": self._provenance[sid].radiology_report_confirmation_status,
                "clinical_adjudication_status": self._provenance[sid].clinical_adjudication_status,
                "reviewer_pseudo_id": "" if self._provenance[sid].reviewer_pseudo_id == "UNASSIGNED" else self._provenance[sid].reviewer_pseudo_id,
                "review_date": "" if self._provenance[sid].review_date == "UNREVIEWED" else self._provenance[sid].review_date,
                "label_provenance": "" if self._provenance[sid].label_provenance in PLACEHOLDER_VALUES else self._provenance[sid].label_provenance,
                "label_ontology_version": self._provenance[sid].label_ontology_version,
                "conflicting_diagnosis_indicator": self._provenance[sid].conflicting_diagnosis_indicator,
                "adjudication_record": cur_status,
            }
            self._history[sid] = [
                ProvenanceTransitionEntry(
                    timestamp=h["timestamp"],
                    sample_id=h["sample_id"],
                    previous_status=h.get("previous_status"),
                    new_status=h["new_status"],
                    actor_pseudo_id=h["actor_pseudo_id"],
                    action=h["action"],
                    rationale=h["rationale"],
                    ontology_version=h.get("ontology_version", ONTOLOGY_VERSION),
                    provenance_snapshot=h.get("provenance_snapshot") or {},
                )
                for h in history
            ]
            loaded += 1
        return loaded
