"""
Automated Tests — Track D AdjudicationEngine & Label Provenance Tracking.

Verifies:
- Status assignment across ACCEPTED, REJECTED, OUT_OF_SCOPE, UNCERTAIN, REQUIRES_ADJUDICATION
- Two-pass batch conflict detection across records sharing a patient_id
- Authorized reviewer sign-off transitions (with pseudonymous reviewer ID validation)
- Prevention of forcing OUT_OF_SCOPE or UNCERTAIN records into ACCEPTED without valid ontology match
- Chronological provenance transition history & JSON ledger persistence/reloading
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ml_service.app.services.adjudication_engine import AdjudicationEngine
from ml_service.datasets.clinical_ontology import (
    ONTOLOGY_VERSION,
    AdjudicationStatus,
    OntologyTargetClass,
)
from ml_service.tests.test_track_d_intake import _valid_record


class TestAdjudicationEngine(unittest.TestCase):
    def test_01_batch_adjudication_statuses_and_provenance_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_rec = _valid_record(root, "AE_OK", "PT_AE_1", "metastasis", "xray", seed=1)

            oos_rec = _valid_record(root, "AE_OOS", "PT_AE_2", "oi", "xray", seed=2)
            oos_rec["source_diagnosis"] = "Osteoporosis"
            oos_rec["diagnostic_evidence"] = "DEXA low bone density osteoporosis"

            unc_rec = _valid_record(root, "AE_UNC", "PT_AE_3", "paget", "ct", seed=3)
            unc_rec["source_diagnosis"] = "Equivocal Paget vs fibrous dysplasia"
            unc_rec["diagnostic_evidence"] = "Borderline indeterminate findings"

            rej_rec = _valid_record(root, "AE_REJ", "PT_AE_4", "healthy", "xray", seed=4)
            rej_rec["source_report_ref"] = ""
            rej_rec["radiology_report_confirmation_status"] = "PENDING"

            pending_rec = _valid_record(root, "AE_PEND", "PT_AE_5", "paget", "bone_scan", seed=5)
            pending_rec["clinical_adjudication_status"] = "PENDING_ADJUDICATION"

            engine = AdjudicationEngine()
            out = engine.adjudicate_batch([ok_rec, oos_rec, unc_rec, rej_rec, pending_rec])

            self.assertEqual(out["ontology_version"], ONTOLOGY_VERSION)
            self.assertEqual(out["total_evaluated"], 5)
            counts = out["summary"]["status_counts"]
            self.assertEqual(counts[AdjudicationStatus.ACCEPTED.value], 1)
            self.assertEqual(counts[AdjudicationStatus.OUT_OF_SCOPE.value], 1)
            self.assertEqual(counts[AdjudicationStatus.UNCERTAIN.value], 1)
            self.assertEqual(counts[AdjudicationStatus.REJECTED.value], 1)
            self.assertEqual(counts[AdjudicationStatus.REQUIRES_ADJUDICATION.value], 1)
            self.assertEqual(
                out["summary"]["accepted_by_class"][OntologyTargetClass.SECONDARY_BONE_METASTASIS.value],
                1,
            )
            self.assertEqual(out["summary"]["healthy_control_eligibility_failures"], 1)
            self.assertEqual(out["summary"]["label_provenance_completeness_pct"], 80.0)

    def test_02_reviewer_decision_transitions_and_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pending_rec = _valid_record(root, "AE_REV_1", "PT_AE_10", "paget", "xray", seed=10)
            pending_rec["clinical_adjudication_status"] = "PENDING_ADJUDICATION"
            pending_rec["reviewer_pseudo_id"] = ""

            oos_rec = _valid_record(root, "AE_REV_OOS", "PT_AE_11", "paget", "xray", seed=11)
            oos_rec["source_diagnosis"] = "Osteoarthritis"
            oos_rec["diagnostic_evidence"] = "Degenerative joint osteoarthritis"

            engine = AdjudicationEngine()
            engine.adjudicate_batch([pending_rec, oos_rec])

            # Reject non-pseudonymized reviewer ID (direct name/MRN disallowed)
            with self.assertRaises(ValueError):
                engine.submit_reviewer_decision(
                    sample_id="AE_REV_1",
                    decision=AdjudicationStatus.ACCEPTED,
                    reviewer_pseudo_id="Dr. Jane Doe",
                    rationale="Attempting non-pseudonymized reviewer ID",
                )

            # Never allow forcing OUT_OF_SCOPE sample directly to ACCEPTED without valid ontology match
            with self.assertRaises(ValueError):
                engine.submit_reviewer_decision(
                    sample_id="AE_REV_OOS",
                    decision=AdjudicationStatus.ACCEPTED,
                    reviewer_pseudo_id="RAD_BOARD_02",
                    rationale="Attempt force accept on out-of-scope osteoarthritis",
                )

            # Valid authorized reviewer sign-off updates status to ACCEPTED and records transition
            updated = engine.submit_reviewer_decision(
                sample_id="AE_REV_1",
                decision=AdjudicationStatus.ACCEPTED,
                reviewer_pseudo_id="RAD_BOARD_02",
                rationale="Confirmed radiographic triad + elevated serum ALP.",
                resolved_target_class=OntologyTargetClass.PAGETS_DISEASE,
                diagnostic_evidence_category="SERUM_ALP_AND_RADIOGRAPH",
                radiology_report_confirmation_status="CONFIRMED",
            )
            self.assertEqual(updated["adjudication"]["adjudication_status"], AdjudicationStatus.ACCEPTED.value)
            hist = engine.get_sample_provenance_history("AE_REV_1")
            self.assertEqual(len(hist["transition_history"]), 2)
            self.assertEqual(hist["transition_history"][0]["new_status"], AdjudicationStatus.REQUIRES_ADJUDICATION.value)
            self.assertEqual(hist["transition_history"][1]["previous_status"], AdjudicationStatus.REQUIRES_ADJUDICATION.value)
            self.assertEqual(hist["transition_history"][1]["new_status"], AdjudicationStatus.ACCEPTED.value)
            self.assertEqual(hist["current_provenance"]["reviewer_pseudo_id"], "RAD_BOARD_02")

    def test_03_patient_conflict_detection_and_ledger_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_file = root / "ml_service" / "data" / "reports" / "track_d_adjudication_ledger.json"
            r1 = _valid_record(root, "AE_C1", "PT_SHARED_99", "metastasis", "ct", seed=20)
            r2 = _valid_record(root, "AE_C2", "PT_SHARED_99", "healthy", "ct", seed=21)

            engine = AdjudicationEngine(ledger_path=ledger_file)
            out = engine.adjudicate_batch([r1, r2])
            self.assertEqual(
                out["summary"]["status_counts"][AdjudicationStatus.REQUIRES_ADJUDICATION.value],
                2,
            )
            saved_path = engine.save_ledger()
            self.assertTrue(saved_path.exists())

            reloaded = AdjudicationEngine(ledger_path=ledger_file)
            self.assertEqual(
                reloaded.get_sample_status("AE_C1")["adjudication_status"],
                AdjudicationStatus.REQUIRES_ADJUDICATION.value,
            )
            self.assertEqual(
                reloaded.get_provenance_summary()["total_tracked_samples"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
