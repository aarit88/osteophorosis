"""
Stage 34 Adversarial Safety Tests — Clinical Label Ontology, Diagnostic Adjudication,
Strict Healthy-Control Eligibility, Provenance Audit, and Recursive Dataset Validator.

Covers all 10 mandated clinical label safety scenarios + validate_dataset.py CLI:
1. Osteoporosis cannot map to OI (OUT_OF_SCOPE)
2. Osteoarthritis cannot map to Paget's (OUT_OF_SCOPE)
3. Contralateral normal-appearing limb cannot pass as Healthy Skeletal Control (REJECTED)
4. Primary bone malignancy cannot map to Secondary Bone Metastasis (OUT_OF_SCOPE)
5. Uncertain / equivocal diagnoses are excluded from training/validation/test splits
6. Missing diagnostic evidence fails the readiness gate
7. Conflicting diagnoses for the same patient require adjudication
8. Out-of-scope diagnoses stay out of scope and out of target class splits
9. Valid documented diagnoses with proper provenance pass into leakage-safe splits
10. Protected holdouts (BTXRD Center 2 & Center 3) remain blocked and excluded
11. Recursive raw dataset validation CLI (`validate_dataset.py`)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from ml_service.datasets.clinical_ontology import (
    ONTOLOGY_VERSION,
    AdjudicationStatus,
    OntologyTargetClass,
    adjudicate_diagnostic_label,
)
from ml_service.datasets.track_d_intake import (
    evaluate_track_d_readiness,
    generate_leakage_safe_splits,
    inspect_image_file,
    run_track_d_intake_pipeline,
    validate_track_d_dataset,
)
from ml_service.scripts.validate_dataset import run_dataset_validation_cli
from ml_service.tests.test_track_d_intake import _make_png_bytes, _valid_record


class TestStage34OntologyAndAdjudication(unittest.TestCase):
    def test_00_wrong_ontology_version_is_rejected_strictly(self) -> None:
        record = {"sample_id": "SMP_001", "label_ontology_version": "OSTEO_ONTOLOGY_V0"}
        with self.assertRaisesRegex(ValueError, "ontology version"):
            adjudicate_diagnostic_label(record)

    def test_01_osteoporosis_cannot_map_to_oi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = _valid_record(root, "ADV_01", "PT_3401", "oi", "xray", seed=11)
            rec["source_diagnosis"] = "Senile Osteoporosis"
            rec["diagnostic_evidence"] = "DEXA T-score -2.8 confirming primary osteoporosis"

            adj = adjudicate_diagnostic_label(rec)
            self.assertEqual(adj["adjudication_status"], AdjudicationStatus.OUT_OF_SCOPE.value)
            self.assertIsNone(adj["candidate_target_class"])

            val = validate_track_d_dataset(root, records=[rec])
            self.assertEqual(val["passed_count"], 0)
            self.assertEqual(val["accepted_by_class"][OntologyTargetClass.OSTEOGENESIS_IMPERFECTA.value], 0)
            self.assertEqual(val["adjudication_counts"][AdjudicationStatus.OUT_OF_SCOPE.value], 1)

    def test_02_osteoarthritis_cannot_map_to_paget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = _valid_record(root, "ADV_02", "PT_3402", "paget", "xray", seed=12)
            rec["source_diagnosis"] = "Knee Osteoarthritis"
            rec["diagnostic_evidence"] = "Subchondral sclerosis and joint space narrowing due to osteoarthritis"

            adj = adjudicate_diagnostic_label(rec)
            self.assertEqual(adj["adjudication_status"], AdjudicationStatus.OUT_OF_SCOPE.value)
            self.assertIsNone(adj["candidate_target_class"])

            val = validate_track_d_dataset(root, records=[rec])
            self.assertEqual(val["passed_count"], 0)
            self.assertEqual(val["accepted_by_class"][OntologyTargetClass.PAGETS_DISEASE.value], 0)

    def test_03_contralateral_limb_and_negative_absence_cannot_pass_as_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Case A: contralateral trauma limb
            contra = _valid_record(root, "ADV_03A", "PT_3403", "healthy", "xray", seed=13)
            contra["source_diagnosis"] = "contralateral unaffected limb"
            contra["diagnostic_evidence"] = "Normal appearing contralateral fellow limb from trauma study"

            # Case B: inferred solely from absence of fracture & missing history
            absence = _valid_record(root, "ADV_03B", "PT_3404", "healthy", "xray", seed=14)
            absence["patient_skeletal_history"] = "incomplete medical history"
            absence["diagnostic_evidence"] = "No fracture detected on emergency screen"

            # Case C: generic unspecified anatomical region & missing criteria
            generic_site = _valid_record(root, "ADV_03C", "PT_3405", "healthy", "xray", seed=15)
            generic_site["anatomical_region"] = "skeleton"
            generic_site["control_selection_criteria"] = ""

            val = validate_track_d_dataset(root, records=[contra, absence, generic_site])
            self.assertEqual(val["passed_count"], 0)
            self.assertEqual(val["quarantined_count"], 3)
            self.assertEqual(val["healthy_control_eligibility_failures"], 3)
            self.assertEqual(val["adjudication_counts"][AdjudicationStatus.REJECTED.value], 3)
            self.assertIn(
                "HEALTHY_ELIGIBILITY_FAILED:CONTRALATERAL_OR_FELLOW_LIMB",
                val["quarantined_records"][0]["rejection_reasons"],
            )
            self.assertIn(
                "HEALTHY_ELIGIBILITY_FAILED:INFERRED_FROM_ABSENCE_OR_INCOMPLETE_HISTORY",
                val["quarantined_records"][1]["rejection_reasons"],
            )
            self.assertIn(
                "HEALTHY_ELIGIBILITY_FAILED:ANATOMICAL_REGION_NOT_DOCUMENTED",
                val["quarantined_records"][2]["rejection_reasons"],
            )

    def test_04_primary_bone_malignancy_cannot_map_to_metastasis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            osteo = _valid_record(root, "ADV_04A", "PT_3406", "metastasis", "xray", seed=16)
            osteo["source_diagnosis"] = "Primary bone malignancy (Osteosarcoma)"
            osteo["diagnostic_evidence"] = "Biopsy confirmed primary osteosarcoma of distal femur"

            ewing = _valid_record(root, "ADV_04B", "PT_3407", "metastasis", "ct", seed=17)
            ewing["source_diagnosis"] = "Ewing sarcoma"
            ewing["diagnostic_evidence"] = "Primary Ewing sarcoma verified on histopathology"

            val = validate_track_d_dataset(root, records=[osteo, ewing])
            self.assertEqual(val["passed_count"], 0)
            self.assertEqual(val["adjudication_counts"][AdjudicationStatus.OUT_OF_SCOPE.value], 2)
            self.assertEqual(val["accepted_by_class"][OntologyTargetClass.SECONDARY_BONE_METASTASIS.value], 0)

    def test_05_uncertain_diagnoses_excluded_from_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = _valid_record(root, "ADV_05_OK", "PT_3408", "metastasis", "xray", seed=18)
            uncertain = _valid_record(root, "ADV_05_UNC", "PT_3409", "paget", "xray", seed=19)
            uncertain["source_diagnosis"] = "Indeterminate Paget vs fibrous dysplasia"
            uncertain["diagnostic_evidence"] = "Equivocal radiograph; rule out Paget disease"

            val = validate_track_d_dataset(root, records=[clean, uncertain])
            self.assertEqual(val["passed_count"], 1)
            self.assertEqual(val["adjudication_counts"][AdjudicationStatus.UNCERTAIN.value], 1)

            splits = generate_leakage_safe_splits(val["passed_records"], seed=42)
            split_sids = {r["sample_id"] for r in splits["records"]}
            self.assertEqual(split_sids, {"ADV_05_OK"})
            self.assertNotIn("ADV_05_UNC", split_sids)

    def test_06_missing_diagnostic_evidence_fails_readiness_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            seed_ctr = 30
            for cls in ("metastasis", "paget", "oi", "healthy"):
                for p in range(3):
                    seed_ctr += 1
                    rec = _valid_record(root, f"RD6_{cls}_{p}", f"PT_RD6_{cls}_{p}", cls, "xray", seed=seed_ctr)
                    if cls == "paget" and p == 0:
                        rec["diagnostic_evidence"] = ""
                        rec["source_report_ref"] = ""
                    records.append(rec)

            val = validate_track_d_dataset(root, records=records)
            splits = generate_leakage_safe_splits(val["passed_records"], seed=42)
            readiness = evaluate_track_d_readiness(val, splits)

            self.assertEqual(readiness["readiness_status"], "BLOCKED")
            self.assertFalse(readiness["ready_for_training"])
            self.assertFalse(readiness["gates"]["diagnostic_labels_documented"])
            self.assertEqual(val["adjudication_counts"][AdjudicationStatus.REJECTED.value], 1)

    def test_07_conflicting_diagnoses_for_same_patient_require_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec_a = _valid_record(root, "ADV_07A", "PT_CONFLICT_1", "paget", "ct", seed=41)
            rec_b = _valid_record(root, "ADV_07B", "PT_CONFLICT_1", "healthy", "ct", seed=42)

            val = validate_track_d_dataset(root, records=[rec_a, rec_b])
            self.assertEqual(val["passed_count"], 0)
            self.assertEqual(val["conflicting_patient_count"], 1)
            self.assertEqual(val["conflicting_diagnoses_count"], 2)
            self.assertEqual(val["adjudication_counts"][AdjudicationStatus.REQUIRES_ADJUDICATION.value], 2)
            for rec in val["records"]:
                self.assertIn("CONFLICTING_PATIENT_DIAGNOSIS", rec["rejection_reasons"])

    def test_08_out_of_scope_stays_out_of_scope_and_out_of_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            myeloma = _valid_record(root, "ADV_08A", "PT_3410", "metastasis", "ct", seed=51)
            myeloma["source_diagnosis"] = "Multiple Myeloma"
            myeloma["diagnostic_evidence"] = "Hematologic multiple myeloma lytic bone involvement"

            osteomalacia = _valid_record(root, "ADV_08B", "PT_3411", "oi", "xray", seed=52)
            osteomalacia["source_diagnosis"] = "Osteomalacia"
            osteomalacia["diagnostic_evidence"] = "Vitamin D deficiency osteomalacia"

            val = validate_track_d_dataset(root, records=[myeloma, osteomalacia])
            splits = generate_leakage_safe_splits(val["passed_records"], seed=42)
            self.assertEqual(splits["total_assigned"], 0)
            self.assertTrue(all(v == 0 for v in val["accepted_by_class"].values()))
            self.assertEqual(val["adjudication_counts"][AdjudicationStatus.OUT_OF_SCOPE.value], 2)

    def test_09_valid_documented_diagnoses_with_provenance_pass_into_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            seed_ctr = 70
            for cls in ("metastasis", "paget", "oi", "healthy"):
                for p in range(3):
                    seed_ctr += 1
                    records.append(
                        _valid_record(root, f"V9_{cls}_{p}", f"PT_V9_{cls}_{p}", cls, "xray", seed=seed_ctr)
                    )

            manifest_path = root / "ml_service" / "data" / "manifests" / "track_d_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps({
                    "manifest_version": "2.0.0",
                    "track": "track_d",
                    "ontology_version": ONTOLOGY_VERSION,
                    "records": records,
                }),
                encoding="utf-8",
            )

            out = run_track_d_intake_pipeline(root_dir=root, write_outputs=True)
            readiness = out["readiness"]
            self.assertEqual(readiness["ontology_version"], ONTOLOGY_VERSION)
            self.assertEqual(readiness["readiness_status"], "READY_FOR_TRAINING")
            self.assertTrue(readiness["ready_for_training"])
            self.assertEqual(readiness["adjudication_summary"][AdjudicationStatus.ACCEPTED.value], 12)
            self.assertEqual(readiness["label_provenance_completeness_pct"], 100.0)
            self.assertTrue(Path(out["outputs"]["adjudication_audit_report"]).exists())

    def test_10_protected_btxrd_center2_and_center3_holdouts_remain_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c2 = _valid_record(root, "BTXRD_C2_01", "PT_3420", "healthy", "xray", seed=91, institution_id="btxrd_center_2")
            c3 = _valid_record(root, "BTXRD_C3_99", "PT_3421", "healthy", "xray", seed=92, institution_id="INST_OTHER")
            c3_sha = inspect_image_file(root / c3["image_path"])["sha256"]

            split_csv = root / "ml_service" / "data" / "splits" / "btxrd_split_manifest.csv"
            split_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(split_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "split", "sha256_hash"])
                w.writerow(["BTXRD_C3_99", "center_3", c3_sha])

            val = validate_track_d_dataset(root, records=[c2, c3])
            self.assertEqual(val["passed_count"], 0)
            self.assertEqual(val["quarantined_count"], 2)
            self.assertIn("PROTECTED_EXTERNAL_HOLDOUT_VIOLATION", val["quarantined_records"][0]["rejection_reasons"])
            self.assertIn("PROTECTED_EXTERNAL_HOLDOUT_SAMPLE_ID", val["quarantined_records"][1]["rejection_reasons"])
            self.assertIn("PROTECTED_EXTERNAL_HOLDOUT_HASH_MATCH", val["quarantined_records"][1]["rejection_reasons"])

    def test_11_validate_dataset_cli_recursive_raw_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "ml_service" / "data" / "raw" / "xray" / "metastasis" / "PT_9001_01.png"
            p2 = root / "ml_service" / "data" / "raw" / "xray" / "metastasis" / "PT_9001_02.png"
            p3 = root / "ml_service" / "data" / "raw" / "ct" / "paget" / "PT_9002_bad.png"
            for p in (p1, p2, p3):
                p.parent.mkdir(parents=True, exist_ok=True)
            png_bytes = _make_png_bytes(64, 64, pattern_seed=5)
            p1.write_bytes(png_bytes)
            p2.write_bytes(png_bytes)  # duplicate image hash + repeated patient ID PT_9001
            p3.write_bytes(b"corrupt")  # corrupt image

            summary = run_dataset_validation_cli(root_dir=root)
            self.assertEqual(summary["raw_directory_scan"]["raw_files_found"], 3)
            self.assertEqual(summary["raw_directory_scan"]["valid_raw_files"], 2)
            self.assertEqual(summary["raw_directory_scan"]["invalid_raw_files"], 1)
            self.assertEqual(summary["raw_directory_scan"]["duplicate_image_hashes_count"], 1)
            self.assertIn("PT_9001", summary["raw_directory_scan"]["repeated_patient_ids_in_raw"])
            self.assertGreater(summary["metadata_and_provenance"]["missing_metadata_fields_count"], 0)


if __name__ == "__main__":
    unittest.main()
