"""
Stage 33 Automated Tests — Track D Clinical Dataset Intake, Quality Validation,
Quarantine, Deterministic Patient-Level Splitting, Holdout Isolation, and Readiness Gate.

Runs cleanly with pure Python stdlib unittest inside AI Studio.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from ml_service.datasets.track_d_intake import (
    evaluate_track_d_readiness,
    generate_leakage_safe_splits,
    inspect_image_file,
    run_track_d_intake_pipeline,
    validate_track_d_dataset,
    verify_cross_split_leakage,
)


def _make_png_bytes(width: int = 64, height: int = 64, pattern_seed: int = 1, noise_px: int = 0) -> bytes:
    """Generates a valid uncompressed/zlib-compressed 8-bit grayscale PNG in pure stdlib."""
    def _chunk(ctype: bytes, data: bytes) -> bytes:
        body = ctype + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows = bytearray()
    freq_x = (pattern_seed % 7) + 1
    freq_y = ((pattern_seed // 7) % 7) + 1
    phase = (pattern_seed * 19) & 0xFF
    for y in range(height):
        rows.append(0)  # filter type 0
        for x in range(width):
            val = (
                ((x // 8) * freq_x * 41 + (y // 8) * freq_y * 59 + phase)
                ^ ((x * freq_y + y * freq_x) * 13)
            ) & 0xFF
            if noise_px > 0 and x == 0 and y == 0:
                val = (val + noise_px) & 0xFF
            rows.append(val)
    idat = zlib.compress(bytes(rows))
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _make_minimal_dicom_ct(rows: int = 64, cols: int = 64, modality: str = "CT") -> bytes:
    """Creates minimal Explicit VR Little Endian DICOM file for modality mismatch tests."""
    preamble = b"\x00" * 128 + b"DICM"

    def _elem_str(group: int, elem: int, vr: bytes, val: str) -> bytes:
        b = val.encode("ascii")
        if len(b) % 2 == 1:
            b += b" "
        return struct.pack("<HH", group, elem) + vr + struct.pack("<H", len(b)) + b

    def _elem_u16(group: int, elem: int, val: int) -> bytes:
        return struct.pack("<HH", group, elem) + b"US" + struct.pack("<HH", 2, val)

    return (
        preamble
        + _elem_str(0x0008, 0x0060, b"CS", modality)
        + _elem_u16(0x0028, 0x0010, rows)
        + _elem_u16(0x0028, 0x0011, cols)
        + _elem_u16(0x0028, 0x0100, 16)
    )


def _valid_record(
    root: Path,
    sample_id: str,
    patient_id: str,
    diagnostic_class: str,
    imaging_modality: str,
    seed: int = 1,
    institution_id: str = "INST_HOSP_A",
) -> dict:
    rel_path = f"ml_service/data/raw/{imaging_modality}/{diagnostic_class}/{patient_id}_{sample_id}.png"
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(_make_png_bytes(64, 64, pattern_seed=seed))

    ontology_map = {
        "metastasis": "SECONDARY_BONE_METASTASIS",
        "paget": "PAGETS_DISEASE",
        "oi": "OSTEOGENESIS_IMPERFECTA",
        "healthy": "HEALTHY_SKELETAL_CONTROL",
    }
    evidence_cat_map = {
        "metastasis": "BIOPSY_HISTOPATHOLOGY",
        "paget": "SERUM_ALP_AND_RADIOGRAPH",
        "oi": "GENETIC_COL1A1_COL1A2",
        "healthy": "RADIOLOGIST_NORMAL_CONFIRMATION",
    }
    evidence_text_map = {
        "metastasis": "Biopsy and PET-CT confirmed secondary bone metastasis",
        "paget": "Radiographic triad and elevated serum ALP confirmed Paget's disease",
        "oi": "Genetic COL1A1 confirmation and specialist skeletal diagnosis of OI",
        "healthy": "Board-certified radiologist confirmed normal trabecular bone and intact cortex",
    }
    rec = {
        "sample_id": sample_id,
        "patient_id": patient_id,
        "source_diagnosis": ontology_map.get(diagnostic_class, diagnostic_class),
        "diagnostic_class": diagnostic_class,
        "imaging_modality": imaging_modality,
        "image_path": rel_path,
        "image_format": "PNG",
        "image_dimensions": {"width": 64, "height": 64, "depth": 1},
        "image_bit_depth": 8,
        "anatomical_region": "proximal_femur",
        "institution_id": institution_id,
        "dataset_source": "VERIFIED_RETROSPECTIVE_PACS",
        "license_permissions": "DUA_APPROVED_RESEARCH_ONLY",
        "ethics_approval_ref": "IEC_2026_101",
        "deidentification_status": "VERIFIED_PSEUDONYMIZED",
        "qc_status": "PENDING",
        "split_assignment": "unassigned",
        "source_report_ref": f"RPT_{sample_id}",
        "diagnostic_evidence_category": evidence_cat_map.get(diagnostic_class, "SPECIALIST_CLINICAL_DIAGNOSIS"),
        "histopathology_confirmation_status": "CONFIRMED" if diagnostic_class == "metastasis" else "NOT_APPLICABLE",
        "radiology_report_confirmation_status": "CONFIRMED",
        "clinical_adjudication_status": "ADJUDICATED_CONFIRMED",
        "reviewer_pseudo_id": "RAD_VERIFIED_01",
        "review_date": "2026-11-20",
        "conflicting_diagnosis_indicator": False,
        "label_provenance": f"{institution_id}_CLINICAL_BOARD",
        "label_ontology_version": "OSTEO_ONTOLOGY_V1.0_STAGE34",
        "diagnostic_evidence": evidence_text_map.get(diagnostic_class, "Specialist confirmed diagnosis"),
    }
    if diagnostic_class == "healthy":
        rec["control_selection_criteria"] = "Asymptomatic screening protocol with verified normal BMD"
        rec["patient_skeletal_history"] = "no_skeletal_pathology"
    return rec


class TestTrackDIntakePipeline(unittest.TestCase):
    def test_01_manifest_required_fields_and_phi_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = _valid_record(root, "SMP_001", "PT_9A1B2C", "metastasis", "xray", seed=1)
            phi_bad = _valid_record(root, "SMP_002", "MRN_1234567", "paget", "xray", seed=2)
            phi_bad["patient_name"] = "Jane Doe"
            phi_bad["email"] = "jane@hospital.org"

            res = validate_track_d_dataset(root, records=[good, phi_bad])
            self.assertEqual(res["passed_count"], 1)
            self.assertEqual(res["quarantined_count"], 1)
            reasons = res["quarantined_records"][0]["rejection_reasons"]
            self.assertIn("PHI_DISALLOWED_FIELD:patient_name", reasons)
            self.assertIn("PHI_DISALLOWED_FIELD:email", reasons)
            self.assertIn("PHI_NON_PSEUDONYMIZED_PATIENT_ID", reasons)

    def test_02_class_modality_uncertain_and_negative_mapping_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uncertain = _valid_record(root, "SMP_UNC", "PT_1001", "paget", "xray", seed=3)
            uncertain["diagnostic_evidence"] = "Borderline equivocal Paget appearance; rule out metastasis"

            neg_map = _valid_record(root, "SMP_OSTEO", "PT_1002", "metastasis", "xray", seed=4)
            neg_map["diagnostic_evidence"] = "Primary osteosarcoma lesion"

            # DICOM CT inside declared xray folder
            dcm_rel = "ml_service/data/raw/xray/healthy/PT_1003_scan.dcm"
            dcm_full = root / dcm_rel
            dcm_full.parent.mkdir(parents=True, exist_ok=True)
            dcm_full.write_bytes(_make_minimal_dicom_ct(64, 64, "CT"))
            mod_mismatch = _valid_record(root, "SMP_MOD", "PT_1003", "healthy", "xray", seed=5)
            mod_mismatch["image_path"] = dcm_rel
            mod_mismatch["image_format"] = "DICOM"

            res = validate_track_d_dataset(root, records=[uncertain, neg_map, mod_mismatch])
            self.assertEqual(res["passed_count"], 0)
            self.assertEqual(res["quarantined_count"], 3)
            self.assertIn("UNCERTAIN_DIAGNOSTIC_LABEL", res["quarantined_records"][0]["rejection_reasons"])
            self.assertIn("ADJUDICATION_OUT_OF_SCOPE", res["quarantined_records"][1]["rejection_reasons"])
            self.assertIn(
                "OUT_OF_SCOPE_PRIMARY_BONE_MALIGNANCY_NOT_METASTASIS",
                res["quarantined_records"][1]["rejection_reasons"],
            )
            self.assertIn("MODALITY_DICOM_MISMATCH:declared=xray:dicom=CT", res["quarantined_records"][2]["rejection_reasons"])

    def test_03_corrupt_empty_and_tiny_image_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_rec = _valid_record(root, "SMP_EMPTY", "PT_2001", "oi", "xray", seed=6)
            (root / empty_rec["image_path"]).write_bytes(b"")

            corrupt_rec = _valid_record(root, "SMP_CORR", "PT_2002", "oi", "xray", seed=7)
            (root / corrupt_rec["image_path"]).write_bytes(b"not-a-valid-png")

            tiny_rec = _valid_record(root, "SMP_TINY", "PT_2003", "oi", "xray", seed=8)
            (root / tiny_rec["image_path"]).write_bytes(_make_png_bytes(16, 16, pattern_seed=8))

            res = validate_track_d_dataset(root, records=[empty_rec, corrupt_rec, tiny_rec])
            self.assertEqual(res["passed_count"], 0)
            self.assertEqual(res["quarantined_count"], 3)
            self.assertIn("EMPTY_FILE", res["quarantined_records"][0]["rejection_reasons"])
            self.assertTrue(any(r.startswith("CORRUPT_IMAGE_HEADER") for r in res["quarantined_records"][1]["rejection_reasons"]))
            self.assertIn("INVALID_DIMENSIONS:16x16", res["quarantined_records"][2]["rejection_reasons"])

    def test_04_exact_near_duplicate_and_conflicting_patient_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r1 = _valid_record(root, "SMP_A1", "PT_3001", "metastasis", "xray", seed=10)
            # Exact byte copy under another patient ID
            r2 = _valid_record(root, "SMP_A2", "PT_3002", "metastasis", "xray", seed=10)
            # Near-duplicate (1 pixel perturbation -> identical 9x8 dHash)
            r3 = _valid_record(root, "SMP_B1", "PT_3003", "healthy", "xray", seed=20)
            r4 = _valid_record(root, "SMP_B2", "PT_3004", "healthy", "xray", seed=20)
            (root / r4["image_path"]).write_bytes(_make_png_bytes(64, 64, pattern_seed=20, noise_px=1))
            # Conflicting diagnosis for same patient ID
            r5 = _valid_record(root, "SMP_C1", "PT_3005", "paget", "ct", seed=31)
            r6 = _valid_record(root, "SMP_C2", "PT_3005", "healthy", "ct", seed=45)

            res = validate_track_d_dataset(root, records=[r1, r2, r3, r4, r5, r6])
            self.assertGreaterEqual(res["exact_duplicate_count"], 1)
            self.assertGreaterEqual(res["near_duplicate_count"], 1)
            self.assertEqual(res["conflicting_patient_count"], 1)
            by_sid = {r["sample_id"]: r["rejection_reasons"] for r in res["records"]}
            self.assertIn("DUPLICATE_EXACT_SHA256_CROSS_PATIENT", by_sid["SMP_A1"])
            self.assertIn("DUPLICATE_NEAR_PERCEPTUAL_CROSS_PATIENT", by_sid["SMP_B2"])
            self.assertIn("CONFLICTING_PATIENT_DIAGNOSIS", by_sid["SMP_C1"])
            self.assertIn("CONFLICTING_PATIENT_DIAGNOSIS", by_sid["SMP_C2"])

    def test_05_deterministic_patient_level_splits_and_modality_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            seed_ctr = 100
            for mod in ("xray", "bone_scan", "ct"):
                for cls in ("metastasis", "paget", "oi", "healthy"):
                    for p_idx in range(4):
                        pid = f"PT_{mod.upper()}_{cls.upper()}_{p_idx}"
                        # 2 images per patient to verify grouping
                        for v in range(2):
                            seed_ctr += 1
                            inst = "INST_EXT" if (p_idx == 3 and cls == "metastasis" and mod == "xray") else "INST_A"
                            records.append(
                                _valid_record(
                                    root,
                                    f"S_{mod}_{cls}_{p_idx}_{v}",
                                    pid,
                                    cls,
                                    mod,
                                    seed=seed_ctr,
                                    institution_id=inst,
                                )
                            )

            val_res = validate_track_d_dataset(root, records=records)
            self.assertEqual(val_res["quarantined_count"], 0)
            s1 = generate_leakage_safe_splits(val_res["passed_records"], seed=42, holdout_institutions=["INST_EXT"])
            s2 = generate_leakage_safe_splits(val_res["passed_records"], seed=42, holdout_institutions=["INST_EXT"])
            self.assertEqual(
                [(r["sample_id"], r["split_assignment"]) for r in s1["records"]],
                [(r["sample_id"], r["split_assignment"]) for r in s2["records"]],
            )
            self.assertTrue(s1["leakage_audit"]["is_clean"])
            self.assertGreater(s1["distribution"]["train"], 0)
            self.assertGreater(s1["distribution"]["val"], 0)
            self.assertGreater(s1["distribution"]["test"], 0)
            self.assertEqual(s1["distribution"]["external_test"], 2)

            # Verify every patient stays in exactly 1 split
            pid_splits = {}
            for r in s1["records"]:
                pid_splits.setdefault(r["patient_id"], set()).add(r["split_assignment"])
            self.assertTrue(all(len(v) == 1 for v in pid_splits.values()))

    def test_06_cross_split_leakage_detection(self) -> None:
        fake_records = [
            {"sample_id": "S1", "patient_id": "PT_1", "sha256": "AAAA" * 16, "perceptual_hash": "0000000000000000", "split_assignment": "train"},
            {"sample_id": "S2", "patient_id": "PT_2", "sha256": "AAAA" * 16, "perceptual_hash": "0000000000000001", "split_assignment": "test"},
            {"sample_id": "S3", "patient_id": "PT_1", "sha256": "BBBB" * 16, "perceptual_hash": "ffffffffffffffff", "split_assignment": "val"},
        ]
        audit = verify_cross_split_leakage(fake_records)
        self.assertFalse(audit["is_clean"])
        joined = " ".join(audit["violations"])
        self.assertIn("PATIENT_OVERLAP", joined)
        self.assertIn("EXACT_HASH_CROSS_SPLIT", joined)
        self.assertIn("NEAR_DUPLICATE_CROSS_SPLIT", joined)

    def test_07_missing_metadata_permissions_and_quarantine_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = _valid_record(root, "SMP_MISS", "PT_4001", "healthy", "xray", seed=55)
            rec["license_permissions"] = "MISSING"
            rec["ethics_approval_ref"] = ""
            rec["institution_id"] = ""

            manifest_path = root / "ml_service" / "data" / "manifests" / "track_d_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"manifest_version": "1.0.0", "track": "track_d", "records": [rec]}), encoding="utf-8")

            out = run_track_d_intake_pipeline(root, copy_quarantined_files=True, write_outputs=True)
            self.assertEqual(out["readiness"]["quarantined_samples"], 1)
            self.assertEqual(out["readiness"]["readiness_status"], "BLOCKED")
            q_report = Path(out["outputs"]["quarantine_report"])
            self.assertTrue(q_report.exists())
            q_data = json.loads(q_report.read_text(encoding="utf-8"))
            self.assertEqual(q_data["quarantined_count"], 1)
            self.assertIn("MISSING_OR_INVALID_PERMISSIONS", q_data["reason_counts"])

    def test_08_readiness_gating_empty_vs_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_out = run_track_d_intake_pipeline(root, write_outputs=False)
            self.assertEqual(empty_out["readiness"]["readiness_status"], "NOT_AVAILABLE")
            self.assertFalse(empty_out["readiness"]["ready_for_training"])
            self.assertEqual(empty_out["readiness"]["valid_samples"], 0)

            records = []
            s_id = 200
            for cls in ("metastasis", "paget", "oi", "healthy"):
                for p in range(3):
                    s_id += 1
                    records.append(_valid_record(root, f"RD_{cls}_{p}", f"PT_RD_{cls}_{p}", cls, "xray", seed=s_id))
            val_res = validate_track_d_dataset(root, records=records)
            sp_res = generate_leakage_safe_splits(val_res["passed_records"], seed=42)
            gate_res = evaluate_track_d_readiness(val_res, sp_res)
            self.assertEqual(gate_res["readiness_status"], "READY_FOR_TRAINING")
            self.assertTrue(gate_res["ready_for_training"])
            self.assertTrue(all(gate_res["gates"].values()))

    def test_09_protected_btxrd_center2_center3_holdout_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout_rec = _valid_record(root, "C2_IMG_001", "PT_5001", "healthy", "xray", seed=77, institution_id="center_2")
            clean_rec = _valid_record(root, "IMG_CLEAN_001", "PT_5002", "healthy", "xray", seed=78, institution_id="INST_NEW")
            sha_clean = inspect_image_file(root / clean_rec["image_path"])["sha256"]

            split_csv = root / "ml_service" / "data" / "splits" / "btxrd_split_manifest.csv"
            split_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(split_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "split", "sha256_hash"])
                w.writerow(["C3_IMG_999", "center_3_external", sha_clean])

            res = validate_track_d_dataset(root, records=[holdout_rec, clean_rec])
            self.assertEqual(res["passed_count"], 0)
            self.assertEqual(res["quarantined_count"], 2)
            self.assertIn("PROTECTED_EXTERNAL_HOLDOUT_VIOLATION", res["quarantined_records"][0]["rejection_reasons"])
            self.assertIn("PROTECTED_EXTERNAL_HOLDOUT_HASH_MATCH", res["quarantined_records"][1]["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
