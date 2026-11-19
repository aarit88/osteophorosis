"""Unit tests for ml_service/scripts/run_training.py (stdlib unittest compatible)."""

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from ml_service.scripts.run_training import (
    TARGET_CLASS_DIRS,
    build_patient_stratified_kfold_splits,
    parse_args,
)


class TestRunTrainingScript(unittest.TestCase):
    def _build_synthetic_records_for_split_test(self):
        # Strictly tests in-memory splitting logic with dummy identifiers (no clinical fabrication).
        records = []
        for cls in TARGET_CLASS_DIRS:
            for p_idx in range(10):
                pid = f"{cls}_PT{p_idx:02d}"
                # 2 scans per patient to verify zero multi-scan patient leakage across folds
                for scan_idx in range(2):
                    records.append(
                        SimpleNamespace(
                            sample_id=f"{pid}_S{scan_idx}",
                            patient_id=pid,
                            label=cls,
                            file_path=f"/tmp/{pid}_S{scan_idx}.png",
                            modality="xray",
                        )
                    )
        return records

    def test_patient_stratified_kfold_zero_leakage_and_full_coverage(self):
        records = self._build_synthetic_records_for_split_test()
        splits = build_patient_stratified_kfold_splits(records, n_folds=5, seed=42)
        self.assertEqual(len(splits), 5)

        seen_val_patients = set()
        for fold_idx, (train_samples, val_samples, train_pts, val_pts) in enumerate(splits):
            self.assertTrue(train_pts.isdisjoint(val_pts), f"Leakage in fold {fold_idx}")
            self.assertEqual(len(train_pts), 32)
            self.assertEqual(len(val_pts), 8)
            self.assertEqual(len(train_samples), 64)
            self.assertEqual(len(val_samples), 16)

            # Verify every class has exactly 2 validation patients per fold
            val_class_counts = {c: 0 for c in TARGET_CLASS_DIRS}
            for s in val_samples:
                val_class_counts[s.label] += 1
            for cls, cnt in val_class_counts.items():
                self.assertEqual(cnt, 4, f"Class {cls} expected 4 val samples in fold {fold_idx}, got {cnt}")

            seen_val_patients.update(val_pts)

        self.assertEqual(len(seen_val_patients), 40)

    def test_insufficient_patients_for_k_raises_value_error(self):
        records = [
            SimpleNamespace(patient_id=f"{cls}_0", label=cls)
            for cls in TARGET_CLASS_DIRS
        ]
        with self.assertRaises(ValueError):
            build_patient_stratified_kfold_splits(records, n_folds=5, seed=42)

    def test_cli_defaults_match_clinical_directories(self):
        args = parse_args([])
        self.assertEqual(args.dataset_dir, "ml_service/data/raw")
        self.assertEqual(args.checkpoints_dir, "ml_artifacts/checkpoints")
        self.assertEqual(args.logs_dir, "ml_artifacts/logs")
        self.assertEqual(args.folds, 5)

    def test_cli_blocks_empty_dataset_without_creating_fake_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            ckpt_dir = root / "checkpoints"
            logs_dir = root / "logs"
            raw_dir.mkdir(parents=True)

            proc = subprocess.run(
                [
                    sys.executable,
                    "ml_service/scripts/run_training.py",
                    "--dataset-dir",
                    str(raw_dir),
                    "--checkpoints-dir",
                    str(ckpt_dir),
                    "--logs-dir",
                    str(logs_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(ckpt_dir.exists())
            self.assertFalse(logs_dir.exists())


if __name__ == "__main__":
    unittest.main()
