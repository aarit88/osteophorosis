"""Stdlib-compatible tests for checkpoint recovery audit, Frozen BTXRD integrity checks,
PyTorch weights_only=False loading guards, and preliminary inference CLI safety.
"""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ml_service.scripts.run_preliminary_inference import (
    FROZEN_BTXRD_EXPECTED_SHA256,
    FROZEN_BTXRD_EXPECTED_SIZE_BYTES,
    check_image_not_in_quarantined_holdouts,
    inspect_checkpoint_inventory,
    parse_args,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPreliminaryInferencePrep(unittest.TestCase):
    def test_verify_only_reports_missing_checkpoints_without_fabricating_files(self):
        report = inspect_checkpoint_inventory(REPO_ROOT)
        self.assertEqual(report["four_class_status"], "BLOCKED")
        self.assertEqual(report["four_class_track_d_checkpoints"], [])
        self.assertEqual(report["checkpoints_present"], 0)

        proc = subprocess.run(
            [sys.executable, "ml_service/scripts/run_preliminary_inference.py", "--verify-only"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["checkpoints_present"], 0)
        self.assertEqual(payload["four_class_status"], "BLOCKED")

    def test_strict_verify_exits_nonzero_when_checkpoints_missing(self):
        proc = subprocess.run(
            [sys.executable, "ml_service/scripts/run_preliminary_inference.py", "--verify-only", "--strict"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)

    def test_frozen_btxrd_constants_match_stage18_manifest(self):
        manifest_path = REPO_ROOT / "ml_artifacts/btxrd_frozen_model/model_freeze_manifest.json"
        sha_txt_path = REPO_ROOT / "ml_artifacts/btxrd_frozen_model/checkpoint_sha256.txt"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sha_txt = sha_txt_path.read_text(encoding="utf-8").strip().upper()

        self.assertEqual(manifest["checkpoint_sha256"], FROZEN_BTXRD_EXPECTED_SHA256)
        self.assertEqual(sha_txt, FROZEN_BTXRD_EXPECTED_SHA256)
        self.assertEqual(manifest["checkpoint_size_bytes"], FROZEN_BTXRD_EXPECTED_SIZE_BYTES)

    def test_tampered_frozen_btxrd_checkpoint_fails_size_and_hash_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frozen_dir = root / "ml_artifacts/btxrd_frozen_model"
            frozen_dir.mkdir(parents=True)
            fake_ckpt = frozen_dir / "best_checkpoint.pt"
            fake_ckpt.write_bytes(b"not-a-real-checkpoint")

            report = inspect_checkpoint_inventory(root)
            btxrd_entry = next(
                item for item in report["inventory"]
                if item["path"] == "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
            )
            self.assertEqual(btxrd_entry["status"], "PRESENT")
            self.assertEqual(btxrd_entry["integrity"], "MISMATCH")

    def test_holdout_exclusion_guard_blocks_sealed_test_hashes_and_allows_new_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            splits_dir = root / "ml_service/data/splits"
            splits_dir.mkdir(parents=True)

            held_out_img = root / "held_out.png"
            held_out_img.write_bytes(b"sealed-holdout-image-bytes")
            held_out_sha = hashlib.sha256(held_out_img.read_bytes()).hexdigest()

            fresh_img = root / "fresh_research.png"
            fresh_img.write_bytes(b"brand-new-non-holdout-xray")

            (splits_dir / "fracatlas_classification_split.csv").write_text(
                "image_id,split,sha256_hash\n"
                f"IMG_001,test,{held_out_sha}\n",
                encoding="utf-8",
            )

            with self.assertRaises(PermissionError):
                check_image_not_in_quarantined_holdouts(held_out_img, base_dir=root)

            clear = check_image_not_in_quarantined_holdouts(fresh_img, base_dir=root)
            self.assertEqual(clear["holdout_status"], "CLEAR")
            self.assertEqual(clear["holdout_manifest_hashes_checked"], 1)

    def test_cli_refuses_missing_checkpoint_cleanly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "sample.png"
            img.write_bytes(b"png-placeholder-for-cli-error-test")
            proc = subprocess.run(
                [
                    sys.executable,
                    "ml_service/scripts/run_preliminary_inference.py",
                    "--track",
                    "track_c",
                    "--image",
                    str(img),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("error", proc.stderr)

    def test_all_inference_and_evaluator_torch_loads_specify_weights_only_false(self):
        target_files = [
            "ml_service/inference/btxrd_inference.py",
            "ml_service/inference/fracatlas_inference.py",
            "ml_service/models/bone_scan_ensemble.py",
            "ml_service/evaluation/btxrd_center2_evaluator.py",
            "ml_service/evaluation/btxrd_center3_evaluator.py",
        ]
        for rel in target_files:
            source = (REPO_ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
            torch_load_calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch"
                        and node.func.attr == "load"
                    ):
                        kw_names = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                        torch_load_calls.append(kw_names)

            self.assertGreater(len(torch_load_calls), 0, f"Expected torch.load in {rel}")
            for kw_map in torch_load_calls:
                self.assertIn("weights_only", kw_map, f"Missing weights_only in {rel}")
                self.assertIsInstance(kw_map["weights_only"], ast.Constant)
                self.assertIs(kw_map["weights_only"].value, False, f"Expected weights_only=False in {rel}")

    def test_cli_parse_args_defaults(self):
        args = parse_args(["--verify-only"])
        self.assertTrue(args.verify_only)
        self.assertEqual(args.track, "track_c")
        self.assertAlmostEqual(args.threshold, 0.31)
        self.assertFalse(args.allow_holdout)


if __name__ == "__main__":
    unittest.main()
