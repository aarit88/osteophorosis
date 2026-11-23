from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from ml_service.utils.checkpoint_inventory import build_checkpoint_inventory


class TestCheckpointInventory(unittest.TestCase):
    def test_inventory_reports_missing_track_b_and_frozen_track_c_without_fabricating_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "ml_artifacts/bone_scan_ensemble/ensemble_config.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"checkpoints": [{"fold_index": 0, "path": "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"}]}))
            inventory = build_checkpoint_inventory(root)
            self.assertEqual(inventory["tracks"][0]["status"], "MISSING")
            self.assertEqual(inventory["tracks"][1]["status"], "MISSING")
            self.assertTrue(inventory["read_only"])

    def test_inventory_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "ml_artifacts/bone_scan_ensemble/ensemble_config.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"checkpoints": [{"fold_index": 0, "path": "../outside.pt"}]}))
            self.assertEqual(build_checkpoint_inventory(root)["tracks"][0]["status"], "INVALID_PATH")

    def test_track_b_sha256_pin_is_verified_or_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "ml_artifacts/bone_scan_ensemble/ensemble_config.json"
            config.parent.mkdir(parents=True)
            artifact = root / "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"test fixture bytes; not model weights")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            row = {"fold_index": 0, "path": "ml_artifacts/bone_scan_cv/fold_0/best_checkpoint.pt", "expected_sha256": digest}
            config.write_text(json.dumps({"checkpoints": [row]}))
            self.assertEqual(build_checkpoint_inventory(root)["tracks"][0]["status"], "VERIFIED")
            row["expected_sha256"] = "0" * 64
            config.write_text(json.dumps({"checkpoints": [row]}))
            self.assertEqual(build_checkpoint_inventory(root)["tracks"][0]["status"], "HASH_MISMATCH")


if __name__ == "__main__":
    unittest.main()
