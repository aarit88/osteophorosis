"""
Unit tests for Stage 32 cross-platform Python checkpoint inventory & recovery CLI
(`ml_service/scripts/recover_checkpoints.py`). Pure Python stdlib; zero fake weights saved
to workspace paths.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import urllib.error

from ml_service.scripts.recover_checkpoints import (
    FROZEN_BTXRD_EXPECTED_SHA256,
    FROZEN_BTXRD_EXPECTED_SIZE_BYTES,
    download_checkpoint_resumable,
    recover_from_local_directories,
    recover_from_remote_manifest,
    resolve_confined_target,
    scan_workspace_weight_files,
    validate_remote_https_url,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class _FakeHTTPResponse:
    def __init__(self, data: bytes, status_code: int = 200):
        self._buf = io.BytesIO(data)
        self.status = status_code

    def read(self, amt: int = -1) -> bytes:
        return self._buf.read(amt)

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._buf.close()
        return False


class TestRecoverCheckpoints(unittest.TestCase):
    def test_workspace_cli_inventory_reports_zero_weights_and_no_remote_storage(self):
        proc = subprocess.run(
            [sys.executable, "ml_service/scripts/recover_checkpoints.py", "--inventory"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "inventory_only")
        self.assertFalse(payload["remote_storage_configured"])
        self.assertEqual(payload["workspace_weight_files_found"], [])
        inv = payload["expected_checkpoint_inventory"]
        self.assertEqual(inv["checkpoints_present"], 0)
        self.assertEqual(inv["checkpoints_total_tracked"], 21)
        self.assertEqual(inv["four_class_status"], "BLOCKED")

    def test_scan_workspace_weight_files_discovers_hidden_and_excludes_node_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hidden = root / ".hidden_weights"
            hidden.mkdir()
            wfile = hidden / "sample.onnx"
            wfile.write_bytes(b"onnx-test-bytes")

            nm = root / "node_modules" / "pkg"
            nm.mkdir(parents=True)
            (nm / "ignored.pt").write_bytes(b"ignore")

            findings = scan_workspace_weight_files(root)
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["rel_path"], ".hidden_weights/sample.onnx")
            self.assertEqual(findings[0]["size_bytes"], len(b"onnx-test-bytes"))

    def test_blocks_non_https_and_ssrf_targets(self):
        with self.assertRaises(ValueError):
            validate_remote_https_url("http://example.com/model.pt", resolve_dns=False)
        with self.assertRaises(ValueError):
            validate_remote_https_url("https://user:pass@example.com/model.pt", resolve_dns=False)
        with self.assertRaises(ValueError):
            validate_remote_https_url("https://127.0.0.1/model.pt", resolve_dns=False)
        with self.assertRaises(ValueError):
            validate_remote_https_url("https://169.254.169.254/latest/meta-data", resolve_dns=False)
        with self.assertRaises(ValueError):
            validate_remote_https_url("https://10.0.0.5/model.pt", resolve_dns=False)
        with self.assertRaises(ValueError):
            validate_remote_https_url("https://localhost/model.pt", resolve_dns=False)

        safe = validate_remote_https_url("https://storage.example.org/ckpt.pt?sig=secret123", resolve_dns=False)
        self.assertTrue(safe.startswith("https://"))

    def test_rejects_arbitrary_rel_paths_and_filename_only_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                resolve_confined_target(root, "../escape.pt")
            with self.assertRaises(ValueError):
                resolve_confined_target(root, "ml_artifacts/arbitrary/model.pt")

            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "rel_path": "ml_artifacts/fracatlas_full_training/best_checkpoint.pt",
                                "url": "https://cdn.example.org/best_checkpoint.pt",
                                # missing expected_sha256 and expected_size_bytes
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                recover_from_remote_manifest(manifest, base_dir=root, resolve_dns=False)

    def test_rejects_weakened_frozen_btxrd_manifest_spec_and_inline_secret(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_btxrd = root / "bad_btxrd.json"
            bad_btxrd.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "rel_path": "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt",
                                "url": "https://cdn.example.org/btxrd.pt",
                                "expected_sha256": "0" * 64,
                                "expected_size_bytes": FROZEN_BTXRD_EXPECTED_SIZE_BYTES,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                recover_from_remote_manifest(bad_btxrd, base_dir=root, resolve_dns=False)

            bad_secret = root / "bad_secret.json"
            bad_secret.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "rel_path": "ml_artifacts/fracatlas_full_training/best_checkpoint.pt",
                                "url": "https://cdn.example.org/frac.pt",
                                "expected_sha256": "A" * 64,
                                "expected_size_bytes": 100,
                                "token": "hardcoded-secret",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                recover_from_remote_manifest(bad_secret, base_dir=root, resolve_dns=False)

    def test_resumable_download_verifies_sha256_places_atomically_and_redacts_secrets(self):
        payload = b"RESUMABLE_CHECKPOINT_PREFIX_" + b"SUFFIX_BYTES_END"
        prefix = payload[:20]
        full_sha = hashlib.sha256(payload).hexdigest().upper()
        rel = "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoints": [
                            {
                                "rel_path": rel,
                                "url": "https://cdn.example.org/fracatlas/best_checkpoint.pt?token=PRESIGNED_SECRET",
                                "expected_sha256": full_sha,
                                "expected_size_bytes": len(payload),
                                "auth_env_var": "OSTEO_TEST_REMOTE_BEARER",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            # Pre-seed partial .part file to simulate resuming from byte 20
            tmp_dir = root / "ml_artifacts" / ".recovery_tmp"
            tmp_dir.mkdir(parents=True)
            safe_key = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]
            part_file = tmp_dir / f"{safe_key}_best_checkpoint.pt.part"
            part_file.write_bytes(prefix)

            captured_headers = {}

            def mock_opener(req, timeout=30):
                captured_headers.update({k.lower(): v for k, v in req.header_items()})
                self.assertEqual(captured_headers.get("range"), "bytes=20-")
                return _FakeHTTPResponse(payload[20:], status_code=206)

            os.environ["OSTEO_TEST_REMOTE_BEARER"] = "super-secret-bearer-token"
            try:
                res = recover_from_remote_manifest(
                    manifest,
                    base_dir=root,
                    resolve_dns=False,
                    opener=mock_opener,
                )
            finally:
                os.environ.pop("OSTEO_TEST_REMOTE_BEARER", None)

            self.assertEqual(captured_headers.get("authorization"), "Bearer super-secret-bearer-token")
            dest = root / rel
            self.assertTrue(dest.is_file())
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(part_file.exists())

            serialized = json.dumps(res)
            self.assertNotIn("super-secret-bearer-token", serialized)
            self.assertNotIn("PRESIGNED_SECRET", serialized)
            self.assertEqual(res["results"][0]["status"], "RECOVERED_RESUMED_AND_VERIFIED")
            self.assertEqual(res["results"][0]["resumed_from_byte"], 20)

    def test_mismatched_frozen_btxrd_download_is_quarantined_and_never_placed(self):
        rel = "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dest = root / rel
            entry = {
                "rel_path": rel,
                "dest_path": dest,
                "url": "https://cdn.example.org/btxrd.pt",
                "redacted_url": "https://cdn.example.org/btxrd.pt",
                "expected_sha256": FROZEN_BTXRD_EXPECTED_SHA256,
                "expected_size_bytes": FROZEN_BTXRD_EXPECTED_SIZE_BYTES,
                "auth_env_var": None,
            }

            bad_bytes = b"corrupt-btxrd-checkpoint-data"

            def mock_opener(_req, timeout=30):
                return _FakeHTTPResponse(bad_bytes, status_code=200)

            with self.assertRaises(ValueError):
                download_checkpoint_resumable(entry, base_dir=root, opener=mock_opener)

            self.assertFalse(dest.exists())
            quarantine_dir = root / "ml_artifacts" / "recovery_quarantine" / "btxrd_mismatch"
            q_files = list(quarantine_dir.glob("*best_checkpoint.pt"))
            self.assertEqual(len(q_files), 1)
            self.assertEqual(q_files[0].read_bytes(), bad_bytes)

    def test_local_recovery_quarantines_mismatched_btxrd_without_deleting_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            root.mkdir()
            src_root = Path(tmpdir) / "backup"
            btxrd_src = src_root / "ml_artifacts" / "btxrd_frozen_model"
            btxrd_src.mkdir(parents=True)
            bad_file = btxrd_src / "best_checkpoint.pt"
            bad_file.write_bytes(b"not-244mb-btxrd")

            frac_src = src_root / "ml_artifacts" / "fracatlas_full_training"
            frac_src.mkdir(parents=True)
            frac_file = frac_src / "best_checkpoint.pt"
            frac_file.write_bytes(b"valid-fracatlas-bytes")

            report = recover_from_local_directories([src_root], base_dir=root)
            statuses = {a["status"] for a in report["actions"]}
            self.assertIn("QUARANTINED_BTXRD_MISMATCH", statuses)
            self.assertIn("RECOVERED_MATCHED_PATH", statuses)

            # Originals untouched, frozen destination uncreated, fracatlas copied safely
            self.assertTrue(bad_file.is_file())
            self.assertFalse((root / "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt").exists())
            self.assertTrue((root / "ml_artifacts/fracatlas_full_training/best_checkpoint.pt").is_file())


if __name__ == "__main__":
    unittest.main()
