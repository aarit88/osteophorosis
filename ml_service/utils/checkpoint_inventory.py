"""Read-only inventory and integrity checks for model checkpoint artifacts.

No weights are downloaded, generated, modified, or substituted by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TRACK_C = {
    "path": "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt",
    "size_bytes": 244_849_703,
    "sha256": "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD",
}


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest().upper()}


def build_checkpoint_inventory(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    config_path = repo_root / "ml_artifacts/bone_scan_ensemble/ensemble_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tracks: list[dict[str, Any]] = []
    for item in config.get("checkpoints", []):
        rel = Path(item["path"])
        # All manifest paths are repository-relative; reject traversal and absolutes.
        if rel.is_absolute() or ".." in rel.parts:
            tracks.append({"track": "B", "fold": item.get("fold_index"), "path": str(rel), "status": "INVALID_PATH"})
            continue
        full = repo_root / rel
        details = _file_fingerprint(full) if full.is_file() else {}
        expected_hash = str(item.get("expected_sha256") or item.get("sha256") or "").upper()
        if not details:
            track_status = "MISSING"
        elif expected_hash:
            track_status = "VERIFIED" if details["sha256"] == expected_hash else "HASH_MISMATCH"
        else:
            track_status = "PRESENT_UNPINNED"
        tracks.append({
            "track": "B", "fold": item.get("fold_index"), "path": rel.as_posix(),
            "expected_sha256": expected_hash or None, "status": track_status, **details,
        })

    frozen_path = repo_root / FROZEN_TRACK_C["path"]
    if frozen_path.is_file():
        actual = _file_fingerprint(frozen_path)
        frozen_status = "VERIFIED" if actual["size_bytes"] == FROZEN_TRACK_C["size_bytes"] and actual["sha256"] == FROZEN_TRACK_C["sha256"] else "INTEGRITY_MISMATCH"
    else:
        actual, frozen_status = {}, "MISSING"
    tracks.append({"track": "C", **FROZEN_TRACK_C, "status": frozen_status, **actual})
    return {"inventory_version": "1.0", "read_only": True, "tracks": tracks}


if __name__ == "__main__":
    print(json.dumps(build_checkpoint_inventory(), indent=2))
