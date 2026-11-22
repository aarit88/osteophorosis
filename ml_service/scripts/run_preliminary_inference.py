"""
OsteoGuard AI - Checkpoint Inventory & Preliminary Single-Track Research Inference CLI.

Capabilities:
  1. `--verify-only`: Pure-stdlib audit of all Track A, B, C, and D expected checkpoint paths
     plus Frozen BTXRD SHA-256 (21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD)
     and byte size (244,849,703 bytes) verification. Never creates fake weights.
  2. `--track track_a|track_c --image <path>`: Runs genuine preliminary inference on a new,
     non-holdout plain radiograph using recovered weights, optional Grad-CAM heatmap generation,
     and automatic holdout-exclusion checks against split manifests if present on disk.

Strictly preserves:
  - Zero fake checkpoints or random-weight substitutes.
  - Zero training or 4-class Track D clinical inference activation.
  - Holdout isolation for FracAtlas test (N=614) and BTXRD Center 2 (N=549) / Center 3 (N=259).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is importable when invoked as a standalone script
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FROZEN_BTXRD_EXPECTED_SHA256 = "21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD"
FROZEN_BTXRD_EXPECTED_SIZE_BYTES = 244849703

CHECKPOINT_INVENTORY = [
    {
        "track": "Track A (FracAtlas ResNet-101 Full Best)",
        "rel_path": "ml_artifacts/fracatlas_full_training/best_checkpoint.pt",
        "priority": "PRIMARY_PRELIMINARY_TARGET",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    {
        "track": "Track A (FracAtlas ResNet-101 Full Last)",
        "rel_path": "ml_artifacts/fracatlas_full_training/last_checkpoint.pt",
        "priority": "AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    {
        "track": "Track A (FracAtlas ResNet-101 Pilot Best)",
        "rel_path": "ml_artifacts/fracatlas_pilot/best_checkpoint.pt",
        "priority": "PILOT_AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    {
        "track": "Track A (FracAtlas ResNet-101 Pilot Last)",
        "rel_path": "ml_artifacts/fracatlas_pilot/last_checkpoint.pt",
        "priority": "PILOT_AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    {
        "track": "Track C (Frozen BTXRD ResNet-34)",
        "rel_path": "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt",
        "priority": "PRIMARY_PRELIMINARY_TARGET_FROZEN",
        "expected_sha256": FROZEN_BTXRD_EXPECTED_SHA256,
        "expected_size_bytes": FROZEN_BTXRD_EXPECTED_SIZE_BYTES,
    },
    {
        "track": "Track C (BTXRD Full Training Best)",
        "rel_path": "ml_artifacts/btxrd_full_training/best_checkpoint.pt",
        "priority": "SOURCE_FOR_FROZEN_COPY",
        "expected_sha256": FROZEN_BTXRD_EXPECTED_SHA256,
        "expected_size_bytes": FROZEN_BTXRD_EXPECTED_SIZE_BYTES,
    },
    {
        "track": "Track C (BTXRD Full Training Last)",
        "rel_path": "ml_artifacts/btxrd_full_training/last_checkpoint.pt",
        "priority": "AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    {
        "track": "Track C (BTXRD Pilot Best)",
        "rel_path": "ml_artifacts/btxrd_pilot/best_checkpoint.pt",
        "priority": "PILOT_AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    {
        "track": "Track C (BTXRD Pilot Last)",
        "rel_path": "ml_artifacts/btxrd_pilot/last_checkpoint.pt",
        "priority": "PILOT_AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    *[
        {
            "track": f"Track B (Bone Scan Ensemble Fold {fold} Best)",
            "rel_path": f"ml_artifacts/bone_scan_cv/fold_{fold}/best_checkpoint.pt",
            "priority": "SECONDARY_ENSEMBLE",
            "expected_sha256": None,
            "expected_size_bytes": None,
        }
        for fold in range(5)
    ],
    *[
        {
            "track": f"Track B (Bone Scan Ensemble Fold {fold} Last)",
            "rel_path": f"ml_artifacts/bone_scan_cv/fold_{fold}/last_checkpoint.pt",
            "priority": "AUXILIARY",
            "expected_sha256": None,
            "expected_size_bytes": None,
        }
        for fold in range(5)
    ],
    {
        "track": "Track B (Bone Scan Pilot Best)",
        "rel_path": "ml_artifacts/bone_scan_pilot/best_checkpoint.pt",
        "priority": "PILOT_AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
    {
        "track": "Track B (Bone Scan Pilot Last)",
        "rel_path": "ml_artifacts/bone_scan_pilot/last_checkpoint.pt",
        "priority": "PILOT_AUXILIARY",
        "expected_sha256": None,
        "expected_size_bytes": None,
    },
]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def inspect_checkpoint_inventory(base_dir: Path = REPO_ROOT) -> Dict[str, Any]:
    """Audits expected checkpoints on disk using pure stdlib without modifying any artifacts."""
    items: List[Dict[str, Any]] = []
    present_count = 0

    for spec in CHECKPOINT_INVENTORY:
        full_path = base_dir / spec["rel_path"]
        exists = full_path.is_file()
        entry: Dict[str, Any] = {
            "track": spec["track"],
            "path": spec["rel_path"],
            "priority": spec["priority"],
            "status": "MISSING",
            "size_bytes": None,
            "sha256": None,
            "integrity": "NOT_PRESENT",
        }
        if exists:
            present_count += 1
            size_bytes = full_path.stat().st_size
            sha = compute_sha256(full_path)
            entry["size_bytes"] = size_bytes
            entry["sha256"] = sha
            entry["status"] = "PRESENT"

            exp_sha = spec["expected_sha256"]
            exp_size = spec["expected_size_bytes"]
            if exp_sha is not None and exp_size is not None:
                if size_bytes == exp_size and sha == exp_sha:
                    entry["integrity"] = "VERIFIED_EXACT_MATCH"
                else:
                    entry["integrity"] = "MISMATCH"
            else:
                entry["integrity"] = "PRESENT_UNPINNED"
        items.append(entry)

    track_d_dir = base_dir / "ml_artifacts/checkpoints"
    track_d_files = []
    if track_d_dir.exists():
        track_d_files = sorted(
            p.name for p in track_d_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pt", ".pth"}
        )

    return {
        "base_dir": str(base_dir),
        "checkpoints_present": present_count,
        "checkpoints_total_tracked": len(CHECKPOINT_INVENTORY),
        "four_class_track_d_checkpoints": track_d_files,
        "four_class_status": "BLOCKED" if not track_d_files else "PRESENT_UNVERIFIED",
        "inventory": items,
    }


def collect_quarantined_holdout_hashes(base_dir: Path = REPO_ROOT) -> Tuple[Set[str], Dict[str, str]]:
    """
    Collects known quarantined holdout image SHA-256 hashes from split manifests if present on disk:
      - FracAtlas holdout test split (N=614)
      - BTXRD Center 2 (N=549) & Center 3 (N=259) external holdout splits
    """
    holdout_hashes: Set[str] = set()
    hash_sources: Dict[str, str] = {}

    fracatlas_manifest = base_dir / "ml_service/data/splits/fracatlas_classification_split.csv"
    if fracatlas_manifest.is_file():
        with open(fracatlas_manifest, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("split") or "").strip().lower() == "test":
                    sha = (row.get("sha256_hash") or "").strip().upper()
                    if sha:
                        holdout_hashes.add(sha)
                        hash_sources[sha] = "FracAtlas holdout test split (N=614)"

    btxrd_manifest = base_dir / "ml_service/data/splits/btxrd_split_manifest.csv"
    if btxrd_manifest.is_file():
        with open(btxrd_manifest, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                split_val = (row.get("center_holdout_split") or "").strip().lower()
                if split_val in {"external_test_center2", "external_test_center3"}:
                    sha = (row.get("sha256_hash") or row.get("image_sha256") or "").strip().upper()
                    if sha:
                        holdout_hashes.add(sha)
                        hash_sources[sha] = f"BTXRD {split_val}"

    return holdout_hashes, hash_sources


def check_image_not_in_quarantined_holdouts(
    image_path: Path,
    base_dir: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Rejects accidental preliminary test execution on sealed holdout images if manifests exist."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    img_sha = compute_sha256(image_path)
    holdout_hashes, hash_sources = collect_quarantined_holdout_hashes(base_dir=base_dir)
    if img_sha in holdout_hashes:
        raise PermissionError(
            f"HOLDOUT_VIOLATION: Image {image_path.name} (SHA-256={img_sha}) matches sealed holdout "
            f"[{hash_sources[img_sha]}]. Preliminary testing must use a new non-holdout image."
        )

    return {
        "image_path": str(image_path),
        "image_sha256": img_sha,
        "holdout_manifest_hashes_checked": len(holdout_hashes),
        "holdout_status": "CLEAR",
    }


def _save_gradcam_overlay(
    image_path: Path,
    cam_heatmap: Any,
    out_path: Path,
    is_letterbox_512: bool = False,
) -> str:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if is_letterbox_512:
            w, h = rgb.size
            scale = 512 / max(w, h)
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
            base_img = Image.new("RGB", (512, 512), (0, 0, 0))
            base_img.paste(resized, ((512 - new_w) // 2, (512 - new_h) // 2))
        else:
            base_img = rgb.resize((224, 224), Image.Resampling.BILINEAR)

    h_arr = np.clip(cam_heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h_arr - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h_arr - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h_arr - 1.0), 0.0, 1.0)
    a = np.where(h_arr > 0.12, h_arr * 0.45, 0.0)
    overlay = Image.fromarray((np.stack([r, g, b, a], axis=-1) * 255.0).astype(np.uint8), mode="RGBA")
    blended = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(out_path, format="PNG")
    return str(out_path)


def run_preliminary_inference(
    track: str,
    image_path: Path,
    checkpoint_path: Optional[Path] = None,
    threshold: float = 0.31,
    gradcam_out: Optional[Path] = None,
    allow_holdout: bool = False,
    base_dir: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Executes genuine preliminary research inference for Track A or Track C."""
    if not allow_holdout:
        holdout_audit = check_image_not_in_quarantined_holdouts(image_path=image_path, base_dir=base_dir)
    else:
        holdout_audit = {
            "image_path": str(image_path),
            "image_sha256": compute_sha256(image_path),
            "holdout_status": "OVERRIDE_ALLOWED",
        }

    if track == "track_a":
        from ml_service.inference.fracatlas_inference import (
            FracAtlasFractureClassifier,
            preprocess_fracatlas_image,
        )
        from osteoguard.explainability.gradcam import GradCAMExplainer

        classifier = FracAtlasFractureClassifier(
            checkpoint_path=checkpoint_path,
            default_threshold=threshold,
        )
        result = classifier.predict_image(image_path, threshold=threshold)
        if gradcam_out is not None:
            tensor = preprocess_fracatlas_image(image_path).unsqueeze(0).to(classifier.device)
            with GradCAMExplainer(classifier.model, target_layer="layer4") as explainer:
                cam = explainer.generate_cam(tensor, target_class_idx=1, input_size=(224, 224))
            result["gradcam_overlay_path"] = _save_gradcam_overlay(
                image_path, cam, gradcam_out, is_letterbox_512=False
            )
    elif track == "track_c":
        from ml_service.datasets.btxrd_dataset import preprocess_btxrd_image
        from ml_service.inference.btxrd_inference import BTXRDFrozenClassifier
        from osteoguard.explainability.gradcam import GradCAMExplainer

        classifier = BTXRDFrozenClassifier(
            checkpoint_path=checkpoint_path,
            verify_sha256=True,
        )
        result = classifier.predict_image(image_path)
        result["track"] = "Track C — Frozen BTXRD ResNet-34 Bone Tumour Classifier"
        result["research_notice"] = (
            "PRELIMINARY SINGLE-TRACK RESEARCH OUTPUT ONLY — Primary bone tumour radiograph "
            "research model (uncalibrated raw softmax), not 4-class differential diagnosis."
        )
        if gradcam_out is not None:
            tensor = preprocess_btxrd_image(image_path, target_size=512, is_train=False).unsqueeze(0).to(classifier.device)
            pred_idx = int(result["predicted_class_idx"])
            with GradCAMExplainer(classifier.model, target_layer="layer4") as explainer:
                cam = explainer.generate_cam(tensor, target_class_idx=pred_idx, input_size=(512, 512))
            result["gradcam_overlay_path"] = _save_gradcam_overlay(
                image_path, cam, gradcam_out, is_letterbox_512=True
            )
    else:
        raise ValueError(f"Unsupported preliminary track: {track}. Choose 'track_a' or 'track_c'.")

    return {
        "holdout_audit": holdout_audit,
        "prediction": result,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OsteoGuard AI Checkpoint Recovery Verifier & Preliminary Inference CLI"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify checkpoint presence and Frozen BTXRD SHA-256/byte-size without running inference.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="With --verify-only, exit non-zero if primary Track A or Frozen Track C checkpoint is missing/invalid.",
    )
    parser.add_argument(
        "--track",
        choices=["track_a", "track_c"],
        default="track_c",
        help="Preliminary research model track to execute.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a new, non-holdout plain radiograph PNG/JPG.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint override path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.31,
        help="Track A binary decision threshold (default: 0.31 high-sensitivity screening).",
    )
    parser.add_argument(
        "--gradcam-out",
        type=str,
        default=None,
        help="Optional output PNG path for Grad-CAM overlay.",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Disable holdout-manifest exclusion check (audit use only).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.verify_only:
        report = inspect_checkpoint_inventory(REPO_ROOT)
        print(json.dumps(report, indent=2))
        if args.strict:
            primary_ok = any(
                item["path"] == "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
                and item["integrity"] == "VERIFIED_EXACT_MATCH"
                for item in report["inventory"]
            )
            track_a_ok = any(
                item["path"] == "ml_artifacts/fracatlas_full_training/best_checkpoint.pt"
                and item["status"] == "PRESENT"
                for item in report["inventory"]
            )
            if not (primary_ok and track_a_ok):
                return 2
        return 0

    if not args.image:
        print(
            json.dumps(
                {
                    "error": "MISSING_IMAGE_ARGUMENT",
                    "detail": "Provide --image <path_to_non_holdout_xray> or run with --verify-only.",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    try:
        output = run_preliminary_inference(
            track=args.track,
            image_path=Path(args.image),
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            threshold=args.threshold,
            gradcam_out=Path(args.gradcam_out) if args.gradcam_out else None,
            allow_holdout=args.allow_holdout,
            base_dir=REPO_ROOT,
        )
        print(json.dumps(output, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "detail": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
