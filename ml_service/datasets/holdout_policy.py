"""Central policy and registry for datasets that must never enter Track D.

This module only records exclusion rules and hashes/IDs already present in the
repository's evaluation manifests. It does not infer or create patient data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set


PROTECTED_SITE_MARKERS = frozenset({
    "center2",
    "btxrdcenter2",
    "center3",
    "btxrdcenter3",
    "fracatlastestholdout",
})


def _canonical(value: object) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def is_protected_site(value: object) -> bool:
    canonical = _canonical(value)
    return any(marker in canonical for marker in PROTECTED_SITE_MARKERS)


@dataclass
class ProtectedHoldoutRegistry:
    sample_ids: Set[str] = field(default_factory=set)
    hashes: Set[str] = field(default_factory=set)
    sources: List[str] = field(default_factory=list)

    def record_violations(self, record: Dict[str, object]) -> List[str]:
        reasons: List[str] = []
        for field_name in ("institution_id", "dataset_source", "image_path"):
            if is_protected_site(record.get(field_name)):
                reasons.append("PROTECTED_EXTERNAL_HOLDOUT_VIOLATION")
                break
        sample_id = str(record.get("sample_id") or "").strip()
        if sample_id and sample_id.casefold() in self.sample_ids:
            reasons.append("PROTECTED_EXTERNAL_HOLDOUT_SAMPLE_ID")
        sha = str(record.get("sha256") or "").strip().upper()
        if sha and sha in self.hashes:
            reasons.append("PROTECTED_EXTERNAL_HOLDOUT_HASH_MATCH")
        return reasons


def load_protected_holdouts(root_dir: Path) -> ProtectedHoldoutRegistry:
    """Aggregate all known protected split and sealed-evaluation registries."""
    registry = ProtectedHoldoutRegistry()
    candidates = (
        ("ml_service/data/splits/btxrd_split_manifest.csv", "btxrd"),
        ("ml_service/data/splits/fracatlas_classification_split.csv", "fracatlas"),
        ("ml_artifacts/btxrd_center2_external_evaluation/center2_predictions.csv", "center2"),
        ("ml_artifacts/btxrd_center3_external_evaluation/center3_predictions.csv", "center3"),
    )
    for relpath, kind in candidates:
        path = root_dir / relpath
        if not path.is_file():
            continue
        registry.sources.append(relpath)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                split = _canonical(row.get("split") or row.get("center") or "")
                protected = (
                    kind == "center2"
                    or kind == "center3"
                    or (kind == "btxrd" and any(x in split for x in ("center2", "center3", "external")))
                    or (kind == "fracatlas" and split == "test")
                )
                if not protected:
                    continue
                sample_id = (row.get("image_id") or row.get("sample_id") or "").strip()
                digest = (row.get("sha256_hash") or row.get("sha256") or "").strip().upper()
                if sample_id:
                    registry.sample_ids.add(sample_id.casefold())
                if len(digest) == 64 and all(ch in "0123456789ABCDEF" for ch in digest):
                    registry.hashes.add(digest)
    return registry


def normalize_holdout_sites(values: Iterable[str]) -> Set[str]:
    return {_canonical(value) for value in values if _canonical(value)}
