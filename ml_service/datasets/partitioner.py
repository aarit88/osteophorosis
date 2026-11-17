"""
OsteoGuard AI - Patient-Level Dataset Partitioner & Leakage Guard.
Enforces:
  1. Strict patient-level grouping: zero patient leakage across train, val, and test splits.
  2. Stratification across target diagnostic classes at patient level.
  3. Configurable split ratios (default: 70% train, 15% val, 15% test).
  4. External-site evaluation holdout strategy (out-of-domain site evaluation).
  5. Deterministic, reproducible pseudo-random splitting.
  6. Split manifest generation (split_manifest.csv).
  7. Strict isolation: transformations/scalers fit exclusively on training data;
     test splits are never used for model selection, tuning, or threshold selection.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ml_service.config.settings import get_settings
from ml_service.datasets.auditor import DatasetAuditor
from ml_service.utils.logging import logger
from osteoguard.constants import TARGET_CLASSES


class PatientLeakageError(Exception):
    """Raised when one or more patients are found in multiple dataset splits."""
    pass


class TrainingIsolationError(Exception):
    """Raised when an attempt is made to fit statistical transforms on non-training data."""
    pass


SPLIT_MANIFEST_COLUMNS = [
    "dataset_id",
    "patient_id",
    "source_id",
    "site_id",
    "target_class",
    "modality",
    "file_path",
    "mask_path",
    "split",
    "sha256_hash",
]


def verify_patient_leakage(
    split_manifest_input: Union[Path, str, List[Dict[str, Any]]]
) -> Tuple[bool, List[str]]:
    """
    Verifies that patient sets across splits (train, val, test, external_test)
    are strictly mutually disjoint.

    Returns:
        (is_clean: bool, violations: List[str])
    """
    split_to_patients: Dict[str, Set[str]] = defaultdict(set)

    if isinstance(split_manifest_input, (str, Path)):
        manifest_path = Path(split_manifest_input)
        if not manifest_path.exists():
            return True, []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("patient_id", "").strip()
                split = row.get("split", "").strip()
                if pid and split:
                    split_to_patients[split].add(pid)
    elif isinstance(split_manifest_input, list):
        for row in split_manifest_input:
            pid = str(row.get("patient_id", "")).strip()
            split = str(row.get("split", "")).strip()
            if pid and split:
                split_to_patients[split].add(pid)
    else:
        raise ValueError(f"Unsupported input type for split verification: {type(split_manifest_input)}")

    splits = list(split_to_patients.keys())
    violations: List[str] = []

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1 = splits[i]
            s2 = splits[j]
            overlap = split_to_patients[s1].intersection(split_to_patients[s2])
            if overlap:
                violations.append(
                    f"Patient leakage between split '{s1}' and '{s2}': {len(overlap)} patients leaked: {sorted(list(overlap))[:10]}"
                )

    is_clean = len(violations) == 0
    return is_clean, violations


def assert_training_only_fit(split: str) -> None:
    """
    Guards statistical transformations, scalers, and normalizers from fitting on non-training data.
    """
    if split.lower() != "train":
        raise TrainingIsolationError(
            f"Fitting scalers, normalizers, or feature selection on split '{split}' is strictly prohibited. "
            f"Statistical parameters must be learned ONLY from the 'train' split."
        )


class PatientLevelPartitioner:
    """
    Partitions medical datasets by patient ID into train, val, test, and optional external-test splits.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        total = round(train_ratio + val_ratio + test_ratio, 5)
        if total != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got train={train_ratio}, val={val_ratio}, test={test_ratio}, sum={total})"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.settings = get_settings()

    def partition_records(
        self,
        records: List[Dict[str, Any]],
        holdout_sites: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Groups image records by patient ID, isolates external holdout sites if requested,
        and stratifies the remaining patients by target class across train, val, and test splits.
        """
        if not records:
            return []

        holdout_set = set(holdout_sites or [])
        rng = random.Random(self.seed)

        # 1. Separate external holdout site patients
        patient_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        patient_to_site: Dict[str, str] = {}
        patient_to_class: Dict[str, str] = {}

        for rec in records:
            pid = rec["patient_id"]
            site = rec.get("site_id", rec.get("source_id", "default_site"))
            cls = rec["target_class"]

            patient_to_records[pid].append(rec)
            patient_to_site[pid] = site
            patient_to_class[pid] = cls

        patient_split_assignment: Dict[str, str] = {}

        # Check external holdout sites
        internal_patients: List[str] = []
        for pid, site in patient_to_site.items():
            if site in holdout_set:
                patient_split_assignment[pid] = "external_test"
            else:
                internal_patients.append(pid)

        # 2. Stratify internal patients by target class
        class_to_patients: Dict[str, List[str]] = defaultdict(list)
        for pid in internal_patients:
            cls = patient_to_class[pid]
            class_to_patients[cls].append(pid)

        # Split patients per class
        for cls, pids in class_to_patients.items():
            # Deterministic shuffle for reproducibility
            sorted_pids = sorted(pids)
            rng.shuffle(sorted_pids)
            n = len(sorted_pids)

            if n == 1:
                # With only 1 patient, place in train
                patient_split_assignment[sorted_pids[0]] = "train"
            elif n == 2:
                patient_split_assignment[sorted_pids[0]] = "train"
                patient_split_assignment[sorted_pids[1]] = "val"
            else:
                n_train = max(1, int(round(n * self.train_ratio)))
                n_val = max(1, int(round(n * self.val_ratio)))
                # Adjust if sum exceeds total or leaves test empty when n >= 3
                if n_train + n_val >= n:
                    n_train = max(1, n - 2)
                    n_val = 1
                n_test = n - (n_train + n_val)
                if n_test <= 0:
                    n_test = 1
                    if n_train > 1:
                        n_train -= 1
                    else:
                        n_val = max(1, n_val - 1)

                train_pids = sorted_pids[:n_train]
                val_pids = sorted_pids[n_train : n_train + n_val]
                test_pids = sorted_pids[n_train + n_val :]

                for p in train_pids:
                    patient_split_assignment[p] = "train"
                for p in val_pids:
                    patient_split_assignment[p] = "val"
                for p in test_pids:
                    patient_split_assignment[p] = "test"

        # 3. Assign split to every record
        partitioned_records: List[Dict[str, Any]] = []
        for rec in records:
            pid = rec["patient_id"]
            split = patient_split_assignment.get(pid, "train")

            updated = dict(rec)
            updated["split"] = split
            # Ensure standard column names
            if "sha256" in updated and "sha256_hash" not in updated:
                updated["sha256_hash"] = updated["sha256"]
            if "dataset_id" not in updated:
                updated["dataset_id"] = updated.get("source_id", "osteoguard_v1")
            if "source_id" not in updated:
                updated["source_id"] = "local"
            if "site_id" not in updated:
                updated["site_id"] = "site_01"
            if "mask_path" not in updated:
                updated["mask_path"] = ""

            partitioned_records.append(updated)

        # 4. Strict leakage assertion
        is_clean, violations = verify_patient_leakage(partitioned_records)
        if not is_clean:
            raise PatientLeakageError(f"Zero patient leakage violated during partitioning: {violations}")

        return partitioned_records

    def export_split_manifest(
        self,
        partitioned_records: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Exports the split manifest to CSV.
        """
        out_file = output_path or (self.settings.base_dir / "ml_service/data/splits/split_manifest.csv")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SPLIT_MANIFEST_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for rec in partitioned_records:
                writer.writerow(rec)

        logger.info(f"Exported split manifest with {len(partitioned_records)} rows to {out_file}")
        return out_file

    def partition_from_audit(
        self,
        audit_result: Dict[str, Any],
        holdout_sites: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], Path]:
        """
        Takes records from DatasetAuditor, partitions them, and writes split_manifest.csv.
        """
        records = audit_result.get("records", [])
        partitioned = self.partition_records(records, holdout_sites=holdout_sites)
        out_path = self.export_split_manifest(partitioned, output_path=output_path)
        return partitioned, out_path

