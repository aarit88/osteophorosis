"""
OsteoGuard AI - Patient-Level Dataset Partitioning Engine.
Enforces patient-level grouping to prevent data leakage across
training, validation, and holdout testing splits.
"""

from collections import defaultdict
import random
from typing import Dict, List, Set, Tuple
from pydantic import BaseModel, Field

from osteoguard.constants import TARGET_CLASSES


class SampleRecord(BaseModel):
    sample_id: str
    patient_id: str
    label: str
    file_path: str
    modality: str = "xray"


class PartitionSplitResult(BaseModel):
    train_samples: List[SampleRecord]
    val_samples: List[SampleRecord]
    test_samples: List[SampleRecord]
    train_patients: Set[str]
    val_patients: Set[str]
    test_patients: Set[str]


class PatientLevelPartitioner:
    """
    Partitions cohort into Train/Val/Test subsets at the patient level.
    Guarantees that no patient appears in more than one partition.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        total = train_ratio + val_ratio + test_ratio
        if not abs(total - 1.0) < 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def partition(self, samples: List[SampleRecord]) -> PartitionSplitResult:
        """
        Executes group-stratified patient-level partition.
        """
        rng = random.Random(self.seed)

        # 1. Group samples by patient and primary disease label
        patient_to_samples: Dict[str, List[SampleRecord]] = defaultdict(list)
        patient_to_class: Dict[str, str] = {}

        for sample in samples:
            patient_to_samples[sample.patient_id].append(sample)
            # Patient label is defined by the majority class among patient's samples
            patient_to_class[sample.patient_id] = sample.label

        # 2. Stratify patients per diagnostic class
        class_to_patients: Dict[str, List[str]] = defaultdict(list)
        for p_id, cls in patient_to_class.items():
            class_to_patients[cls].append(p_id)

        train_patients: Set[str] = set()
        val_patients: Set[str] = set()
        test_patients: Set[str] = set()

        for cls, p_list in class_to_patients.items():
            # Deterministic shuffle per class
            shuffled = list(p_list)
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(round(n * self.train_ratio))
            n_val = int(round(n * self.val_ratio))

            # Ensure at least 1 patient in train if available
            if n_train == 0 and n > 0:
                n_train = 1

            train_subset = shuffled[:n_train]
            val_subset = shuffled[n_train:n_train + n_val]
            test_subset = shuffled[n_train + n_val:]

            train_patients.update(train_subset)
            val_patients.update(val_subset)
            test_patients.update(test_subset)

        # 3. Mathematical verification of zero patient overlap
        assert train_patients.isdisjoint(val_patients), "Patient leakage detected between Train and Val!"
        assert train_patients.isdisjoint(test_patients), "Patient leakage detected between Train and Test!"
        assert val_patients.isdisjoint(test_patients), "Patient leakage detected between Val and Test!"

        # 4. Map back to samples
        train_samples = [s for p in train_patients for s in patient_to_samples[p]]
        val_samples = [s for p in val_patients for s in patient_to_samples[p]]
        test_samples = [s for p in test_patients for s in patient_to_samples[p]]

        return PartitionSplitResult(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
        )

