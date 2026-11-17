# Multimodal Uncertainty Architecture & Missing Modality Handling

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Date**: November 16, 2026  
**Status**: **ARCHITECTURAL BLUEPRINT**

---

## 1. Pathway Uncertainty Computation

```mermaid
flowchart TD
    subgraph Pathways ["Modality Pathways"]
        X["Radiography Stream
(ResNet-101)"]
        S["Scintigraphy Stream
(ResNet-34)"]
        CT["CT Stream
(Volumetric CT)"]
        RAD["Radiomics Stream
(102-D IBSI)"]
    end

    subgraph LateFusion ["Multimodal Late Fusion (614-D)"]
        X & S & CT & RAD --> FUSE["Late Fusion Classifier
Softmax Probabilities p_fusion"]
    end

    subgraph UncertaintyGate ["Uncertainty & Abstention Gate"]
        FUSE --> METRICS["Compute H_norm(p_fusion)
Compute OOD Flag"]
        METRICS --> TRIAGEH_norm > 0.60 OR
OOD Flag == True?
        TRIAGE -->|Yes| ABSTAIN["ABSTAIN
Amber Review Banner"]
        TRIAGE -->|No| ACCEPT["ACCEPT
High-Confidence Output"]
    end
```

---

## 2. Missing Modality Invariant Rules

> [!IMPORTANT]
> **Rule 1: Missing Modality NEVER Implies Zero Uncertainty**:
> When a patient scan lacks one or more modalities (e.g. X-ray present, but CT and Bone Scan absent), the system must NOT set epistemic uncertainty to zero.
> 
> **Rule 2: Absence of Corroboration Elevates Epistemic Uncertainty**:
> In unimodal inference, absence of cross-modal verification increases the probability of out-of-distribution artifacts and must be reflected in wider predictive confidence intervals.
> 
> **Rule 3: No Invented Combination Formulas**:
> Unimodal uncertainty values must not be merged using ad-hoc weighted sums. Uncertainty must be computed from the coherent output distribution of the multimodal fusion network or marginal ensemble.
