# Conventional Mirels Scoring Specification & Clinical Boundary Protocol

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI: Biomechanical Fracture Risk Engine  
**Standard**: Classical Mirels Criteria (1989)  
**Date**: November 16, 2026  
**Status**: **VERIFIED & SCOPED TO METASTATIC LONG BONES**

---

## 1. Clinical Background & Intended Scope

The Mirels scoring system was published by Dr. Hilmert Mirels in 1989 (*Clinical Orthopaedics and Related Research*, 249:256-264) to provide orthopedic surgeons with an objective numerical framework for predicting impending pathological fractures in patients with **metastatic malignant disease involving the long bones**.

### Mandatory Clinical Scope Boundary
> [!IMPORTANT]
> **Metastatic Long-Bone Disease ONLY**:
> Mirels scoring is clinically and mathematically validated **solely for long-bone metastases** (e.g., femur, humerus, tibia).
> It is **NOT VALID** and **MUST NOT BE APPLIED** to:
> 1. **Paget's Disease of Bone**: Characterized by disordered osteoclastic/osteoblastic hyper-remodeling, bone expansion, cortical coarsening, and transverse "chalk-stick" microfractures.
> 2. **Osteogenesis Imperfecta (OI)**: A hereditary Type I collagen disorder characterized by diffuse bone fragility, recurrent childhood fractures, gracile bones, and bowing deformities.
> 3. **Healthy Skeletal Controls**: Lack osteolytic destruction; baseline score would be clinically meaningless.

---

## 2. Four-Factor Scoring Architecture

The system evaluates four clinical and radiographic variables, scoring each from 1 to 3:

| Variable | 1 Point | 2 Points | 3 Points |
| :--- | :--- | :--- | :--- |
| **1. Anatomical Site** | Upper extremity (humerus, radius, ulna) | Lower extremity (femur diaphysis, tibia, fibula) | Peritrochanteric region of femur |
| **2. Pain Severity** | Mild (non-limiting ache) | Moderate (interferes with activity) | Functional (pain exacerbated by weight-bearing) |
| **3. Lesion Nature** | Osteoblastic (sclerotic / dense) | Mixed (lytic and blastic components) | Osteolytic (cortical bone resorption) |
| **4. Lesion Size** | Less than 1/3 cortical transverse diameter | 1/3 to 2/3 cortical transverse diameter | Greater than 2/3 cortical transverse diameter |

### Cumulative Score Range
- **Minimum Score**: 1 + 1 + 1 + 1 = 4
- **Maximum Score**: 3 + 3 + 3 + 3 = 12
- **Total Input Combinations**: 3^4 = 81 discrete configurations.

---

## 3. Historical Clinical Risk Mapping & Literature Guidelines

Based on the original 1989 cohort of 78 irradiated lesions:

| Total Score | Literature Fracture Rate | Clinical Recommendation Tier | Orthopedic Action Guidance |
| :---: | :---: | :--- | :--- |
| **4 - 7** | 0.0% - 4.0% | **LOW_RISK** | Non-operative surveillance; palliative radiation therapy without surgical stabilization. |
| **8** | approx 15.0% | **INTERMEDIATE_RISK** | Borderline zone; clinical judgment based on functional demands and expected survival. |
| **9 - 12** | 33.3% - 100.0% | **HIGH_RISK** | Prophylactic internal fixation strongly recommended prior to radiation therapy. |

---

## 4. Combinatorial Distribution Analysis

Across all 81 possible input states:
- **Low Risk (<= 7)**: 31 combinations (38.3%)
- **Intermediate Risk (= 8)**: 19 combinations (23.5%)
- **High Risk (>= 9)**: 31 combinations (38.3%)

Distribution of total scores across the 81 configurations:
```
Score  4: 1 combinations
Score  5: 4 combinations
Score  6: 10 combinations
Score  7: 16 combinations
Score  8: 19 combinations
Score  9: 16 combinations
Score 10: 10 combinations
Score 11: 4 combinations
Score 12: 1 combinations
```

---

## 5. Software Implementation Seam

Implemented in `osteoguard.biomechanics.mirels`:
- `ConventionalMirelsScorer.calculate_score(...)`
- Accepts canonical integer ratings `[1, 2, 3]` or case-insensitive string synonyms.
- Enforces strict validation and raises `ValueError` on unrecognized anatomical sites, pain descriptions, or lesion morphologies.
- Explicitly warns when called outside the metastatic disease context.
