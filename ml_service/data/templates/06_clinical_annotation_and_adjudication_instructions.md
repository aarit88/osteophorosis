# Clinical Annotation & Diagnostic Adjudication Instructions (Template 06)

> **STATUS**: `DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`  
> **Enforced Ontology**: `OSTEO_ONTOLOGY_V1.0_STAGE34` (`ml_service/datasets/clinical_ontology.py`)

---

## 1. Authorized Adjudicators
Only board-certified radiologists, orthopedic specialists, or designated clinical adjudicators assigned a pseudonymous ID (`RAD_*`, `CLIN_*`, `ADJ_*`, `REV_*`) may enter `ACCEPTED` clinical decisions.

## 2. Canonical Classes & Strict Non-Equivalence Boundaries
1. **`SECONDARY_BONE_METASTASIS` (`metastasis`)**
   - Separate `HISTOPATHOLOGY_CONFIRMED` (`BIOPSY_HISTOPATHOLOGY`) from `RADIOLOGY_AND_CLINICAL_CONCORDANCE` (`PET_CT_AND_RADIOLOGY`).
   - **Never** map primary bone sarcomas (osteosarcoma, Ewing sarcoma, chondrosarcoma, BTXRD Class 2) or multiple myeloma -> assign `OUT_OF_SCOPE`.
2. **`PAGETS_DISEASE` (`paget`)**
   - Confirm radiographic triad (cortical thickening, coarsened trabeculae, osseous expansion) + serum ALP/BSAP or biopsy corroboration.
   - **Never** map osteoarthritis, subchondral sclerosis, fibrous dysplasia, or osteopetrosis -> assign `OUT_OF_SCOPE`.
3. **`OSTEOGENESIS_IMPERFECTA` (`oi`)**
   - Keep `GENETIC_COL1A1_COL1A2` confirmation distinct from `SPECIALIST_CLINICAL_DIAGNOSIS` (Sillence Type I/III/IV); note pediatric growth-plate status.
   - **Never** map osteoporosis, osteopenia, rickets, or osteomalacia -> assign `OUT_OF_SCOPE`. Reject any non-human/animal micro-CT.
4. **`HEALTHY_SKELETAL_CONTROL` (`healthy`)**
   - Requires explicit `RADIOLOGIST_NORMAL_CONFIRMATION`, specific `anatomical_region`, explicit `control_selection_criteria`, and `patient_skeletal_history = none_documented / normal_skeletal_history`.
   - **Never** label contralateral/fellow trauma limbs or "absence of fracture" as healthy -> assign `REJECTED`.

## 3. Uncertain & Conflicting Cases
- Equivocal, borderline, or rule-out findings -> `UNCERTAIN` (never coerced into a target class; excluded from training splits).
- Conflicting patient diagnoses across records -> `REQUIRES_ADJUDICATION` until resolved with `source_report_ref` and clinical rationale.
