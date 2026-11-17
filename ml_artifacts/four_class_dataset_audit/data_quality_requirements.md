# Data Quality, Provenance, and Clinical Validation Requirements

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 22 — Four-Class Dataset Readiness and Gap Audit  
**Artifact Path:** `ml_artifacts/four_class_dataset_audit/data_quality_requirements.md`  

---

## 1. Executive Quality Framework

To achieve publication-grade scientific validity, clinical interpretability, and regulatory compliance, any candidate dataset prospective to the final OsteoGuard AI four-class framework must satisfy the stringent data governance and engineering requirements defined below.

---

## 2. Mandatory Data Quality Standards

### 2.1 Verified Diagnostic Ground Truth
1. **Secondary Bone Metastasis:**
   - Ground truth must be established by histopathological biopsy of the primary tumor or metastatic lesion, or corroborated by multi-modality concordant imaging (e.g., $^{99\text{m}}\text{Tc}$-MDP hotspot corresponding to osteolytic/sclerotic focus on CT/MRI) and longitudinal clinical follow-up.
2. **Paget's Disease of Bone:**
   - Must be diagnosed by board-certified radiologists based on pathognomonic architectural remodeling (cortical expansion, coarse trabeculation, cotton-wool appearance) accompanied by clinical and biochemical correlation (serum alkaline phosphatase elevation).
   - Generic sclerotic lesions, osteoarthritis, or fibrous dysplasia must be explicitly ruled out.
3. **Osteogenesis Imperfecta (OI):**
   - Diagnosis must be confirmed by clinical genetic testing (`COL1A1`/`COL1A2` pathogenic variants) or clinical genetics evaluation under the Sillence classification.
   - Simple osteopenia, rickets, or accidental trauma must never be mapped to OI.
4. **Healthy Skeletal Control:**
   - Must be explicitly documented as radiologically normal by board-certified radiologists, demonstrating intact cortical margins, normal bone mineral density, and absence of focal lesions, fractures, or metabolic alterations.
   - Non-fractured emergency trauma radiographs or contralateral cancer scans are strictly disqualified.

---

### 2.2 Patient-Level Identification & Cluster-Aware Partitioning
1. **Cryptographic Patient Identifiers:** Every image must be linked to an immutable, de-identified patient key (e.g., `Patient_ID`).
2. **Zero Intra-Patient Cross-Partition Leakage:**
   - Splitting must strictly employ **Group K-Fold** partitioning.
   - All multi-view, bilateral, or serial temporal radiographs belonging to a single individual must reside exclusively in the training, validation, or test split.
   - Image-level random splitting is strictly prohibited.

---

### 2.3 Image Integrity & Deduplication Protocols
1. **Exact-Duplicate Elimination:** Compute SHA-256 cryptographic hashes for all incoming files; identical bit-level files must be flagged and consolidated.
2. **Perceptual Near-Duplicate Auditing:** Compute perceptual hashes (pHash / dHash with Hamming distance threshold $\le 4$) to detect multi-resolution exports, cropped variants, or compressed duplicates.
3. **Lossless Image Decoding:** All images must be verified for header integrity, correct color space parsing (16-bit grayscale or 8-bit RGB), valid dynamic range, and absence of corrupt pixel byte streams.

---

### 2.4 Explicit Modality and Physical Metadata
Every ingested instance must contain standardized metadata:
- `modality`: `PLANAR_XRAY`, `BONE_SCINTIGRAPHY`, or `VOLUMETRIC_CT`
- `anatomical_region`: `SPINE`, `PELVIS`, `FEMUR`, `TIBIA`, `HAND`, `WRIST`, `TOTAL_BODY`
- `view_projection`: `AP`, `PA`, `LATERAL`, `OBLIQUE`, `ANTERIOR_WB`, `POSTERIOR_WB`
- `acquisition_vendor`: Manufacturer and detector type (CR vs DR) where available in DICOM tags
- `pixel_spacing`: Calibrated spatial resolution (mm/pixel)

---

### 2.5 Multi-Reader Annotation & Radiological Consensus
- Lesion localization masks (bounding boxes or polygonal contours) must be curated by or reviewed by at least two board-certified MSK radiologists, with an independent third radiologist acting as arbiter for discordant delineations (Cohen's Kappa $\kappa \ge 0.80$).

---

### 2.6 Independent Hospital-Level External Validation
- To satisfy cross-center generalization mandates (as proven essential by Stages 19–21), final models must be evaluated on at least **two fully independent hospital cohorts** that were never seen during training, validation, or hyperparameter selection.

---

## 3. Compliance Audit of Existing In-Tree Datasets

| Dataset | Diagnostic Ground Truth | Patient IDs Available | Group K-Fold Feasible | Modality Metadata | External Hospital Diversity | Compliance Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **FracAtlas** ($N=4,083$) | Verified for fracture; negative class is confounded by trauma | **FAIL** (No patient IDs) | **FAIL** (Image-level only) | Incomplete (No DICOM tags) | Single hospital (Bangladesh) | **NON-COMPLIANT** for 4-class differential model |
| **Bone Scintigraphy** ($N=582$) | Verified for breast cancer metastasis | **PASS** (Patient 1–291) | **PASS** (Patient-level 5-fold) | Partial (Paired A/P views) | Single hospital (Paraguay) | **CONDITIONALLY COMPLIANT** (Scintigraphy pathway only) |
| **BTXRD** ($N=3,746$) | Verified for primary bone tumours | **FAIL** (No patient IDs) | **FAIL** (Cluster risk) | Incomplete (8-bit JPEG/PNG) | Multi-Center (Centers 2 & 3 burned) | **NON-COMPLIANT** (Lacks patient IDs; external centers burned) |

---

## 4. Summary of In-Tree Deficiencies

1. **FracAtlas:** Cannot provide valid healthy controls; lacks patient IDs.
2. **Bone Scintigraphy:** Cannot provide healthy controls; restricted to nuclear medicine.
3. **BTXRD:** Cannot represent secondary metastases; lacks patient IDs; Center 2 and Center 3 are burned holdouts.
4. **Paget's & OI:** Zero data in-tree.

Therefore, the existing in-tree datasets **cannot** satisfy the clinical data quality and provenance standards required for developing the final four-class OsteoGuard AI model.

