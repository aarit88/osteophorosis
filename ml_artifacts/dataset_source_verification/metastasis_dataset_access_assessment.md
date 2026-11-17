# Forensic Source and Access Assessment: Secondary Bone Metastasis

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 23 — Verified Dataset Sources, Access and Acquisition Gate  
**Artifact Path:** `ml_artifacts/dataset_source_verification/metastasis_dataset_access_assessment.md`  

---

## 1. Executive Summary & Critical Forensic Discoveries

Stage 23 re-audited the candidate data sources for Target Class 1 (**Secondary Bone Metastasis**), resulting in two major forensic discoveries:

1. **Zenodo DOI `10.5281/zenodo.13900966` is an Identical Duplicate of the In-Tree Dataset:**
   - Forensic analysis confirmed that Zenodo record `13900966` is titled *"Bone scan images dataset for study of bone metastases in adult breast cancer patients at IICS-UNA Paraguay"*.
   - This is **the exact same dataset** already acquired, partitioned, and evaluated in `ml_service/data/raw/bone_scan_metastasis` (Stages 11–14).
   - Identifying this duplicate eliminates redundant download overhead and prevents false accounting of candidate volume.
2. **TCIA Spine-Mets-CT-SEG Cohort Size Verified at 55 Subjects:**
   - While Stage 22 literature review tentatively cited up to 238 cases, official TCIA indexing confirms the cohort consists of **55 unique patients** (~20.36 GB of CT DICOM series).
   - The dataset represents volumetric CT, making it suitable exclusively for 3D CT pathways, not 2D plain X-ray classifiers.

---

## 2. In-Depth Candidate Source Audits

### Candidate 1: IICS-UNA Paraguay Bone Scintigraphy Dataset (In-Tree)
- **Zenodo DOI:** [10.5281/zenodo.13900966](https://doi.org/10.5281/zenodo.13900966)
- **Local In-Tree Directory:** `ml_service/data/raw/bone_scan_metastasis/`
- **Institution:** Instituto de Investigaciones en Ciencias de la Salud, Universidad Nacional de Asunción (IICS-UNA), Paraguay.
- **Species:** Human (*Homo sapiens*), adult female breast cancer staging cohort ($N=289$ female, $N=2$ male).
- **Imaging Modality:** Whole-Body Planar Bone Scintigraphy ($^{99\text{m}}\text{Tc}$-MDP gamma emission camera).
- **Cohort Volume:** 291 unique patients; exactly 582 images (paired Anterior `A*.png` and Posterior `P*.png` scans).
- **Metastasis Ground Truth:**
  - Positive for Bone Metastases: 60 patients (120 images).
  - Negative for Bone Metastases: 231 patients (462 images).
- **Patient Identifiers:** Explicitly mapped (`Patient 1` to `Patient 291`).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Classification Status:** **`PUBLIC_DOWNLOAD_VERIFIED` (ALREADY ACQUIRED IN-TREE)**.
- **Clinical Governance Limitation:** Represents secondary metastasis in nuclear medicine scintigraphy exclusively from breast carcinoma primaries; does not provide plain X-ray radiographs.

---

### Candidate 2: TCIA Spine-Mets-CT-SEG (The Cancer Imaging Archive)
- **Official DOI:** [10.7937/kh36-ds04](https://doi.org/10.7937/kh36-ds04)
- **Collection Name:** *Spine metastatic bone cancer: pre and post radiotherapy CT*
- **Institution / Host:** The Cancer Imaging Archive (TCIA) / National Cancer Institute (NCI).
- **Species:** Human (*Homo sapiens*).
- **Imaging Modality:** Computed Tomography (CT), axial slice series.
- **Verified Cohort Volume:** 55 unique patients; 108 CT examination series (~20.36 GB uncompressed).
- **Ground Truth & Annotations:** Pathologist-confirmed primary carcinomas (breast, lung, prostate, renal); radiologist contour segmentations of vertebrae bodies and metastatic lesions (lytic, blastic, and mixed) in DICOM-SEG format.
- **Patient Metadata:** Fully anonymized TCIA subject keys (`Spine-Mets-001` through `Spine-Mets-055`), patient age, sex, primary cancer histology.
- **Licensing & Citation:** CC BY 3.0 / 4.0 with TCIA Data Use Agreement. Mandatory citation of TCIA collection.
- **Download Mechanism:** Requires TCIA NBIA Data Retriever desktop application and `.tcia` manifest file.
- **Classification Status:** **`CONTROLLED_ACCESS_REQUEST_REQUIRED`**.
- **Audit Verdict:** Suitable exclusively for a 3D volumetric CT radiomic pathway; cannot be interchanged with 2D plain X-ray classifiers.

---

### Candidate 3: BTXRD Class 2 (Primary Malignant Bone Tumours — In-Tree)
- **Local Directory:** `ml_service/data/raw/btxrd/`
- **Cohort Size:** 342 radiographs across 3 Chinese medical centers.
- **Diagnostic Entities:** Osteosarcoma ($N=297$) and other primary bone malignancies ($N=45$).
- **Classification Status:** **`UNSUITABLE_FOR_TARGET_CLASS`**.
- **Governance Mandate:**
  > [!CAUTION]
  > **STRICT PATHOLOGY BOUNDARY:**  
  > Primary sarcomas (osteosarcoma, chondrosarcoma, Ewing sarcoma) arise from native mesenchymal bone cells, present predominantly in children and young adults, and exhibit aggressive local cortical destruction and periosteal reaction. Secondary metastases arise from epithelial carcinomas (breast, prostate, lung, kidney) metastasizing hematogenously in older adults. Equating BTXRD primary sarcomas with secondary carcinoma metastases is medically and scientifically invalid.

---

## 3. Modality and Target Compatibility Summary

| Source Candidate | Primary Modality | Target Pathology | Verified Patient Count | Verified Image Count | Suitable for Plain X-Ray Model? | Suitable for Multimodal Framework? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **IICS-UNA Paraguay** | Planar Scintigraphy ($^{99\text{m}}\text{Tc}$-MDP) | Breast Cancer Bone Mets | 60 positive (291 total) | 120 pos (582 total) | **NO** (Modality mismatch) | **YES** (Nuclear medicine stream) |
| **TCIA Spine-Mets** | Volumetric CT | Spinal Carcinoma Mets | 55 patients | 108 series (~20 GB) | **NO** (Volumetric 3D CT) | **YES** (CT radiomic stream) |
| **BTXRD Class 2** | Plain Radiography (X-ray) | Primary Sarcomas | Unknown | 342 images | **NO** (Pathology mismatch) | **NO** (Strictly excluded) |

---

## 4. Final Class Readiness Determination: Secondary Bone Metastasis

$$\mathbf{\text{Class Readiness: CONDITIONALLY READY (MODALITY-RESTRICTED)}}$$

- **Nuclear Medicine Scintigraphy Stream:** **READY** (60 metastasis-positive patients in-tree with verified 5-fold CV checkpoints from Stages 11–14).
- **Plain Radiography Stream:** **NOT READY** (Zero plain X-ray radiographs of secondary metastases currently in-tree).
- **Volumetric CT Stream:** **CONDITIONALLY VIABLE** (TCIA Spine-Mets verified at 55 subjects; requires DUA registration).

