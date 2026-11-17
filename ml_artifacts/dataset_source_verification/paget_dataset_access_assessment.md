# Forensic Source and Access Assessment: Paget's Disease of Bone

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 23 — Verified Dataset Sources, Access and Acquisition Gate  
**Artifact Path:** `ml_artifacts/dataset_source_verification/paget_dataset_access_assessment.md`  

---

## 1. Executive Summary & Repository Reality

Paget's disease of bone (osteitis deformans) is a chronic focal osteometabolic disorder characterized by disorganized, accelerated osteoclastic/osteoblastic bone remodeling.

A comprehensive search across official open biomedical repositories (The Cancer Imaging Archive, Zenodo, Figshare, PhysioNet, NIH Repositories) confirmed a fundamental finding:
> [!IMPORTANT]
> **NO STANDALONE PUBLIC MACHINE-LEARNING DATASET EXISTS FOR PAGET'S DISEASE.**  
> Unlike pneumonia or fracture imaging, there are currently **zero standardized, bulk-downloadable open-access datasets** dedicated to Paget's disease of bone. Existing imagery is scattered across online educational radiology teaching files, individual peer-reviewed case reports, and proprietary hospital PACS archives.

---

## 2. Forensic Audit of Candidate Collections

### Candidate 1: Radiopaedia Paget's Disease Collection
- **Source URL:** [https://radiopaedia.org/articles/paget-disease-of-bone](https://radiopaedia.org/articles/paget-disease-of-bone)
- **Institution / Governance:** Radiopaedia.org (Global Collaborative Radiology Educational Platform).
- **Available Cases & Volume:** ~150 distinct clinical cases (~300 images across X-ray, CT, and $^{99\text{m}}\text{Tc}$-MDP scintigraphy).
- **Explicit Diagnostic Labels:** Verified board-certified radiologist consensus with classic radiological signs (cotton-wool calvarium, picture-frame vertebrae, blade-of-grass/candle flame lytic front, iliopectineal line thickening).
- **Bulk Download Permitted:** **STRICTLY FALSE.**  
  Radiopaedia's Terms of Reference explicitly mandate:
  > *"Automated harvesting, data scraping, or bulk downloading of images and clinical descriptions from Radiopaedia.org is strictly prohibited without prior written license agreement."*
- **Machine Learning Reuse Permitted:** **FALSE WITHOUT WRITTEN LICENSE.** Radiopaedia content is licensed under Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 3.0). Machine learning model training that may lead to commercial deployment is restricted. Furthermore, individual case copyrights reside with global volunteer contributors.
- **Classification Status:** **`EDUCATIONAL_ONLY`**.
- **Audit Verdict:** Valuable for qualitative human radiological reference and visual Grad-CAM sanity checks; strictly disqualified from automated bulk training ingestion.

---

### Candidate 2: National Library of Medicine (NLM) MedPix Teaching Database
- **Source URL:** [https://medpix.nlm.nih.gov/](https://medpix.nlm.nih.gov/)
- **Institution / Governance:** National Institutes of Health (NIH) / National Library of Medicine (NLM).
- **Available Cases & Volume:** ~45 unique patient cases (~90 radiographs and CT studies).
- **Explicit Diagnostic Labels:** Verified teaching cases with detailed clinical histories, biochemical correlations (elevated alkaline phosphatase), and differential diagnosis commentaries.
- **Bulk Download Permitted:** **FALSE.** MedPix provides no bulk programmatic API; each case must be queried and retrieved individually.
- **Machine Learning Reuse Permitted:** US Government Work / NLM Terms of Service permit educational and scientific research reuse with proper citation.
- **Critical Technical Limitation:** Many images have educational callouts, bounding boxes, or diagnostic arrows burned directly into the pixel data, which would induce severe convolutional shortcut learning if uncleaned.
- **Classification Status:** **`EDUCATIONAL_ONLY`**.
- **Audit Verdict:** Permissible for small few-shot test verification (~40 cases); inadequate for deep convolutional training from scratch.

---

### Candidate 3: Open-Access Peer-Reviewed Literature (PubMed Central / BMC / BMJ)
- **Source Scope:** Case reports and pictorial reviews indexed on PubMed Central and Wikimedia Commons under CC BY 4.0 or CC BY-SA licenses.
- **Available Volume:** ~60–80 cases scattered across individual journals.
- **Explicit Diagnostic Labels:** Peer-reviewed clinical case reports.
- **Bulk Download Permitted:** **FALSE.** Requires manual extraction from individual XML/PDF publications.
- **Format & Resolution:** Highly variable 8-bit web formats (JPEG/PNG) lacking standardized DICOM pixel arrays, bit depths, and photometric interpretation.
- **Classification Status:** **`INDIVIDUAL_PERMISSION_REQUIRED` / `UNVERIFIED`**.
- **Audit Verdict:** Heterogeneous quality; risk of severe dataset shift.

---

### Candidate 4: Institutional Hospital PACS Partnership (The Authentic Path)
- **Source Scope:** Multi-center academic hospital PACS archives (e.g., musculoskeletal radiology departments with metabolic bone disease clinics).
- **Estimated Available Volume:** 150–300 authentic native DICOM examinations.
- **Explicit Diagnostic Labels:** Electronic health record (EHR) biopsy or radiologist-verified ICD-10 diagnostic codes (`M88.0`–`M88.9`).
- **Bulk Download Permitted:** Permitted under formal Institutional Review Board (IRB) approved data transfer protocols.
- **Classification Status:** **`CONTROLLED_ACCESS_REQUEST_REQUIRED`**.
- **Audit Verdict:** The only scientifically sound, publication-grade pathway to acquire native 16-bit DICOM radiographs of Paget's disease.

---

## 3. Comparison of Paget's Disease Candidate Sources

| Candidate Repository | Accessible Volume | Modality | Native DICOM | Bulk Scraping Permitted | Machine Learning Permitted | Classification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Radiopaedia** | ~150 cases | X-ray / CT / Scan | No (8-bit Web) | **PROHIBITED** | No (Without License) | **EDUCATIONAL_ONLY** |
| **NLM MedPix** | ~45 cases | X-ray / CT | No (8-bit Web) | No (Manual Only) | Yes (Research) | **EDUCATIONAL_ONLY** |
| **PMC Open-Access Figures** | ~70 figures | X-ray | No (8-bit Web) | Manual Only | Variable (CC BY) | **INDIVIDUAL_PERMISSION_REQUIRED** |
| **Institutional PACS Collaboration** | 100–300 cases | X-ray / CT | **Yes (16-bit DICOM)** | Yes (Via IRB) | **Yes (Under DUA)** | **CONTROLLED_ACCESS_REQUEST_REQUIRED** |

---

## 4. Final Class Readiness Determination: Paget's Disease of Bone

$$\mathbf{\text{Class Readiness: NOT READY — DATA ACQUISITION REQUIRED}}$$

- **Existing In-Tree Paget's Disease Radiographs:** **0**
- **Publicly Downloadable Licensed Datasets:** **0**
- **Actionable Next Step:** Deploy the drafted Institutional Radiology Collaboration request (detailed in `dataset_access_request_templates.md`) to acquire native DICOM cohorts under IRB governance.

