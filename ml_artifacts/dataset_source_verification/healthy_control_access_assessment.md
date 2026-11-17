# Forensic Source and Access Assessment: Healthy Skeletal Controls

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 23 — Verified Dataset Sources, Access and Acquisition Gate  
**Artifact Path:** `ml_artifacts/dataset_source_verification/healthy_control_access_assessment.md`  

---

## 1. Executive Summary & Clinical Control Definition

In clinical artificial intelligence, a **Healthy Skeletal Control** requires explicit radiological verification demonstrating:
1. Normal anatomical alignment and bone morphology.
2. Normal bone mineral density (absence of osteopenia, osteoporosis, or coarse trabecular expansion).
3. Complete absence of acute fractures, healing calluses, osteolytic/osteoblastic lesions, subchondral sclerosis, joint effusions, or hardware.

> [!WARNING]
> **REJECTION OF CONVENIENCE CONTROLS:**  
> Neither emergency trauma radiographs with negative fracture findings (FracAtlas non-fractured) nor active cancer staging scans (bone scintigraphy negative) satisfy this clinical standard.

---

## 2. Forensic Audit of Stanford AIMI MURA

- **Source URL:** [https://aimi.stanford.edu/datasets/mura-msk-xrays](https://aimi.stanford.edu/datasets/mura-msk-xrays)
- **Foundational Paper:** Rajpurkar et al., *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs*, arXiv:1712.06957.
- **Institution:** Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI), Stanford University Medical Center.
- **Modality:** Conventional Plain Radiography (X-ray, multi-view studies).
- **Verified Cohort Volume:**
  - Total Patients: 12,173 patients.
  - Total Studies: 14,863 musculoskeletal studies (40,561 multi-view radiographs).
  - **Normal Cohort:** Exactly **9,045 normal studies** (~20,000 normal radiographs).
  - Abnormal Cohort: 5,818 abnormal studies.
- **Clinical Meaning of the "Normal" Label:**
  - Each study was evaluated by board-certified Stanford musculoskeletal radiologists.
  - Normal studies demonstrated absolute absence of fracture, dislocation, osteolysis, osteosclerosis, surgical hardware, or high-grade degenerative joint changes.
- **Patient Identifiers:** Explicit patient keys (e.g., `patient00001` through `patient12173`), enabling cluster-aware **Group K-Fold** splitting with zero intra-patient leakage.
- **Access Procedure:**
  1. Access the Stanford AIMI dataset repository portal.
  2. Register with verified institutional credentials.
  3. Electronically accept the Stanford Research Data Use Agreement (DUA).
- **Licensing Restrictions:** Stanford University Research DUA permits **Non-Commercial Academic Research Only**. Commercial deployment requires formal university licensing.
- **Classification Status:** **`CONTROLLED_ACCESS_REQUEST_REQUIRED`**.

### Critical Clinical Boundary of MURA:
> [!CAUTION]
> **ANATOMICAL SCOPE RESTRICTION:**  
> MURA encompasses the **upper extremity exclusively**:
> - Elbow (`XR_ELBOW`)
> - Finger (`XR_FINGER`)
> - Forearm (`XR_FOREARM`)
> - Hand (`XR_HAND`)
> - Humerus (`XR_HUMERUS`)
> - Shoulder (`XR_SHOULDER`)
> - Wrist (`XR_WRIST`)
> 
> MURA does **NOT** contain radiographs of the pelvis, spine, femur, or tibia. Therefore, MURA radiographs can serve as a **verified upper-extremity normal comparator**, but must **NEVER** be described or deployed as general whole-skeleton healthy controls.

---

## 3. Re-Audit of Existing In-Tree Candidates

| Candidate Cohort | In-Tree Path | Raw Image Count | Patient IDs | Clinical Setting | Compliance Verdict |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **FracAtlas Non-Fractured** | `ml_service/data/raw/fracatlas` | 3,364 | None | Emergency department acute trauma triage | **REJECTED (UNSUITABLE)** |
| **Bone Scintigraphy Negative** | `ml_service/data/raw/bone_scan_metastasis` | 462 (231 pts) | Yes | Adult female breast cancer staging | **REJECTED (UNSUITABLE)** |
| **BTXRD Center 1 Normal** | `ml_service/data/raw/btxrd` | 1,600 clean | None | Chinese orthopedic hospital non-tumor intake | **CONDITIONALLY USABLE (UNVERIFIED)** |

### Detailed Defect Analysis of In-Tree Candidates:
1. **FracAtlas Non-Fractured Class:** Patients presented with acute skeletal trauma, pain, or fall. The absence of a discrete fracture line does not establish normal bone mineral density or absence of joint pathology.
2. **Bone Scintigraphy Negative Class:** Staging oncology patients undergo systemic anti-estrogen or cytotoxic therapies that alter bone mineral metabolism. Furthermore, functional gamma scintigraphy cannot serve as a healthy control for plain radiography.
3. **BTXRD Center 1 Normal Class:** While non-tumorous, the absence of patient IDs introduces clustering risks, and the lack of clinical history prevents ruling out underlying osteopenia or osteoarthritis.

---

## 4. Final Class Readiness Determination: Healthy Skeletal Controls

$$\mathbf{\text{Class Readiness: CONDITIONALLY READY / RESTRICTED}}$$

- **Verified In-Tree Radiographs:** 1,600 clean BTXRD Center 1 images (conditionally usable under explicit documentation of missing patient IDs).
- **Top External Candidate:** Stanford AIMI MURA (9,045 normal studies; verified suitable for upper-extremity comparator).
- **Actionable Next Step:** Submit electronic application to Stanford AIMI for MURA access under academic non-commercial terms.

