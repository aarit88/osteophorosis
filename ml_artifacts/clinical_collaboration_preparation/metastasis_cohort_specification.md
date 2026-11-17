# Clinical Cohort Acquisition Specification: Secondary Bone Metastasis

**Target Class 1**: Secondary Bone Metastasis (Metastatic Bone Disease)  
**Document Status**: Formal Cohort Acquisition Specification  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Clinical Definition and Oncological Context

Secondary bone metastasis represents the hematogenous or lymphatic dissemination of malignant epithelial cells from an extra-skeletal primary solid tumor to bone. The bone microenvironment provides a fertile niche ("seed and soil" hypothesis) wherein tumor-derived factors stimulate osteoclastogenesis or aberrant osteoblast activity, producing:
1. **Osteolytic Metastases**: Severe focal bone destruction mediated by tumor-induced osteoclasts (frequent in lung carcinoma, renal cell carcinoma, thyroid carcinoma, and melanoma).
2. **Osteoblastic / Sclerotic Metastases**: Disorganized, dense new bone deposition stimulated by tumor paracrine signaling (characteristic of prostate adenocarcinoma and carcinoid tumors).
3. **Mixed Metastases**: Simultaneous areas of lysis and sclerosis (common in breast adenocarcinoma and gastrointestinal malignancies).

Bone metastases frequently cause skeletal-related events (SREs), including intractable bone pain, hypercalcemia of malignancy, spinal cord compression, and catastrophic pathological fractures.

---

## 2. Decoupled Modality Pathways and Existing Assets

Metastatic bone disease presents distinctly across diagnostic imaging modalities. OsteoGuard AI enforces a decoupled multi-pathway architecture:

```
+--------------------------------------------------------------------------------------------------+
|                                  METASTASIS MODALITY PATHWAYS                                    |
+--------------------------+--------------------------------------+--------------------------------+
| Modality Pathway         | Clinical Diagnostic Role             | Project Asset Status           |
+--------------------------+--------------------------------------+--------------------------------+
| Pathway A: Radiography   | - Focal lytic / sclerotic lesions    | To be acquired via hospital    |
| (Plain Digital X-Ray)    | - Cortical breach & impending        | collaboration (150-250 pts)    |
|                          |   pathological fracture assessment   |                                |
+--------------------------+--------------------------------------+--------------------------------+
| Pathway B: Scintigraphy  | - Whole-body osteoblastic metabolic  | ACQUIRED & VERIFIED IN-TREE:   |
| (Planar 99mTc-MDP)       |   survey                             | IICS-UNA breast cancer cohort  |
|                          | - High sensitivity for skeletal-wide | (291 patients, 582 images)     |
|                          |   metastatic distribution            | Evaluated in Stages 11-14      |
+--------------------------+--------------------------------------+--------------------------------+
| Pathway C: Volumetric CT | - 3D axial and appendicular osteo-   | CANDIDATE IDENTIFIED:          |
| (Helical CT Bone Window) |   lysis and vertebral stability      | TCIA Spine-Mets-CT-SEG         |
|                          | - Quantitative volumetric radiomics  | (55 patients, 108 CT series)   |
+--------------------------+--------------------------------------+--------------------------------+
```

---

## 3. Diagnostic Reference Standards

Candidate metastasis cases must be authenticated by one of the following diagnostic reference standards:

### Standard A: Histopathological / Biopsy Confirmation (Gold Standard)
- Core needle biopsy or surgical excision of the skeletal lesion confirming metastatic carcinoma consistent with the known primary malignancy.

### Standard B: Multimodal Imaging Concordance + Documented Clinical Course
- Unequivocal focal skeletal lesions identified on $\ge 2$ complementary imaging modalities (e.g., $^{18}\text{F}$-FDG PET-CT, whole-body MRI, contrast CT, and $^{99\text{m}}\text{Tc}$-MDP bone scintigraphy).
- Corroborated by progressive skeletal lesion enlargement or response to systemic endocrine/chemotherapy or targeted radiation therapy on serial follow-up.

---

## 4. Fundamental Distinction from Primary Bone Sarcomas

> [!IMPORTANT]
> **STRICT REJECTION OF PRIMARY SARCOMA CONFLATION**:  
> In Stages 15–21, OsteoGuard AI extensively evaluated primary bone tumours from the BTXRD archive (Class 0: Normal, Class 1: Benign, Class 2: Primary Malignant).  
> - **Biological Distinction**: Primary bone sarcomas (osteosarcoma, chondrosarcoma, Ewing sarcoma) originate directly from mesenchymal bone cells, occur primarily in children and young adults (peak age 10–25 for osteosarcoma/Ewing), and exhibit distinct cytogenetic drivers (e.g., *EWSR1-FLI1* translocation).  
> - **Radiological Distinction**: Primary sarcomas present with aggressive aggressive periosteal reactions (Codman's triangle, "sunburst" spiculation, "onion-skin" lamellation) and soft tissue extension.  
> - **Epidemiological Distinction**: Secondary bone metastases are over 100 times more common than primary bone sarcomas and occur predominantly in patients aged $\ge 45$ years.  
> - **Governance Rule**: Primary bone sarcomas **MUST NEVER BE EQUATED** with secondary bone metastases. The BTXRD Class 2 cohort is strictly segregated and shall never be injected into the secondary metastasis class.

---

## 5. View Pairing and Patient-Level Grouping Rules

For planar bone scintigraphy (Pathway B):
- **Mandatory Paired Views**: Both Anterior and Posterior projections must be acquired under identical detector geometry and gamma camera calibration.
- **Atomic Subject Grouping**: Both views are bound to a single immutable Subject ID. Under no circumstances may an anterior scan reside in the training split while its posterior scan resides in the validation or test split.
