# Forensic Source and Access Assessment: Osteogenesis Imperfecta (OI)

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 23 — Verified Dataset Sources, Access and Acquisition Gate  
**Artifact Path:** `ml_artifacts/dataset_source_verification/oi_dataset_access_assessment.md`  

---

## 1. Executive Summary & Critical Re-Audit Findings

In Stage 22, two external candidates were initially cataloged for Target Class 3 (**Osteogenesis Imperfecta**):
1. A published clinical study indexed in PubMed Central (`PMC11324408`).
2. A Figshare collection referenced by DOI (`10.6084/m9.figshare.c.3795019`).

In Stage 23, rigorous forensic inspection of the underlying digital repositories revealed two critical discoveries:
1. **The Figshare Collection is Murine Animal Micro-CT, NOT Human Clinical Radiography:** The Figshare data represents micro-computed tomography of mouse tibiae (*oim* model), making it completely unsuitable for human clinical radiograph models.
2. **The PMC11324408 Radiographs are NOT Publicly Downloadable:** While the study describes human clinical data from the Brittle Bone Disorders Consortium, the raw radiographic image series are restricted under consortium data governance and require a formal controlled-access application.

---

## 2. In-Depth Candidate Audits

### Candidate 1: PMC11324408 / Brittle Bone Disorders Consortium (BBDC)
- **Publication Title:** *Cross-sectional and longitudinal analysis of bone age maturation during peri-pubertal growth in children with type I, III and IV osteogenesis imperfecta*
- **Journal Citation:** *Bone*, Vol. 185, 2024, Article 117192. DOI: [10.1016/j.bone.2024.117192](https://doi.org/10.1016/j.bone.2024.117192). PMCID: `PMC11324408`.
- **Primary Institution / Consortium:** Brittle Bone Disorders Consortium (BBDC) of the Rare Diseases Clinical Research Network (RDCRN), funded by NIH/NCATS/NIDCR.
- **Species:** Human (*Homo sapiens*), pediatric cohort (Types I, III, and IV).
- **Imaging Modality:** Conventional Planar Radiography (Left hand/wrist bone age radiographs and appendicular skeletal surveys).
- **Participant & Image Count:** 38–42 longitudinal pediatric participants (~85–120 total radiographic examinations).
- **Public Download Availability:** **FALSE (NO DIRECT DOWNLOAD).** PubMed Central hosts the open-access manuscript text and supplementary Word document (`NIHMS2008512-supplement-1.docx`), but does **NOT** provide raw full-resolution DICOM or lossless image files.
- **Data Availability Statement:** Explicitly directs investigators to the RDCRN Data Management and Coordinating Center (DMCC) data-sharing mechanism.
- **Access Tier:** **`CONTROLLED_ACCESS_REQUEST_REQUIRED`**.
- **Access Procedure:**
  1. Prepare a formal scientific data request detailing research scope, institutional affiliation, and computing security protocols.
  2. Obtain Institutional Review Board (IRB) exemption/approval for secondary analysis of de-identified pediatric data.
  3. Execute the standard RDCRN Data Use Agreement (DUA).
- **Suitability Verdict:** **CONDITIONALLY VIABLE PENDING CONSORTIUM APPROVAL**.

---

### Candidate 2: Figshare DOI `10.6084/m9.figshare.c.3795019` (Ranzoni et al.)
- **Dataset Title:** *Micro-computed tomography reconstructions of tibiae of stem cell transplanted osteogenesis imperfecta mice*
- **Authors:** Anna Maria Ranzoni, Michelangelo Corcelli, Tim Arnett, Pascale V. Guillot
- **Source Journal:** *Scientific Data* (Nature Portfolio), 2017. Associated Figshare Collection: `3795019`.
- **Species:** **ANIMAL (Murine — Mus musculus, B6C3Fe a/a-Col1a2oim/oim mouse model).**
- **Imaging Modality:** **Micro-Computed Tomography (Micro-CT) and 3D surface meshes (`.ply` format and 8-bit cross-sectional bitmap slices).**
- **Cohort Count:** 18 mice (wild-type, non-transplanted *oim*, and stem cell-transplanted *oim* mice).
- **Public Download Availability:** **TRUE** (Files are freely downloadable via Figshare REST API).
- **Classification Status:** **`ANIMAL_OR_SYNTHETIC_ONLY`**.
- **Critical Re-Audit Conclusion:**
  > [!CAUTION]
  > **STRICT REJECTION FOR CLINICAL DIAGNOSTIC AI:**  
  > In Stage 22, this collection was preliminarily noted under general OI keywords. Stage 23 forensic verification confirms this is **micro-CT of murine rodent bones**. Rodent tibiae scanned at micro-meter resolution cannot be mixed into a human clinical diagnostic model for plain X-rays. It is **UNSUITABLE** and formally excluded from the human clinical pipeline.

---

## 3. Alternative Clinical Sources for Human Osteogenesis Imperfecta

Because no standalone open-access human OI radiograph dataset exists on open download portals, the following viable clinical acquisition pathways have been identified:

| Alternative Source | Source Type | Clinical Entity | Modality | Governance Status | Actionable Step |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BBDC / RDCRN Consortium** | Multi-Center NIH Rare Disease Consortium | Types I, III, IV Pediatric OI | Planar Radiographs (Hand, Spine, Limbs) | Controlled Access DUA | Submit research application to RDCRN DMCC |
| **Shriners Children's Hospital Network** | Specialized Pediatric Orthopedic Hospital | Complete Sillence Spectrum (Types I–VIII) | Native DICOM Radiography | Academic Research Collaboration | Establish multi-center pediatric orthopedic IRB protocol |
| **Osteogenesis Imperfecta Foundation (OIF)** | Patient Registry & Clinical Advisory Panel | Confirmed Genetic OI (`COL1A1`/`COL1A2`) | De-identified Radiographic Studies | Research Committee Review | Submit formal protocol to OIF Medical Advisory Council |

---

## 4. Final Class Readiness Determination: Osteogenesis Imperfecta

$$\mathbf{\text{Class Readiness: NOT READY — DATA ACQUISITION REQUIRED}}$$

- **Existing In-Tree Human OI Radiographs:** **0**
- **Verified Human Open Download Portals:** **0**
- **Primary Viable Next Action:** Submit formal data use request to the Rare Diseases Clinical Research Network (RDCRN) for the Brittle Bone Disorders Consortium (BBDC) imaging collection.

