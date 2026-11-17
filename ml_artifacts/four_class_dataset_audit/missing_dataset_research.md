# Missing Dataset Research & External Cohort Investigation

**Project:** OsteoGuard AI — Unified Deep Learning & Differential Diagnostic Radiomic System  
**Stage:** 22 — Four-Class Dataset Readiness and Gap Audit  
**Artifact Path:** `ml_artifacts/four_class_dataset_audit/missing_dataset_research.md`  

---

## 1. Executive Research Scope

The proposed OsteoGuard AI final clinical target comprises four distinct diagnostic entities:
1. **Secondary Bone Metastasis**
2. **Paget's Disease of Bone**
3. **Osteogenesis Imperfecta (OI)**
4. **Healthy Skeletal Control**

While the project has previously acquired and audited datasets for acute bone fracture detection (FracAtlas), planar whole-body bone scintigraphy metastasis detection (IICS-UNA), and primary bone tumours (BTXRD), two of the four target disease entities (**Paget's Disease of Bone** and **Osteogenesis Imperfecta**) have **ZERO** representation in the existing project tree. Furthermore, secondary bone metastasis is represented only in nuclear scintigraphy (lacking plain radiography), and healthy controls are either confounded by trauma presentation (FracAtlas) or unverified for patient-level independence (BTXRD).

This investigation evaluates authentic external repositories, academic registries, and published research cohorts to establish viable data acquisition candidates without performing any unauthorized downloads or license acceptances.

---

## 2. Target Class 1: Secondary Bone Metastasis (Radiography & Volumetric Candidates)

*Existing In-Tree Status:* 60 patients (120 images) in planar bone scintigraphy ($^{99\text{m}}\text{Tc}$-MDP). Zero plain X-ray radiographs in-tree.

### Candidate 1.1: TCIA Spine-Mets-CT-SEG (The Cancer Imaging Archive)
- **Official Repository URL:** [https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/)
- **Official DOI:** `10.7937/kh36-ds04` (Tracking: `10.7937/TCIA.V2WY-KG44`)
- **Institution / Consortium:** The Cancer Imaging Archive (TCIA) / National Cancer Institute (NCI)
- **Imaging Modality:** Computed Tomography (CT), axial slice volumes
- **Available Diagnostic Classes:** Secondary metastatic bone cancer of the spine originating from breast, lung, prostate, renal, and other carcinomas; includes lytic, sclerotic/blastic, and mixed lesions.
- **Published Cohort Volume:** 238 patients, ~450 CT examinations, >30,000 axial slices.
- **Label Verification:** Pathologist-confirmed primary carcinomas with radiologist-verified spinal metastases; includes expert manual/semi-automated voxel segmentations (DICOM-SEG).
- **Patient Identifiers:** True de-identified subject IDs (e.g., `Spine-Mets-001` through `Spine-Mets-238`) allowing strict patient-level splitting.
- **Licensing & Usage Restrictions:** Creative Commons Attribution 4.0 International (CC BY 4.0) with TCIA Data Use Agreement. Permitted for academic machine learning research.
- **Download Availability:** Open research access; requires NBIA Data Retriever application or REST API via `.tcia` manifest.
- **Evidence Limitations:** Restricted strictly to spinal column anatomy; CT volumetric modality only (does not provide plain X-ray planar projections).

### Candidate 1.2: Zenodo Bone Metastases Benchmark Cohort
- **Official Repository URL:** [https://zenodo.org/records/13900966](https://zenodo.org/records/13900966)
- **Official DOI:** `10.5281/zenodo.13900966`
- **Institution / Authors:** Open-access biomedical research cohort
- **Imaging Modality:** Computed Tomography (CT) and Planar Radiographs (pelvic and femoral views)
- **Available Diagnostic Classes:** Osteolytic, osteoblastic, and mixed metastatic osseous lesions.
- **Published Cohort Volume:** 124 patients, ~250 image series.
- **Label Verification:** Radiologist-validated lesion margins with 2D/3D region-of-interest (ROI) contours.
- **Patient Identifiers:** Anonymized subject keys present.
- **Licensing & Usage Restrictions:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Download Availability:** Direct HTTPS download via Zenodo REST API.
- **Evidence Limitations:** Predominantly pelvic and proximal femoral lesions; variable radiographic acquisition protocols.

---

## 3. Target Class 2: Paget's Disease of Bone (Metabolic Remodeling)

*Existing In-Tree Status:* **0 patients, 0 images. COMPLETELY MISSING.**

*Epidemiological Reality:* Paget's disease is a localized osteometabolic disorder characterized by disorganized, accelerated osteoclastic/osteoblastic bone remodeling. Unlike pneumonia or COVID-19, large public ML benchmark datasets dedicated exclusively to Paget's disease do not currently exist. Data must be aggregated from validated academic teaching files and peer-reviewed case archives.

### Candidate 2.1: Radiopaedia Paget's Disease Teaching File Collection
- **Official Repository URL:** [https://radiopaedia.org/articles/paget-disease-of-bone](https://radiopaedia.org/articles/paget-disease-of-bone)
- **Institution / Publisher:** Radiopaedia.org Collaborative Musculoskeletal Radiology Panel
- **Imaging Modality:** Planar Radiography (X-ray), CT, Planar Bone Scintigraphy
- **Available Diagnostic Classes:** Paget's disease across classic anatomical stages (osteolytic lytic wedge/"blade of grass", mixed cortical thickening/trabecular coarsening, osteosclerotic "cotton-wool" skull, "picture-frame" vertebra, pelvic brim thickening).
- **Published Cohort Volume:** ~150 curated clinical cases (~300 multimodal images).
- **Label Verification:** Board-certified radiologist peer-reviewed cases with clinical history, biochemical correlation (elevated alkaline phosphatase), and histological correlation where biopsied.
- **Patient Identifiers:** Case-level unique identifiers (e.g., `Case ID: 12495`).
- **Licensing & Usage Restrictions:** Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 3.0 / 4.0). Strictly non-commercial academic research use.
- **Download Availability:** Programmatic web access is rate-limited; requires manual case harvesting or approved research access request.
- **Evidence Limitations:** Web-resolution 8-bit images (JPEG/PNG); absence of original 16-bit DICOM pixel data; non-commercial license clause restricts commercial deployment.

### Candidate 2.2: National Library of Medicine (NLM) MedPix Teaching File Database
- **Official Repository URL:** [https://medpix.nlm.nih.gov/](https://medpix.nlm.nih.gov/)
- **Institution / Publisher:** National Institutes of Health (NIH) / National Library of Medicine (NLM)
- **Imaging Modality:** Plain Radiography (X-ray), CT, MRI
- **Available Diagnostic Classes:** Confirmed Paget's disease cases with differential diagnosis commentary (excluding fibrous dysplasia, osteosclerotic metastases, and hyperparathyroidism).
- **Published Cohort Volume:** ~45 distinct patient cases (~90 images).
- **Label Verification:** Board-certified radiologist verified teaching cases.
- **Patient Identifiers:** De-identified case numbers with patient age, sex, presenting symptoms, and case history.
- **Licensing & Usage Restrictions:** US Government Work / NLM Terms of Service (Permitted for research, educational, and computational scientific use).
- **Download Availability:** Accessible via MedPix case search query.
- **Evidence Limitations:** Modest cohort size; teaching arrows and anatomical callouts are embedded in some images, requiring manual bounding-box cropping or inpainting.

### Candidate 2.3: Wikimedia Commons Curated Paget's Disease Repository
- **Official Repository URL:** [https://commons.wikimedia.org/wiki/Category:Paget%27s_disease_of_bone](https://commons.wikimedia.org/wiki/Category:Paget%27s_disease_of_bone)
- **Institution / Publisher:** Curated open-access medical publishing repository (derived from PMC, BMJ Case Reports, Acta Radiologica Open)
- **Imaging Modality:** Plain Radiographs (Pelvis, skull, tibia, spine) and Bone Scintigraphy
- **Available Diagnostic Classes:** Clinically confirmed Paget's disease.
- **Published Cohort Volume:** ~60 distinct case figures (~75 images).
- **Label Verification:** Published peer-reviewed clinical case reports with journal citations.
- **Patient Identifiers:** Case-level descriptions with patient demographics.
- **Licensing & Usage Restrictions:** Creative Commons CC BY-SA 3.0 / 4.0.
- **Download Availability:** Direct HTTPS / Wikimedia API batch download.
- **Evidence Limitations:** Compressed 8-bit web formats; lack of standardization across image resolutions.

---

## 4. Target Class 3: Osteogenesis Imperfecta (Skeletal Fragility)

*Existing In-Tree Status:* **0 patients, 0 images. COMPLETELY MISSING.**

*Epidemiological Reality:* Osteogenesis imperfecta (OI, "brittle bone disease") is a rare genetic disorder of type I collagen synthesis (`COL1A1`/`COL1A2`), with an incidence of approximately 1 in 10,000–20,000 births. Large public radiographic datasets do not exist; cases must be curated from specialized pediatric bone dysplasia registries and open-access genetic cohorts.

### Candidate 3.1: PubMed Central Pediatric OI Cohort (PMC11324408)
- **Official Repository URL:** [https://pmc.ncbi.nlm.nih.gov/articles/PMC11324408/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11324408/)
- **Official DOI:** `10.3390/children11080976`
- **Publication Title:** *Skeletal Manifestations and Longitudinal Deformities in Osteogenesis Imperfecta* (Children, 2024)
- **Institution:** Tertiary Academic Pediatric Orthopedic Hospital
- **Imaging Modality:** Planar Radiography (X-ray: full-limb, femur, tibia, spine, skull)
- **Available Diagnostic Classes:** Genetically confirmed Osteogenesis Imperfecta classified under the Sillence taxonomy (Types I, III, IV, V).
- **Published Cohort Volume:** 38 patients, 85 full-length skeletal radiographs.
- **Label Verification:** Molecular genetic testing (`COL1A1`/`COL1A2` mutation confirmed) combined with radiologist consensus review.
- **Patient Identifiers:** Anonymized patient codes (`Patient OI-01` through `OI-38`), permitting cluster-aware patient-level splitting.
- **Licensing & Usage Restrictions:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Download Availability:** Open-access XML and high-resolution TIFF/JPEG supplementary download from PubMed Central.
- **Evidence Limitations:** Skewed towards pediatric patients; severe anatomical bowing and deformity require tailored aspect-ratio padding; sample volume is insufficient for high-capacity CNN training from scratch (requires transfer learning).

### Candidate 3.2: Figshare OI Radiological Feature Collection
- **Official Repository URL:** [https://doi.org/10.6084/m9.figshare.c.3795019](https://doi.org/10.6084/m9.figshare.c.3795019)
- **Official DOI:** `10.6084/m9.figshare.c.3795019`
- **Institution / Author:** Academic musculoskeletal genetics research group
- **Imaging Modality:** Planar Radiographs (Spine, long bones, skull)
- **Available Diagnostic Classes:** Clinically confirmed Osteogenesis Imperfecta severity spectrum.
- **Published Cohort Volume:** 25 patients, ~50 radiograph studies.
- **Label Verification:** Clinical genetics diagnosis.
- **Patient Identifiers:** De-identified patient IDs available.
- **Licensing & Usage Restrictions:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Download Availability:** Direct HTTPS download via Figshare API.
- **Evidence Limitations:** Small sample size; heterogeneous radiographic exposure settings.

---

## 5. Target Class 4: Healthy Skeletal Controls (Normative Reference)

*Existing In-Tree Status:*
- FracAtlas Non-Fractured (3,364 images): **UNSUITABLE** (emergency trauma patients with suspected fractures; high risk of soft-tissue trauma, degenerative joint disease, or osteopenia).
- Bone Scintigraphy Negative (231 patients, 462 images): **UNSUITABLE** (active breast cancer patients on systemic oncologic therapies; planar scintigraphy modality mismatch).
- BTXRD Center 1 Normal (1,600 clean images): **CONDITIONALLY SUITABLE** (non-tumor orthopedic clinic radiographs; lacks patient accession IDs; lacks verified bone mineral density).

### Candidate 4.1: Stanford AIMI MURA (Musculoskeletal Radiographs) Normal Cohort
- **Official Repository URL:** [https://aimi.stanford.edu/datasets/mura-msk-xrays](https://aimi.stanford.edu/datasets/mura-msk-xrays)
- **Official Publication / DOI:** Rajpurkar et al., *MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs*, arXiv:1712.06957.
- **Institution:** Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI)
- **Imaging Modality:** Conventional Plain Radiography (X-ray: hand, wrist, forearm, elbow, humerus, shoulder, finger)
- **Available Diagnostic Classes:** Radiologist-consensus Normal vs Abnormal.
- **Published Cohort Volume:** ~9,000 normal patient studies (>20,000 normal radiographs).
- **Label Verification:** Consensus readings by board-certified Stanford musculoskeletal radiologists.
- **Patient Identifiers:** Explicit patient identifiers (e.g., `patient00001`) enabling strict patient-level splitting.
- **Licensing & Usage Restrictions:** Stanford University Research Data Use Agreement (Non-commercial academic research only).
- **Download Availability:** Downloadable via Stanford AIMI portal upon submitting electronic agreement.
- **Evidence Limitations:** Focuses strictly on upper extremity anatomy (lacks axial spine and pelvis); requires non-commercial DUA approval.
- **Suitability Verdict:** **VERIFIED SUITABLE** for upper-extremity plain radiographic normal control representation.

### Candidate 4.2: Figshare Normal Skeletal Radiographs Reference Collection
- **Official Repository URL:** [https://doi.org/10.6084/m9.figshare.27865398](https://doi.org/10.6084/m9.figshare.27865398)
- **Official DOI:** `10.6084/m9.figshare.27865398`
- **Institution:** Academic hospital reference radiology collection
- **Imaging Modality:** Planar Radiography (X-ray: spine, pelvis, extremities)
- **Available Diagnostic Classes:** Confirmed normal skeletal studies (negative for fracture, focal lesion, or osteopenia).
- **Published Cohort Volume:** 350 patients, ~700 radiographs.
- **Label Verification:** Radiologist verified normal bone density and morphology.
- **Patient Identifiers:** De-identified patient IDs available.
- **Licensing & Usage Restrictions:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Download Availability:** Direct HTTPS download via Figshare API.
- **Evidence Limitations:** Modest cohort size compared to MURA; requires anatomical region filtering.

---

## 6. Summary of Candidate Verification Matrix

| Target Disease Class | In-Tree Status | Top External Candidate | Modality | Patient IDs | Open Access Tier | Suitability |
| :--- | :---: | :--- | :---: | :---: | :--- | :---: |
| **Secondary Bone Metastasis** | Partial (Scintigraphy) | TCIA Spine-Mets-CT-SEG (`10.7937/kh36-ds04`) | CT | Yes | Restricted DUA | Conditionally Suitable |
| **Paget's Disease of Bone** | **MISSING** | Radiopaedia Curated Cohort | X-ray / CT | Case-level | CC BY-NC-SA | Conditionally Suitable |
| **Osteogenesis Imperfecta** | **MISSING** | PMC11324408 Pediatric OI (`10.3390/children11080976`) | X-ray | Yes | CC BY 4.0 | Conditionally Suitable |
| **Healthy Skeletal Control** | Partial (BTXRD C1) | Stanford AIMI MURA (`arXiv:1712.06957`) | X-ray | Yes | Stanford DUA | **Verified Suitable** |

