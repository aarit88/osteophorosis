# OsteoGuard AI — Dataset Acquisition, Provenance & Ingestion Framework

## 1. Executive Summary & Clinical AI Governance

OsteoGuard AI requires authentic, clinically validated medical imaging cohorts for multi-class differential diagnosis across four primary target skeletal conditions:

1. **Secondary Bone Metastasis** (`secondary_bone_metastasis` / `metastasis`)
2. **Paget's Disease of Bone** (`pagets_disease` / `paget`)
3. **Osteogenesis Imperfecta** (`osteogenesis_imperfecta` / `oi`)
4. **Healthy Skeletal Control** (`healthy_control` / `healthy`)

### 1.1 Pre-Training Readiness Status
```yaml
DATASET_READY: false
```
> [!IMPORTANT]
> The `DATASET_READY` flag is strictly set to **`FALSE`**. Currently, zero physical imaging datasets are downloaded into the repository. In compliance with **Rule 4**, **Rule 5**, **Rule 6**, and **Rule 7**, no synthetic images, phantom patient records, or fabricated evaluation metrics may be substituted. Deep learning model training and inference remain hard-blocked until authentic clinical cohorts are acquired and verified.

---

## 2. Strict Negative Mapping Protocols (Pathology Boundary Integrity)

To preserve scientific validity and diagnostic safety, data curators and automated validation pipelines must strictly adhere to the following **Negative Mapping Rules**:

> [!CAUTION]
> **Rule 2.1 — NEVER Map Primary Malignant Bone Tumours to Secondary Metastasis**
> - Primary osteosarcoma, chondrosarcoma, and Ewing sarcoma exhibit distinct genomic drivers, local periosteal reactions (e.g., Codman triangle, sunburst appearance), and architectural patterns that differ fundamentally from secondary haematogenous metastases (breast, prostate, renal, lung).
> - Multiple myeloma, lymphoma, and solitary plasmacytomas must also not be mapped to secondary carcinoma metastasis without discrete histological subtyping.

> [!CAUTION]
> **Rule 2.2 — NEVER Map Generic Abnormal Images to Paget's Disease**
> - Degenerative sclerosis, severe osteoarthritis, fibrous dysplasia, osteopetrosis, or hyperparathyroidism ("brown tumours") must NEVER be lazily mapped to Paget's disease.
> - Paget's disease requires radiologist confirmation of pathognomonic architectural remodeling: cortical thickening, coarse trabecular expansion, loss of corticomedullary differentiation, or "cotton-wool" calvarial appearance.

> [!CAUTION]
> **Rule 2.3 — NEVER Map Age-Related Osteoporosis to Osteogenesis Imperfecta**
> - Post-menopausal, senile, steroid-induced, or disuse osteoporosis involves accelerated bone resorption in mature bone. Osteogenesis imperfecta (OI) is a genetic defect in type I collagen synthesis (`COL1A1`/`COL1A2`), resulting in congenital bone fragility, gracile long bones, bowing deformities, and Wormian bones.
> - Osteoporosis, osteopenia, rickets, and osteomalacia must NEVER be categorized as OI.

> [!CAUTION]
> **Rule 2.4 — NEVER Map 'Unaffected' Limbs or Scans as Confirmed Healthy Controls**
> - Contralateral or unaffected limbs in cancer, trauma, or fracture patients must NEVER be assumed healthy. Sub-clinical systemic metastasis, diffuse osteopenia, or occult stress fractures frequently contaminate such images.
> - Healthy skeletal controls require an explicit radiologist finding of normal bone density and morphology with zero osteolytic, osteoblastic, or metabolic pathology.

> [!WARNING]
> **Rule 2.5 — Do Not Assume Research-Paper Class Distributions Are Available**
> - Clinical data sources are naturally imbalanced. Public open-access sources for rare congenital conditions (such as Osteogenesis Imperfecta) contain far fewer subjects than high-volume trauma repositories.
> - Models must employ Multiclass Focal Loss ($\gamma = 2.0$, $\alpha = [0.25, 0.25, 0.25, 0.25]$) and patient-level stratified k-fold partitioning rather than artificially balancing classes with synthetic data.

---

## 3. Comprehensive Initial Data Sources Dossier

The framework registers nine authentic biomedical imaging repositories across the four target disease classes:

| Source ID | Target Class | Modality | Primary Format | Access Tier | Download Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TCIA_SPINE_METS_CT_SEG` | Secondary Metastasis | CT | DICOM / NIfTI | **Restricted Approval** | TCIA NBIA Data Retriever |
| `ZENODO_BONE_METS_13900966` | Secondary Metastasis | CT & X-ray | NIfTI / PNG | **Public Download** | Direct HTTPS / Zenodo API |
| `WIKIMEDIA_PAGET_COMMONS` | Paget's Disease | X-ray & Bone Scan | JPEG / PNG / TIFF | **Public Download** | Direct HTTPS / Wikimedia API |
| `NLM_MEDPIX_PAGET` | Paget's Disease | X-ray & CT | JPEG / PNG | **Restricted Approval** | MedPix Case Query Export |
| `PMC_11324408_OI` | Osteogenesis Imperfecta | X-ray | JPEG / TIFF / ZIP | **Public Download** | PubMed Central Open Access |
| `FIGSHARE_OI_COLLECTION_3795019` | Osteogenesis Imperfecta | X-ray | DICOM / TIFF | **Public Download** | Figshare Direct Download |
| `FIGSHARE_NORMAL_SKELETAL_27865398` | Healthy Controls | X-ray | PNG / DICOM | **Public Download** | Figshare Direct Download |
| `STANFORD_AIMI_MURA` | Healthy Controls | X-ray | PNG (16-bit) | **Restricted Approval** | Stanford DUA Portal |
| `TCIA_HEALTHY_TOTAL_BODY_CTS` | Healthy Controls | CT | DICOM | **Restricted Approval** | TCIA NBIA Data Retriever |

---

## 4. Source-by-Source Specifications & Acquisition Instructions

### 4.1 Secondary Bone Metastasis

#### Source 1: TCIA Spine-Mets-CT-Seg
- **Repository URL**: [https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/)
- **DOI**: `10.7937/TCIA.V2WY-KG44`
- **Available Format**: Raw DICOM axial slice series, NIfTI volumetric lesion masks.
- **Available Modality**: Computed Tomography (CT), primarily spine/thoracolumbar.
- **Available Labels**: Verified spinal metastatic lesions with primary tumor histopathology (breast, prostate, lung, renal).
- **Patient-Level Metadata**: De-identified subject IDs (e.g., `Spine-Mets-001`), patient age, sex, primary site.
- **Segmentation Annotations**: Complete 3D voxel segmentations of vertebrae bodies and osteolytic/osteoblastic metastatic lesions.
- **License & Access Restrictions**: Creative Commons Attribution 4.0 International (CC BY 4.0) with TCIA Data Use Agreement. Requires registration with TCIA.
- **Intended ML Use**: U-Net skeletal segmentation pre-training, ResNet-101 deep CT feature extraction, 3D CT radiomics.
- **Known Limitations**: Restricted to spine; slice thickness varies between 1.0 mm and 2.5 mm.
- **Download Instructions (Manual Approval Required)**:
  1. Register for an account at [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/).
  2. Install the **NBIA Data Retriever** desktop application.
  3. Navigate to the Spine-Mets-CT-Seg collection page and click **Download Selected Series** to obtain the `.tcia` manifest.
  4. Open the `.tcia` manifest in the NBIA Data Retriever to download DICOM series to local secure storage.
  5. Copy/symlink de-identified DICOM files into `ml_service/data/raw/metastasis/`.

#### Source 2: Zenodo Bone Metastases Cohort
- **Repository URL**: [https://zenodo.org/records/13900966](https://zenodo.org/records/13900966)
- **DOI**: `10.5281/zenodo.13900966`
- **Available Format**: NIfTI (`.nii.gz`), PNG exported slices, CSV feature tables.
- **Available Modality**: CT and planar pelvic/femoral radiographs.
- **Available Labels**: Verified osteolytic, osteoblastic, and mixed metastatic osseous lesions.
- **Patient-Level Metadata**: De-identified patient IDs, age, gender, anatomic location.
- **Segmentation Annotations**: 2D and 3D binary ROI lesion contours.
- **License & Access Restrictions**: Creative Commons Attribution 4.0 International (CC BY 4.0). Completely public.
- **Intended ML Use**: 102-D IBSI radiomic extraction benchmarking, multimodal late fusion training.
- **Known Limitations**: Focuses on pelvis and proximal femur; variable radiographic exposure.
- **Download Instructions (Publicly Downloadable)**:
  ```bash
  # Using Python or curl
  curl -L -O https://zenodo.org/records/13900966/files/bone_metastases_dataset.zip
  unzip bone_metastases_dataset.zip -d ml_service/data/raw/metastasis/
  ```

---

### 4.2 Paget's Disease of Bone

#### Source 3: Wikimedia Commons Paget's Disease Repository
- **Repository URL**: [https://commons.wikimedia.org/wiki/Category:Paget%27s_disease_of_bone](https://commons.wikimedia.org/wiki/Category:Paget%27s_disease_of_bone)
- **Available Format**: JPEG, PNG, TIFF.
- **Available Modality**: Plain radiographs (tibia, pelvis, skull), 99mTc-MDP Bone Scintigraphy.
- **Available Labels**: Verified Paget's disease cases published in peer-reviewed open-access journals.
- **Patient-Level Metadata**: Publication DOI citations, anatomic site, clinical presentation notes.
- **Segmentation Annotations**: None (requires expert manual bounding box or automated U-Net ROI extraction).
- **License & Access Restrictions**: Creative Commons Attribution-ShareAlike (CC BY-SA 3.0 / 4.0). Completely public.
- **Intended ML Use**: Planar X-ray feature extraction, bone scintigraphy radionuclide hotspot calibration.
- **Known Limitations**: 8-bit compressed images rather than 16-bit DICOM; limited sample size requiring few-shot evaluation.
- **Download Instructions (Publicly Downloadable)**:
  1. Query the Wikimedia Commons MediaWiki API:
     ```bash
     curl "https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Paget%27s_disease_of_bone&cmtype=file&format=json"
     ```
  2. Filter for radiograph and scintigram files.
  3. Download high-resolution originals into `ml_service/data/raw/paget/`.

#### Source 4: NLM MedPix Teaching File Database
- **Repository URL**: [https://medpix.nlm.nih.gov/](https://medpix.nlm.nih.gov/)
- **Available Format**: JPEG, PNG, teaching DICOM format.
- **Available Modality**: Planar Radiographs, Computed Tomography, MRI.
- **Available Labels**: Board-certified radiologist consensus diagnosis with teaching commentary.
- **Patient-Level Metadata**: Age, sex, presenting symptoms, anatomical location.
- **Segmentation Annotations**: Radiologist-annotated anatomical pointers and boundary regions.
- **License & Access Restrictions**: U.S. National Library of Medicine Terms of Service (Free for non-commercial educational & research use).
- **Intended ML Use**: Qualitative Grad-CAM saliency alignment, validation holdout cohort.
- **Known Limitations**: Automated batch scraping is restricted by rate limits; manual query export required.
- **Download Instructions (Manual Acquisition Required)**:
  1. Access [MedPix Search](https://medpix.nlm.nih.gov/).
  2. Search for `"Paget disease of bone"` or `"osteitis deformans"`.
  3. Filter by modality: **Plain Radiograph** and **CT Bone Window**.
  4. Download cases and extract metadata into `ml_service/data/dataset_manifest.csv`.
  5. Store files under `ml_service/data/raw/paget/`.

---

### 4.3 Osteogenesis Imperfecta

#### Source 5: PMC11324408 Osteogenesis Imperfecta Radiological Study
- **Repository URL**: [https://pmc.ncbi.nlm.nih.gov/articles/PMC11324408/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11324408/)
- **DOI**: `10.3390/children11080976`
- **Available Format**: High-resolution JPEG, TIFF, supplementary ZIP archive.
- **Available Modality**: Planar Radiographs (Full limb, spine, femur, tibia).
- **Available Labels**: Clinically and genetically confirmed OI Sillence classifications (Types I, III, IV).
- **Patient-Level Metadata**: De-identified subject IDs, pediatric/adult age, biological sex, genetic mutation.
- **Segmentation Annotations**: None.
- **License & Access Restrictions**: Creative Commons Attribution 4.0 International (CC BY 4.0). Completely public.
- **Intended ML Use**: Cortical thinning radiomic profiling, biomechanical fracture risk calibration ($R_{\text{fracture}}$).
- **Known Limitations**: Predominantly pediatric population; severe deformities require specialized data augmentation.
- **Download Instructions (Publicly Downloadable)**:
  1. Download the article's supplementary files and high-resolution figure bundles from PMC.
  2. Extract images to `ml_service/data/raw/oi/`.

#### Source 6: Figshare OI Radiological Collection
- **Repository URL**: [https://doi.org/10.6084/m9.figshare.c.3795019](https://doi.org/10.6084/m9.figshare.c.3795019)
- **DOI**: `10.6084/m9.figshare.c.3795019`
- **Available Format**: DICOM, TIFF, PNG.
- **Available Modality**: Planar Radiographs (Long bones, pelvis, skull).
- **Available Labels**: Clinically confirmed Osteogenesis Imperfecta cases.
- **Patient-Level Metadata**: De-identified IDs, age, gender, severity scale.
- **Segmentation Annotations**: None.
- **License & Access Restrictions**: Creative Commons Attribution 4.0 International (CC BY 4.0). Completely public.
- **Intended ML Use**: ResNet-101 semantic feature extraction and GLSZM radiomics.
- **Known Limitations**: Variable radiographic exposures requiring CLAHE pre-processing.
- **Download Instructions (Publicly Downloadable)**:
  ```bash
  # Download via Figshare API or direct browser download
  curl -L -O https://api.figshare.com/v2/collections/3795019/articles
  # Extract verified cases to:
  # ml_service/data/raw/oi/
  ```

---

### 4.4 Healthy Skeletal Controls

#### Source 7: Figshare Normal Skeletal Radiographs
- **Repository URL**: [https://doi.org/10.6084/m9.figshare.27865398](https://doi.org/10.6084/m9.figshare.27865398)
- **DOI**: `10.6084/m9.figshare.27865398`
- **Available Format**: PNG, DICOM.
- **Available Modality**: Planar Radiography (X-ray: spine, pelvis, extremities).
- **Available Labels**: Verified normal skeletal studies (negative for fractures, lesions, or osteopenia).
- **Patient-Level Metadata**: De-identified IDs, adult age ranges, sex.
- **Segmentation Annotations**: None.
- **License & Access Restrictions**: Creative Commons Attribution 4.0 International (CC BY 4.0). Completely public.
- **Intended ML Use**: Baseline normative distribution calibration for 4-class Focal Loss.
- **Known Limitations**: Restricted to adult subjects.
- **Download Instructions (Publicly Downloadable)**:
  ```bash
  curl -L -O https://api.figshare.com/v2/articles/27865398/files
  # Extract verified normal controls to:
  # ml_service/data/raw/healthy/
  ```

#### Source 8: Stanford AIMI MURA (Musculoskeletal Radiographs)
- **Repository URL**: [https://aimi.stanford.edu/datasets/mura-msk-xrays](https://aimi.stanford.edu/datasets/mura-msk-xrays)
- **DOI**: `10.48550/arXiv.1712.06957`
- **Available Format**: PNG (16-bit and 8-bit), CSV patient manifests.
- **Available Modality**: Planar Radiographs (Upper and lower extremity studies).
- **Available Labels**: Radiologist consensus evaluation: `0` (Normal) vs. `1` (Abnormal).
- **Patient-Level Metadata**: Patient IDs (`patient00001` through `patient14863`), study types (`XR_WRIST`, `XR_SHOULDER`, `XR_HUMERUS`).
- **Segmentation Annotations**: None.
- **License & Access Restrictions**: Stanford University Research Data Use Agreement. Restricted to non-commercial academic research.
- **Intended ML Use**: Large-scale normal negative class representation for extremity bone radiographs.
- **Known Limitations**: Only extremity radiographs; does not include axial spine or pelvis.
- **Download Instructions (Approval Required)**:
  1. Submit the Stanford AIMI Research Data Use Agreement online.
  2. Upon approval, receive a secure download link via email.
  3. Extract studies labeled `normal` into `ml_service/data/raw/healthy/`.

#### Source 9: TCIA Healthy Total Body CTs
- **Repository URL**: [https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/](https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/)
- **DOI**: `10.7937/TCIA.2019.4Q8A8J7V`
- **Available Format**: DICOM.
- **Available Modality**: Computed Tomography (Whole Body).
- **Available Labels**: Asymptomatic skeletal scans confirmed negative for osseous lesions.
- **Patient-Level Metadata**: De-identified TCIA IDs, kVp, slice thickness, patient sex.
- **Segmentation Annotations**: None.
- **License & Access Restrictions**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Intended ML Use**: 3D CT normative Hounsfield Unit modeling (+400 HU WL / 1500 HU WW).
- **Known Limitations**: High data volume (multi-gigabyte CT volumes).
- **Download Instructions (Approval Required)**:
  1. Download `.tcia` manifest from TCIA.
  2. Open in NBIA Data Retriever.
  3. Extract to `ml_service/data/raw/healthy/`.

---

## 5. File Verification & Integrity Utilities

The framework provides an automated verification tool:
- `ml_service/datasets/registry_verifier.py`
- Programmatic class `DatasetRegistry`:
  - Parses `dataset_registry.yaml`.
  - Verifies cryptographic SHA-256 file hashes.
  - Generates `dataset_manifest.csv` from acquired directories.
  - Enforces `DATASET_READY == False` hard gate until all validation rules pass.

---

## 6. Version Control Policy (.gitignore Safeguards)

To comply with HIPAA privacy regulations, prevent repo bloat, and uphold scientific integrity:
- Raw DICOM/PNG images (`ml_service/data/raw/**`) are strictly excluded from Git.
- Model weights (`ml_artifacts/checkpoints/**`) are strictly excluded from Git.
- The `dataset_manifest.csv` and `dataset_registry.yaml` track data provenance without checking in raw binary images.

---

## 7. Stage 06.1 Ingestion, Extraction & Forensic Audit Results

In Stage 06.1, three physical medical imaging dataset archives were supplied, forensically audited, and safely extracted into `ml_service/data/raw/`:

### 7.1 FracAtlas (`ml_service/data/raw/fracatlas/`)
- **Archive**: `ml_service/data/raw/_archives/FracAtlas.zip` (338.4 MB compressed, 342.8 MB uncompressed)
- **Modality**: Planar Radiography (X-ray)
- **Total Images**: 4,085 image files
- **Authoritative Classification File**: `dataset.csv` (4,083 records)
  - `fractured = 1`: **719 images**
  - `fractured = 0`: **3,364 images**
- **Annotations**:
  - `COCO_fracture_masks.json`: 719 fractured images, 924 polygonal fracture segmentations.
  - PASCAL VOC XML: 4,083 files.
  - YOLO TXT: 4,087 files.
  - VGG JSON: `VGG_fracture_masks.json`.
- **Pre-existing Splits**: Fracture-only detection split (`train: 574, valid: 82, test: 63`, Sum = 719).
- **Forensic Duplicate Finding**: `IMG0003375.jpg` and `IMG0003376.jpg` exist in both `Fractured/` and `Non_fractured/` directories with identical SHA-256 digests. `dataset.csv` authoritatively classifies both as `fractured = 1`.
- **Intended ML Use**: Biomechanical fracture risk stratification calibration and fracture detection module.
- **Readiness Status**: `READY_FOR_FRACTURE_EXPERIMENTS`.

### 7.2 Bone Scan Metastasis Dataset (`ml_service/data/raw/bone_scan_metastasis/`)
- **Archive**: `ml_service/data/raw/_archives/Bone Scan Images Dataset for Study of Bone Metastases in Adult Breast Cancer Patients at IICS-UNA Paraguay.zip` (79.9 MB compressed, 80.3 MB uncompressed)
- **Modality**: Planar Whole-Body Scintigraphy (Nuclear Medicine, $^{99\text{m}}\text{Tc}$-MDP)
- **Cohort**: 291 adult breast cancer patients (289 Female, 2 Male)
- **Total Images**: 582 PNG images (Paired Anterior `A*.png` and Posterior `P*.png` for every patient)
- **Class Distribution**:
  - `Bone metastases`: **60 patients** (120 images)
  - `No bone metastases`: **231 patients** (462 images)
- **Missing & Duplicate Analysis**: 0 missing files, 0 duplicate image hashes.
- **Strict Clinical Governance**:
  - Treat as an independent binary nuclear-medicine classification task.
  - NEVER combine planar scintigraphy with ordinary X-rays or CT scans.
  - NEVER consider 'no bone metastases' oncology scans as healthy skeletal controls.
- **Readiness Status**: `READY_FOR_BONE_SCAN_EXPERIMENTS`.

### 7.3 BTXRD Bone Tumor Dataset (`ml_service/data/raw/btxrd/`)
- **Archive**: `ml_service/data/raw/_archives/BTXRD.zip` (840.5 MB compressed, 903.5 MB uncompressed)
- **Modality**: Conventional Plain Radiography (X-ray)
- **Total Images**: 3,746 JPEG images across 3 acquisition centers (Center 1: 2938, Center 2: 549, Center 3: 259)
- **Class Distribution**:
  - `tumor = 1`: **1,867 images**
    - `benign = 1`: **1,525 images** (754 Osteochondroma, 263 Multiple Osteochondromas, 206 Simple Bone Cyst, 93 Giant Cell Tumor, 44 Osteofibroma, 51 Synovial Osteochondroma, 115 Other BT)
    - `malignant = 1`: **342 images** (297 Osteosarcoma, 45 Other MT)
  - `tumor = 0` (Non-tumor / Normal bone): **1,879 images**
- **Annotations**: 1,867 LabelMe JSON files with polygonal tumor boundaries.
- **Forensic Duplicate Finding**: 21 duplicate image content hashes (repeated scans of identical patient anatomy).
- **Strict Clinical Governance**:
  - NEVER map primary bone malignancy (Osteosarcoma) to secondary bone metastasis.
  - Non-tumor normal bone radiographs (1,879 cases) are candidate negative skeletal controls.
- **Readiness Status**: `READY_FOR_TUMOR_EXPERIMENTS`.

### 7.4 Master Four-Class Pipeline Status
- **Status**: **`NOT_READY`** (`dataset_ready: false`).
- Target classes `pagets_disease` and `osteogenesis_imperfecta` currently lack verified clinical patient cohorts. Model training for the unified 4-class pipeline remains locked until all four classes are acquired.


