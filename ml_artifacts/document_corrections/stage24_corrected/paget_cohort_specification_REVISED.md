# Clinical Cohort Acquisition Specification (Revised): Paget's Disease of Bone

**Target Class 2**: Paget's Disease of Bone (*Osteitis Deformans*)  
**Document Status**: Formal Cohort Acquisition Specification (Revised Draft)  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Disease Definition and Pathophysiology

Paget's disease of bone is a chronic focal skeletal disorder characterized by abnormal, accelerated bone remodeling. Disordered bone turnover progresses through three overlapping phases:
1. **Initial Osteolytic Phase**: Intense localized osteoclastic bone resorption (e.g., advancing resorption wedge or "blade of grass" sign).
2. **Intermediate Mixed Phase**: Concurrent rapid bone resorption and disorganized osteoblastic new bone formation.
3. **Late Osteosclerotic Phase**: Exhaustion of cellular remodeling resulting in dense, chaotic, poorly mineralized woven bone ("mosaic pattern"), prominent trabecular thickening, and skeletal enlargement (osteomegaly).

Bone affected by Paget's disease is mechanically compromised, architecturally disorganized, highly vascular, and susceptible to bowing deformities, secondary osteoarthritis, and pathological fractures.

---

## 2. Dataset Availability and Sourcing Context

> [!NOTE]
> **DATASET-SOURCE AUDIT STATEMENT**:  
> Our current dataset-source audit did not identify a suitable, publicly accessible and appropriately licensed clinical imaging cohort for Paget's disease within the project's requirements.  
> Web-based educational teaching files (e.g., Radiopaedia, Eurorad) cannot be utilized as bulk clinical training data due to copyright restrictions, selection bias, absence of verified longitudinal clinical metadata, and lossy compression artifacts.  
> Therefore, establishing a verified Paget's disease cohort requires a proposed retrospective hospital-based collaboration under Institutional Ethics Committee (IEC) oversight.

---

## 3. Diagnostic Reference Standards (Requiring Radiologist Review)

Candidate cases must satisfy established clinical and radiological diagnostic criteria, subject to formal review and verification by an appropriate radiologist or clinician:

### Primary Standard: Classic Radiographic Triad
Independent confirmation by a qualified musculoskeletal radiologist of the classic triad:
1. **Cortical Thickening**: Expansion of outer cortical boundaries with indistinct endosteal margins.
2. **Coarsened / Accentuated Trabeculae**: Prominent, thickened trabecular lines conforming to altered mechanical stress.
3. **Osseous Expansion (Osteomegaly)**: True enlargement of the bone contour compared to normal contralateral bone or standard anatomical norms.

### Corroborating Standard: Biochemical & Scintigraphic Markers
- Documented elevation of total serum alkaline phosphatase (ALP) or bone-specific alkaline phosphatase (BSAP) with normal serum calcium and phosphate.
- Where available, whole-body planar $^{99\text{m}}\text{Tc}$-MDP bone scintigraphy showing intense, uniform radiotracer uptake throughout the involved bone.

---

## 4. Anatomical Distribution and Technical Imaging Parameters

- **Primary Skeletal Sites**: Pelvis (~75%), lumbar/thoracic spine (~50%), femur (~35%), skull (~30%), tibia (~25%), and humerus (~15%).
- **Imaging Modality**: Plain Digital Radiography (Direct Radiography [DR] or Computed Radiography [CR]).
- **Image Resolution**: Minimum $1024 \times 1024$ pixels; preferred $\ge 2048 \times 2048$ pixels.
- **Grayscale Depth**: 12-bit or 16-bit DICOM; uncompressed or losslessly compressed.
- **Target Cohort Scale**: Approximately 150 to 250 confirmed human cases across multiple skeletal regions, accompanied by anatomically matched controls.
