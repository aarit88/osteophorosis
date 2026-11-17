# Clinical Cohort Acquisition Specification (Revised): Osteogenesis Imperfecta (OI)

**Target Class 3**: Osteogenesis Imperfecta (Human Clinical)  
**Document Status**: Formal Cohort Acquisition Specification (Revised Draft)  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Clinical Definition and Molecular Genetics

Osteogenesis imperfecta (OI, "Brittle Bone Disease") is a heterogeneous group of genetic connective tissue disorders characterized by bone fragility, reduced bone mineral density, recurrent low-trauma fractures, and skeletal deformities.

Approximately 85–90% of individuals have autosomal dominant pathogenic variants in *COL1A1* or *COL1A2*, which encode type I procollagen. Additional recessive and X-linked forms affect collagen biosynthesis, post-translational modification, folding, and osteoblast differentiation.

Bone tissue in OI demonstrates thin cortical borders, reduced trabecular volume, altered osteocyte density, and elevated mineralization heterogeneity, predisposing to recurrent skeletal injury.

---

## 2. Published Research Cohort Evidence and Limitations (PMC11324408)

### 2.1 Study Evidence Summary
A major published study investigating human pediatric OI imaging is:
- **Reference**: *Bone*, 2024;185:117192 (PMC11324408).
- **Consortium**: Rare Diseases Clinical Research Network (RDCRN) Brittle Bone Disorders Consortium (BBDC Protocol 7701).
- **Cohort Composition**:
  - Baseline: 159 pediatric participants with clinical/genetic OI.
  - Follow-up: 55 participants with repeated examinations approximately 24 months later.
  - Phenotypes: Sillence Types I, III, and IV.
  - **Anatomical Coverage**: Left hand and wrist radiographs evaluated for automated bone age and bone mineral density assessment.

### 2.2 Sourcing and Scope Limitations
- **Candidate Status**: The published study documents the existence of this research cohort; however, **no data access has been granted merely by citing the paper**. Access requires formal application to and approval by the RDCRN BBDC Data Access Committee.
- **Anatomical Boundary**: The cohort is **strictly limited to hand and wrist bone-age imaging**. It cannot automatically represent whole-skeleton manifestations, femoral bowing, tibial deformities, or vertebral compression fractures.
- **Pediatric Controls**: Analysis of pediatric radiographs requires age- and growth-plate-matched controls to prevent automated models from confounding normal open epiphyseal plates with radiolucent fracture lines or lytic lesions.

---

## 3. Diagnostic Reference Standards

Candidate OI cases must satisfy one or more of the following clinical standards:
1. **Molecular Confirmation**: Documented pathogenic or likely pathogenic variant in *COL1A1*, *COL1A2*, or recognized recessive OI genes.
2. **Clinical Phenotype (Sillence Criteria)**: Verified diagnosis by a pediatric geneticist or endocrinologist confirming Sillence Type I (mild), Type III (progressively deforming), or Type IV (moderate).
3. **Clinical Fragility History**: Documented history of recurrent low-trauma fragility fractures, supported by clinical features (blue sclerae, dentinogenesis imperfecta, hearing impairment, or DXA $Z$-score $\le -2.0$).

---

## 4. Absolute Disqualification of Animal Models

> [!CAUTION]
> **STRICT REJECTION OF ANIMAL MICRO-CT DATA**:  
> Candidate datasets identified in earlier audits containing benchtop micro-CT ($\mu\text{CT}$) scans of murine (*oim* mouse) bones (e.g., Figshare DOI `10.6084/m9.figshare.c.3795019`) are **STRICTLY DISQUALIFIED**:  
> - Murine rodent bones lack Haversian remodeling systems and differ substantially in cortical dimensions and trabecular morphology.  
> - Benchtop 3D $\mu\text{CT}$ reconstructions and surface meshes bear zero physical or mathematical equivalence to human 2D projection radiographs.  
> - **Mandatory Rule**: Only ethically approved, verified human clinical radiographs may be considered for project intake.
