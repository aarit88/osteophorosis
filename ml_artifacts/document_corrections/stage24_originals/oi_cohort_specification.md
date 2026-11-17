# Clinical Cohort Acquisition Specification: Osteogenesis Imperfecta (OI)

**Target Class 3**: Osteogenesis Imperfecta (Human Clinical)  
**Document Status**: Formal Cohort Acquisition Specification  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Disease Definition and Pathophysiology

Osteogenesis imperfecta (OI, "Brittle Bone Disease") is a genetically heterogeneous group of connective tissue disorders characterized by bone fragility, recurrent low-trauma fractures, skeletal deformities, and growth deficiency. 

In approximately 85–90% of individuals, OI is caused by autosomal dominant pathogenic variants in the *COL1A1* or *COL1A2* genes encoding the pro-$\\alpha 1(\\text{I})$ and pro-$\\alpha 2(\\text{I})$ chains of type I collagen. Autosomal recessive and X-linked forms involve defects in collagen post-translational modification, folding, intracellular transport, and mineralization (e.g., *CRTAP*, *P3H1*, *PPIB*, *SERPINF1*, *IFITM5*).

The resultant bone matrix exhibits reduced bone volume, severe cortical thinning, disordered trabecular microarchitecture, and hypermineralized but structurally fragile bone tissue.

---

## 2. Diagnostic Reference Standards

Candidate subjects must have a documented diagnosis of human OI confirmed by one or more of the following clinical standards:

### Standard A: Molecular Genetic Confirmation (Gold Standard)
- Targeted gene panel, whole-exome sequencing (WES), or Sanger sequencing identifying a pathogenic or likely pathogenic variant in *COL1A1*, *COL1A2*, or recognized recessive OI loci.

### Standard B: Validated Clinical Consensus (Sillence Criteria)
Classification by a pediatric endocrinologist, geneticist, or pediatric orthopaedic surgeon into the classic Sillence phenotypic spectrum:
- **Type I (Mild)**: Generalized osteopenia, recurrent fractures in childhood decreasing post-puberty, blue sclerae, normal or near-normal stature, absence of major bone deformity.
- **Type II (Perinatal Lethal)**: Extreme bone fragility, crumpled long bones, beaded ribs, severe calvarial unmineralization. *(Generally neonatal imaging; excluded unless dedicated infant sub-model is trained)*.
- **Type III (Severe Progressively Deforming)**: Severe short stature, multiple recurrent fractures occurring with minimal trauma, progressive limb bowing, scoliosis/kyphosis, dentinogenesis imperfecta, triangular facies.
- **Type IV (Moderate)**: Mild-to-moderate skeletal deformities, variable short stature, normal sclerae, recurrent fragility fractures.

### Standard C: Biochemical / Densitometric Corroboration
- Marked reduction in bone mineral density (BMD) assessed via Dual-Energy X-ray Absorptiometry (DXA) with pediatric $Z$-scores $\\le -2.0$, accompanied by documented fragility fracture history.

---

## 3. Radiographic Phenotype and Imaging Requirements

### Characteristic Radiographic Manifestations
Plain skeletal radiographs of OI patients exhibit distinct phenotypic features:
1. **Cortical Thinning**: "Pencil-thin" cortical borders of long bone diaphyses.
2. **Generalized Osteopenia**: Diffuse radiolucency with prominent, sparse trabecular outlines.
3. **Bowing Deformities**: Multi-planar bowing of the femur (anterior/lateral) and tibia (saber-shin deformity).
4. **Fracture Sequelae**: Multiple fractures in varying stages of healing, abundant hyperplastic callus formation, and pseudoarthroses.
5. **Epiphyseal/Metaphyseal Irregularities**: "Popcorn" calcifications at the growth plates (especially in Type III/IV).
6. **Axial Deformities**: Codfish/biconcave vertebrae, platyspondyly, basilar invagination, and progressive thoracic kyphoscoliosis.
7. **Skull Findings**: Platybasia, delayed suture closure, and numerous Wormian bones along cranial sutures.

### Imaging Technical Specifications
- **Modality**: Plain Digital Radiography (CR / DR).
- **Projections**: Skeletal surveys including AP/Lateral bilateral femur, tibia, humerus, forearm; AP Pelvis; AP and Lateral Spine; AP/Lateral Skull.
- **Resolution**: $\\ge 1024 \\times 1024$ pixels; DICOM 12-16 bit format.
- **Radiation Safety**: Strict ALARA (As Low As Reasonably Achievable) adherence; pediatric imaging protocols must be verified.

---

## 4. Cohort Volume Targets and Demographics

- **Target Patient Volume**: 100 to 200 confirmed human patients (yielding 500 to 1,200 radiographic views).
- **Age Distribution**: 
  - Pediatric Cohort: Age 1 to 17 years (must be stratified by skeletal age and growth plate status: open vs. fusing vs. closed).
  - Adult Cohort: Age $\ge 18$ years.
- **Matched Pediatric Controls**: Non-OI pediatric trauma radiographs with normal bone density, matched by age group and growth plate maturation status, to prevent the machine learning model from confounding open growth plates (epiphyseal lines) with pathological fractures or radiolucent lesions.

---

## 5. Absolute Exclusion of Animal Micro-CT Models

> [!CAUTION]
> **ABSOLUTE DISQUALIFICATION OF ANIMAL MODELS**:  
> In Stage 23, candidate online OI datasets (e.g., Figshare DOI `10.6084/m9.figshare.c.3795019`) were forensically investigated and confirmed to consist of **murine (oim mouse) tibiae scanned via benchtop micro-CT ($\\mu\\text{CT}$)** at nominal resolutions of $5\\text{ }\\mu\\text{m}$ to $10\\text{ }\\mu\\text{m}$.  
> - **Morphological Divergence**: Murine rodent bone lacks a true Haversian remodeling system, differs in cortical thickness by orders of magnitude, and exhibits distinct trabecular architecture.  
> - **Signal / Acquisition Divergence**: High-resolution synchrotron or benchtop $\\mu\\text{CT}$ reconstructions represented as 3D surface meshes (`.ply`) or isotropic 8-bit slices bear zero physical or mathematical resemblance to clinical human 2D projection radiographs.  
> - **Clinical Validity Requirement**: Any and all animal imaging data is **STRICTLY DISQUALIFIED**. Only human clinical diagnostic radiographs shall be acquired and ingested.
