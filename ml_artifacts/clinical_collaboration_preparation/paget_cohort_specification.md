# Clinical Cohort Acquisition Specification: Paget's Disease of Bone

**Target Class 2**: Paget's Disease of Bone (*Osteitis Deformans*)  
**Document Status**: Formal Cohort Acquisition Specification  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Disease Definition and Pathophysiology

Paget's disease of bone is a chronic metabolic bone disorder characterized by focal areas of accelerated, disordered bone remodeling. The normal coordinated activity of osteoclasts and osteoblasts is disrupted, leading to:
1. **Osteolytic Phase**: Intense focal osteoclastic resorption (e.g., advancing resorption wedge or "blade of grass" sign).
2. **Mixed Phase**: Concurrent osteoclastic resorption and disorganized osteoblastic bone formation.
3. **Osteosclerotic Phase**: Exhausted osteoclastic activity with dense, disorganized, woven bone ("mosaic pattern"), thickened trabeculae, and cortical expansion.

Bone affected by Paget's disease is architecturally chaotic, structurally fragile, highly vascularized, and predisposed to bowing deformities, osteoarthritis, pathological fractures, and rare sarcomatous degeneration (<1%).

---

## 2. Diagnostic Reference Standards

To ensure pristine diagnostic fidelity, candidate subjects must satisfy at least one of the following validated diagnostic reference standards:

### Standard A: Radiographic Triad + Clinical Corroboration (Primary Standard)
Evaluation by $\\ge 2$ board-certified musculoskeletal (MSK) radiologists confirming the classic radiographic triad:
1. **Cortical Thickening**: Expansion of outer cortical boundaries with loss of distinct endosteal margin.
2. **Coarsened / Accentuated Trabeculae**: Thick, sclerotic trabecular struts aligned along lines of mechanical stress.
3. **Osseous Expansion (Osteomegaly)**: True enlargement of the affected bone contour compared to normal contralateral bone or anatomical norms.

### Standard B: Biochemical and Scintigraphic Correlation (Secondary Standard)
- Marked elevation of total serum alkaline phosphatase (ALP) or bone-specific alkaline phosphatase (BSAP) with normal serum calcium and phosphate.
- Whole-body $^{99\\text{m}}\\text{Tc}$-MDP bone scintigraphy showing intense, uniform, geometrically defined tracer uptake throughout the entire anatomic contour of the involved bone.

### Standard C: Histopathological Confirmation (Definitive Standard, where available)
- Bone biopsy demonstrating characteristic mosaic "jigsaw" pattern of woven and lamellar bone separated by prominent, scalloped, irregular cement lines.

---

## 3. Anatomical Distribution and Imaging Modality Requirements

Paget's disease is typically monostotic (~30%) or polyostotic (~70%) with a strong predilection for the axial skeleton:
- **Pelvis**: ~75% of cases (thickened iliopubic/ilioischial lines, teardrop sign, pelvic brim thickening).
- **Lumbar and Thoracic Spine**: ~50% of cases ("picture frame" vertebra, ivory vertebra, vertebral enlargement).
- **Femur**: ~35% of cases (anterolateral bowing, transverse insufficiency fractures on convex cortex).
- **Skull**: ~30% of cases (osteoporosis circumscripta in early phase; "cotton wool" appearance, diploic widening, basilar invagination).
- **Tibia**: ~25% of cases (anterior saber shin bowing, advancing lytic wedge).
- **Humerus and Scapula**: ~15% of cases.

### Imaging Technical Specifications
- **Modality**: Plain Digital Radiography (CR / DR).
- **Matrix Resolution**: Minimum $1024 \\times 1024$, preferred $2048 \\times 2048$ or higher.
- **Grayscale Depth**: 12-bit or 16-bit DICOM.
- **Standard Projections**: AP and Lateral of long bones; AP Pelvis; AP and Lateral Spine; Townes and Lateral Skull.

---

## 4. Cohort Volume Targets and Demographics

- **Target Patient Volume**: 150 to 250 verified human patients (yielding approximately 400 to 700 radiograph views).
- **Age Profile**: Typical adult population aged $\\ge 50$ years (Paget's disease is exceedingly rare below age 40).
- **Sex Distribution**: Mild male predominance (~3:2 male-to-female ratio).
- **Matched Controls**: Must include anatomically matched normal radiographs from an age-matched cohort ($\ge 50$ years) to prevent models from learning age-related osteoarthritic confounders instead of pagetic remodeling.

---

## 5. Absolute Exclusion of Educational Web Harvesting

> [!WARNING]
> **STRICT PROHIBITION ON EDUCATIONAL WEB HARVESTING**:  
> Web repositories such as Radiopaedia, Eurorad, and online teaching files are **STRICTLY PROHIBITED** from bulk ML cohort ingestion:  
> 1. **Terms of Service**: Radiopaedia Terms of Reference explicitly prohibit automated scraping, bulk extraction, and training of commercial or non-commercial automated models without bespoke negotiated licensing.  
> 2. **Selection and Presentation Bias**: Cases published online represent extreme, visually dramatic teaching anomalies, leading to severe distribution mismatch with routine hospital imaging.  
> 3. **Image Degradation**: Online assets are lossy JPEG screen captures with compressed 8-bit dynamic range, non-linear gamma curves, stripped DICOM tags, and burned-in presentation annotations.  
> 4. **Lack of Patient Provenance**: Online cases lack longitudinal patient IDs, multi-view parity, laboratory markers, and negative control counterparts.  
> *Acquisition must occur exclusively via institutional PACS extraction under research ethics oversight.*
