# Modality-Specific Architecture Plan: Decoupled Multi-Pathway Deep Learning

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Technical Architecture Specification  
**Author**: OsteoGuard AI Deep Learning Architecture Group  

---

## 1. Physical Sensor Incompatibility and the Multi-Pathway Rationale

In conventional computer vision, multi-modal tasks often utilize shared early feature encoders. In medical imaging of bone pathology, this approach is **fundamentally invalid** due to distinct physical signal generation mechanisms:

1. **Plain Digital Radiography (X-Ray)**: Transmission imaging measuring the line integral of X-ray attenuation through 3D anatomical structures projected onto a 2D detector plane. Governed by Compton scattering and photoelectric effect.
2. **Planar Bone Scintigraphy**: Emission imaging measuring gamma photons (140 keV) emitted by osteotropic radiopharmaceuticals ($^{99\text{m}}\text{Tc}$-MDP) adsorbed onto hydroxyapatite crystal surfaces, reflecting osteoblast metabolic activity rather than anatomical density.
3. **Computed Tomography (CT)**: 3D volumetric cross-sectional reconstruction of calibrated linear attenuation coefficients expressed in Hounsfield Units (HU).

> [!CRITICAL]
> **ARCHITECTURAL MANDATE**:  
> Forcing structurally and physically incompatible imaging modalities into a single shared deep convolutional or Vision Transformer image encoder is **STRICTLY REJECTED**.  
> OsteoGuard AI enforces a **Decoupled Multi-Pathway Architecture** with clinical late fusion.

---

## 2. Decoupled Pipeline Specifications

```
                             +-----------------------------------+
                             |     RAW PATIENT IMAGING DATA      |
                             +-----------------+-----------------+
                                               |
         +-------------------------------------+-------------------------------------+
         |                                     |                                     |
         v                                     v                                     v
+------------------+                 +--------------------+                +-------------------+
|  PATHWAY A: X-RAY|                 |PATHWAY B: SCINTIGR.|                |   PATHWAY C: CT   |
| (Plain Radiograph|                 | (Planar Bone Scan) |                |  (Volumetric CT)  |
+------------------+                 +--------------------+                +-------------------+
| - Backbone:      |                 | - Dual-Stream View |                | - 3D-UNet / 3D-   |
|   DenseNet-121 / |                 |   Fusion (Ant/Post)|                |   ResNet Backbone |
|   ConvNeXt-S     |                 | - Backbone:        |                | - Vertebral Mask  |
| - Pretrained on  |                 |   ResNet-34        |                |   Segmentation    |
|   Musculoskel.   |                 | - Cross-Attention  |                | - Volumetric HU   |
| - Target: Paget, |                 | - Target: Secondary|                |   Texture         |
|   OI, Controls,  |                 |   Metastases,      |                | - Target: Spinal  |
|   Lytic Lesions  |                 |   Metabolic Normal |                |   Metastases      |
+--------+---------+                 +---------+----------+                +---------+---------+
         |                                     |                                     |
         | Feature Vector f_A                  | Feature Vector f_B                  | Feature Vector f_C
         v                                     v                                     v
+------------------+                 +--------------------+                +-------------------+
| Softmax Prob. p_A|                 | Softmax Prob. p_B  |                | Softmax Prob. p_C |
+--------+---------+                 +---------+----------+                +---------+---------+
         |                                     |                                     |
         +-------------------------------------+-------------------------------------+
                                               |
                                               v
                             +-----------------------------------+
                             |     CLINICAL COVARIATE VECTOR     |
                             | (Age, Sex, Site, Serum ALP, etc.) |
                             +-----------------+-----------------+
                                               |
                                               v
                             +-----------------------------------+
                             |    CLINICAL LATE-FUSION LAYER     |
                             |  (Calibrated Bayesian Meta-Model  |
                             |   / Gradient-Boosted Gating)      |
                             +-----------------+-----------------+
                                               |
                                               v
                             +-----------------------------------+
                             | FINAL FOUR-CLASS POSTERIOR PROB.  |
                             |  + BIOMECHANICAL FRACTURE RISK    |
                             +-----------------------------------+
```

---

## 3. Pathway Details

### Pathway A: Plain Radiography Pathway
- **Input**: High-resolution 2D radiograph ($1024 \\times 1024$ or $512 \\times 512$ ROI).
- **Network Architecture**: DenseNet-121 or ConvNeXt-Small with feature pyramid pooling.
- **Pretraining**: Transfer learning from ImageNet-1k followed by musculoskeletal radiography domain adaptation (Stanford MURA).
- **Output**: Feature embedding $\mathbf{f}_A \in \mathbb{R}^{512}$ and class posterior $\mathbf{p}_A$.

### Pathway B: Planar Bone Scintigraphy Pathway
- **Input**: Paired Anterior and Posterior whole-body planar scans ($1024 \\times 256$ each).
- **Network Architecture**: Siamese dual-stream ResNet-34 with intermediate cross-view multi-head attention and channel concatenation (established in Stage 13).
- **Pretraining**: Validated on IICS-UNA breast cancer metastasis cohort.
- **Output**: Feature embedding $\mathbf{f}_B \in \mathbb{R}^{512}$ and class posterior $\mathbf{p}_B$.

### Pathway C: Volumetric CT Pathway (Contingent on TCIA Acquisition)
- **Input**: 3D axial CT series reconstructed with bone kernel ($\le 1.5\text{ mm}$ slice thickness).
- **Network Architecture**: 3D ResNet-50 / 3D DenseNet with automated spinal vertebral segmentation masking.
- **Output**: Volumetric radiomic vector $\mathbf{f}_C \in \mathbb{R}^{256}$ and class posterior $\mathbf{p}_C$.

---

## 4. Clinical Late-Fusion Architecture

The integration layer operates on the decoupled modality outputs without backpropagating raw pixel gradients across modalities:
$$\mathbf{p}_{\text{final}} = \mathcal{G}\left(\mathbf{p}_A, \mathbf{p}_B, \mathbf{p}_C, \mathbf{x}_{\text{clinical}}\right)$$
Where:
- $\mathbf{p}_A, \mathbf{p}_B, \mathbf{p}_C$ represent the modality-specific posterior probability distributions.
- $\mathbf{x}_{\text{clinical}}$ represents standardized clinical covariates (Patient Age, Biological Sex, Anatomical Site, Serum Alkaline Phosphatase, Prior Fracture Count).
- $\mathcal{G}$ is a calibrated meta-learner (regularized multinomial logistic regression or LightGBM decision tree ensemble with Platt scaling / temperature scaling).

### Missing Modality Handling
In clinical reality, a patient rarely undergoes plain X-ray, scintigraphy, and CT simultaneously. The late-fusion layer naturally supports missing modalities by applying conditional Bayesian marginalization:
$$P(y = c \mid \mathcal{M}_{\text{available}}, \mathbf{x}) \propto P(y = c) \prod_{m \in \mathcal{M}_{\text{available}}} \frac{P(y = c \mid \mathbf{z}_m)}{P(y = c)}$$
This guarantees that OsteoGuard AI can operate robustly whether a patient presents with an X-ray alone, a bone scan alone, or a comprehensive multi-modal workup.

---

## 5. Implementation Status Disclaimer
*This architectural blueprint serves as a research planning and integration roadmap. No model training, fine-tuning, or architectural coding has been executed in Stage 24.*
