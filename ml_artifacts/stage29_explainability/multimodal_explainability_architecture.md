# Multimodal Explainability Architecture & Integration Blueprint

**Document Version**: 1.0.0  
**Project**: OsteoGuard AI  
**Date**: November 16, 2026  
**Status**: **ARCHITECTURAL BLUEPRINT (PRE-CLINICAL)**

---

## 1. Modality-Specific Explanation Mapping

The OsteoGuard AI research proposal specifies a synchronized dual-stream architecture combining macroscopic deep vision representations with high-frequency radiomics and biomechanical metrics.

```mermaid
flowchart TD
    subgraph Pathways ["Modality Ingestion Pathways"]
        X["Pathway A: Plain Radiography (2D)"]
        S["Pathway B: Bone Scintigraphy (2D)"]
        CT["Pathway C: Volumetric CT (3D)"]
        TAB["Pathway D: Late Fusion Tabular (614-D)"]
    end

    subgraph Explainers ["Appropriate Explainability Engine"]
        X --> G2D_A["2D Grad-CAM
(ResNet-101 layer4)"]
        S --> G2D_B["2D Grad-CAM
(ResNet-34 layer4)"]
        CT --> G3D["3D Volumetric Grad-CAM
(3D CNN Backbone)"]
        TAB --> SHAP["Tabular SHAP
(KernelSHAP / TreeSHAP)"]
    end

    subgraph ClinicalUI ["Separated Clinical UI Presentation"]
        G2D_A & G2D_B & G3D --> VIS["Visual Anatomical Heatmap Overlay
(Spatial Localization)"]
        SHAP --> WATERFALL["SHAP Feature Waterfall Plot
(Biomarker Attributions)"]
    end

    classDef path fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef expl fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef ui fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    class X,S,CT,TAB path;
    class G2D_A,G2D_B,G3D,SHAP expl;
    class VIS,WATERFALL ui;
```

---

## 2. Architectural Guidelines & Disclaimers

### 2.1 Strict Separation of Heatmaps and Tabular Scores
- **No Pseudo-Metric Conflation**: Spatial Grad-CAM heatmaps and tabular SHAP feature attributions address fundamentally different mathematical entities. They must be presented in separate UI panels and NEVER collapsed into a single unsupported "confidence score".

### 2.2 Causality vs. Attribution
- **Correlation, Not Causation**: SHAP values indicate how much a feature shifted the model's internal decision boundary relative to a background baseline. They do NOT prove biological or medical causality.

### 2.3 Grad-CAM Resolution Boundaries
- Grad-CAM maps reflect coarse features from deep convolutional layers (e.g. $16 \times 16$ or $7 \times 7$) upsampled to input resolution. They must never be represented to clinicians as precise anatomical segmentation boundaries.
