# Application Architecture Audit: End-to-End System Integration

## 1. Executive Architecture Summary
OsteoGuard AI is an experimental AI research prototype designed around the multi-modal differential diagnosis and biomechanical fracture risk stratification proposal submitted to the Karnataka State Council for Science and Technology (KSCST).

Across Stages 1 through 30, individual mathematical, radiomic, biomechanical, explainability, and uncertainty engines were designed and validated on controlled synthetic digital phantoms. Stage 31 executes the end-to-end integration of these modular components into a demonstrable, auditable full-stack application (FastAPI backend + React/Vite/TypeScript frontend).

Crucially, **the four-class clinical diagnostic model is NOT READY** because verified, ethics-approved cohorts for Paget's disease, human Osteogenesis Imperfecta (OI), and matched healthy controls have not been acquired. The application is built honestly around this limitation: the synthetic workbench is active for engineering verification, but four-class diagnostic inference is strictly disabled behind an immutable HTTP 503 `BLOCKED` safety gate.

---

## 2. Full-Stack System Topology

```
+----------------------------------------------------------------------------------------------------+
|                                    REACT / VITE CLIENT (PORT 3000)                                 |
|                                                                                                    |
|  [Analysis & Upload]    [Synthetic Workbench]    [Research Tracks]    [Feature Explorer]           |
|  [Biomechanics Engine]  [Explainability Hub]    [Uncertainty Triage]  [Dataset Governance]         |
|  [Evidence Registry]    [Safety Gate Testing]    [Audit Log (Art 14)] [Model Architecture]         |
+---------------------------------------------------+------------------------------------------------+
                                                    |
                                                    | REST API (HTTP Proxy /api -> :8000)
                                                    v
+----------------------------------------------------------------------------------------------------+
|                                    FASTAPI BACKEND SERVICE (PORT 8000)                             |
|                                                                                                    |
|  API Gateway & Middleware:                                                                         |
|    - CORS & Security Headers                                                                       |
|    - Structured Request Validation & Pydantic V2 Schemas                                           |
|    - RFC 7807 Error Handling & Audit Logging                                                       |
|                                                                                                    |
|  Research Routes & Services (/api/v1/research/*, /api/v1/*):                                       |
|    +-----------------------------+-----------------------------+-----------------------------+     |
|    | Track Separation Controller | Historical Evidence Registry| Clinical Governance Ledger  |     |
|    +-----------------------------+-----------------------------+-----------------------------+     |
|    | Radiomics Feature Engine    | Biomechanics Geometry Engine| Conventional Mirels Scorer  |     |
|    | (81 Verified / 21 Excluded) | (2D Moments Ixx, Iyy, J)    | (Classical 1989 Rule Set)   |     |
|    +-----------------------------+-----------------------------+-----------------------------+     |
|    | Grad-CAM Visualizer (conv1) | Exact Shapley Calculator    | Uncertainty Triage Engine   |     |
|    | (Spatial CNN Fixture)       | (Low-Dim Synthetic Tests)   | (Tri-State H_norm >= 0.60)  |     |
|    +-----------------------------+-----------------------------+-----------------------------+     |
|    | IMMUTABLE FOUR-CLASS SAFETY GATE: POST /api/v1/diagnostic/four-class -> HTTP 503 BLOCKED     |     |
+---------------------------------------------------+------------------------------------------------+
                                                    |
                                                    | Cryptographic Hash Verification
                                                    v
+----------------------------------------------------------------------------------------------------+
|                                    ARTIFACTS & STORAGE SUB-SYSTEM                                  |
|                                                                                                    |
|  Frozen Primary Checkpoint:                                                                        |
|    - Path: ml_artifacts/btxrd_frozen_model/best_checkpoint.pt                                      |
|    - SHA-256: 21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD (IMMUTABLE)        |
|  Burned Holdouts (Preserved):                                                                      |
|    - BTXRD Center 2 External Test Partition (Stage 19)                                             |
|    - BTXRD Center 3 External Test Partition (Stage 20)                                             |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Component Architecture Register

| Component | Layer / Subsystem | Implementation Status | Implementation Details | Known Gaps / Disclosures |
|---|---|---|---|---|
| **Data Intake & Preprocessing** | Pipeline / Loader | PARTIALLY_IMPLEMENTED | Synthetic digital phantom generators (checkerboard, gradient, annulus, uniform) | Volumetric CT bone window loaders unverified on clinical series |
| **IBSI Radiomics Feature Engine** | Feature Extraction | PARTIALLY_IMPLEMENTED | 81 verified IBSI features (18 First Order, 26 GLCM, 16 GLRLM, 16 GLSZM, 5 GLDM) | 21 heuristic features unverified; 102-D spec marked NOT VERIFIED |
| **Cross-Sectional Biomechanics** | Physical Modeling | IMPLEMENTED | 2D area, centroid, moments of inertia ($I_{xx}, I_{yy}$), polar moment $J$ | Evaluated on 2D synthetic phantoms; does NOT compute clinical fracture risk |
| **Conventional Mirels Scorer** | Clinical Decision Support | IMPLEMENTED | Classical 4-parameter scoring (site, pain, nature, size) with risk stratification | Pure empirical scoring rule (1989); no automated segmentation inputs |
| **Grad-CAM Explainability** | Model Interpretability | IMPLEMENTED | Production hook manager, spatial localization verified on `ControlledSpatialCNN` | Verified on synthetic fixture; does NOT explain clinical patient radiographs |
| **SHAP Radiomic Explainability** | Feature Attribution | PARTIALLY_IMPLEMENTED | Exact combinatorial Shapley calculator verified on low-dimensional tests | Full DeepSHAP/KernelSHAP for high-dimensional clinical CNNs unverified |
| **Uncertainty & Selective Prediction** | Safety / Triage Gate | IMPLEMENTED | Normalized Shannon entropy $H_{\text{norm}}$, margin confidence, tri-state triage | Candidate threshold $H_{\text{norm}} \ge 0.60$ is an engineering heuristic |
| **Multi-Modal Late Fusion (614-D)** | Deep Learning Architecture | ARCHITECTURALLY_SPECIFIED | 512-D DenseNet CNN + 102-D Radiomics concatenation contract | Scientifically NOT VERIFIED; zero clinical weights trained |
| **Four-Class Output Classifier Head**| Diagnostic Inference | BLOCKED_BY_DATA | Output schema defined (metastasis, paget, oi, healthy) | STRICTLY DISABLED. Returns HTTP 503 SERVICE UNAVAILABLE |

---

## 4. Architectural Boundaries and Safeguards
1. **Separation of Concerns**: Sub-track models (FracAtlas fracture detection, bone scintigraphy metastasis screening, BTXRD primary bone tumour classification) are maintained in strictly isolated registries. They are never conflated with or mapped to the final four-class differential diagnosis.
2. **Deterministic Reproducibility**: All synthetic tests utilize fixed pseudo-random seeds (`seed=42`) and analytical ground truths.
3. **No Patient Data Ingestion**: Zero clinical patient datasets are accessed during application execution. All active demos run on deterministic digital fixtures.

