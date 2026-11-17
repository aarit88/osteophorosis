# OsteoGuard AI — Unified Deep Learning & Radiomic Framework for Differential Bone Diagnosis and Fracture Risk Stratification

[![Research Prototype](https://img.shields.io/badge/status-Research%20Prototype-red.svg)](#)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2Bcu126-orange.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](#)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue.svg)](#)
[![React 18](https://img.shields.io/badge/React-18.3-61dafb.svg)](#)
[![Vite 6](https://img.shields.io/badge/Vite-6.2-purple.svg)](#)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-38bdf8.svg)](#)
[![Tests Passing](https://img.shields.io/badge/tests-108%20passed%20(pytest)-success.svg)](#)

> [!CAUTION]
> **RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL USE — ZERO REAL-TIME PATIENT DATA**  
> Under institutional governance, clinical four-class diagnostic inference is permanently disabled behind an immutable **HTTP 503 `BLOCKED` safety gate** (`FOUR_CLASS_MODEL_NOT_READY`). Open-access human imaging datasets for Paget's disease of bone, human osteogenesis imperfecta (OI), and matched healthy skeletal controls have not been acquired. The system operates strictly as an engineering research prototype evaluating deterministic digital phantoms and historical research tracks.

---

## 🌟 Overview & System Architecture

OsteoGuard AI integrates deep multi-modal feature representations, IBSI-compliant radiomic texture analysis, cross-sectional biomechanical fracture-risk geometry, and selective prediction (uncertainty-based abstention) into a single unified research platform.

```
┌────────────────────────────────────────────────────────────────────────┐
│               OsteoGuard AI Full-Stack Research Prototype              │
├──────────────────────────────────┬─────────────────────────────────────┤
│  Frontend Client (Port 3000)     │  FastAPI ML Service (Port 8000)     │
│  React 18 + Vite 6 + Tailwind    │  PyTorch + Scipy + NumPy + Pydantic │
│  • Synthetic Research Workbench  │  • Radiomics Engine (81 IBSI feats) │
│  • 4 Research Tracks Explorer    │  • Biomechanical Geometry (I_xx, J) │
│  • Radiomics & Biomechanics UI   │  • Mirels Impending Fracture Scorer │
│  • Explainability Heatmaps (CAM) │  • Grad-CAM Heatmap Generator       │
│  • Uncertainty & Abstention Deck │  • Shannon Entropy Abstention Gate  │
│  • Model Evidence Registry       │  • Synthetic Demonstration Pipeline │
│  • Architecture & Safety Gate UI │  • HTTP 503 Blocked Diagnostic Gate │
└──────────────────────────────────┴─────────────────────────────────────┘
```

---

## 🔬 Four Discrete Research Tracks

The project rigorously separates historical empirical experiments from proposal-aligned future diagnostic targets:

| Track | Dataset & Modality | Architecture | Key Empirical / Scientific Finding | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Track A: Fracture Detection** | FracAtlas ($N=614$ Holdout)<br>Plain Radiography | ResNet-101<br>(42.5M params) | ROC-AUC: **0.9117**, PR-AUC: **0.7537**.<br>At screening threshold $\theta=0.31$: Sensitivity **90.00%**, Specificity **74.40%**, Balanced Acc **82.20%**. | **FROZEN** (Stage 10) |
| **Track B: Bone Scintigraphy** | UNA Paraguay ($N=44$ Holdout / 88 scans)<br>Whole-Body Bone Scan | 5-Fold ResNet-18 Ensemble<br>+ Clinical Max-Pooling | ROC-AUC: **0.8317**, PR-AUC: **0.7257**.<br>Accuracy: **84.09%**, Specificity: **88.57%**, Sensitivity: **66.67%** ($\theta=0.60$) / **77.78%** ($\theta=0.45$). | **FROZEN** (Stage 14) |
| **Track C: Multi-Center Shift** | BTXRD Multi-Center ($N=1,396$ total)<br>Center 1 ($N=588$), C2 ($N=549$), C3 ($N=259$) | ResNet-101 Pretrained<br>Frozen Checkpoint (`21064670...`) | **Severe multi-center domain collapse documented**:<br>• Center 1 Val Acc: **81.80%**, Macro-F1: **0.8103**<br>• Center 2 Ext Acc: **51.73%**, Macro-F1: **0.5069**<br>• Center 3 Stress Acc: **47.10%**, Macro-F1: **0.3592** (Normal class sensitivity collapsed to **18.52%**). | **AUDITED & FROZEN** (Stage 18-21) |
| **Track D: Four-Class Unified** | Intended: Metastasis, Paget's, OI, Healthy<br>Radiographs, CT, Scintigraphy | Modular Late Fusion<br>(512-D Deep + 102-D Radiomics) | **BLOCKED BY CLINICAL DATA**:<br>Paget's disease, human OI, and matched controls are unavailable. Four-class inference permanently returns **HTTP 503 `BLOCKED`**. | **ENGINEERING VERIFIED; CLINICAL MODEL BLOCKED** |

---

## ⚙️ Core Engineering Engines (`osteoguard/`)

1. **Proposal-Aligned Radiomics Engine (`osteoguard/radiomics/`)**:
   - 102-feature extraction pipeline adhering to Image Biomarker Standardisation Initiative (IBSI) benchmarks.
   - 81 features mathematically certified on synthetic digital phantoms (first-order intensity, GLCM, GLRLM, GLSZM, GLDM, NGTDM).
   - 21 non-IBSI heuristic features isolated and flagged in telemetry.

2. **Biomechanical Fracture-Risk Engine (`osteoguard/biomechanics/`)**:
   - Cross-sectional bone geometry evaluation calculating cortical area ($A$), second moments of area ($I_{xx}$, $I_{yy}$), product moment ($I_{xy}$), and polar moment ($J = I_{xx} + I_{yy}$) using deterministic green's theorem cross-sections.
   - Validated Mirels scoring module for metastatic long-bone lesions (Scores 4–12; Score $\ge 9$ triggers high-risk prophylactic fixation recommendation).

3. **Explainability Engine (`osteoguard/explainability/`)**:
   - Grad-CAM activation mapping with spatial resolution verification and synthetic perturbation sensitivity analysis.
   - Global SHAP feature attribution audit framework for radiomic late-fusion vectors.

4. **Uncertainty & Selective Prediction Engine (`osteoguard/uncertainty/`)**:
   - Information-theoretic normalized Shannon entropy ($H_{\text{norm}} = -\sum p_i \ln p_i / \ln K \in [0, 1]$).
   - Tri-state selective prediction triage: `ACCEPT_PREDICTION`, `BORDERLINE_UNCERTAIN`, and `ABSTAIN_HUMAN_REVIEW_REQUIRED` (triggered when $H_{\text{norm}} \ge 0.60$ or OOD detected).

---

## 🔒 Four-Class Diagnostic Safety Gate

Active diagnostic inference (`POST /api/v1/diagnostic/four-class`) strictly returns **HTTP 503 SERVICE UNAVAILABLE**:

```json
{
  "error_code": "FOUR_CLASS_MODEL_NOT_READY",
  "detail": "Clinical diagnostic inference is disabled. The OsteoGuard four-class diagnostic model cannot be deployed or executed because approved clinical datasets for Paget's disease, human osteogenesis imperfecta (OI), and matched healthy controls have not been acquired.",
  "details": {
    "status": "BLOCKED",
    "error_code": "FOUR_CLASS_MODEL_NOT_READY",
    "target_classes": [
      "Secondary bone metastasis",
      "Paget's disease of bone",
      "Osteogenesis imperfecta (OI)",
      "Healthy skeletal control"
    ],
    "blockers": [
      "Unavailability of open-access human Paget's disease radiographs",
      "Human osteogenesis imperfecta datasets pending institutional ethics and data transfer agreements",
      "Matched healthy skeletal controls pending clinical site harmonization",
      "Unresolved 21 engineering radiomics features (only 81/102 IBSI features verified)",
      "Absence of institutional review board (IRB/IEC) approved clinical validation protocol"
    ]
  }
}
```

---

## 🚀 Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Node.js 18+ (Tested on Node.js 22)
- npm 9+

### 1. Launch Backend ML Service (FastAPI)
```powershell
# From project root
$env:PYTHONPATH = "."
python -m uvicorn ml_service.app.main:app --host 127.0.0.1 --port 8000
```
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Probe: `http://127.0.0.1:8000/health`
- Research Status: `http://127.0.0.1:8000/api/v1/research/status`

### 2. Launch Frontend Research Prototype (React + Vite)
```powershell
# From osteophorosis-main/ directory
cd osteophorosis-main
npm run dev -- --host 127.0.0.1 --port 3000
```
- Web Application: `http://127.0.0.1:3000`
- Synthetic Workbench: `http://127.0.0.1:3000/workbench`
- Research Tracks: `http://127.0.0.1:3000/tracks`
- Safety Gate Testing: `http://127.0.0.1:3000/architecture`
- Full-Stack Swagger UI: `http://127.0.0.1:3000/docs`

### 3. Run Automated Test Suites
```powershell
# Run Stage 31 Integration Test Suite (14 tests)
pytest ml_service/tests/test_stage_31_application_integration.py -v

# Run Stages 25 to 31 Full Regression Suite (108 tests)
pytest ml_service/tests/test_stage_25_dataset_intake.py ml_service/tests/test_stage_26_advisor_review.py ml_service/tests/test_stage_27_radiomics.py ml_service/tests/test_stage_28_biomechanics.py ml_service/tests/test_stage_29_explainability.py ml_service/tests/test_stage_30_uncertainty.py ml_service/tests/test_stage_31_application_integration.py -q

# Run Frontend Vitest Suite (6 tests)
cd osteophorosis-main
npm run test

# Build Frontend Production Bundle
npm run build
```

---

## 📂 Repository Structure

```
osteophorosis/
├── osteophorosis-main/           # React 18 + Vite 6 Frontend Application
│   ├── src/
│   │   ├── pages/                # Routed views: Workbench, Tracks, Radiomics, Biomechanics, etc.
│   │   ├── components/           # UI components: gauges, heatmaps, disclaimer banners
│   │   ├── context/              # Analysis and persistent state context
│   │   └── api/                  # API clients and TypeScript interfaces
│   ├── tests/                    # Vitest test suites (error, upload, results)
│   ├── server.ts                 # Express + Vite middleware server with backend reverse proxy
│   ├── openapi.yaml              # OpenAPI 3.1 specification
│   └── package.json              # Frontend dependencies and npm scripts
├── osteoguard/                   # Core Proposal-Aligned Mathematical Packages
│   ├── radiomics/                # IBSI-certified texture & feature extraction (102-D)
│   ├── biomechanics/             # Cross-sectional moment calculations & Mirels scoring
│   ├── explainability/           # Grad-CAM heatmap generator & SHAP audit framework
│   └── uncertainty/              # Normalized Shannon entropy & selective abstention gate
├── ml_service/                   # FastAPI Backend Application & Evaluation Suites
│   ├── app/
│   │   ├── routes/               # Modular REST endpoints (/api/v1/*)
│   │   ├── services/             # Research, workbench, and demo service providers
│   │   └── main.py               # Application factory, CORS, exception handlers
│   ├── evaluation/               # Multicenter, CV, holdout, and forensic evaluators
│   ├── models/                   # Neural network architectures (ResNet-101, ResNet-18)
│   └── tests/                    # Comprehensive pytest test suites (Stages 01–31)
├── ml_artifacts/                 # Authoritative Evidence, Checkpoints & Audit Logs
│   ├── stage_reports/            # Authoritative markdown audit reports (Stages 01–31)
│   ├── stage31_application_integration/ # Model evidence registry & track manifests
│   └── btxrd_frozen_model/       # Frozen model checkpoint metadata & SHA-256 hash
├── docs/                         # Clinical protocols, ML architecture plans, forensic audits
├── datasets/                     # Reference archives
├── pyproject.toml                # Python packaging specification
├── requirements-gpu-lock.txt     # Locked production Python environment
├── .gitignore                    # Medical data, weights & cache exclusions
└── README.md                     # Project documentation & release guide
```

---

## ⚖️ Ethical Governance & Medical Compliance

- **EU AI Act Article 14**: System enforces human-in-the-loop oversight through selective prediction abstention ($H_{\text{norm}} \ge 0.60$) and persistent provenance logging.
- **FDA Good Machine Learning Practice (GMLP)**: Historical multi-center domain collapse is transparently reported (Track C); models are strictly isolated from cross-dataset remapping.
- **Cryptographic Immutability**: All benchmarked models are cryptographically locked (BTXRD Checkpoint SHA-256: `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD`). Burned holdout partitions are preserved and uncompromised.
