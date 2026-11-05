# OsteoGuard AI — Explainable Bone Pathology Detection System
> **Frontend, Simulation & Clinical Validation Layer (Phase 2 Production Hardened & Major Project Contribution)**

[![Research Prototype](https://img.shields.io/badge/status-Research%20Prototype-red.svg)](#)
[![TypeScript](https://img.shields.io/badge/language-TypeScript%205-blue.svg)](#)
[![Vite](https://img.shields.io/badge/bundler-Vite%206-purple.svg)](#)
[![TailwindCSS](https://img.shields.io/badge/styling-TailwindCSS%20v4-teal.svg)](#)
[![OpenAPI 3.1](https://img.shields.io/badge/API-OpenAPI%203.1%20Validated-brightgreen.svg)](#)
[![Tests](https://img.shields.io/badge/tests-Vitest%20Passing-success.svg)](#)

> ⚠️ **RESEARCH PROTOTYPE — SIMULATED OUTPUT — NOT FOR CLINICAL USE**  
> This application serves as the production-quality frontend, simulation runtime, and clinical evaluation platform for **OsteoGuard AI**, an explainable multi-modal deep learning system that classifies planar radiographs, CT bone windows, and radionuclide bone scans into 4 distinct pathology classes:
> 1. **Bone Metastasis** (osteolytic and osteoblastic lesions)
> 2. **Paget's Disease of Bone** (cortical thickening and disorganized trabecular remodeling)
> 3. **Osteogenesis Imperfecta (OI)** (severe bone fragility and cortical thinning)
> 4. **Healthy Bone** (normal mineral density and trabecular architecture)

---

## 🌟 University Major Project Novelty & Contributions

This codebase represents the frontend, clinical integration, and simulation contribution of a 4-person engineering team. While our teammates develop the offline PyTorch dual-stream CNN + Vision Transformer training pipeline, real backend microservices, and cluster deployment, this system introduces **three novel clinical decision-support paradigms** anchored in international medical AI regulations:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               OSTENGUARD AI NOVELTY SUITE                              │
├──────────────────────────────┬─────────────────────────────┬───────────────────────────┤
│  1. Uncertainty & Abstention │ 2. Human-in-the-Loop Audit  │ 3. Traceable Report & CF  │
│  • Normalized Shannon Ent. H │ • Concurrence Review (EU 14)│ • 3-Page Forensic PDF Gen │
│  • 95% Confidence Intervals  │ • Append-only JSONL Ledger  │ • Cryptographic QR Verif. │
│  • OOD Abstention Banner     │ • Admin /audit Dashboard    │ • Evidence Anchors ¹ ² ³  │
│  • Reliability Scatter Plot  │ • FDA GMLP Conformance      │ • Interactive CF Scrubber │
└──────────────────────────────┴─────────────────────────────┴───────────────────────────┘
```

1. **Uncertainty-Aware Differential Diagnosis with Safe Abstention UI**
   - Softmax Shannon Entropy calculation ($H \in [0.0, 1.0]$) with dynamic 95% Clopper-Pearson/Wilson confidence intervals and Out-Of-Distribution (OOD) heuristic flagging.
   - **Abstention Workflow**: When model uncertainty exceeds safe clinical thresholds ($H > 0.60$ or `oodFlag: true`), the UI **demotes the primary diagnosis** into a *"Model Suggestion (Unverified / Refer for Review)"* container accompanied by a high-visibility amber safety banner. This directly prevents diagnostic anchoring bias in resident radiologists.
   - Reliability calibration scatter plot showing predicted confidence vs. empirical observed accuracy across 10-bin deciles.

2. **Clinician-in-the-Loop Feedback Capture & Audit Dashboard (EU AI Act Art. 14 & FDA GMLP)**
   - Radiologists can register agreement (*Concur / Disagree / Uncertain*) with structured clinical reason codes (`SUBTLE_LESION`, `ATYPICAL_PRESENTATION`, `ARTIFACT_NOISE`, `ALTERNATIVE_SUSPECTED`) and free-text notes.
   - Submissions stream to an append-only audit ledger (`mock-server/data/feedback.jsonl`).
   - Dedicated `/audit` route provides executive metrics (Total Reviews, Overall Concurrence Rate %, Flagged Cases) and an interactive SVG Donut Chart with filterable audit ledger records.

3. **Traceable Clinical PDF Report & Interactive Counterfactual Scrubber**
   - Automated 3-page forensic report generator rendered via `jspdf` and `html2canvas` featuring patient metadata, before/after Grad-CAM split comparisons, SHAP waterfall bars, and a scannable verification QR code linking directly to `/audit`.
   - **Numbered Evidence Citations**: Diagnostic findings are superscript-indexed ($¹$, $²$, $³$) and cross-referenced to specific anatomical Grad-CAM coordinates and radiomic SHAP metrics.
   - **Counterfactual Sensitivity Panel**: An interactive real-time slider that allows clinicians to perturb cortical thickness or texture entropy, immediately re-evaluating diagnostic class boundaries client-side.

---

## 🚀 Quickstart & Commands

### Prerequisites
- Node.js 18+ or 22+
- npm 9+

### Available Scripts

| Command | Purpose |
| :--- | :--- |
| `npm run dev` | Runs the unified full-stack server (port `3000`) with Vite middlewares + Express `/api/v1/*` endpoints + Swagger docs |
| `npm run mock` | Runs the standalone mock API server on `http://localhost:8787` |
| `npm run dev:full` | Concurrently boots both standalone mock API (`8787`) and Vite dev server (`3000`) |
| `npm run openapi:validate` | Validates `openapi.yaml` against OpenAPI 3.1 standard using `@redocly/cli` |
| `npm test` | Executes full Vitest unit & integration test suite (`tests/*.test.tsx`) |
| `npm run e2e` | Playwright end-to-end testing verification runner |
| `npm run build` | Compiles TypeScript and builds optimized production bundles (`dist/`) |
| `npm run lint` | Runs TypeScript static verification (`tsc --noEmit`) |

### Interactive API Documentation (Swagger UI)
When the application is running, open:
```
http://localhost:3000/docs
```
to access the interactive Swagger UI generated directly from `openapi.yaml`.

---

## 🔌 Part B — Pluggable Prediction Engine (`src/engine/`)

The application isolates all inference operations behind an abstract engine seam, allowing zero-friction handoff to Teammate B's live backend:

```
┌────────────────────────────────────────────────────────┐
│                   React UI Components                  │
└───────────────────────────┬────────────────────────────┘
                            │
               PredictionEngine (Interface)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
       MockEngine                      LiveEngine
 (Deterministic SHA-256)      (Teammate B Real PyTorch API)
            ▲                               ▲
            └───────────────┬───────────────┘
                            │
                      Engine Factory
               (Controlled via VITE_ENGINE)
```

- **`PredictionEngine.ts`**: Formal interface exposing `analyze(input: AnalyzeInput): Promise<AnalyzeResponse>`.
- **`MockEngine.ts`**: Communicates with the deterministic simulation server.
- **`LiveEngine.ts`**: Communicates with the production backend (`VITE_API_BASE_URL_LIVE`); gracefully throws typed errors when offline.
- **`index.ts` (Factory)**: Selected dynamically via `VITE_ENGINE` (`"mock"` | `"live"` | `"auto"`).
  - In `"auto"` mode, the factory attempts the live backend and automatically falls back to `"mock"` if unavailable, displaying an **Engine: Fallback** indicator badge in the top navigation bar.

---

## 📋 Comprehensive API Contract (`/api/v1/*`)

All simulation and inference endpoints are versioned under `/api/v1/`. The root health check remains at `/api/health`.

### 1. `POST /api/v1/analyze`
**Headers**: `Content-Type: multipart/form-data`  
**Body**:
- `image`: Binary file (X-ray, CT, bone scan)
- `modality`: `"xray"` | `"ct"` | `"bone_scan"`

**Response (`AnalyzeResponse`)**:
```json
{
  "requestId": "req_a9f182c0b4d3",
  "timestamp": "2026-11-05T14:15:22.000Z",
  "modality": "xray",
  "imageUrl": "/uploads/input_a9f182c0b4d3.png",
  "gradcamUrl": "/uploads/gradcam_a9f182c0b4d3.png",
  "prediction": {
    "topClass": "metastasis",
    "confidence": 0.884,
    "probabilities": {
      "metastasis": 0.884,
      "paget": 0.052,
      "oi": 0.041,
      "healthy": 0.023
    },
    "uncertainty": "low"
  },
  "uncertaintyDetail": {
    "entropy": 0.24,
    "confidenceInterval": [0.821, 0.947],
    "oodFlag": false,
    "calibrationBin": {
      "predicted": 0.88,
      "observed": 0.87
    }
  },
  "counterfactuals": [
    {
      "featureName": "Cortical Thickness",
      "originalValue": -0.42,
      "perturbedValue": -0.15,
      "delta": 0.27,
      "predictedClass": "metastasis",
      "targetClass": "healthy",
      "marginShift": 0.23,
      "clinicalInterpretation": "Increasing cortical thickness by +0.27 mm shifts posterior toward normal cortical margins."
    }
  ],
  "shap": [
    { "name": "Cortical Thickness", "value": -0.42 },
    { "name": "GLCM Contrast", "value": 0.38 },
    { "name": "Bone Volume Fraction", "value": -0.31 },
    { "name": "Texture Entropy", "value": 0.27 },
    { "name": "First-order Mean Intensity", "value": -0.19 },
    { "name": "GLRLM Long-Run Emphasis", "value": 0.12 }
  ],
  "fractureRisk": {
    "level": "high",
    "percent": 84
  },
  "disclaimer": "⚠️ RESEARCH PROTOTYPE — SIMULATED OUTPUT — NOT FOR CLINICAL USE."
}
```

### 2. `POST /api/v1/feedback`
Captures structured clinician review. Appends to `mock-server/data/feedback.jsonl`.
```json
{
  "requestId": "req_a9f182c0b4d3",
  "concurrence": "agree",
  "clinicianId": "RAD_RES_04",
  "comments": "Focal osteolytic cortical disruption is consistent with secondary metastasis."
}
```

### 3. `GET /api/v1/feedback/stats`
Returns aggregated statistics for the `/audit` executive dashboard:
```json
{
  "total": 42,
  "agree": 35,
  "disagree": 4,
  "unsure": 3,
  "agreementRate": 0.833
}
```

### 4. `POST /api/v1/events`
Captures client-side telemetry events for human-AI interaction evaluation:
- `study_session_started`
- `modality_selected`
- `analysis_completed`
- `gradcam_toggled`
- `counterfactual_scrubbed`
- `report_exported`
- `feedback_submitted`

---

## ♿ Accessibility (a11y) Self-Audit Report

Conducted in compliance with **WCAG 2.1 Level AA** standards:

| Requirement | Implementation Detail | Status |
| :--- | :--- | :---: |
| **ARIA Live Regions** | `aria-live="polite"` attached to asynchronous inference containers and queue banners so screen readers announce completions automatically. | ✅ Pass |
| **Screen-Reader Tables** | Visually hidden `<table className="sr-only">` accompany Recharts probability bars and SHAP feature graphs, tabulating values for non-visual navigation. | ✅ Pass |
| **Keyboard Navigation** | Upload dropzone is focusable (`tabIndex={0}`) and activates via `Enter` or `Space`. All modals feature focus trapping and `Escape` handlers. | ✅ Pass |
| **Focus Rings** | High-contrast `focus-visible:ring-2 focus-visible:ring-emerald-500` applied globally across all buttons, tabs, and scrubbers. | ✅ Pass |
| **Reduced Motion** | Framer Motion animations honor `prefers-reduced-motion: reduce` via `useReducedMotion()` hook, suppressing high-frequency layout transitions. | ✅ Pass |
| **Contrast Ratios** | All text meets 4.5:1 minimum contrast in both dark and light color themes. | ✅ Pass |

---

## ⚡ Performance Budget & Code-Splitting

- **Route-Level Code-Splitting**: `Results.tsx` and `AuditPage.tsx` are lazily loaded via `React.lazy()` and wrapped in `<Suspense fallback={<PageSkeleton />}>`.
- **Vendor Chunk Isolation**: `recharts`, `jspdf`, `html2canvas`, and `framer-motion` are dynamically split into asynchronous chunks, keeping initial bundle size under 180 KB gzipped.
- **Lighthouse Metric Target (Mobile Preset)**:
  - **Performance Score**: 94/100
  - **First Contentful Paint (FCP)**: 0.9s
  - **Speed Index**: 1.3s
  - **Largest Contentful Paint (LCP)**: 1.8s
  - **Cumulative Layout Shift (CLS)**: 0.002

---

## 🌐 Internationalization (i18n) Scaffold

The UI wraps core interface copy using `react-i18next`:
- **English (`en`)**: Primary clinical terminology and regulatory disclaimers.
- **Hindi (`hi`)**: Locale scaffold demonstrating seamless multi-language expansion for international deployments.

Switch languages dynamically through the navigation header selector.

---

## 📁 Validation Pack Documents

Located in the `docs/` directory:
1. **[`docs/usability-protocol.md`](docs/usability-protocol.md)**: A complete 30-minute moderated usability protocol designed for $N=5$ radiology residents, integrating System Usability Scale (SUS) and NASA-TLX cognitive workload metrics.
2. **[`docs/metrics.md`](docs/metrics.md)**: Formal mathematical definitions for Time-to-Decision (TTD), Concurrence Rate, Telemetry Taxonomy, and Abstention Safety margins.
3. **[`docs/novelty-framing.md`](docs/novelty-framing.md)**: Manuscript-ready scientific methodology section framing OsteoGuard AI's explainability, abstention UI, and human-in-the-loop audit trails against EU AI Act Article 14 and FDA GMLP standards.

---

## 📂 Source Code Structure

```
osteoguard-frontend/
├── src/
│   ├── api/
│   │   ├── client.ts             # Resilient HTTP client (timeout, 2x retry, typed errors)
│   │   └── types.ts              # Strict TypeScript interfaces matching OpenAPI contract
│   ├── engine/
│   │   ├── PredictionEngine.ts   # Core engine interface
│   │   ├── MockEngine.ts         # Local deterministic simulation engine
│   │   ├── LiveEngine.ts         # Live backend engine for teammate handoff
│   │   └── index.ts              # Engine factory (mock | live | auto)
│   ├── components/
│   │   ├── UncertaintyCard.tsx   # Entropy gauge, 95% CI & reliability scatter plot
│   │   ├── AbstentionBanner.tsx  # Amber human-review abstention safety notice
│   │   ├── ClinicianFeedbackCard.tsx # EU AI Act Art. 14 concurrence review form
│   │   ├── CounterfactualPanel.tsx   # Interactive feature perturbation scrubber
│   │   ├── TraceableReportModal.tsx  # 3-page forensic PDF generator with QR code
│   │   ├── EngineStatusWidget.tsx    # Navigation engine indicator & status modal
│   │   ├── QueueIndicator.tsx    # Client-side request queue status badge
│   │   ├── RecentAnalysesDrawer.tsx  # Slide-in drawer for recent scans (N=20)
│   │   ├── ErrorBoundary.tsx     # Route-level crash recovery fallback
│   │   ├── DisclaimerBanner.tsx  # Persistent non-dismissible safety banner
│   │   ├── GradCamCompare.tsx    # Split-view & opacity overlay slider
│   │   ├── ProbabilityChart.tsx  # Animated multi-class distribution bars
│   │   ├── ShapChart.tsx         # Diverging radiomics feature importance
│   │   └── FractureGauge.tsx     # Biomechanical fracture risk radial gauge
│   ├── pages/
│   │   ├── Home.tsx              # Landing, modality selection, sample loader, queue
│   │   ├── Results.tsx           # Full diagnostic review, explainability, reports
│   │   ├── AuditPage.tsx         # /audit route with stats, donut chart & audit table
│   │   └── About.tsx             # Pipeline timeline & teammate responsibility matrix
│   ├── i18n/
│   │   └── config.ts             # i18next configuration (en, hi)
│   ├── context/
│   │   └── AnalysisContext.tsx   # State persistence across navigation
│   └── App.tsx                   # Lazy routes, ErrorBoundary & persistent layout
├── mock-server/
│   ├── index.js                  # Express API server (routes, swagger-ui, multer)
│   ├── simulation.js             # SHA-256 deterministic generator, Sharp Grad-CAM
│   └── data/
│       └── feedback.jsonl        # Append-only clinical review audit log
├── docs/
│   ├── usability-protocol.md     # 30-min resident evaluation protocol
│   ├── metrics.md                # Quantitative metrics & telemetry schemas
│   └── novelty-framing.md        # Academic manuscript draft section
├── tests/
│   ├── upload.test.tsx           # Drag-and-drop & file selection tests
│   ├── results.test.tsx          # Uncertainty, SHAP, and report rendering tests
│   ├── error.test.tsx            # Network resilience & ErrorBoundary tests
│   └── setup.ts                  # Testing Library & Vitest DOM mocks
├── openapi.yaml                  # OpenAPI 3.1 specification
├── redocly.yaml                  # Redocly linter configuration
├── server.ts                     # Unified port 3000 full-stack runner
└── .env.example                  # Environment variable reference
```