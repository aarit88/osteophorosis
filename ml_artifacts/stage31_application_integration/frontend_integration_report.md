# Frontend Integration Report: UI Components, Routing, and Production Build

## 1. Executive Summary
The OsteoGuard AI frontend is a single-page application built with React 18, TypeScript 5.7, Tailwind CSS v4, and Vite 6.2. In Stage 31, the frontend was expanded from a simple single-track viewer into an auditable research platform that provides dedicated views for every engineering component developed across Stages 1–30.

---

## 2. Integrated Views and Page Components

| Route | Component Name | File Path | Key Features & Modules Demonstrated |
|---|---|---|---|
| `/` | `Home` | `src/pages/Home.tsx` | Analysis upload, multi-modal intake, processing queue, telemetry logging |
| `/workbench` | `ResearchWorkbenchPage` | `src/pages/ResearchWorkbenchPage.tsx` | Synchronized multi-panel demonstrator executing Radiomics, Biomechanics, Explainability, and Uncertainty on synthetic fixtures |
| `/tracks` | `ResearchTracksPage` | `src/pages/ResearchTracksPage.tsx` | Strict separation of Tracks A, B, C, and D with historical metrics and limitation badges |
| `/radiomics` | `RadiomicsPage` | `src/pages/RadiomicsPage.tsx` | Interactive digital phantom selector, 81 verified vs 21 unverified features breakdown |
| `/biomechanics` | `BiomechanicsPage` | `src/pages/BiomechanicsPage.tsx` | 8 bone cross-section phantoms, section modulus ($I_{xx}, I_{yy}, J$), and classical Mirels (1989) calculator |
| `/explainability`| `ExplainabilityPage` | `src/pages/ExplainabilityPage.tsx` | Grad-CAM activation heatmap matrix sample, spatial shift testing, SHAP partial status |
| `/uncertainty` | `UncertaintyPage` | `src/pages/UncertaintyPage.tsx` | Simulated probability presets, normalized Shannon entropy meter, candidate threshold ($H_{\text{norm}} \ge 0.60$), triage badge |
| `/governance` | `GovernancePage` | `src/pages/GovernancePage.tsx` | Four-class clinical cohort acquisition status, ICMR/IEC ethics checklist, bilateral DUA status |
| `/evidence` | `EvidenceRegistryPage` | `src/pages/EvidenceRegistryPage.tsx` | Central model ledger, SHA-256 frozen checkpoint verification, multi-center domain collapse audit |
| `/architecture` | `ArchitecturePage` | `src/pages/ArchitecturePage.tsx` | Multi-modal fusion flowchart, component readiness table, and interactive Four-Class Safety Gate test |
| `/audit` | `AuditPage` | `src/pages/AuditPage.tsx` | EU AI Act Article 14 compliant forensic audit trail, session events, radiologist review log |
| `/about` | `About` | `src/pages/About.tsx` | System specifications, institutional acknowledgments, KSCST proposal alignment |

---

## 3. Navigation Architecture (`NavBar.tsx`)
The top navigation bar provides:
1. Brand header with active AI status indicator.
2. Quick-access navigation:
   - `Analysis` (`/`)
   - `Workbench` (`/workbench`)
   - `Tracks` (`/tracks`)
   - `Engines` Dropdown Menu:
     - Radiomics Engine (`/radiomics`)
     - Biomechanics & Mirels (`/biomechanics`)
     - Explainability & SHAP (`/explainability`)
     - Uncertainty Triage (`/uncertainty`)
   - `Governance` (`/governance`)
   - `Evidence` (`/evidence`)
   - `Safety Gate` (`/architecture`)
   - `Audit` (`/audit`)
   - `About` (`/about`)
3. Utility widgets:
   - Active analysis queue indicator
   - Recent analyses history drawer
   - Language locale toggle (en/hi)
   - Engine mode widget
   - Theme toggle (dark/light)
4. Mobile-responsive secondary horizontal scrolling navigation bar.

---

## 4. TypeScript Client & API Resilience (`client.ts`)
The API client layer (`src/api/client.ts`) implements typed asynchronous wrappers for all 12 backend endpoints with automated timeout management, network retry logic, and error mapping:
- `getResearchStatus()`
- `getResearchTracks()`
- `getEvidenceRegistry()`
- `getArchitectureReadiness()`
- `getDatasetGovernance()`
- `runRadiomicsDemo(payload)`
- `runBiomechanicsGeometry(payload)`
- `runBiomechanicsMirels(payload)`
- `runExplainabilityDemo(payload)`
- `runUncertaintyDemo(payload)`
- `runSyntheticWorkbench(payload)`
- `requestFourClassInference()`

---

## 5. Production Build Verification
The frontend was built using `npm run build` (`tsc && vite build`) inside `osteophorosis-main/`:
- **Modules Transformed**: 3,285
- **TypeScript Compile Errors**: 0
- **Build Exit Code**: 0
- **Bundle Optimization**: All 12 pages are code-split into distinct dynamic import chunks:
  - `ResearchWorkbenchPage-*.js`: 14.80 kB
  - `BiomechanicsPage-*.js`: 12.89 kB
  - `ArchitecturePage-*.js`: 9.78 kB
  - `UncertaintyPage-*.js`: 9.12 kB
  - `ExplainabilityPage-*.js`: 8.42 kB
  - `RadiomicsPage-*.js`: 7.77 kB
  - `GovernancePage-*.js`: 5.79 kB
  - `EvidenceRegistryPage-*.js`: 5.78 kB
  - `ResearchTracksPage-*.js`: 5.20 kB

