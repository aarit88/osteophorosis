# API Integration Report: Endpoints, Schemas, and Verification

## 1. Overview
The OsteoGuard AI backend exposes 12 production REST endpoints under `/api/v1` implemented using FastAPI and Pydantic v2. These endpoints decouple the experimental sub-track models, surface historical evidence, provide interactive synthetic demonstrations for the proposal-aligned engines, and enforce an immutable safety gate on four-class clinical inference.

---

## 2. API Route Specifications & Test Verification

| Method | Endpoint Path | Function Name | Status Code | Test Status | Purpose |
|---|---|---|---|---|---|
| `GET` | `/api/v1/research/status` | `get_research_status` | 200 OK | PASSED | Returns aggregate research status, active track count, and locked checkpoint verification. |
| `GET` | `/api/v1/research/tracks` | `get_research_tracks` | 200 OK | PASSED | Returns full specifications, historical metrics, and limitations for Tracks A, B, C, and D. |
| `GET` | `/api/v1/evidence/registry` | `get_evidence_registry` | 200 OK | PASSED | Returns the model evidence registry with generalization findings and checkpoint hashes. |
| `GET` | `/api/v1/architecture/readiness` | `get_architecture_readiness` | 200 OK | PASSED | Component-by-component readiness audit against KSCST proposal specifications. |
| `GET` | `/api/v1/datasets/status` | `get_dataset_governance_status` | 200 OK | PASSED | Audits four-class clinical cohort availability, DUA status, and ICMR/IEC requirements. |
| `POST` | `/api/v1/radiomics/demo` | `run_radiomics_demo` | 200 OK | PASSED | Extracts 102 features from digital phantoms (81 verified IBSI / 21 unverified heuristic). |
| `POST` | `/api/v1/biomechanics/geometry` | `run_biomechanics_geometry` | 200 OK | PASSED | Computes cross-sectional area, moments of inertia ($I_{xx}, I_{yy}$), and polar moment $J$. |
| `POST` | `/api/v1/biomechanics/mirels` | `run_biomechanics_mirels` | 200 OK | PASSED | Classical Mirels 1989 scoring (site, pain, nature, size) and surgical threshold. |
| `POST` | `/api/v1/explainability/demo` | `run_explainability_demo` | 200 OK | PASSED | Generates Grad-CAM saliency heatmaps on synthetic spatial CNN fixtures; reports SHAP status. |
| `POST` | `/api/v1/uncertainty/demo` | `run_uncertainty_demo` | 200 OK | PASSED | Evaluates Shannon entropy, normalized entropy, and tri-state triage ($H_{\text{norm}} \ge 0.60$). |
| `POST` | `/api/v1/workbench/run-demo` | `run_synthetic_workbench` | 200 OK | PASSED | Executes coordinated multi-engine synthetic demonstration across all 4 validated engines. |
| `POST` | `/api/v1/diagnostic/four-class` | `diagnostic_four_class_blocked` | 503 SERVICE UNAVAILABLE | PASSED | **Strictly blocked.** Rejects clinical inference with detailed blockers checklist. |

---

## 3. Gating Implementation: Four-Class Blocked Inference
The endpoint `POST /api/v1/diagnostic/four-class` is the primary safety gate. When requested, it executes the following logic:

```python
@router.post(
    "/diagnostic/four-class",
    response_model=BlockedInferenceResponse,
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    summary="Four-Class Diagnostic Inference (Strictly Blocked)",
)
def diagnostic_four_class_blocked(
    service: ResearchService = Depends(get_research_service),
) -> BlockedInferenceResponse:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=service.get_blocked_inference().model_dump(),
    )
```

### JSON Response Payload (HTTP 503):
```json
{
  "error_code": "FOUR_CLASS_MODEL_NOT_READY",
  "detail": "Clinical diagnostic inference is disabled. The OsteoGuard four-class diagnostic model cannot be deployed or executed because approved clinical datasets for Paget's disease, human osteogenesis imperfecta (OI), and matched healthy controls have not been acquired.",
  "timestamp": "2026-11-16T17:15:00.000Z",
  "path": "/api/v1/diagnostic/four-class",
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
    ],
    "mandatory_safeguard": "Under research governance rules, synthetic phantoms must not be fed into clinical models, and preliminary sub-track models (FracAtlas, Bone Scan, BTXRD) must never be remapped into four-class predictions."
  }
}
```

---

## 4. Verification and Automated Testing
All 12 endpoints were comprehensively tested using `pytest` with `starlette.testclient.TestClient`.
- Execution command: `python -m pytest ml_service/tests/test_stage_31_application_integration.py -v`
- Test result: **14 passed in 1.45s (100% success rate)**.
- Full regression suite across stages 25-31: **108 passed in 7.92s (100% success rate)**.

