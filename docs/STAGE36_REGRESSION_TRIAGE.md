# Stage 36 Regression Triage & Integration Stabilization

## Scope and safety

This records the configured repository-wide pytest run and the targeted regression fixes performed during Stage 36. The existing `docs/CODEX_REPOSITORY_AUDIT.md` was not changed. No clinical datasets, frozen checkpoints, metrics, approvals, institutional agreements, or reviewer identities were created. Track D remains gated on genuine data, governance, and a SHA-256 pinned checkpoint.

## Full Python suite

Command:

```powershell
$env:TEMP='C:\Users\Admin\Documents\Codex\stage36-pytest-tmp'
$env:TMP=$env:TEMP
python -m pytest -q --tb=short
```

Exit code: **1**. Final result after fixes: **68 failed, 373 passed, 15 skipped, 54 errors, 1 warning** (53.88 seconds). Pytest's configured test discovery includes both `tests_ml/` and `ml_service/tests/`. Before correcting the checkout-relative historical preservation assertions, the first fresh rerun had 84 failures and 357 passes.

| Result group | Count | Finding / action |
| --- | ---: | --- |
| Passed | 373 | Unit, API, governance, and other tests passed. |
| Skipped | 15 | Existing tests intentionally skipped. |
| Errors during setup | 54 | Historical dataset/training tests require absent manifests, raw cohorts, and original checkpoints. Representative absent resources: `ml_service/data/splits/fracatlas_classification_split.csv`, `bone_scan_split_manifest.csv`, `bone_scan_train_cv_manifest.csv`, `btxrd_split_manifest.csv`, and `ml_artifacts/btxrd_frozen_model/best_checkpoint.pt`. These cannot be synthesized: doing so would fabricate protected clinical or model evidence. |
| Failed assertions | 68 | Remaining failures are historical Stage 06–30 assertions requiring absent cohorts, manifests, prior trained artifacts, or the protected frozen weights. The checkout-relative frontend/Python assertions were repaired; their dedicated check had 16 passes and one combined test blocked only by its absent Stage 16 checkpoint. |
| Warning | 1 | Starlette deprecation warning for `HTTP_422_UNPROCESSABLE_ENTITY`; no test failure. |

The initial run (89 failed, 352 passed, 15 skipped, 54 errors) also wrote pytest temporary checkpoint-shaped files below the repository because its temp root was under `work/`. This contaminated the checkpoint inventory assertion. I removed that task-created temporary directory and reran using an external temp root; subsequent results no longer had that inventory false positive. No existing model artifacts were changed.

## Confirmed integration regressions fixed

- The root endpoint advertised `/health` under `endpoints.health` while the existing API contract expects `/api/ml/health`. The response now advertises the contract path and includes the registered dataset and model readiness routes.
- The readiness state-machine tests created unpinned dummy checkpoint files. They now configure the temporary test checkpoint as active and supply its computed SHA-256, exercising the actual trust gate instead of bypassing it.
- The SHAP status test assumed the optional `shap` dependency was always absent. It now validates the adapter's dependency status against the runtime availability/version.

Focused endpoint, readiness, and checkpoint inventory command:

```powershell
python -m pytest -q ml_service/tests/test_health_endpoints.py ml_service/tests/test_model_state_machine.py ml_service/tests/test_recover_checkpoints.py --tb=short
```

Result: **21 passed**. The updated runtime-dependent SHAP status test also passed in the focused regression set below.

Stage 31–35 governance, intake, security, inventory, guardrail, API, and SHAP regression command:

```powershell
python -m pytest -q ml_service/tests/test_stage_29_explainability.py::test_14_tabular_shap_adapter_status_and_schema_validation ml_service/tests/test_track_d_intake.py ml_service/tests/test_stage34_ontology_adjudication.py ml_service/tests/test_stage35_acquisition_governance.py ml_service/tests/test_stage_31_application_integration.py ml_service/tests/test_security_api_contracts.py ml_service/tests/test_checkpoint_inventory.py ml_service/tests/test_zero_dataset_guardrails.py --tb=short
```

Result: **70 passed**.

## Frontend verification

- `npm ci`: **passed**, 384 packages installed, audit reported 0 vulnerabilities.
- `npm test -- --reporter=dot`: **passed**, 7 test files / 13 tests.
- `npm run lint`: **passed** (`tsc --noEmit`).
- `npm run build`: **passed** (TypeScript and Vite production build). Vite emitted a non-failing chunk-size advisory for the large Results bundle.

## CI

Added `.github/workflows/ci.yml` with independent frontend and Python governance jobs. Frontend CI performs `npm ci`, Vitest, TypeScript lint, and production build. Python CI installs `requirements-dev.txt` and runs the maintained intake, governance, API integration, security contract, artifact inventory, and zero-data guardrail tests. The historical artifact-dependent full suite remains a local diagnostic because CI cannot fabricate the external cohorts and frozen weights those assertions require.

## Remaining blockers

- Genuine clinical cohort data and authenticated source manifests are absent for the legacy Stage 06–23 split/training/multicenter tests.
- The protected frozen BTXRD checkpoint and historical trained checkpoints are absent. They were not regenerated or replaced.
- The full configured Python suite therefore remains red with the counts above until genuine cohort manifests/data and historical checkpoints are made available. This is an evidence-backed blocker, not a green suite.
