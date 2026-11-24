# Stage 49 — Release and Deployment Notes

## Verification commands

Run from the repository root:

```powershell
npm ci
npm test -- --reporter=dot
npm run lint
npm run build
npm run e2e:install
npm run e2e
python -m pytest -q ml_service/tests/test_track_d_intake.py ml_service/tests/test_stage34_ontology_adjudication.py ml_service/tests/test_stage35_acquisition_governance.py ml_service/tests/test_stage_31_application_integration.py ml_service/tests/test_security_api_contracts.py ml_service/tests/test_deployment_health.py ml_service/tests/test_stage38_cohort_onboarding.py ml_service/tests/test_stage39_dataset_preparation.py ml_service/tests/test_stage40_source_qualification.py ml_service/tests/test_checkpoint_inventory.py ml_service/tests/test_zero_dataset_guardrails.py ml_service/tests/test_deployment_smoke_script.py
npm run test:deployment
python -m compileall -q ml_service
git diff --check
```

`npm run e2e:install` installs Playwright's headless Chromium shell into the ignored `.playwright-browsers/` directory, and `npm run e2e` runs the browser checks against the real Express/Vite development gateway.

On the current Windows development checkout, `npm ci` encountered `EPERM` while removing an existing native Lightning CSS module. `npm install` restored the dependency tree and reported zero vulnerabilities; CI's clean-install job remains the clean-install verification environment. Do not interpret a local `npm ci` attempt that exits nonzero as a passed check.

The deployment smoke test starts real local FastAPI and Express processes with isolated temporary storage. It uses a tiny non-clinical PNG only to verify the upload parser and the Track D 503 gate. It does not train a model or create clinical evidence.

## Production configuration

Follow the existing Stage 37 deployment guide and `.env.example`. Production requires externally managed secrets, exact CORS origins, the institution-provisioned reviewer registry, and persistent integrity-protected ledgers. Do not copy development values into production. Do not place datasets or model weights in container images.

## Clinical readiness

Track D remains gated with `HTTP 503 FOUR_CLASS_MODEL_NOT_READY`. The software release does not claim that a suitable clinical cohort, ethics approval, data-use agreement, clinical validation, or trained four-class model exists. Track A/B/C artifact and evaluation protections remain in force.
