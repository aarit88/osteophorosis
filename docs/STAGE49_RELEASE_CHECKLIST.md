# Stage 49 — Release Checklist

## Release status

This checklist records repository verification for the current implementation. It does not certify clinical readiness, institutional approval, production compliance, or dataset authorization.

| Area | Status | Evidence |
|---|---|---|
| Frontend unit tests | Passed: 18 files, 52 tests | `npm test -- --reporter=dot` |
| TypeScript | Passed | `npm run lint` |
| Production frontend build | Passed | `npm run build` |
| Browser E2E | Passed: 3 tests | `npm run e2e` against Express/Vite |
| FastAPI governance/security regressions | Passed: 105 passed, 3 skipped | Stage 33–40 and security/deployment regression command |
| Live Express/FastAPI smoke | Passed | `npm run test:deployment` |
| Python compilation | Passed | `python -m compileall -q ml_service` |
| CI YAML parsing | Passed | Parsed `.github/workflows/ci.yml` with the `yaml` package |
| Diff whitespace | Passed | `git diff --check` |
| Clean dependency install | Not verified on this Windows checkout | `npm ci` hit Windows `EPERM` while removing an existing native module; `npm install` restored dependencies successfully |
| Clinical Track D inference | Intentionally unavailable | HTTP 503 `FOUR_CLASS_MODEL_NOT_READY` |
| Clinical datasets and approvals | Not present in this release | External acquisition and governance dependency |
| Docker build | Not claimed | Must be recorded from an actual Docker run |

The frontend suite prints expected React/jsdom error-boundary output from its intentional crash test; the test run still passes. The local deployment smoke reports that Windows cleanup terminates the service process; graceful shutdown is verified by Linux CI.

## Release gate

Do not mark a release as ready if required automated checks fail. A passing software build does not establish clinical validity. Track D remains unavailable until genuine authorized data, governance, training, validation, and model readiness gates are met.
