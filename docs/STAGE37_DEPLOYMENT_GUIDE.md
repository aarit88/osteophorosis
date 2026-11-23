# Stage 37 — Deployment Guide

## 1. Summary and operational scope

Stage 37 adds a local controlled-research deployment using the repository's existing Express gateway, Vite-built React application, and FastAPI ML API. The gateway serves the compiled frontend and is the only host-published service. FastAPI stays private on the Compose network. Liveness, API-process readiness, and clinical model readiness have separate endpoints and meanings.

This setup is **not a clinical deployment**. The repository does not contain the historical clinical cohorts or protected checkpoints needed for Track D. The image build does not include datasets or model weights, and diagnosis remains blocked until real data, governance, model validation, and an externally verified checkpoint are available.

## 2. Architecture

| Component | Container / port | Host exposure | Health |
| --- | --- | --- | --- |
| Express 5 + React static build | `gateway:3000` | `127.0.0.1:${HOST_PORT:-3000}` | `/health` liveness; `/readyz` checks FastAPI connectivity |
| FastAPI ML / governance API | `ml-service:8000` | None; Compose network only | `/health/live`, `/health/ready`, `/health/model` |

Express proxies the existing `/api/v1/research`, datasets, evidence, architecture, and related API routes to `BACKEND_URL`. The frontend uses same-origin requests by default. `NODE_ENV=production` serves `dist/`; development mode mounts Vite middleware. Production behavior loads Vite dynamically only in development.

Compose waits for FastAPI's **API-process** health before starting the gateway. That check does not claim the model is ready. `/health/model` returns HTTP 503 with a stable not-ready code until the Track D checkpoint is valid and the readiness requirements pass. The four-class diagnosis endpoint remains blocked in the current repository state.

## 3. Requirements and files

- Docker Engine with Compose v2 (`docker compose`), for container startup.
- Node.js 20 and npm for local frontend checks and the live smoke test.
- Python 3.11 for the ML image and deployment smoke test. Python runtime direct dependencies are pinned in `ml_service/requirements-runtime.txt`.
- Files: `Dockerfile`, `Dockerfile.ml-service`, `compose.yaml`, optional `compose.production.yaml`, and `.dockerignore`.

The Python image installs pinned runtime dependencies without pytest, formatters, or optional SHAP. The Node image uses `npm ci`, builds the UI, prunes development-only dependencies, and runs as the built-in non-root `node` user. FastAPI runs as UID 10001. Both services use a read-only root filesystem, drop Linux capabilities, and disallow privilege escalation. Dockerfiles and image build were not executed on this workstation because Docker is unavailable; see verification below.

## 4. Environment and secret configuration

Copy the sample and keep the real `.env` private:

```powershell
Copy-Item .env.example .env
```

`.env.example` contains no credentials and is tracked. `.env`, `.env.*.local`, private reviewer registries, datasets, keys, and weights must not be committed.

| Variable | Consumer | Purpose |
| --- | --- | --- |
| `NODE_ENV`, `HOST`, `PORT`, `HOST_PORT` | Express / Compose | Mode, bind and local published port |
| `BACKEND_URL` | Express | FastAPI service URL; Compose sets `http://ml-service:8000` |
| `OSTEO_ENV`, `OSTEO_HOST`, `OSTEO_PORT`, `OSTEO_LOG_LEVEL` | FastAPI | Runtime mode, bind, and logging |
| `OSTEO_CORS_ORIGINS` | Express and FastAPI | Comma-separated exact browser origins; production requires explicit HTTPS origins (HTTP localhost is allowed for local work) |
| `OSTEO_CLINICAL_API_TOKEN` | Express | Bearer token for protected gateway routes; mandatory in production |
| `OSTEO_GATEWAY_HMAC_SECRET` | Express and FastAPI | Signs and validates gateway-to-ML inference requests; mandatory in production |
| `OSTEO_AUDIT_ENCRYPTION_KEY` | Express | AES-256-GCM audit/feedback storage key; externally supplied in production |
| `OSTEO_AUDIT_LEDGER_HMAC_KEY` | FastAPI | HMAC integrity key for adjudication ledger; mandatory in production |
| `OSTEO_ONBOARDING_LEDGER_HMAC_KEY` | FastAPI | HMAC integrity key for Stage 38 dataset provenance ledger; mandatory in production |
| `OSTEO_ONBOARDING_REGISTRY_FILE`, `OSTEO_ONBOARDING_ROOT`, `OSTEO_STAGING_ROOT` | FastAPI | Writable persistent onboarding registry, acquired package, and Stage 33 staging paths |
| `OSTEO_ADJUDICATION_LEDGER_FILE` | FastAPI | Append-only ledger path on the persistent ledger volume |
| `OSTEO_REVIEWER_AUTHORIZATION_FILE` | FastAPI | Institution-provisioned reviewer registry path; required by the production overlay |
| `OSTEO_REVIEWER_REGISTRY_HOST_PATH` | Compose production overlay | External host path to that read-only institution registry |
| `OSTEO_DATA_DIR`, `OSTEO_RAW_DATA_DIR`, `OSTEO_SPLITS_DATA_DIR` and related paths | FastAPI | Dataset tree; Compose mounts it read-only |
| `OSTEO_ARTIFACTS_DIR`, `OSTEO_CHECKPOINTS_DIR`, `OSTEO_LOGS_DIR` | FastAPI | Persistent artifact volume; no weights are included in the image |
| `OSTEO_ACTIVE_CHECKPOINT_PATH`, `OSTEO_TRACK_D_CHECKPOINT_SHA256` | FastAPI | Optional explicit Track D checkpoint and externally verified SHA-256; leave unset until authorized genuine model release |
| `OSTEO_NODE_DATA_DIR` | Express | Persistent encrypted gateway event/feedback data |
| `OSTEO_STRICT_CLINICAL_MODE` | Express | Keeps the diagnostic bridge blocked instead of falling back to a demo prediction |
| `OSTEO_DISABLE_DEMO_SEED` | Express | Prevents demo feedback rows from being seeded into the Compose state volume |
| `VITE_API_BASE_URL` | Vite build | Empty by default so the frontend calls the same-origin gateway |

Generate distinct secrets outside source control and put them into the private `.env` file. For example:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"  # API bearer token
python -c "import secrets; print(secrets.token_hex(32))"  # run separately for gateway HMAC, audit encryption, and ledger HMAC
```

Do not reuse one key across purposes. The production startup checks reject missing/weak Node credentials, missing ML gateway, adjudication-ledger, or onboarding-ledger keys, non-exact production CORS, absent ledger paths, or a missing reviewer registry. The reviewer registry itself must be provisioned by the responsible institution; this repository does not create reviewer identities.

For local development, the sample leaves secret values blank and uses development mode. Protected adjudication routes report unavailable until a real reviewer registry and ledger key are configured. Compose disables demo feedback seeding and enables strict clinical mode.

## 5. Start and verify

After copying `.env.example` to `.env`:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs --tail 100 ml-service gateway
Invoke-RestMethod http://127.0.0.1:3000/health
Invoke-RestMethod http://127.0.0.1:3000/readyz
Invoke-RestMethod http://127.0.0.1:3000/api/v1/security/status
```

Open `http://127.0.0.1:3000/` for the existing React application. Port 8000 is not published to the host. Use `docker compose exec ml-service` for private-network diagnostics rather than opening the ML port.

In production-like controlled research mode, configure both `NODE_ENV=production` and `OSTEO_ENV=production`, set a real HTTPS `OSTEO_CORS_ORIGINS`, populate all externally generated secrets, and point `OSTEO_REVIEWER_REGISTRY_HOST_PATH` to an institution-provisioned registry file. Start with:

```powershell
docker compose -f compose.yaml -f compose.production.yaml up --build -d
```

Place a TLS-terminating reverse proxy in front of the loopback-only gateway if remote browser access is required. Do not publish the FastAPI port directly.

## 6. Stop, persistence, and access

```powershell
docker compose down
```

This stops containers and preserves named volumes: `node-state`, `model-artifacts`, `adjudication-ledger`, and `onboarding-state`. The dataset bind mount is read-only and points at `ml_service/data`; the gateway sees only its raw-data subtree. The ML service sees the mounted dataset tree read-only. Node can read artifact status but cannot write checkpoint artifacts. Only FastAPI has write access to the model-artifact, adjudication-ledger, and onboarding volumes.

Back up named volumes through the organization's approved encrypted backup process. Restrict host access to source datasets, the reviewer registry, and `.env`; encrypt host storage and backups. No backup or restore operation is performed by this repository. `docker compose down -v` deletes persistent ledger and artifact volumes; use it only when you explicitly intend to destroy that local state.

Do not put protected holdouts in generic artifact/checkpoint paths. Do not mount Track C weights writable. Generic discovery must not promote Track A, B, or C weights to Track D. Track C hash and size protections remain separate from this deployment setup.

## 7. Tests and configuration checks

```powershell
npm ci
npm test
npm run lint
npm run build
python -m pytest -q ml_service/tests/test_deployment_health.py ml_service/tests/test_security_api_contracts.py
npm run test:deployment
```

`scripts/deployment-smoke.py` starts actual FastAPI and Express processes on temporary loopback ports with isolated temporary storage. It checks startup, health contracts, gateway-to-FastAPI connectivity, dataset governance response, redacted public readiness, missing/invalid bearer credentials, permitted and invalid upload content/types, the 15 MiB upload limit, dependency failure readiness, production fail-closed startup, and continued HTTP 503 model gating. Its single-pixel PNG is a non-clinical parser fixture. It creates no reviewer identity or checkpoint. The Linux CI smoke test is configured to verify graceful SIGTERM shutdown; it has not run in this workspace. This Windows host cannot reliably deliver console signals to detached child processes, so local smoke cleanup uses process termination and does not claim graceful shutdown was verified here.

The Stage 33 intake regression tests cover valid and invalid isolated manifests. Existing Stage 34 tests cover reviewer role authorization and HMAC ledger integrity/tampering; Stage 35 tests cover production security configuration, upload validation, and protected checkpoint discovery. Generic Track A/B/C exclusion remains covered by `test_track_a_b_or_c_checkpoints_are_not_discovered_as_track_d`.

Vitest also parses the Compose files and checks service exposure, health dependencies, read-only mounts, user/capability hardening, and Docker build-context exclusions. CI runs these checks, the Python governance group, and the live process smoke test. A separate CI job runs Compose validation and builds both images without clinical data, weights, or production credentials.

## 8. Health semantics and limitations

- `GET /health` on Express and `GET /health/live` on FastAPI are liveness probes only.
- `GET /readyz` on Express checks its configured FastAPI dependency; an unreachable dependency returns HTTP 503 without exposing internal host details.
- `GET /health/ready` on FastAPI means the API process can serve requests and status reports. It does not mean data, governance, or a model is ready.
- `GET /health/model` returns HTTP 503 until genuine clinical model readiness is established. `/api/ml/models/status` and dataset/governance APIs expose their specific status separately; do not interpret their HTTP 200 status-report response as clinical authorization.
- `/api/health` reports `research` with clinical readiness `blocked` when FastAPI is reachable but Track D is unavailable; it returns HTTP 503 when the configured ML dependency is unavailable. The public security summary omits checkpoint hashes/names, filesystem paths, and raw class counts.
- Missing data is reported as not ready. Production config failures stop startup. Health responses do not disclose credentials, filesystem paths, reviewer rows, or exception tracebacks.

This repository currently lacks the historical cohorts/split manifests and frozen checkpoint noted in `docs/STAGE36_REGRESSION_TRIAGE.md`. No dataset readiness or clinical validation is inferred from container health. Ethics approvals, institutional agreements, access authorization, reviewer identities, a genuine trained model, external validation, and clinical governance remain external requirements. The service is research infrastructure and does not provide clinical diagnosis in this state. Abstention is never a negative or healthy classification.

## 9. Troubleshooting

| Symptom | Check |
| --- | --- |
| Compose says `.env` is missing | Copy `.env.example` to `.env`; do not commit secrets. |
| Gateway exits in production | Check presence and length of clinical API, gateway HMAC, and audit encryption secrets; inspect logs without printing secret values. |
| ML service exits in production | Check gateway HMAC, ledger HMAC, explicit CORS, ledger path, and institution-provisioned reviewer registry mount. |
| Port 3000 is occupied | Set `HOST_PORT` to another free loopback port in `.env`. |
| Gateway `/readyz` returns 503 | Check `docker compose ps` and `docker compose logs ml-service`; `/health/model` may still be 503 while API readiness is healthy. |
| Image build fails on Python wheels | Verify Docker has sufficient disk and memory for CPU PyTorch, and check package index/network access. |
| Node dependency installation fails | Run `npm ci` with network access and verify `package-lock.json` is present. |
| Dataset/model status is blocked | Expected without genuine authorized data/manifests and a valid explicitly SHA-pinned Track D checkpoint. Do not create dummy artifacts to clear the gate. |
| Reviewer endpoints return 503 | Confirm the externally provisioned registry path is mounted read-only and ledger HMAC key/persistence are configured. No default reviewer is supplied. |

## 10. Verification results in this workspace

- Docker availability: **not available** (`docker` command was not found). Dockerfiles, Compose startup, container health, and image builds therefore remain unverified locally. The CI image-build job is configured but has not run in this workspace.
- `npm ci`: **passed**, 384 packages installed; npm reported 0 vulnerabilities.
- Python deployment/governance CI regression command: **72 passed**.
- `npm test -- --reporter=dot`: **20 passed** across 8 files. The existing error-boundary test logs its simulated component error while passing.
- `npm run lint`: **passed** (`tsc --noEmit`).
- `npm run build`: **passed**. Vite emitted its non-failing large-chunk advisory.
- `npm run test:deployment`: **passed** against actual local Express and FastAPI processes, not containers. It verified health/status, gateway calls, auth failures, upload constraints, dependency failure, production config rejection, and blocked Track D.
- `python -m compileall -q ml_service osteoguard scripts/deployment-smoke.py`: **passed**.
- Vitest validates Compose YAML and deployment contracts. `docker compose config`, Dockerfile lint/build, container health, and container networking were **not run locally** because Docker is unavailable. CI is configured to validate Compose and build both images.
- Graceful shutdown is implemented in Express, and the Linux CI smoke test is configured to exercise it. That CI job has not run in this workspace. This Windows run used bounded process termination because detached children could not receive console signals here.
- No real clinical cohort, protected checkpoint, production credential, or reviewer registry was supplied or created.
