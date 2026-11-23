# Implemented security and artifact controls

## Reviewer authorization and adjudication ledger

Reviewer authorization is provisioned outside source control with `OSTEO_REVIEWER_AUTHORIZATION_FILE`. The JSON file has `registry_version: "1.0"` and a `reviewers` list containing each pseudonymous ID, allowed roles, active status, and SHA-256 of a long random bearer token. Store token digests only; distribute the raw token through the institution's approved secret channel. Workflow transitions require a bearer token and the submitted reviewer ID and role must match the authenticated principal.

Set `OSTEO_AUDIT_LEDGER_HMAC_KEY` to a secret with at least 32 bytes and `OSTEO_ADJUDICATION_LEDGER_FILE` to an institution-managed ledger path to enable adjudication writes. Ledger version 2 uses canonical JSON and HMAC-SHA256; load rejects tampering and ontology-version mismatches. There is no generated fallback key.

## Production API settings

Set `NODE_ENV=production`, `OSTEO_CLINICAL_API_TOKEN` and `OSTEO_GATEWAY_HMAC_SECRET` to independent secrets of at least 32 characters, and `OSTEO_CORS_ORIGINS` to a comma-separated list of exact trusted origins. Wildcard production origins are rejected. The FastAPI service also requires the gateway HMAC secret in production. The Express image API accepts PNG, JPEG, WebP, TIFF, and BMP raster uploads up to 15 MiB, validates content through the image sanitizer, and rejects unsupported or corrupt uploads.

## Checkpoint inventory

Run `python -m ml_service.utils.checkpoint_inventory` for a read-only inventory with SHA-256 checks. Track B fold paths in `ml_artifacts/bone_scan_ensemble/ensemble_config.json` are repository-relative. Track C is checked against its recorded size and SHA-256; missing weights are reported as `MISSING`, never downloaded, regenerated, or replaced. Historical evaluation metrics remain historical evidence and do not imply that a checkpoint file is present in this checkout.

## Track D gate

The four-class inference route returns HTTP 503 with `FOUR_CLASS_MODEL_NOT_READY` until Track D intake readiness and institutional governance approval both pass, regardless of any A, B, or C checkpoint. A future Track D model must be explicitly configured through `OSTEO_ACTIVE_CHECKPOINT_PATH` and pinned by `OSTEO_TRACK_D_CHECKPOINT_SHA256`; generic checkpoint-directory discovery is disabled. Stage 33 dataset readiness responses include the live manifest schema validation state. Readiness and governance screens read these API responses.
