# Stage 39 — Dataset Harmonization, QC and Split Preparation

## Scope and safety boundary

Stage 39 prepares verified, staged Track D packages from Stage 38 for future research. It inventories files, runs technical image QC, writes separately versioned harmonized outputs for supported configurations, prepares patient-grouped internal split assignments, and produces JSON and Markdown reports. It does not train, evaluate, calibrate, select, deploy, or mark a model ready. The Track D diagnostic routes remain gated by the existing `FOUR_CLASS_MODEL_NOT_READY` response.

Stage 39 accepts only a Stage 33 manifest with `track: track_d`, backed by a Stage 38 source whose package is verified and staged. It checks the staged file inventory against `provenance.json` before every operation. Stage 33 remains the downstream intake gate; Stage 39 is not a replacement for its validation or for institutional governance review.

## Lifecycle and operations

Runs record source/version, staging reference, manifest/config version, run ID, timestamps, pseudonymous operator ID, state history, validation status, blockers, and report integrity. The lifecycle is `SOURCE_STAGED → INVENTORY_PENDING → INVENTORY_COMPLETED → QC_PENDING → QC_COMPLETED → HARMONIZATION_PENDING → HARMONIZATION_COMPLETED → SPLIT_AUDIT_PENDING → SPLIT_AUDIT_COMPLETED → PREPARATION_COMPLETED`. A blocking QC or failed operation stops the run in `BLOCKED`; failed gates never advance automatically. Split preparation may follow passed QC directly when harmonization is not requested.

The authenticated operator API is under `/api/v1/dataset-preparation` and is proxied through the Express gateway:

- `GET /runs`, `POST /runs`, `GET /runs/{run_id}`
- `POST /runs/{run_id}/inventory`, `/qc`, `/harmonize`, `/splits`
- `GET /runs/{run_id}/report?format=json|markdown`

Only reviewers with `DATA_STEWARD` or `PRINCIPAL_INVESTIGATOR` roles can create or operate runs. Other endpoints use the same reviewer bearer authorization as Stage 38. The frontend keeps the token in component memory only. Operations are synchronous and bounded to 50,000 manifest records per run; large cohorts should be partitioned into separately governed source versions.

## Inventory and privacy

The inventory records source/version, file reference, file size/type, SHA-256, dimensions when a common image decoder can read the file, exact duplicates, optional perceptual hashes, class/modality counts, missing metadata keys, and a deterministic inventory fingerprint (timestamp excluded). Patient and sample identifiers are never emitted: stable pseudonymous group/file references are derived for the report. Institution names are represented by hashed references in inventory details. Raw manifests and source files are left untouched.

Before processing, the service rejects path traversal, symlinks, missing files, non-Track-D manifests, and staged files that differ from the Stage 38 provenance inventory. It also checks the provenance inventory fingerprint against the verified Stage 38 registry record and pins the staged provenance bytes into the preparation run, so later changes are detected. Hashes establish integrity relative to the trusted registry record; they do not prove source authenticity or clinical validity.

## QC rules and limitations

Findings include rule ID, severity (`INFO`, `WARNING`, `ERROR`, `BLOCKER`), status, technical finding, non-identifying evidence, recommendation, and whether the issue blocks preparation.

- Raster X-ray/scintigraphy: Pillow decode/verify, two-dimensional non-empty pixels, and non-zero dynamic range. Corrupt and uniform images block. No clinical interpretation is attempted.
- DICOM: pydicom decode, pixel decode/finite-value check, and screening for PatientName, PatientID, PatientBirthDate, PatientAddress, and OtherPatientIDs. Detected tag names (never tag values) block processing. The current screening is a limited direct-identifier screen, not a complete de-identification certification.
- CT DICOM: modality, rescale slope/intercept, and `RescaleType=HU` are required before windowing; arbitrary pixel values are never labeled HU.
- NIfTI: nibabel header bounds (2–4 dimensions, each axis ≤8192), finite positive voxel spacing, and decoded finite voxel values are checked.

QC is technical and deterministic for a fixed file and installed decoder version. It is not a clinical assessment. The pipeline does not infer missing scintigraphy views or fill missing CT slices. Current DICOM processing handles individual image objects/slices; series reconstruction and gap detection are not implemented and must be completed before workflows that need reconstructed CT volumes.

## Harmonization configuration

Each run stores a strict, versioned `PreparationConfig` (`1.0.0`). X-ray defaults use grayscale, percentile clipping at 0.5/99.5, aspect-preserving 1024×1024 resize, and zero padding. Bone scintigraphy defaults clip source intensities at 25,000 counts, then produce an aspect-preserving 256×1024 8-bit research image. No augmentation or label-conditioned transform is used. Source files are never modified; outputs live under the configured Stage 39 runs directory in a separate `prepared/` folder. Existing outputs cannot be overwritten.

CT uses configurable bone-window center/width. A DICOM object must explicitly declare CT and HU rescale type with valid finite slope/intercept. NIfTI requires an operator-supplied source-documented HU evidence reference in the run configuration because NIfTI storage units alone do not establish HU semantics. CT outputs retain shape and geometry in a JSON sidecar; no spatial resampling or multi-slice DICOM series reconstruction is currently performed. Unsupported inputs/configurations fail closed.

Preprocessing parameters are recorded, but no dataset-level normalization statistics are fit. The default transform is per-image and does not use labels. Any future training pipeline must independently review suitability and fit any learned normalization only on its permitted training partition.

## Patient grouping, splits and protected data

Stage 39 calls the existing patient-level split generator and cross-split leakage auditor from Stage 33. A patient remains a single group across records, views, studies, modalities, and institutions represented in that manifest. Deterministic seed and configurable positive train/validation/internal-test ratios are recorded. Missing classes are reported rather than synthesized or silently removed. Sparse strata may yield empty internal test partitions; no balance is fabricated.

The centralized protected holdout registry is checked per record (site/source/sample ID/hash); protected markers such as Center 2/3 and the FracAtlas held-out test partition are excluded before split generation. Protected external data never enter Stage 39 preprocessing or split generation. Track A/B/C manifests are rejected; no Track A/B/C checkpoints, splits, or evaluation procedures are changed. Institution overlap is reported as an analysis item, while patient-level exclusivity remains mandatory. The service prepares one registered source version per run; it does not merge patient identities across separately registered sources, so cross-source cohort union requires a separately governed linkage process before use.

## Reports, storage and operations

The HMAC-protected registry is stored at `OSTEO_PREPARATION_REGISTRY_FILE`; run reports and prepared outputs are stored under `OSTEO_PREPARATION_ROOT`. In production, `OSTEO_PREPARATION_HMAC_KEY` must be externally provisioned and at least 32 bytes. Registry writes and output creation use temporary files/directories and atomic replacement. Reports are persisted as `report.json` and readable `report.md`; the JSON report includes a reproducible SHA-256 integrity reference. Report integrity is checked on access.

Reports explicitly state `clinical_validity=NOT_ESTABLISHED`, `clinical_readiness=NOT_ESTABLISHED`, and `training_authorized=false`. API responses avoid raw patient IDs and direct DICOM identifier values. No image upload endpoint is added, and prepared directories are not publicly served.

## Example workflow

1. Register and acquire data through Stage 38 using the institution's actual approvals and terms.
2. Verify package integrity and Stage 33 intake, then stage the documented Track D package.
3. A data steward or PI creates a run with a versioned configuration.
4. Generate inventory, review QC, and address blockers at the source.
5. Run supported modality harmonization if needed; otherwise proceed directly to split audit after QC.
6. Review patient-level leakage, protected holdout exclusions, missing classes, and institution overlap.
7. Export the preparation report for governance review. This does not authorize model training.

## Requirements before training

Real authorized clinical data, appropriate institutional governance and data-use terms, completed de-identification/privacy review, label validity/adjudication, Stage 33 intake approval, a justified statistical analysis plan, independent validation, and all model-specific readiness gates remain required. Stage 39 does not satisfy or waive those requirements.
