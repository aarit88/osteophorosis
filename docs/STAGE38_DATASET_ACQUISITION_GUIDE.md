# Stage 38 — Dataset Acquisition and Cohort Onboarding

**Status:** Infrastructure implemented. No external dataset was downloaded, registered as acquired, or staged in this work. Track D remains unavailable and this workflow does not authorize training.

## Operating principles

Register a source before transfer. Keep acquired packages under the authorized `OSTEO_ONBOARDING_ROOT/<source_id>/<version>/` directory on storage approved for the source's access restrictions. Place the Stage 33 intake manifest at `manifest.json`; the existing Draft 2020-12 Track D validator is run during package verification. No upload endpoint is provided for large or potentially restricted medical images. Transfer must use the institution's approved secure method, with access control and encryption appropriate to the agreement.

An authorized reviewer uses the authenticated `/api/v1/dataset-onboarding` routes to register a source, record lifecycle events with evidence references, inspect provenance, verify the package, and stage it for Stage 33. Access approval requires a reference to external authorization; entering a reference does not itself prove approval. Verification requires recorded license/data-use and ethics references. These references are recorded as documentation, not independently attested by this service. `governance_approved` remains false. Staging copies to a versioned directory and does not initiate model training.

The registry file is atomically replaced. Its event history is hash chained and integrity checked on load; in production, set an externally supplied `OSTEO_ONBOARDING_LEDGER_HMAC_KEY` of at least 32 characters. Development without that key uses an unkeyed checksum for accidental tamper detection only. Local hashes and chain integrity do not prove provenance authenticity, permission, ethics approval, or clinical validity. Back up the persistent registry, acquired source packages, and staged packages under the institution's approved encrypted backup procedure.

Package verification creates a sorted file inventory with byte sizes and SHA-256. It rejects symlinks, paths outside the configured package root, empty files, unsupported file extensions, and packages without the Stage 33 manifest. Stage 33 performs the imaging, privacy, manifest, ontology, patient-grouping, holdout and dataset checks. A successful image or manifest check is not a clinical validity decision. Reverify before stage; staging re-hashes the copied tree and rolls back on mismatch. Staged data is only a handoff to Stage 33 intake and remains outside training.

## API and lifecycle

Every endpoint below requires an institutional bearer token from the externally provisioned reviewer registry. Role checks apply per action: source creation and package staging require `DATA_STEWARD` or `PRINCIPAL_INVESTIGATOR`; access authorization requires `PRINCIPAL_INVESTIGATOR`, `DATA_PROTECTION_OFFICER`, `IRB_OFFICER`, or `LEGAL_COUNSEL`; other state changes have their own restricted roles.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/dataset-onboarding/sources` | List authorized source registry entries |
| `POST /api/v1/dataset-onboarding/sources` | Register a documented source (schema in OpenAPI) |
| `GET /api/v1/dataset-onboarding/sources/{source_id}` | Read acquisition, verification, integrity and staging states |
| `POST /api/v1/dataset-onboarding/sources/{source_id}/events` | Record a validated acquisition lifecycle transition |
| `GET /api/v1/dataset-onboarding/sources/{source_id}/provenance` | Read source details and verified event chain |
| `POST /api/v1/dataset-onboarding/sources/{source_id}/verify` | Re-run package inventory and Stage 33 validation |
| `POST /api/v1/dataset-onboarding/sources/{source_id}/stage` | Atomically stage a verified source version for Stage 33 |

The lifecycle is `REGISTERED → ACCESS_PENDING → ACCESS_APPROVED → ACQUIRED → VERIFICATION_PENDING → VERIFIED → ARCHIVED`, with explicit `REJECTED` and `RESTRICTED` paths. Invalid or skipped transitions fail. `VERIFIED` indicates package integrity and Stage 33 intake checks, not dataset readiness. Acquisition and staging do not change the Track D readiness state.

Example source registration body (replace every example reference with real operator-provided information; this example does not assert any approval):

```json
{
  "source_id": "SRC_LOCAL_001",
  "name": "Operator supplied source name",
  "source_type": "INSTITUTIONAL",
  "source_reference": "operator supplied reference",
  "description": "Operator supplied description",
  "modalities": ["xray"],
  "target_categories": ["PAGETS_DISEASE"],
  "acquisition_method": "Operator supplied secure transfer method",
  "access_restriction": "RESTRICTED",
  "version": "v1",
  "intended_use": "Operator supplied permitted research purpose"
}
```

No source record is preloaded. Omitted license, DUA, ethics, acquisition date, owner, contact, and fingerprint fields remain unknown. The interface does not invent their values.

## Candidate sources reviewed (23 November 2026)

These are **potential sources only**, not registered or approved by OsteoGuard. Review the current dataset page, terms, project eligibility, and institutional obligations before making any access request or transfer.

| Cohort / modality | Evidence and access | Fit and limitations |
| --- | --- | --- |
| Bone metastasis / CT — [TCIA Spine-Mets-CT-SEG](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/) | TCIA documents 55 human subjects with CT, segmentations and vertebral lesion classification; its page requires following the TCIA Data Usage Policy and dataset citation. | Potential CT lesion research cohort only. Lesion categories and primary cancers do not directly establish the Track D four-way patient-level diagnosis labels. Small cohort, treatment-related repeated scans, and patient-level grouping need review. Confirm current access/terms and permissible secondary use with TCIA and the responsible institution. |
| Bone metastasis / bone scintigraphy — [IICS-UNA repository record](https://repositorio.conacyt.gov.py/handle/20.500.14066/4505?locale-attribute=en) | The repository page describes 582 images from 291 patients and identifies CC BY-NC 4.0 terms. | Potential research candidate; review the dataset article, actual delivered file format, label definitions, patient grouping, restrictions and whether the non-commercial license fits the intended use. A repository listing is not an authorization to download or train. |
| Bone scintigraphy — [BS-80K article](https://doi.org/10.1016/j.compbiomed.2022.106221) | The article describes over 82,000 images from 3,247 patients, including specialist labels and anterior/posterior views, and points to a public download location. | Potential metastasis benchmark only. The paper's “publicly available” description is not a complete use license or a governance approval. Confirm license, terms, label provenance, redistribution rules, and access with the authors before acquisition. Patient-level splits must keep repeated views/images together. |
| Healthy skeletal CT — [TCIA Healthy-Total-Body-CTs](https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/) | TCIA describes 30 healthy adult volunteers imaged with low-dose whole-body CT and tissue segmentations; collection metadata lists NIH Controlled Data Access Policy and CC BY 4.0. | Potential healthy-control imaging only. Small healthy adult cohort, repeated timepoints for some volunteers, protocol differences, and controlled-access conditions require review. Confirm precise permitted use and whether its health definition satisfies the project's control criteria. |
| Osteogenesis imperfecta / clinical registry — [EuRREB OI module](https://eurreb.eu/osteogenesis-imperfecta-oi/) | The registry describes OI clinical data elements and says access requires joining its Core Registry. | Not a medical-image dataset. May help define clinical variables or identify a collaboration pathway; does not supply X-ray/CT or grant image access. Seek registry participation terms and a separate imaging collaboration/DUA. |
| Osteogenesis imperfecta / imaging | A published OI CT study reports the image data are not public because CT data and DICOM headers contain patient information; the authors describe access on reasonable request ([article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12131355/)). | Institutional author request and governance review would be required. No dataset access is asserted. A paper with radiograph figures is not a reusable image cohort. |
| Paget disease / imaging | The sources reviewed did not identify a verified, openly licensed, patient-level Paget X-ray/CT/scintigraphy dataset appropriate for this four-class task. A clinical imaging review describes radiography and scintigraphy as relevant modalities but is not a dataset ([review record](https://pubmed.ncbi.nlm.nih.gov/35672089/)). | Treat as an institutional acquisition need. Search through academic/clinical collaborations and request access under approved protocol and DUA; validate independent diagnosis and label definitions. Do not substitute teaching images or synthetic samples. |

### Access, licensing and suitability notes

- TCIA collection policies and licensing can vary by collection; use the current collection page and [TCIA Data Usage Policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/), retain required attribution, and check any restricted collection agreement.
- PhysioNet explains that access levels may be open, restricted (registered users and DUA), credentialed (credentials and DUA), or contributor-review (project plan approved by the data contributor) ([access policy](https://physionet.org/content/credentialed-access/)). These requirements illustrate that repository availability does not equal permission. No Stage 38 candidate above is assumed to be on PhysioNet.
- License compatibility, transfer authority, data minimization, de-identification, retention, publication, redistribution, commercial use, and cross-border transfer must be reviewed by the data owner and the responsible institutional ethics/privacy/legal authorities.
- Dataset-level diagnosis does not automatically create validated per-image labels. Keep acquisition, source authorization, clinical label adjudication, image QC, patient-level split construction and governance approval as separate recorded gates.
- Do not include protected external holdouts, Track C evaluation data, or unauthorized research data in the Track D acquisition root.

## Intake checklist

Before transfer: identify data owner, obtain documented authority and any required ethics/DUA/licensing approvals, define intended use and restrictions, confirm label and modality availability, and provision a least-privilege storage location. At receipt: record version and date, preserve a source-provided checksum when available, transfer using approved encrypted channels, and pseudonymize patient identifiers under the approved process. Verify and review actionable findings. Reconcile all patients across institutions and modalities before Stage 33 intake, enforce patient-level grouping and protected-holdout exclusions, and maintain a documented rollback/quarantine route. A data steward must approve each gate using the institutional identity registry; these application records are not substitutes for institutional decisions.

## Not performed

No records were added to the source registry during implementation. No clinical data, images, approvals, agreements, reviewer identities, or model weights were obtained, created, downloaded, or copied. Any proposed candidate remains unverified against the project's specific ethics protocol, data-use purposes, per-patient labels, modalities, and split requirements until an authorized institution completes those checks.

## Verification performed

- Stage 33 intake, Stage 34 ontology/adjudication, Stage 35 governance/security, Stage 36 readiness/checkpoint regressions, Stage 37 deployment health, Stage 38 onboarding, and Track D no-data guardrails: **102 passed, 1 skipped**. The skipped test could not create an OS symlink under this Windows account; symlink rejection is implemented and covered where symlink fixtures are available.
- `npm ci`: **passed**, 384 packages installed; 0 vulnerabilities reported.
- Vitest: **23 passed** across 10 files, including onboarding API contracts and lifecycle display.
- `npm run lint`: **passed**.
- `npm run build`: **passed**; Vite reported the existing non-failing large-chunk advisory.
- `npm run test:deployment`: **passed** against live FastAPI and Express processes, including unauthenticated rejection on the proxied Stage 38 registry route and Track D remaining blocked.
- Python compilation and `git diff --check`: **passed**.
- The full historical Python suite remains blocked by absent cohort data/manifests and model artifacts: **68 failed, 387 passed, 16 skipped, 54 errors**. This matches the documented Stage 36 external-input limitation; no missing data or checkpoints were fabricated.
- Docker is unavailable in this workspace, so container image/Compose execution was not repeated or claimed as verified.
