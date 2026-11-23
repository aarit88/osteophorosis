# Stage 44 — Research Modules

## Scope and safety

Stage 44 improves the existing `/tracks`, `/radiomics`, and `/biomechanics` pages. It adds no model training, checkpoint loading, dataset upload, or clinical image inference. The research calculation APIs accept synthetic phantom configurations or hypothetical Mirels category values; they do not accept a patient, image, study, or Stage 38/39 dataset reference. Those interfaces must not be used to bypass the dataset onboarding and preparation gates.

Track A and B historical holdout evaluations remain protected. Track C remains frozen; its external center holdouts must not be re-evaluated or used for model selection. Track D's four-class inference remains blocked with `FOUR_CLASS_MODEL_NOT_READY` until its existing readiness requirements are satisfied. No checkpoint is exposed by these pages.

## Tracks page

`/tracks` uses `GET /api/v1/research/tracks` and renders only the tracks returned by the backend. It shows their registry task, modality, target classes, governance/source context, limitations, and a cautious status summary. It does not show raw historical metric values, paths, hashes, checkpoint files, or training commands. “Historical frozen” is registry status and does not prove a checkpoint can currently be loaded; current artifact availability is not tested by this page. Track A/B holdout status and Track C generalization restrictions are surfaced as restrictions. Track D links to Governance and is shown as blocked.

## Radiomics

### Implemented endpoint and inputs

`POST /api/v1/radiomics/demo` generates a 2D mathematical phantom and runs `IBSIRadiomicsEngine.extract_features_dict(image, mask, spacing)`. The page requires an explicit action and accepts one of four generated phantom types (`checkerboard`, `linear_gradient`, `annulus`, `uniform`), grid sizes 32/64/128 pixels, and spacing 0.5/1/2 mm per pixel. The schema permits grid sizes 8–128 and spacing 0.1–5 mm. Unknown phantom names are rejected with HTTP 422 rather than silently routed to the uniform phantom.

### Feature families and validation state

The engine produces a 102-value engineering vector across first-order statistics (18), GLCM (26), GLRLM (16), GLSZM (16), and a combined GLDM/NGTDM-related family (26). The current project verification records identify 81 features as mathematically verified and 21 as heuristic/unverified. Therefore the UI calls this specification partially verified; it does not claim the full 102-vector is IBSI certified. The current API returns representative samples (up to eight from each reported subset), not all feature values. Shape features are not exposed by this radiomics endpoint. Geometry measurements are provided separately by the synthetic biomechanics endpoint.

The returned values can be read as numerical outputs of the selected phantom and configuration only. They are not clinical measurements. The UI shows the API's failure response, the actual phantom configuration, and client-observed receipt time. This response does not contain a server timestamp, software build identifier, or input dataset reference; the page reports those as unavailable rather than inventing them.

## Biomechanics

### Synthetic 2D cross-section geometry

`POST /api/v1/biomechanics/geometry` accepts one of the eight `SyntheticBonePhantomSuite` identifiers and a scalar spacing between 0.1 and 5 mm/pixel. The engine validates a non-empty binary 2D mask with at least one foreground pixel. It calculates cortical area (mm²), centroid (mm), second moments (`I_xx`, `I_yy`, `I_xy`, mm⁴), polar moment (`J`, mm⁴), and dimensionless retention ratios against the intact synthetic ring. Regular phantoms use the selected isotropic spacing. `anisotropic_spacing_annulus` preserves its factory-defined `dy=0.4`, `dx=0.8` mm spacing; the API returns effective spacing explicitly. The API rejects unsupported phantom names with HTTP 422. The calculation does not estimate material properties, patient-specific bone strength, or fracture risk.

### Conventional Mirels score rule

`POST /api/v1/biomechanics/mirels` accepts four integer categories, each 1–3: site, pain, radiographic nature, and relative lesion size. It returns the total and a conventional rule tier, along with other historical rule fields. The page requires all four hypothetical category selections and displays only the score and rule tier. It suppresses the endpoint's historical percentage and treatment-recommendation fields to avoid presenting them as a patient-specific fracture-risk prediction. The score was developed for metastatic long-bone lesions and is not a substitute for clinical assessment. No patient-specific evidence is accepted or submitted.

## Traceability and limitations

Each calculation is explicit and uses the existing API client and backend routes. The UI records the exact submitted configuration and the browser-observed response time. The APIs do not currently return a software version, server timestamp, or governed study reference. No PHI is rendered. No progress is shown because these endpoints provide no progress events. Recent activity is not shown because no activity history API is connected.

The synthetic calculations are not tied to Stages 38/39. Any future governed image analysis must use approved source onboarding, verification/staging, preparation, and patient-level safeguards. This stage does not establish clinical validity, enable Track D, grant dataset access, or authorize model training/evaluation.

## Verification coverage

Frontend regression tests cover track metadata and restrictions, explicit/no-fabrication behavior for radiomics, API validation/error states, supported phantom selection, geometry units and effective spacing, and required Mirels inputs with suppression of risk/treatment claims. Backend API regression tests cover supported geometry spacing, anisotropic spacing metadata, and rejection of unknown phantom names.
