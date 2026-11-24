# Stage 45 — Explainability, Uncertainty, and Results

## Implemented

- Explainability and uncertainty pages now report method availability and limitations without generating synthetic heatmaps, probability scenarios, or patient-specific plots.
- Both pages query the existing Track D inference gate and show its actual response, or a safe retryable unavailable state.
- The Results page no longer restores prior responses from browser storage. Results are held in memory for the current application session only.
- Results with missing or invalid probability and uncertainty fields, an out-of-distribution flag, or a high candidate entropy value are withheld. Abstention never becomes a negative or healthy label.
- Grad-CAM, SHAP, counterfactual, fracture-risk, and report panels are not generated without a verified model-specific service.
- Synthetic radiomics, geometry, explanation, uncertainty, and workbench endpoints return `SYNTHETIC_DEMO_NOT_AVAILABLE` in the runtime service. Python phantom tests run only in the isolated pytest environment.

## Current availability

Track D remains blocked by `HTTP 503 FOUR_CLASS_MODEL_NOT_READY`. No patient-specific Track D explanation, calibration, uncertainty, or diagnostic result is available. The proposed normalized entropy cutoff of 0.60 remains an engineering candidate and is not a clinically validated decision threshold.

The Results page can render only a response held in application memory. It does not create reports, retain images, or expose protected patient records. The current backend gate prevents Track D inference from creating such a response.

## Safety boundary

An automated explanation describes model behavior; it does not establish medical correctness or causality. Abstention requires human review and is not a negative diagnosis. The manual Mirels arithmetic calculator is separate from image-derived biomechanical analysis and is not a clinical risk determination.
