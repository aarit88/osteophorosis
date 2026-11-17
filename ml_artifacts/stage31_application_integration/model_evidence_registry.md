# Model Evidence Registry & Multi-Center Evaluation Ledger

## 1. Scope and Purpose
This registry provides an auditable, transparent ledger of every model checkpoint developed across the OsteoGuard AI research trajectory. It documents training stages, architecture backbones, internal and external validation metrics, cryptographic hashes, generalization findings, and clinical deployment statuses verified against original stage reports.

---

## 2. Central Model Ledger

| Model ID | Research Track | Architecture Backbone | Checkpoint Hash (SHA-256) | Training Status | Key Metrics | External Holdout Status | Clinical Clearance |
|---|---|---|---|---|---|---|---|
| `osteo-trackA-fracatlas-v1` | Track A: FracAtlas | ResNet-101 (Pretrained IMAGENET1K_V2) | Stage 09 Epoch 9 Verified | FROZEN (Stage 09) | ROC-AUC: 0.9117, PR-AUC: 0.7537, Sens: 90.00%, Spec: 74.40%, Acc: 77.20% (θ=0.31) / 88.27% (θ=0.50) | Evaluated in Stage 10 (Burned Holdout N=614) | RESEARCH_ONLY |
| `osteo-trackB-scintigraphy-v1` | Track B: Scintigraphy | 5-Fold ResNet-18 Patient-Aware Ensemble (Clinical Max-Pooling) | Stage 12 CV Verified | FROZEN (Stage 13) | ROC-AUC: 0.8317, PR-AUC: 0.7257, Acc: 84.09%, Bal Acc: 77.62%, Sens: 66.67%, Spec: 88.57% (θ=0.60) | Evaluated in Stage 14 (Burned Holdout N=44 Patients, 88 Scans) | RESEARCH_ONLY |
| `osteo-trackC-btxrd-frozen-v1` | Track C: BTXRD | ResNet-34 Aspect-Ratio Letterbox (512x512) + Focal Loss | `21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD` | FROZEN (Stage 18 Locked) | C1 Val Acc: 81.80%, Macro-F1: 0.8103; C2 Test Acc: 51.73%, Macro-F1: 0.5069; C3 Test Acc: 47.10%, Macro-F1: 0.3592 | Center 2 (Stage 19, N=549) & Center 3 (Stage 20, N=259) Burned Holdouts | NOT_SUITABLE_FOR_DEPLOYMENT |
| `osteo-trackD-fourclass-unified` | Track D: Four-Class Unified | Dual-Stream Swin + IBSI (512+102 Fusion) | None (Untrained) | BLOCKED_BY_DATA | N/A (Zero clinical weights trained) | PENDING_DATA_ACQUISITION | BLOCKED |

---

## 3. Cryptographic Checkpoint Verification
The primary frozen deep learning artifact is located at:
- **File System Location**: `ml_artifacts/btxrd_frozen_model/best_checkpoint.pt`
- **Algorithm**: Secure Hash Algorithm 256-bit (SHA-256)
- **Verified Digest**: `21064670fa6277b0e96135333708de4692ddb80e428b792f3675405b5dc109cd`
- **Status**: Verified intact during Stage 31 integration testing.

---

## 4. Multi-Center Generalization Collapse Findings (Track C)

During Stages 19, 20, and 21, the frozen BTXRD model (`osteo-trackC-btxrd-frozen-v1`) was evaluated against independent hospital holdouts to test generalizability across clinical imaging sites:

```
Center 1 (Internal Val Cohort, N=588):      [████████████████████░░░░░]  81.80% Acc (Macro-F1: 0.8103, Bal Acc: 82.90%)
Center 2 (Primary External Holdout, N=549): [█████████████░░░░░░░░░░░░]  51.73% Acc (Macro-F1: 0.5069, Bal Acc: 53.74%)  [-30.07%]
Center 3 (Tertiary Stress Test, N=259):     [███████████░░░░░░░░░░░░░░]  47.10% Acc (Macro-F1: 0.3592, Bal Acc: 44.67%)  [-34.70%]
```

### Detailed Metric Breakdown Across Clinical Centers:

| Evaluation Metric | Center 1 Internal Val ($N=588$) | Center 2 External Holdout ($N=549$) | Center 3 External Stress ($N=259$) |
|---|---|---|---|
| **Overall Accuracy** | **81.80%** | **51.73%** [47.72%, 55.74%] | **47.10%** [40.93%, 53.28%] |
| **Balanced Accuracy** | **82.90%** | **53.74%** [49.16%, 58.44%] | **44.67%** [36.94%, 52.60%] |
| **Macro-F1 Score** | **0.8103** | **0.5069** [0.4648, 0.5494] | **0.3592** [0.3007, 0.4183] |
| **Weighted-F1 Score** | **0.8141** | **0.5210** [0.4802, 0.5598] | **0.5285** [0.4687, 0.5866] |
| **Macro ROC-AUC (OvR)** | **0.9114** | **0.7180** [0.6841, 0.7512] | **0.6468** [0.5853, 0.7043] |
| **Macro PR-AUC (OvR)** | **0.8700** | **0.5683** [0.5240, 0.6212] | **0.4558** [0.4060, 0.5327] |
| **Malignant Sensitivity** | **90.24%** (37/41) | **59.21%** (45/76) | **67.74%** (21/31) |
| **Malignant Specificity** | **97.81%** (535/547) | **84.36%** (399/473) | **69.30%** (158/228) |
| **Malignant Precision (PPV)** | **75.51%** | **37.82%** | **23.08%** |

### Root Cause Analysis of Generalization Collapse:
1. **Radiographic Parameter Drift**: Center 1 utilized high-kVp digital radiography with automated exposure compensation, whereas Centers 2 and 3 utilized computed radiography (CR) plates with heterogeneous exposure levels.
2. **Post-Processing Edge Enhancement**: Center 3 images underwent aggressive vendor-specific sharpening filters, introducing high-frequency spatial artifacts that distorted deep convolutional feature maps.
3. **Epidemiological Referral Bias (Center 3)**: Center 3 had an extreme tumour referral prevalence (89.58% tumours, only 10.42% normal), causing normal-class sensitivity to collapse to **18.52%** (5/27).
4. **Clinical Governance Conclusion**:
   The model demonstrated acute site overfitting. Evaluated partitions from Center 2 ($N=549$) and Center 3 ($N=259$) are permanently categorized as **burned test holdouts** and must not be retroactively used for model fine-tuning.

---

## 5. Metadata Traceability Limitations (Track A)
- FracAtlas open dataset does not provide unique patient identifier fields.
- Although images were divided using stratified random splits in Stage 06.2 ($N=614$ test), multi-projection radiographs (AP, lateral, oblique) from the same patient may have crossed partition boundaries.
- Track A accuracy represents image-level recognition rather than guaranteed patient-level generalization.
