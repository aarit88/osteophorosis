# STAGE 11B — BONE SCINTIGRAPHY PILOT EXPERIMENT SUMMARY

**Pilot Status:** PASS  
**Model:** BoneScanResNet18 (ImageNet Pretrained ResNet-18, 1-channel adapted)  
**Evaluated Split:** Training CV Fold 0 (Validation: 42 patients; Training: 161 patients)  
**Original Validation & Test Sets:** Strictly untouched (0 access)  

### Patient-Level Validation Performance (Epoch 1):
- **Patient ROC-AUC:** 0.8822
- **Patient PR-AUC:** 0.6594 (Baseline prevalence: 0.2143)
- **Sensitivity:** 100.00% (9 / 9)
- **Specificity:** 18.18% (6 / 33)
- **Precision:** 25.00%
- **F1-Score:** 0.4000
- **Balanced Accuracy:** 59.09%
- **Peak VRAM:** 195.0 MiB (<18% of 4 GB RTX 3050 A)
