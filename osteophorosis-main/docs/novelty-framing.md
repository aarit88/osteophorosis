# OsteoGuard AI — Academic Novelty & Contribution Framing
> **Manuscript Section:** Introduction & Interface Contribution  
> **Target Venue:** IEEE Transactions on Medical Imaging / JAMIA / MICCAI Clinical XAI Workshop  
> **Authorship Separation Disclaimer:** *Deep learning model training, backbone selection, and GPU optimization belong to the machine learning sub-team. The author's primary novelty and undergraduate capstone contribution resides in the uncertainty-aware abstention interface, clinician-in-the-loop oversight audit seam, and traceable superscript explainability narrative.*

---

## Academic Novelty Statement (Draft Paragraph)

> Recent deep learning architectures for musculoskeletal radiograph analysis demonstrate expert-level sensitivity across isolated pathology benchmarks (e.g., Rajpurkar et al., 2022; Kim et al., 2023). However, clinical translation remains hindered by the **"black-box overconfidence" dilemma**, wherein neural networks produce uncalibrated high-confidence predictions on ambiguous or out-of-distribution presentations without transparent failure modes. To address this fundamental translational gap, this work introduces **OsteoGuard AI**, an uncertainty-aware, human-in-the-loop, and traceable explainability interface for multi-class bone pathology classification.
>
> Grounded in the statutory mandates of **Article 14 of the European Union Artificial Intelligence Act (EU AI Act, 2024/1689)** and the **FDA/Health Canada/MHRA Good Machine Learning Practice (GMLP) Principle 9**, our framework introduces three interface-level contributions:
>
> 1. **Uncertainty-Gated Selective Classification (Abstention UI)**: Rather than forcing categorical predictions, our interface evaluates normalized Shannon entropy ($H > 0.60$) and empirical out-of-distribution (OOD) geometry. When diagnostic uncertainty is elevated, the interface automatically triggers an active abstention protocol, withholding the prediction from primary display and demoting it to an unverified secondary suggestion accompanied by reliability calibration diagrams.
> 2. **Clinician-in-the-Loop Audit Trails (Feedback Seam)**: A structured oversight mechanism captures clinical concurrence, discordance, and taxonomy-based disagreement rationales (e.g., motion artifact, missing orthogonal view) into an auditable JSONL ledger exposed via an administrative audit endpoint (`/audit`), operationalizing post-market surveillance requirements.
> 3. **Traceable Narrative Explainability with Counterfactual Sensitivity**: Existing tools present Grad-CAM heatmaps and SHAP feature vectors as isolated, non-auditable graphics. Our system synthesizes these components into a multi-page clinical dossier wherein diagnostic assertions are tethered to specific visual and radiomic evidence blocks via superscript citations ($¹$). Furthermore, an interactive counterfactual panel allows radiologists to test local decision boundary robustness by scrubbing feature perturbations, transforming static explainability into an interactive audit tool.

---

## Regulatory & Literature Citations

1. **European Union (2024)**: *Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (AI Act)*. Article 14: Human Oversight for High-Risk AI Systems.
2. **U.S. FDA, Health Canada, UK MHRA (2021)**: *Good Machine Learning Practice for Medical Device Development: Guiding Principles*. Principle 9: "Users are provided clear, essential information regarding model performance and uncertainty."
3. **Rajpurkar, P., et al. (2022)**: *AI in health and medicine*. Nature Medicine, 28(1), 31–38.
4. **Kim, D., et al. (2023)**: *Explainable deep learning for bone metastasis detection in planar skeletal scintigraphy*. IEEE Transactions on Radiation and Plasma Medical Sciences, 7(4), 382–391.
5. **Ghoshal, B., & Tucker, A. (2021)**: *Estimating uncertainty and out-of-distribution detection in deep neural networks for medical imaging*. Medical Image Analysis, 73, 102174.
