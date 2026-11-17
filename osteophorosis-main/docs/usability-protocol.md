# OsteoGuard AI — Radiology Usability Study Protocol
> **Document Ref:** OG-STUDY-2026-v1  
> **Study Design:** 30-Minute Moderated Usability & Cognitive Workload Evaluation  
> **Participant Cohort:** $N = 5$ Radiology Residents / Orthopedic Fellows  
> **Regulatory Anchor:** EU AI Act (2024/1689) Article 14 & FDA GMLP Principle 9

---

## 1. Study Objectives
1. **Explainability Transparency**: Quantify whether Grad-CAM salience maps and SHAP radiomics attributions assist clinicians in identifying osteolytic lesions, cortical thickening, and trabecular microstructural changes.
2. **Abstention Acceptability**: Assess clinician trust and psychological safety when the model withholds low-confidence predictions (Softmax entropy $> 0.60$ or OOD flag) and activates human referral.
3. **Traceable Narrative Verification**: Evaluate the usability of superscript-referenced evidence in the 3-page clinical PDF export.
4. **Cognitive Workload & Usability**: Measure workload via NASA-TLX and usability via the System Usability Scale (SUS).

---

## 2. Participant Tasks (Timed Moderated Protocol)

### Task 1: Image Upload & Modality Configuration (3 min)
- **Instructions**: "You are reviewing a patient with suspected localized bone pathology. Select the Plain Radiograph (X-Ray) modality, upload the provided demo DICOM-converted femur scan, and trigger analysis."
- **Success Criteria**: Correct modality selected; file ingested without validation error.
- **Metric Captured**: Time-to-Submit (TTS).

### Task 2: Diagnostic Evaluation & Attention Inspection (7 min)
- **Instructions**: "Inspect the predicted primary diagnosis, uncertainty rating, and Grad-CAM visual attention overlay. Toggle between Blend and Split-Slider views to evaluate whether the attention hotspot aligns with cortical bone margins."
- **Telemetry Logged**: `gradcam_toggle` event, `gradcam_intensity_change` event.
- **Cognitive Inquiry**: "Does the model focus on anatomical regions you consider clinically salient?"

### Task 3: Counterfactual Sensitivity Exploration (6 min)
- **Instructions**: "Navigate to the 'What would change the diagnosis?' panel. Select the top counterfactual scenario and scrub the slider to test what level of Cortical Thickness or Contrast shift would flip the classification."
- **Telemetry Logged**: `counterfactual_scrubbed` event with slider delta.
- **Cognitive Inquiry**: "Does this sensitivity perturbation clarify the decision boundary?"

### Task 4: Clinician-in-the-Loop Feedback & Audit Logging (4 min)
- **Instructions**: "Record your formal concurrence or discordance using the Clinician Verification card. If disagreeing or unsure, specify the primary clinical reason code (e.g., artifact, missing lateral projection) and submit."
- **Telemetry Logged**: `clinician_feedback_submitted` event with decision and reasonCode.

### Task 5: Traceable PDF Export & Audit Review (5 min)
- **Instructions**: "Generate the 3-Page Traceable Clinical Dossier. Verify the superscript evidence references and review the audit ledger entry on the `/audit` page."
- **Success Criteria**: PDF generated; audit entry confirmed.

---

## 3. Post-Study Questionnaires

### System Usability Scale (SUS) — 5-Point Likert Scale (1 = Strongly Disagree, 5 = Strongly Agree)
1. I think that I would like to use OsteoGuard AI frequently in my diagnostic workflow.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most radiology residents would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

*Target Benchmark: Composite SUS Score $\ge 80.0$ (Grade A).*

### NASA-TLX (Task Load Index) — 100-Point Subscales
- **Mental Demand**: How much mental and perceptual activity was required?
- **Physical Demand**: How much physical activity was required?
- **Temporal Demand**: How hurried or rushed was the pace of the task?
- **Performance**: How successful do you think you were in evaluating the radiograph?
- **Effort**: How hard did you have to work to accomplish your level of performance?
- **Frustration**: How insecure, discouraged, irritated, or stressed did you feel?

---

## 4. Qualitative Debrief Prompts
1. "When the model withheld its prediction due to high entropy, how did that affect your trust in the tool?"
2. "How did the counterfactual scrubber compare to standard static confidence scores?"
3. "Would the traceable superscript citations satisfy your department's audit and peer-review standards?"
