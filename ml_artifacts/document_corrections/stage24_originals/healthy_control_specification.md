# Clinical Cohort Acquisition Specification: Healthy Skeletal Controls

**Target Class 4**: Healthy Skeletal Controls  
**Document Status**: Formal Cohort Acquisition Specification  
**Author**: OsteoGuard AI Clinical Integration Team  

---

## 1. Clinical Definition of True Skeletal Controls

A healthy skeletal control study represents imaging of a patient whose skeletal architecture demonstrates:
1. Normal cortical thickness, intact trabecular microarchitecture, and age-appropriate bone density.
2. Complete absence of focal osteolytic, osteoblastic, or mixed lesions.
3. Complete absence of metabolic bone disorders (Paget's disease, osteomalacia, hyperparathyroidism, rickets).
4. Complete absence of genetic bone dysplasias (Osteogenesis Imperfecta, osteopetrosis, fibrous dysplasia).
5. Complete absence of acute, subacute, or healing skeletal fractures.
6. Verified board-certified radiologist diagnostic report concluding: *"Normal radiographic appearance of examined skeleton; no focal bone lesion, fracture, or suspicious osteo-articular pathology."*

---

## 2. Forensic Analysis of Candidate Control Datasets

### A. FracAtlas "Non-Fractured" Radiographs: Forensic Limitations
In Stages 06–10 and Stage 22, the FracAtlas dataset (Bangabandhu Sheikh Mujib Medical University) was audited:
- **Emergency Room Trauma Context**: FracAtlas images originate from patients presenting to an acute orthopaedic emergency department following traumatic injury.
- **Presence of Confounders**: While labeled "non-fractured", these radiographs frequently display acute soft-tissue hematomas, joint effusions, incidental degenerative osteoarthrosis, osteopenia, or subtle subcortical microtrabecular trauma.
- **Audit Disposition**: FracAtlas non-fractured images **CANNOT BE UNCRITICALLY EQUATED** to pristine healthy controls. They represent "non-fractured trauma presentations" and may only serve as trauma confounder stress tests, not pristine baseline controls.

### B. Stanford AIMI MURA (Musculoskeletal Radiographs): Upper-Extremity Boundary
In Stages 22 and 23, the Stanford AIMI MURA benchmark was evaluated:
- **Cohort Composition**: 9,045 verified normal studies across 40,561 multi-view digital radiographs.
- **Anatomical Restriction**: MURA is strictly confined to seven **upper-extremity** anatomical regions:
  1. Elbow
  2. Finger
  3. Forearm
  4. Hand
  5. Humerus
  6. Shoulder
  7. Wrist
- **Audit Disposition**: MURA is an exceptionally high-quality control dataset **EXCLUSIVELY FOR UPPER-EXTREMITY** comparisons. It cannot serve as a healthy control cohort for pelvic, spinal, or lower-extremity (femur/tibia) pathologies (which dominate Paget's disease and skeletal metastases).

---

## 3. Anatomical Matching and Acquisition Specifications

To avoid shortcut learning where the neural network predicts disease based solely on anatomical site (e.g., learning that pelvis images are Paget's and wrists are normal):

```
+--------------------------------------------------------------------------------------------------+
|                                    ANATOMICAL MATCHING STRATEGY                                  |
+--------------------------+------------------------------------+----------------------------------+
| Anatomical Region        | Target Pathology Candidate         | Required Healthy Control Source  |
+--------------------------+------------------------------------+----------------------------------+
| Pelvis & Hip             | Paget's disease, Bone metastasis   | Asymptomatic pelvic X-rays from  |
|                          |                                    | hospital screening / abdominal CT|
+--------------------------+------------------------------------+----------------------------------+
| Lumbar / Thoracic Spine  | Bone metastasis, Paget's disease   | Normal spinal trauma surveys     |
|                          |                                    | cleared by CT / MRI              |
+--------------------------+------------------------------------+----------------------------------+
| Femur & Tibia (Lower)    | Paget's disease, OI, Metastasis    | Contralateral normal long bone   |
|                          |                                    | radiographs / screening views    |
+--------------------------+------------------------------------+----------------------------------+
| Upper Extremity (Arm)    | OI, Bone metastasis, Paget's       | Stanford AIMI MURA Normal Cohort |
| (Humerus, Forearm, Hand) |                                    | (9,045 verified normal studies)  |
+--------------------------+------------------------------------+----------------------------------+
| Whole-Body Scintigraphy  | Secondary bone metastasis          | IICS-UNA Normal Scintigraphy     |
|                          |                                    | (Physiological tracer clearance) |
+--------------------------+------------------------------------+----------------------------------+
```

---

## 4. Exclusion Criteria for Healthy Skeletal Controls

Any candidate control image exhibiting any of the following characteristics must be excluded:
1. Significant osteopenia or osteoporosis (T-score $\le -2.5$ if DXA is linked, or marked radiographical cortical thinning).
2. Prior orthopaedic hardware, joint arthroplasty, intramedullary nails, screws, or plates.
3. Severe degenerative osteoarthritis with subchondral cysts or large osteophytes that could mimic lytic or sclerotic lesions.
4. Active systemic neoplastic disease, even if skeletal involvement is not visibly apparent.
