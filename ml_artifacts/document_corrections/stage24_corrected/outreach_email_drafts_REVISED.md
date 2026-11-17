# Clinical and Academic Outreach Email Drafts (Revised)

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Prepared Formal Outreach Communications (DRAFT FOR REVIEW)  
**Author**: OsteoGuard AI Governance Office  

> [!CAUTION]
> **COMMUNICATION PROTOCOL RULE (DRAFT ONLY — NOT SENT)**:  
> These drafts are strictly intended for internal academic review and discussion with faculty mentors.  
> **NO EMAILS HAVE BEEN TRANSMITTED, AND NO OUTREACH HAS BEEN INITIATED.**  
> Text enclosed in brackets (`[Placeholders]`) must be verified and approved by the academic supervisor prior to any transmission.

---

## DRAFT 1: Faculty Research Advisor / Department Mentor (RECOMMENDED FIRST COMMUNICATION)

**Subject**: Academic Guidance & Protocol Review: OsteoGuard AI Research Framework

Dear Professor `[Faculty Advisor Name]`,

I hope this email finds you well.

I am writing to seek your academic guidance and review of our research protocol draft for the next phase of our bone disease artificial intelligence project, **OsteoGuard AI**.

Over the previous project phases, we developed single-modality baseline models for bone scintigraphy metastases and primary bone tumours. In our recent multi-center evaluations of the BTXRD primary bone tumour classifier (Stages 19–21), we observed substantial domain shift when evaluating the model on independent hospital cohorts:
- Internal Development Validation (Center 1, N=588): Macro-F1 = 0.8103 (Balanced Acc = 0.8290, Overall Acc = 0.8180).
- External Evaluation (Center 2, N=549): Macro-F1 = 0.5069 (an absolute decrease of 0.3034 [-30.34 percentage points]; relative decrease of 37.44%).
- External Stress Test (Center 3, N=259, specialized oncology referral center): Macro-F1 = 0.3592 (an absolute decrease of 0.4511 [-45.11 percentage points]; relative decrease of 55.67%).

Furthermore, our forensic dataset audits confirmed that primary bone sarcomas (osteosarcoma, Ewing sarcoma) differ fundamentally from secondary bone metastases and cannot be conflated with them. In addition, candidate open-access datasets for Osteogenesis Imperfecta were found to be non-human murine micro-CT data, and no suitable, publicly accessible clinical imaging cohorts were identified for Paget's disease.

To establish a scientifically sound foundation for a four-class differential diagnosis framework (Secondary Metastasis, Paget's Disease, Osteogenesis Imperfecta, and Healthy Controls), we have prepared a draft Institutional Research Protocol (`OG-PROT-2026-04C-REVISED`) aligned with the ICMR National Ethical Guidelines (2017) and the Digital Personal Data Protection Act framework.

I would be extremely grateful for an opportunity to meet with you for 15–20 minutes to discuss this protocol, receive your guidance on our proposed methodology, and discuss how to properly approach our university hospital's radiology department under appropriate faculty mentorship.

Thank you very much for your time and support.

Respectfully yours,  
`[Student Researcher Name]`  
`[Student ID / Degree Program]`  
`[Department / Institution Name]`  
`[Contact Email]` | `[Phone Number]`  

---

## DRAFT 2: Head of Musculoskeletal (MSK) Radiology (Proposed Secondary Outreach)

**Subject**: Exploratory Research Inquiry: Multi-Class Bone Disorder Imaging & Paget's Disease Evaluation

Dear Dr. `[Radiology Department Chair / Senior MSK Radiologist Name]`,

I hope you are doing well.

I am writing from `[Institution / University Department]` under the academic mentorship of Professor `[Faculty Advisor Name]`. We are conducting research into machine learning methods for musculoskeletal imaging, focused on differentiating challenging bone disorders.

In our preliminary evaluations of deep learning models on multi-center bone radiograph datasets, we identified significant cross-institutional performance degradation, highlighting the critical need for verified clinical reference standards and diverse multi-center validation.

Our current dataset audit did not identify a suitable, publicly accessible and appropriately licensed clinical imaging cohort for **Paget's disease of bone (*Osteitis Deformans*)**. We are exploring the academic feasibility of developing a collaborative retrospective imaging protocol with your department to evaluate the classic radiographic features of Paget's disease alongside anatomically matched normal skeletal radiographs.

Key elements of our proposed protocol draft:
- **Ethics and Privacy**: All work would be conducted under formal Institutional Ethics Committee (IEC) review in accordance with ICMR Ethical Guidelines (2017).
- **Honest Broker Extraction**: Image de-identification and extraction would be handled entirely within the hospital firewall by authorized personnel; no direct patient identifiers would be accessed by the machine learning team.
- **Academic Contributions**: Contributor roles and co-authorship on any resulting scientific publications would follow ICMJE guidelines based on substantive clinical and radiological contributions.

We have prepared an initial protocol concept draft for faculty review. If this area might be of interest to you or a faculty member in your division, would it be possible to schedule a brief 15-minute introductory meeting with Professor `[Faculty Advisor Name]` and myself to discuss feasibility?

Thank you very much for considering our inquiry.

Sincerely,  
`[Student Researcher Name]` (on behalf of `[Faculty Advisor / Senior Sponsor Name]`)  
`[Department / Academic Unit]`  
`[Institution Name]`  
`[Contact Email]`  

---

## DRAFT 3: Director of Pediatric Orthopaedics / Metabolic Bone Clinic

**Subject**: Academic Research Inquiry: Quantitative Radiomic Evaluation in Osteogenesis Imperfecta

Dear Dr. `[Director / Senior Clinician Name]`,

I hope this email finds you well.

My name is `[Student Researcher Name]`, and I am a research student working under the supervision of Professor `[Faculty Advisor Name]` at `[Institution / University Name]`. Our research focuses on computer-assisted analysis of skeletal radiographs and quantitative assessment of bone fragility.

We are currently investigating computational methods to evaluate bone texture, cortical thinning, and deformity progression in **Osteogenesis Imperfecta (OI)**. As part of our preliminary planning, we are reviewing candidate clinical datasets, including published pediatric bone-age radiograph collections from the Brittle Bone Disorders Consortium (BBDC Protocol 7701).

Because pediatric bone maturation introduces complex radiological confounders (such as open epiphyseal plates mimicking fracture lines), expert clinical insight is essential to ensure our research questions are clinically meaningful and scientifically valid.

We would be deeply grateful for an opportunity to consult with you or a designated clinical colleague for 15 minutes to share our draft research concept and receive your perspective on key clinical priorities in OI fracture risk assessment.

Thank you very much for your time and consideration.

Respectfully yours,  
`[Student Researcher Name]`  
`[Department / Institution Name]`  
`[Faculty Supervisor: Prof. Faculty Advisor Name]`  
`[Contact Email]`  

---

## DRAFT 4: Institutional Ethics Committee (IEC) / Research Office (Procedural Guidance Request)

**Subject**: Procedural Guidance Request: Retrospective De-Identified Human Imaging Protocol Draft

Dear `[Ethics Committee Secretariat / Member Secretary]`,

I hope you are doing well.

I am writing under the guidance of my faculty advisor, Professor `[Faculty Advisor Name]`, to request preliminary procedural guidance regarding the preparation of a research protocol for Institutional Ethics Committee review.

We are drafting a retrospective, non-interventional imaging protocol entitled:  
*"OsteoGuard AI: Deep Learning and Radiomic Framework for Differential Diagnosis of Bone Disorders"* (`OG-PROT-2026-04C-REVISED`).

The proposed research will involve the computational analysis of existing, fully de-identified diagnostic radiographs and CT scans across four diagnostic categories, accessed through authorized institutional repositories and collaborative hospital archives.

Study parameters:
- **Nature of Study**: Strictly retrospective review of existing de-identified imaging; no direct patient interaction or interventional procedures.
- **De-Identification**: De-identification to be conducted by the data provider's honest broker prior to research team access, with all direct identifiers removed.
- **Guidance Framework**: The protocol is being prepared under the ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017).

Could you please advise us on:
1. The standard institutional application format and documentation checklist required for retrospective de-identified imaging studies.
2. The specific institutional criteria and documentation required when applying for consideration of a waiver of informed consent under ICMR Guidelines (Section 5, Box 5.2).
3. Upcoming submission deadlines for the Institutional Ethics Committee.

Thank you very much for your guidance and assistance.

Sincerely,  
`[Student Researcher Name]`  
`[Faculty Sponsor: Prof. Faculty Advisor Name, Designation]`  
`[Department]`  
`[Institution Name]`  
`[Contact Email]`  
