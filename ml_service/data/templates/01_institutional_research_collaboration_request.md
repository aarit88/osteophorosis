# Institutional Research Collaboration Request (Template 01)

> **STATUS**: `DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW`  
> **Project**: OsteoGuard AI — Multi-Center Four-Class Bone Pathology & Fracture Risk Research  
> **Target Ontology**: `OSTEO_ONTOLOGY_V1.0_STAGE34` (`SECONDARY_BONE_METASTASIS`, `PAGETS_DISEASE`, `OSTEOGENESIS_IMPERFECTA`, `HEALTHY_SKELETAL_CONTROL`)

---

## 1. Administrative Header (To Be Completed by Principal Investigator)
- **Recipient Institution / Department**: `[Institution Name — Department of Musculoskeletal Radiology / Orthopaedics / Oncology / Pediatric Genetics]`
- **Recipient Chair / Clinical Lead**: `[Name, Title, Email]`
- **Requesting Principal Investigator**: `[PI Name, Institution, Department]`
- **Draft Protocol Reference**: `OG-PROT-2026-04C-DRAFT` *(Pending IEC/IRB Review)*

## 2. Scientific Rationale & Clinical Gap
Differentiating **secondary bone metastases**, **Paget's disease of bone**, **osteogenesis imperfecta (OI)**, and **matched healthy skeletal controls** presents overlapping radiographic phenotypes (osteolysis, cortical remodeling, osteosclerosis, fragility fractures). No verified public multi-class clinical benchmark covers all four human diagnostic entities across standardized modalities.

## 3. Requested Cohort Scope & Modality Separation
Specify **only** the applicable modality streams (never pooled across modalities):
- [ ] **Pathway A — Plain Digital Radiography (CR/DR)**: Target classes `[metastasis / paget / oi / healthy]`, native 12/16-bit DICOM (`>= 1024x1024`).
- [ ] **Pathway B — Planar Bone Scintigraphy (`99mTc-MDP`)**: Paired Anterior + Posterior views bound atomically by pseudonymous `patient_id`.
- [ ] **Pathway C — Volumetric Computed Tomography (CT)**: Axial bone-window series (`<= 1.5 mm` slice thickness).

## 4. Clinical Governance & Authorship Commitments
1. **Zero PHI Transfer**: All DICOM headers and burned-in pixel text are scrubbed locally by the collaborating hospital's Honest Broker prior to transfer (see Template 05).
2. **IEC/IRB & DUA Precedence**: Zero image extraction or transfer occurs prior to formal Ethics Committee approval (Template 03) and bilateral DUA execution (Template 04).
3. **Co-Investigator Authorship & Oversight**: Clinical radiologists and specialists leading cohort extraction and ground-truth adjudication participate as co-investigators.
4. **Non-Clinical Research Scope**: Software outputs remain investigational research artifacts; not cleared for autonomous clinical diagnosis.

## 5. Institutional Sign-Off Placeholders
- **Collaborating Department Chair**: `[Signature / Date — PENDING]`
- **Requesting Principal Investigator**: `[Signature / Date — PENDING]`
