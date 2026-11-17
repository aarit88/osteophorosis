# Stage 25 Acquisition Blockers and Real-World Next Actions

**Project**: OsteoGuard AI Multi-Class Differential Diagnosis  
**Document Status**: Official Acquisition Blocker Log  
**Date**: November 16, 2026  

---

## 1. Summary of Hard Blockers

Model training for the four-class differential diagnosis model is **STRICTLY BLOCKED**. The project cannot proceed to model architecture implementation or training due to the following hard real-world blockers:

```
+--------------------------------------------------------------------------------------------------+
|                                    PROJECT HARD ACQUISITION BLOCKERS                             |
+----+-----------------------------+----------------------------------+----------------------------+
| #  | Target Cohort               | Exact Blocker Description        | Blocking Real-World Action |
+----+-----------------------------+----------------------------------+----------------------------+
| 1  | Paget's Disease of Bone     | Zero verified clinical imaging   | Formal Institutional Ethics|
|    |                             | data exists in-tree. No public   | Committee (IEC) review and |
|    |                             | benchmark available.             | hospital DUA execution.    |
+----+-----------------------------+----------------------------------+----------------------------+
| 2  | Osteogenesis Imperfecta     | Zero verified human clinical     | Formal data access request |
|    |                             | radiographs in-tree. Animal      | to RDCRN BBDC Data Access  |
|    |                             | micro-CT data disqualified.      | Committee (DMCC).          |
+----+-----------------------------+----------------------------------+----------------------------+
| 3  | Healthy Skeletal Controls   | Matched normal pelvic, spinal,   | Inclusion of asymptomatic  |
|    | (Axial / Pelvis / Lower)    | and long-bone X-rays missing.    | screening controls in      |
|    |                             | FracAtlas trauma rejected.       | hospital IEC protocol.     |
+----+-----------------------------+----------------------------------+----------------------------+
| 4  | Stanford AIMI MURA          | Upper-extremity control dataset  | Non-commercial research    |
|    |                             | not yet staged or ingested.      | registration under DUA.    |
+----+-----------------------------+----------------------------------+----------------------------+
| 5  | TCIA Spine-Mets-CT-SEG      | Volumetric 3D spinal CT cohort   | Acknowledgment of TCIA     |
|    |                             | not yet staged or ingested.      | terms and NBIA staging.    |
+----+-----------------------------+----------------------------------+----------------------------+
```

---

## 2. Immediate Real-World Next Actions

To resolve these blockers in compliance with ethical and academic governance:
1. **Academic Alignment (Immediate Priority)**:
   - Present the revised Institutional Research Protocol (`OG-PROT-2026-04C-REVISED`) and Draft 1 of the outreach communication to the designated **Faculty Research Advisor**.
   - Obtain academic review, protocol refinements, and formal departmental mentorship.
2. **Clinical Partnership Outreach (Secondary Priority)**:
   - Schedule introductory discussions with the university hospital's Musculoskeletal Radiology division using Draft 2.
   - Seek preliminary clinical guidance on retrospective PACS retrieval feasibility and clinician co-investigator appointment.
3. **Ethics Guidance (Tertiary Priority)**:
   - Submit procedural inquiry (Draft 4) to the Institutional Ethics Committee (IEC) to confirm standard application templates and waiver of consent requirements under ICMR Guidelines (2017).
4. **Controlled Repository Applications**:
   - Prepare formal registration for Stanford AIMI MURA (upper-extremity controls).
   - Prepare formal data access application to the RDCRN BBDC for pediatric hand/wrist radiographs.

---

## 3. Mandatory Stopping Rule
*In accordance with project integrity constraints, no datasets may be automatically downloaded, scraped, or fabricated. Model training remains completely halted.*
