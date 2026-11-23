# Stage 40 — Dataset Source Qualification

**Review snapshot: 23 November 2026.** This is a research-source qualification register and acquisition planning workflow. It does not acquire images, approve access, establish ethics, attest authenticity, create a clinical cohort, or authorize model development.

## Architecture and boundaries

`ml_service/app/services/source_qualification.py` implements a persistent, versioned source registry and acquisition-plan ledger. It uses atomic file replacement, integrity digests, an append-only hash-linked event log, and an externally supplied HMAC key in production (`OSTEO_SOURCE_QUALIFICATION_HMAC_KEY`). The file is stored under the existing persistent onboarding volume in Compose. Reviewer references come only from the authenticated reviewer principal; the service does not create reviewer identities.

The Stage 40 API is authenticated and role-gated. `DATA_STEWARD` and `PRINCIPAL_INVESTIGATOR` may register sources, add evidence, record reviews, create plans, and transition plans. Other existing onboarding roles can read source, coverage, feasibility, and plan records. Reviewer credentials stay in request headers and the governance UI holds the token in component memory only.

Stage 40 is separate from, and cannot bypass, the existing gates:

1. Qualification records describe evidence and acquisition feasibility.
2. An acquisition plan documents a proposed external request or transfer.
3. Stage 38 remains responsible for actual source registration, authorization lifecycle, package provenance, integrity verification, and staging.
4. Stage 39 remains responsible for inventory, QC, harmonization, patient grouping, and split/leakage audits.
5. Neither qualification, planning, registration, nor staging changes Track D model readiness.

Qualification states are `DISCOVERED`, `INFORMATION_VERIFIED`, `LICENSE_REVIEW_PENDING`, `ACCESS_RESTRICTED`, `ACCESS_REQUEST_REQUIRED`, `GOVERNANCE_REVIEW_REQUIRED`, `QUALIFIED_FOR_REQUEST`, `QUALIFIED_FOR_ACQUISITION`, `REJECTED`, and `ARCHIVED`. Qualification transitions require an explicit reviewer action and references to evidence already attached to the source. `QUALIFIED_FOR_ACQUISITION` means the source has been assessed as an acquisition candidate; it does **not** mean an external data owner granted access. The acquisition-plan lifecycle independently records `DRAFT`, review/request states, external authorization decision, handoff, blockers, rejection, or archive. An `APPROVED_EXTERNALLY` plan status requires an external authorization reference supplied by an authorized reviewer.

Evidence kinds separate `VERIFIED_FACT`, `SOURCE_CLAIM`, `PUBLISHED_CLAIM`, `REVIEWER_ASSESSMENT`, and `UNVERIFIED_ASSUMPTION`. Unverified assumptions cannot support a qualification decision. Source-reported counts are stored in fields named `source_reported_*`, must cite attached evidence IDs, and remain separate from `verified_*` fields. They are never used as actual cohort counts. Coverage shows evidence-backed candidates by research interest and separately lists ontology-mapped sources; the current candidate catalog has no four-class label mappings. Actual patient count is always reported as unavailable. Handoff only links an already-created Stage 38 source ID; it never creates approvals or package data.

## API

Base path: `/api/v1/dataset-qualification` (proxied by the Express gateway).

| Method and path | Operation |
|---|---|
| `GET /sources` | List qualification records |
| `POST /sources` | Register a `DISCOVERED` candidate |
| `GET /sources/{source_id}` | Inspect a record |
| `POST /sources/{source_id}/evidence` | Add a cited source or review finding |
| `POST /sources/{source_id}/review` | Record an explicit evidence-linked reviewer decision |
| `POST /sources/{source_id}/assessment` | Update license, access, data-use, label, and patient-grouping assessments against cited evidence |
| `GET /coverage` | Candidate-only class/modality matrix, plus separately observed Stage 38 state |
| `GET /feasibility` | Theoretical cohort feasibility with missing classes and blockers; never estimates patient totals |
| `GET /plans`, `POST /plans`, `GET /plans/{plan_id}` | List, create, and inspect acquisition plans |
| `POST /plans/{plan_id}/status` | Advance a plan through validated lifecycle transitions |
| `POST /sources/{source_id}/stage38-handoff` | Link a pre-existing Stage 38 registration after acquisition qualification and explicit confirmation of matching source URL and modality |

No dataset download/upload endpoint is provided. Access requests, contracts, ethics submissions, and secure transfers remain external operational processes.

## Coverage and feasibility semantics

Coverage rows are produced for each fixed Track D diagnosis and each supported modality. They distinguish candidate interest from explicit ontology mapping, qualification status, access requirements, label availability, grouping feasibility, and Stage 38 registered/acquired/verified/staged state. Stage 39 preparation appears only when a completed preparation run exists. These are source-level indicators, not sample counts, and candidate coverage is not actual cohort coverage.

Current candidate evidence does not establish suitable representation for all four Track D classes or all three modalities. In particular, available metastasis sources use lesion-level or binary metastasis labels that cannot be mapped automatically to the Track D four-way diagnosis ontology. A no-metastasis case from a cancer cohort is not a healthy control.

## Candidate source findings

These findings come from linked repository, dataset, or publication documentation. They are leads for qualification, not source-registry reviewer decisions. No data were downloaded or acquired during this implementation.

### Secondary bone metastasis

| Lead | Verified documentation and fit | Access and blockers |
|---|---|---|
| [TCIA Spine-Mets-CT-SEG](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/) | Official collection describes human spine metastatic disease, CT and SEG in DICOM, 55 subjects, and associated lesion classifications. Its table lists CC BY 4.0. Labels describe vertebrae as normal, osteolytic, osteosclerotic, or mixed under a lesion/SINS framework; these are not the four target diseases. | Download requires TCIA Data Retriever. Confirm actual collection data terms and intended use before use; inspect patient IDs, pre/post-radiotherapy repeats, and primary-cancer labels. Candidate for a separate CT lesion/segmentation research question only; no automatic Track D label mapping. |
| [IICS-UNA bone scan images](https://zenodo.org/records/13900966) and [institutional repository record](https://repositorio.conacyt.gov.py/handle/20.500.14066/4505?show=full) | Dataset/paper documentation reports 582 planar bone scintigraphy images from 291 adult breast-cancer patients (anterior/posterior views), PNG images and labels. The article is CC BY-NC 4.0; image collection licensing and any separate dataset terms still require direct confirmation. | Candidate only for a limited adult breast cancer metastasis/no-metastasis task. “No metastases” among breast-cancer patients is not a healthy-control diagnosis. Confirm patient-level identifiers and terms in the actual record/package before any use. |
| [BS-80K paper](https://doi.org/10.1016/j.compbiomed.2022.106221) | The publication describes 82,544 whole-body and region-wise bone scan images associated with 3,247 patients, with anterior/posterior views and specialist annotations. The authors link a Google Drive download. | “Open-access” in the paper is not sufficient license verification. No official license terms were confirmed for the image package in this review. Restrict status to license review/access request until data terms, identifiers, annotation definitions, and permitted use are confirmed. Counts are paper claims, not acquired cohort counts. |

### Healthy skeletal controls

| Lead | Verified documentation and fit | Access and blockers |
|---|---|---|
| [TCIA Healthy-Total-Body-CTs](https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/) | Official TCIA page reports 30 subjects, 111,778 CT images, segmentation and clinical data. The image table specifies NIH Controlled Data Access Policy; segmentation and clinical-data rows show CC BY 4.0. The collection notes potential facial reconstruction in some images and says access follows NIH controlled-data procedures. | CT imaging requires the controlled-data access path. Separate licenses per data component; CC BY for segmentation/clinical data does not relicense the controlled CT images. Confirm approvals, identity-risk controls, repeated scans, and appropriate healthy-control definition. |

### Osteogenesis imperfecta

| Lead | Verified documentation and fit | Access and blockers |
|---|---|---|
| [EuRREB OI module](https://eurreb.eu/osteogenesis-imperfecta-oi/) and [registry data access](https://eurreb.eu/obtaining-registry-data/) | Official module describes anthropometry, fractures/BMD, genetics, clinical features, and treatment. The public module page does not describe an imaging collection. Registry data access requires Data Access Policy/request process; contributing centres use a Joint Data Registry Agreement. | This is a clinical registry lead, not a verified image dataset. Access requires formal request/governance; imaging availability is unverified. Do not claim that registry records contain radiographs. |
| [Bone2Gene collaboration notice](https://oife.org/2026/08/10/get-in-touch-helping-develop-ai-for-rare-bone-diseases/) | Patient-organization notice says a team is collecting de-identified hand radiographs with basic clinical information and invites contact from centres and stakeholders. | This is a collaboration lead, not an open dataset or access grant. Request an institutional collaboration discussion; confirm ethics, ownership, consent, data-use, modality, class scope, and access conditions directly with the project. |

### Paget’s disease

| Lead | Verified documentation and fit | Access and blockers |
|---|---|---|
| [Rotterdam Study radiological survey](https://pubmed.ncbi.nlm.nih.gov/15005843/) DOI `10.1359/jbmr.2004.19.4.566` | Publication reports a nested epidemiologic/radiographic survey and images of multiple skeletal sites. The paper is evidence that a study existed, not evidence that reusable images are available from a repository. | No public image package or dataset license identified in this review. Any request would require contacting study owners and institutional governance. |
| [New England Paget’s Disease Registry study](https://pubmed.ncbi.nlm.nih.gov/20858558/) | Publication describes a voluntary registry/questionnaire linked to radiographic records and recruited patients. | Publication does not grant access to images; no public image repository/terms identified. Institutional request and patient privacy/governance review required. |

### Additional leads assessed

- [EuRREB registry access](https://eurreb.eu/obtaining-registry-data/) is an application-based institutional process, not a public download.
- [Spinal-Multiple-Myeloma-SEG](https://www.cancerimagingarchive.net/collection/Spinal-Multiple-Myeloma-SEG/) is myeloma, a primary marrow malignancy, and is not equivalent to secondary bone metastasis; it is not mapped to the Track D metastasis class.
- A report or educational case with images is not a dataset and does not establish permission to reuse images. No web scraping or automatic acquisition is performed.

## Operational acquisition workflow

1. A Data Steward or PI creates a candidate record with original source URL and documented fields. New records start `DISCOVERED`.
2. Add individually cited evidence entries and record missing or conflicting source information. Use source claims versus independent verified facts accurately.
3. A reviewer records the decision with selected evidence IDs and an explicit confirmation. Qualified statuses cannot be assigned by registration or URL alone.
4. Create an acquisition plan describing the external request, application, approvals, data-use agreement, transfer and protected storage references, expected manifest/provenance, and verification gates. The plan does not send a request or accept terms.
5. After the data owner grants applicable authorization, an authorized operator separately registers the source through Stage 38. Stage 40 can link to that existing source record; it cannot mark Stage 38 access approved.
6. Stage 38 performs acquisition tracking, provenance, package integrity checks and staging. Stage 39 performs preparation and split audits. These still do not establish clinical validity or model readiness.

## Known limitations and next operational steps

- Source evidence is a point-in-time review; source pages, policies, versions, and legal terms can change. Re-check them at application and acquisition time.
- Publicly visible cohort counts are not considered verified local counts and never become actual cohort statistics through this system.
- Automated source qualification cannot replace legal review, ethics committees, data-protection officers, institution agreements, or clinical adjudication.
- Candidate mappings to Track D classes must be independently validated against each source’s clinical label definitions. The four-class ontology is unchanged.
- No candidate above has been accessed, approved, staged, or prepared in this implementation. The registry is pre-populated with seven `DISCOVERED` catalogue leads and source/paper claims with citations; no reviewer identity, review timestamp, ontology mapping, or acquisition approval is pre-populated. Operators must attach and review source evidence before advancing a record.
- No candidate source identified here fills every class and modality with compatible labels and patient-level grouping. Paget and OI image acquisition remain significant blockers; matched healthy control definition and data access also require governance review.
- Track A, B, and C data/artifacts and protected holdouts are outside this qualification and acquisition route. Track D remains untrained and its diagnostic endpoint stays gated.
