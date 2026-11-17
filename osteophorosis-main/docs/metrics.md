# OsteoGuard AI — Quantitative Clinical Evaluation & Metrics Guide
> **Document Ref:** OG-METRICS-2026  
> **Telemetry Schema:** POST `/api/v1/events` JSONL Event Streaming  
> **Storage:** `mock-server/data/events.jsonl`

---

## 1. Primary Evaluation Metrics

### 1. Time-to-Decision (TTD)
- **Definition**: Total elapsed seconds from initial radiograph presentation on `/results` until the clinician submits feedback or downloads the clinical report.
- **Formula**:
  $$\text{TTD} = t_{\text{action}} - t_{\text{mount}}$$
- **Clinical Target**: Mean $\text{TTD} \le 120\text{ seconds}$ without compromising diagnostic accuracy.

### 2. Clinical Concurrence Rate ($CR$)
- **Definition**: The proportion of automated predictions validated as clinically sound by the participating physician.
- **Formula**:
  $$CR = \frac{N_{\text{agree}}}{N_{\text{total}}} \times 100\%$$
- **Discordance Stratification**: Discordant submissions are grouped by reason code:
  - Artifact ($R_1$)
  - Insufficient Context ($R_2$)
  - Missed Lesion ($R_3$)
  - Class Confusion ($R_4$)

### 3. System Usability Scale (SUS) Score
- **Formula**:
  $$\text{SUS} = 2.5 \times \left[ \sum_{i \in \text{odd}} (S_i - 1) + \sum_{i \in \text{even}} (5 - S_i) \right]$$
- **Target Threshold**: Score $\ge 80.0$ (Grade A, Top 10th percentile of medical software).

### 4. NASA-TLX Raw Workload Index (RTLX)
- **Formula**:
  $$\text{RTLX} = \frac{\text{Mental} + \text{Physical} + \text{Temporal} + \text{Performance} + \text{Effort} + \text{Frustration}}{6}$$
- **Target Benchmark**: Score $\le 35.0$ on a 100-point scale.

---

## 2. Behavioral Interaction Telemetry Taxonomy

All user micro-interactions are logged via `POST /api/v1/events` to `mock-server/data/events.jsonl`:

| Event Type | Trigger | Payload Metadata | Purpose |
|---|---|---|---|
| `page_view` | Route navigation | `{ path: string, referrer: string }` | Navigation funnel tracking |
| `modality_selected` | User clicks modality pill | `{ modality: 'xray' \| 'ct' \| 'bone_scan' }` | Modality distribution |
| `demo_image_loaded` | User clicks "Load Demo Image" | `{ assetName: string }` | Demo scenario usage |
| `analysis_started` | User clicks "Run Analysis" | `{ modality: string, fileSize: number }` | Inference pipeline start |
| `analysis_completed` | Results returned from engine | `{ requestId: string, topClass: string, entropy: number }` | Model response baseline |
| `gradcam_mode_changed` | User switches Blend / Slider / Side-by-Side | `{ viewMode: 'toggle' \| 'slider' \| 'sideBySide' }` | XAI interface preference |
| `gradcam_toggled` | User shows/hides heatmap | `{ isVisible: boolean, opacity: number }` | Visual attention inspection rate |
| `counterfactual_scrubbed` | User drags sensitivity scrubber | `{ scenario: string, factor: number }` | Decision boundary testing |
| `clinician_feedback_submitted`| User clicks Agree / Disagree / Unsure | `{ decision: string, reasonCode?: string }` | Human oversight audit |
| `report_modal_opened` | User clicks "Download Traceable Report" | `{ requestId: string }` | Audit export intent |
| `pdf_exported` | User saves PDF | `{ requestId: string, pages: 3 }` | Clinical documentation complete |

---

## 3. Data Extraction Example (JSONL)
```json
{"type":"gradcam_toggled","requestId":"req_506842b450dc","timestamp":"2026-11-05T14:24:02.110Z","metadata":{"isVisible":true,"opacity":0.85}}
{"type":"counterfactual_scrubbed","requestId":"req_506842b450dc","timestamp":"2026-11-05T14:24:18.420Z","metadata":{"scenario":"Cortical Thickness","value":125}}
{"type":"clinician_feedback_submitted","requestId":"req_506842b450dc","timestamp":"2026-11-05T14:24:35.812Z","metadata":{"decision":"agree"}}
```
