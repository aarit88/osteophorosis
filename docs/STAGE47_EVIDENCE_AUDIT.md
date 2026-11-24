# Stage 47 — Evidence, Results, and Audit Views

## Evidence registry

The evidence page presents historical aggregate evaluation artifacts separately from live readiness checks. It exposes the expected Track C SHA-256 and indicates that the checkpoint is verified only when the configured local artifact exists and matches that digest. It does not expose individual prediction rows or protected external holdout samples.

When a public API response unexpectedly includes sample rows, the page still withholds them. Missing local metrics remain unavailable instead of being filled with example values. Historical evaluation evidence does not establish clinical validity or current model availability.

## Results and feedback log

Results use the current in-memory response only. They are cleared when the app session ends and are not restored from browser storage. The Audit route is explicitly a feedback log; it is not represented as the signed adjudication ledger or a complete immutable governance audit. Feedback counts describe submitted feedback records and do not establish clinician agreement, regulatory compliance, or clinical performance.

## Privacy and access

The public evidence summary returns aggregate fields only. The Track C external evaluation data remains protected. The feedback API uses its existing gateway authentication controls. UI errors are generalized and do not display raw service exceptions.
