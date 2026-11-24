# Stage 46 — Governance Workspace

## Existing integrated workspace

The Governance route reuses the established Stage 38–40 workflow panels. It includes dataset source qualification, Stage 38 provenance and acquisition onboarding, and Stage 39 inventory/QC/harmonization/split preparation. Stage 40 reports candidate source qualification and feasibility separately from acquired cohorts.

The route does not create a second registry or bypass the FastAPI authorization checks. Restricted actions use the current in-memory reviewer credential flow; credentials are not saved to local or session storage. Authorization remains a server-side responsibility.

## State distinctions

The UI must keep these states distinct: source candidate, source approved for request, source acquisition authorized, package acquired, package integrity verified, package staged, preparation completed, split audited, and governance approved. A candidate or completed local verification does not establish that a real cohort exists or is approved for training.

## Limits

Registry identifiers, access references, source publications, reviewer actions, and ethics or agreement status must come from actual records. A populated reference field is not evidence of approval. No clinical dataset or institutional authorization is added by this workspace implementation.

Track D inference remains blocked until its existing readiness and governance requirements pass. Stage 38 remains the onboarding and provenance gate; Stage 39 remains the preparation gate.
