"""
Stage 35 Automated Verification Suite — Institutional Dataset Acquisition,
Research Governance, Workflow State Machine, Cohort Protocols, Cohort Planning,
and Preservation of Stage 34 Ontology & Protected Holdouts.

Runs cleanly with pure Python stdlib unittest inside AI Studio.
"""

from __future__ import annotations

import copy
import csv
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

from ml_service.app.services.acquisition_governance import (
    AcquisitionGovernanceService,
    AcquisitionStatus,
    GovernanceRole,
    WorkflowStage,
    evaluate_governance_record,
    transition_workflow_state,
    validate_cohort_protocol_record,
)
from ml_service.app.services.adjudication_engine import AdjudicationEngine
from ml_service.datasets.clinical_ontology import (
    ONTOLOGY_VERSION,
    AdjudicationStatus,
    adjudicate_diagnostic_label,
)
from ml_service.datasets.track_d_intake import (
    generate_leakage_safe_splits,
    inspect_image_file,
    validate_track_d_dataset,
    verify_cross_split_leakage,
)
from ml_service.tests.test_track_d_intake import _valid_record


def _fully_verified_governance_source(
    source_id: str = "SRC_TEST_VERIFIED",
    target_cohort: str = "metastasis",
    workflow_state: str = WorkflowStage.DUA_PENDING.value,
    expiry_date: str = "2028-12-31",
) -> dict:
    return {
        "source_id": source_id,
        "target_cohort": target_cohort,
        "ontology_target_class": "SECONDARY_BONE_METASTASIS",
        "source_classification": "VERIFIED_EXTERNAL_CANDIDATE",
        "acquisition_status": AcquisitionStatus.ACCESS_REQUESTED.value,
        "workflow_state": workflow_state,
        "evidence_reference": "GOV_EVIDENCE_REF_2026_001",
        "governance_record": {
            "irb_iec_approval": {
                "status": "APPROVED",
                "protocol_ref": "IEC/2026/OSTEO-04C",
                "committee": "Institutional Ethics Committee",
            },
            "research_protocol_id": "OG-PROT-2026-04C",
            "dua": {
                "status": "EXECUTED",
                "agreement_ref": "DUA-2026-MULTI-01",
            },
            "institutional_authorization": {
                "status": "APPROVED",
                "signoff_ref": "INST-AUTH-2026-88",
            },
            "consent_or_waiver_status": "IEC_RETROSPECTIVE_WAIVER_APPROVED",
            "deidentification_approval": {
                "status": "APPROVED",
                "sop_ref": "SOP-DEID-PS315-V2",
            },
            "data_retention_requirements": "5 years post-publication followed by cryptographic purge",
            "permitted_research_purposes": ["Four-class bone differential diagnosis research"],
            "redistribution_restrictions": "Strictly prohibited",
            "approval_date": "2026-06-01",
            "expiry_date": expiry_date,
            "evidence_reference": "GOV_DOSSIER_2026_001",
            "reviewer_pseudo_id": "REV_IRB_001",
            "reviewer_role": GovernanceRole.IRB_OFFICER.value,
        },
        "workflow_history": [],
    }


class TestStage35AcquisitionGovernance(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.svc = AcquisitionGovernanceService(root_dir=self.repo_root)

    def test_01_dataset_source_registry_validation_and_truthfulness(self) -> None:
        payload = self.svc.get_sources_payload()
        self.assertEqual(payload["ontology_version"], ONTOLOGY_VERSION)
        self.assertGreaterEqual(payload["total_sources"], 8)
        self.assertEqual(payload["approved_sources_count"], 0)

        by_id = {s["source_id"]: s for s in payload["sources"]}
        # All 4 cohorts represented
        cohorts_present = {s["target_cohort"] for s in payload["sources"]}
        self.assertEqual(cohorts_present, {"metastasis", "paget", "oi", "healthy"})

        # Allowed statuses strictly respected
        allowed = {s.value for s in AcquisitionStatus}
        for s in payload["sources"]:
            self.assertIn(s["acquisition_status"], allowed)

        # Proposed institutional sources are NOT falsely marked url_verified or approved
        self.assertEqual(by_id["SRC_PAGET_HOSP_PACS_XRAY"]["source_classification"], "PROPOSED_INSTITUTIONAL")
        self.assertFalse(by_id["SRC_PAGET_HOSP_PACS_XRAY"]["url_verified"])
        self.assertIsNone(by_id["SRC_PAGET_HOSP_PACS_XRAY"]["verified_patient_count"])

        # Forensic Stage 23 rejections preserved
        self.assertEqual(by_id["SRC_OI_FIGSHARE_MOUSE_REJECTED"]["acquisition_status"], "REJECTED")
        self.assertEqual(by_id["SRC_PAGET_WEB_EDUCATIONAL_REJECTED"]["acquisition_status"], "REJECTED")
        self.assertEqual(by_id["SRC_METS_BTXRD_PRIMARY_MALIGNANT_REJECTED"]["acquisition_status"], "REJECTED")
        self.assertEqual(by_id["SRC_HEALTHY_FRACATLAS_TRAUMA_REJECTED"]["acquisition_status"], "REJECTED")

        # Modalities strictly separated per source
        self.assertEqual(by_id["SRC_METS_TCIA_SPINE_CT"]["available_modalities"], ["ct"])
        self.assertEqual(by_id["SRC_METS_IICS_UNA_BONESCAN"]["available_modalities"], ["bone_scan"])

    def test_02_missing_institutional_permission_or_irb_detection(self) -> None:
        src = _fully_verified_governance_source()
        # Non-empty string "PENDING" must NOT be treated as verified
        src["governance_record"]["irb_iec_approval"] = {"status": "PENDING", "protocol_ref": "IEC_DRAFT"}
        src["governance_record"]["institutional_authorization"] = {"status": "SUBMITTED", "signoff_ref": "pending"}
        audit = evaluate_governance_record(src, reference_date=date(2026, 11, 22))
        self.assertFalse(audit["irb_iec_approved"])
        self.assertFalse(audit["governance_verified"])
        self.assertIn("MISSING_OR_PENDING:irb_iec_approval", audit["missing_or_pending"])
        self.assertIn("MISSING_OR_PENDING:institutional_authorization", audit["missing_or_pending"])

    def test_03_missing_dua_detection(self) -> None:
        src = _fully_verified_governance_source()
        src["governance_record"]["dua"] = {"status": "PENDING", "agreement_ref": None}
        audit = evaluate_governance_record(src, reference_date=date(2026, 11, 22))
        self.assertFalse(audit["dua_approved"])
        self.assertFalse(audit["governance_verified"])
        self.assertIn("MISSING_OR_PENDING:dua", audit["missing_or_pending"])

        # Cannot transition from DUA_PENDING to ACCESS_APPROVED without executed DUA
        with self.assertRaises(ValueError):
            transition_workflow_state(
                source=src,
                target_state=WorkflowStage.ACCESS_APPROVED,
                reviewer_pseudo_id="REV_LEGAL_01",
                reviewer_role=GovernanceRole.LEGAL_COUNSEL,
                reason="Attempting premature access approval",
                evidence_reference="DUA_REF_DRAFT",
                reference_date=date(2026, 11, 22),
            )

    def test_04_invalid_workflow_transition_blocking(self) -> None:
        src = _fully_verified_governance_source(workflow_state=WorkflowStage.SOURCE_IDENTIFIED.value)
        # Cannot skip straight from SOURCE_IDENTIFIED to ACCESS_APPROVED
        with self.assertRaises(ValueError):
            transition_workflow_state(
                source=src,
                target_state=WorkflowStage.ACCESS_APPROVED,
                reviewer_pseudo_id="REV_PI_01",
                reviewer_role=GovernanceRole.PRINCIPAL_INVESTIGATOR,
                reason="Skipping required ethics and DUA stages",
                evidence_reference="DOC_SKIP_01",
                reference_date=date(2026, 11, 22),
            )

        # Terminal REJECTED state cannot advance
        rejected_src = _fully_verified_governance_source(workflow_state=WorkflowStage.REJECTED.value)
        with self.assertRaises(ValueError):
            transition_workflow_state(
                source=rejected_src,
                target_state=WorkflowStage.SOURCE_VERIFIED,
                reviewer_pseudo_id="REV_PI_01",
                reviewer_role=GovernanceRole.PRINCIPAL_INVESTIGATOR,
                reason="Attempting to reopen terminal rejection",
                evidence_reference="DOC_REOPEN_01",
            )

    def test_05_expired_ethics_and_dua_approval_blocking(self) -> None:
        yesterday = (date(2026, 11, 22) - timedelta(days=1)).isoformat()
        src = _fully_verified_governance_source(
            workflow_state=WorkflowStage.DUA_PENDING.value,
            expiry_date=yesterday,
        )
        audit = evaluate_governance_record(src, reference_date=date(2026, 11, 22))
        self.assertTrue(audit["is_expired"])
        self.assertFalse(audit["governance_verified"])
        self.assertTrue(any(m.startswith("EXPIRED_GOVERNANCE_APPROVAL:") for m in audit["missing_or_pending"]))

        with self.assertRaises(ValueError):
            transition_workflow_state(
                source=src,
                target_state=WorkflowStage.ACCESS_APPROVED,
                reviewer_pseudo_id="REV_IRB_02",
                reviewer_role=GovernanceRole.IRB_OFFICER,
                reason="Advancing with expired approval",
                evidence_reference="IEC_RENEWAL_EXPIRED",
                reference_date=date(2026, 11, 22),
            )

    def test_06_missing_governance_evidence_reference_blocking(self) -> None:
        src = _fully_verified_governance_source(workflow_state=WorkflowStage.SOURCE_IDENTIFIED.value)
        for bad_ref in ("", "   ", "pending", "TBD", "N/A"):
            with self.assertRaises(ValueError):
                transition_workflow_state(
                    source=src,
                    target_state=WorkflowStage.SOURCE_VERIFIED,
                    reviewer_pseudo_id="REV_DATA_01",
                    reviewer_role=GovernanceRole.DATA_STEWARD,
                    reason="Source verified against provenance matrix",
                    evidence_reference=bad_ref,
                    reference_date=date(2026, 11, 22),
                )

    def test_07_unauthorized_reviewer_decision_rejection(self) -> None:
        src = _fully_verified_governance_source(workflow_state=WorkflowStage.DUA_PENDING.value)
        # DATA_STEWARD is not authorized to grant ACCESS_APPROVED
        with self.assertRaises(PermissionError):
            transition_workflow_state(
                source=src,
                target_state=WorkflowStage.ACCESS_APPROVED,
                reviewer_pseudo_id="REV_STEWARD_01",
                reviewer_role=GovernanceRole.DATA_STEWARD,
                reason="Attempting access sign-off without legal/DPO/IRB authority",
                evidence_reference="SIGNED_DUA_01",
                reference_date=date(2026, 11, 22),
            )

        # Non-pseudonymous / invalid reviewer ID blocked
        with self.assertRaises(ValueError):
            transition_workflow_state(
                source=src,
                target_state=WorkflowStage.ACCESS_APPROVED,
                reviewer_pseudo_id="Dr.JohnDoe@hospital.org",
                reviewer_role=GovernanceRole.IRB_OFFICER,
                reason="Invalid non-pseudonymous reviewer ID",
                evidence_reference="SIGNED_DUA_01",
                reference_date=date(2026, 11, 22),
            )

        # Authorized IRB_OFFICER with valid pseudonymous ID succeeds
        updated = transition_workflow_state(
            source=src,
            target_state=WorkflowStage.ACCESS_APPROVED,
            reviewer_pseudo_id="REV_IRB_009",
            reviewer_role=GovernanceRole.IRB_OFFICER,
            reason="Verified IEC + executed DUA + DPO de-identification signoff",
            evidence_reference="DUA-2026-MULTI-01",
            reference_date=date(2026, 11, 22),
        )
        self.assertEqual(updated["workflow_state"], WorkflowStage.ACCESS_APPROVED.value)
        self.assertEqual(updated["acquisition_status"], AcquisitionStatus.APPROVED.value)
        self.assertEqual(len(updated["workflow_history"]), 1)

    def test_08_planned_vs_actual_cohort_count_separation(self) -> None:
        planning = self.svc.get_cohort_planning_payload()
        self.assertTrue(planning["synthetic_data_excluded"])
        self.assertEqual(planning["statistical_sample_size_justification_ref"], "PENDING_BIOSTATISTICAL_AND_IEC_REVIEW")
        self.assertEqual(planning["statistical_adequacy_status"], "NOT_ESTABLISHED_IN_SOFTWARE")

        # Totals strictly separate planned targets from actual zero disk samples
        self.assertEqual(planning["totals"]["planned_patient_count"], 940)
        self.assertEqual(planning["totals"]["planned_image_count"], 2270)
        self.assertEqual(planning["totals"]["actual_received_patients"], 0)
        self.assertEqual(planning["totals"]["actual_validated_patients"], 0)
        self.assertEqual(planning["totals"]["actual_validated_images"], 0)

        # Zero data received when acquisition status queried on repo
        acq = self.svc.get_acquisition_status_payload()
        self.assertEqual(acq["total_validated_patients"], 0)
        self.assertEqual(acq["total_received_valid_images"], 0)
        self.assertTrue(acq["overall_acquisition_status"].startswith("BLOCKED"))

        # Check all 7 draft templates exist and carry DRAFT banner
        gov = self.svc.get_governance_status_payload()
        self.assertEqual(len(gov["available_templates"]), 7)
        self.assertEqual(gov["templates_status"], "DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW")
        templates_dir = self.repo_root / "ml_service" / "data" / "templates"
        for name in gov["available_templates"]:
            txt = (templates_dir / name).read_text(encoding="utf-8")
            self.assertIn("DRAFT — REQUIRES INSTITUTIONAL AND LEGAL REVIEW", txt)

    def test_09_metastasis_confirmation_standard_separation(self) -> None:
        protocols = self.svc.load_protocols()["protocols"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            biopsy_rec = _valid_record(root, "M_BIO", "PT_M1", "metastasis", "xray", seed=1)
            biopsy_rec["confirmation_tier"] = "HISTOPATHOLOGY_CONFIRMED"
            res_bio = validate_cohort_protocol_record(biopsy_rec, protocols)
            self.assertTrue(res_bio["valid"])

            rad_rec = _valid_record(root, "M_RAD", "PT_M2", "metastasis", "ct", seed=2)
            rad_rec["diagnostic_evidence_category"] = "PET_CT_AND_RADIOLOGY"
            rad_rec["histopathology_confirmation_status"] = "NOT_PERFORMED"
            rad_rec["confirmation_tier"] = "RADIOLOGY_AND_CLINICAL_CONCORDANCE"
            res_rad = validate_cohort_protocol_record(rad_rec, protocols)
            self.assertTrue(res_rad["valid"])

            # Claiming HISTOPATHOLOGY_CONFIRMED without confirmed biopsy must fail
            fake_bio = copy.deepcopy(rad_rec)
            fake_bio["confirmation_tier"] = "HISTOPATHOLOGY_CONFIRMED"
            res_fake = validate_cohort_protocol_record(fake_bio, protocols)
            self.assertFalse(res_fake["valid"])
            self.assertIn("METASTASIS_HISTOPATHOLOGY_NOT_CONFIRMED", res_fake["protocol_errors"])

            # Primary bone tumor remains OUT_OF_SCOPE
            sarcoma = copy.deepcopy(biopsy_rec)
            sarcoma["source_diagnosis"] = "Osteosarcoma"
            sarcoma["diagnostic_evidence"] = "Primary osteosarcoma"
            res_sarc = validate_cohort_protocol_record(sarcoma, protocols)
            self.assertFalse(res_sarc["valid"])
            self.assertEqual(res_sarc["adjudication_status"], AdjudicationStatus.OUT_OF_SCOPE.value)

    def test_10_paget_triad_and_disqualification_validation(self) -> None:
        protocols = self.svc.load_protocols()["protocols"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_paget = _valid_record(root, "P_01", "PT_P1", "paget", "xray", seed=3)
            self.assertTrue(validate_cohort_protocol_record(valid_paget, protocols)["valid"])

            # Radiopaedia / MedPix web figure rejected
            web_paget = copy.deepcopy(valid_paget)
            web_paget["dataset_source"] = "Radiopaedia teaching archive"
            res_web = validate_cohort_protocol_record(web_paget, protocols)
            self.assertFalse(res_web["valid"])
            self.assertIn("PAGET_EDUCATIONAL_WEB_SOURCE_DISQUALIFIED:radiopaedia", res_web["protocol_errors"])

            # Osteoarthritis cannot map to Paget's
            oa_rec = copy.deepcopy(valid_paget)
            oa_rec["source_diagnosis"] = "Osteoarthritis"
            oa_rec["diagnostic_evidence"] = "Subchondral sclerosis from hip osteoarthritis"
            res_oa = validate_cohort_protocol_record(oa_rec, protocols)
            self.assertFalse(res_oa["valid"])
            self.assertEqual(res_oa["adjudication_status"], AdjudicationStatus.OUT_OF_SCOPE.value)

    def test_11_oi_genetic_vs_clinical_pediatric_and_animal_exclusion(self) -> None:
        protocols = self.svc.load_protocols()["protocols"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Pediatric genetic-confirmed OI missing pediatric consent fails
            ped_oi = _valid_record(root, "OI_P1", "PT_OI1", "oi", "xray", seed=4)
            ped_oi["patient_age_years"] = 9
            ped_oi["confirmation_track"] = "GENETIC_CONFIRMED"
            res_no_consent = validate_cohort_protocol_record(ped_oi, protocols)
            self.assertFalse(res_no_consent["valid"])
            self.assertIn("OI_PEDIATRIC_CONSENT_OR_IEC_WAIVER_MISSING", res_no_consent["protocol_errors"])

            # With verified parental consent + assent it passes
            ped_oi["pediatric_consent_status"] = "PARENTAL_CONSENT_AND_ASSENT_VERIFIED"
            self.assertTrue(validate_cohort_protocol_record(ped_oi, protocols)["valid"])

            # Sillence clinical track kept distinct & passes
            sillence_oi = _valid_record(root, "OI_S1", "PT_OI2", "oi", "xray", seed=5)
            sillence_oi["diagnostic_evidence_category"] = "SPECIALIST_CLINICAL_DIAGNOSIS"
            sillence_oi["confirmation_track"] = "CLINICAL_RADIOLOGICAL_SILLENCE"
            self.assertTrue(validate_cohort_protocol_record(sillence_oi, protocols)["valid"])

            # Murine micro-CT / Figshare 3795019 disqualified
            murine = copy.deepcopy(sillence_oi)
            murine["dataset_source"] = "Figshare 3795019 murine oim micro-ct"
            res_mouse = validate_cohort_protocol_record(murine, protocols)
            self.assertFalse(res_mouse["valid"])
            self.assertTrue(any("OI_ANIMAL_OR_MICROCT_DISQUALIFIED" in e for e in res_mouse["protocol_errors"]))

            # Scintigraphy modality disallowed for OI protocol
            oi_bs = _valid_record(root, "OI_BS", "PT_OI3", "oi", "bone_scan", seed=6)
            res_bs = validate_cohort_protocol_record(oi_bs, protocols)
            self.assertFalse(res_bs["valid"])
            self.assertTrue(any("PROTOCOL_MODALITY_DISALLOWED_FOR_CLASS" in e for e in res_bs["protocol_errors"]))

    def test_12_healthy_control_eligibility_and_mura_anatomical_scope(self) -> None:
        protocols = self.svc.load_protocols()["protocols"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_healthy = _valid_record(root, "H_01", "PT_H1", "healthy", "xray", seed=7)
            self.assertTrue(validate_cohort_protocol_record(ok_healthy, protocols)["valid"])

            # Contralateral limb fails
            contra = copy.deepcopy(ok_healthy)
            contra["source_diagnosis"] = "Contralateral healthy femur"
            contra["diagnostic_evidence"] = "Normal contralateral fellow femur from trauma series"
            res_contra = validate_cohort_protocol_record(contra, protocols)
            self.assertFalse(res_contra["valid"])
            self.assertIn("HEALTHY_ELIGIBILITY_FAILED:CONTRALATERAL_OR_FELLOW_LIMB", res_contra["protocol_errors"])

            # Stanford MURA misused on pelvis/spine fails anatomical scope
            mura_pelvis = copy.deepcopy(ok_healthy)
            mura_pelvis["dataset_source"] = "Stanford AIMI MURA"
            mura_pelvis["anatomical_region"] = "pelvis"
            res_mura = validate_cohort_protocol_record(mura_pelvis, protocols)
            self.assertFalse(res_mura["valid"])
            self.assertIn("HEALTHY_MURA_ANATOMICAL_SCOPE_VIOLATION:upper_extremity_only", res_mura["protocol_errors"])

    def test_13_preservation_of_stage34_ontology_and_adjudication_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = AdjudicationEngine()
            rec = _valid_record(root, "S34_PRES", "PT_S34", "paget", "xray", seed=8)
            adj_direct = adjudicate_diagnostic_label(rec)
            self.assertEqual(adj_direct["adjudication_status"], AdjudicationStatus.ACCEPTED.value)

            engine.adjudicate_batch([rec])
            updated = engine.submit_reviewer_decision(
                sample_id="S34_PRES",
                decision=AdjudicationStatus.UNCERTAIN,
                reviewer_pseudo_id="CLIN_REV_34",
                rationale="Borderline trabecular pattern flagged during dual review",
            )
            self.assertEqual(updated["adjudication"]["adjudication_status"], AdjudicationStatus.UNCERTAIN.value)
            hist = engine.get_sample_provenance_history("S34_PRES")
            self.assertEqual(len(hist["transition_history"]), 2)

    def test_14_patient_level_leakage_and_btxrd_holdout_isolation_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Build clean 4-class cohort (3 patients per class) + BTXRD C2 holdout + C3 hash match
            records = []
            seed_ctr = 20
            for cls in ("metastasis", "paget", "oi", "healthy"):
                for idx in range(3):
                    seed_ctr += 1
                    records.append(
                        _valid_record(root, f"CLEAN_{cls}_{idx}", f"PT_CLN_{cls}_{idx}", cls, "xray", seed=seed_ctr)
                    )

            c2 = _valid_record(root, "BTXRD_C2_HOLD", "PT_HOLD_C2", "healthy", "xray", seed=99, institution_id="btxrd_center_2")
            c3 = _valid_record(root, "BTXRD_C3_HOLD", "PT_HOLD_C3", "healthy", "xray", seed=100, institution_id="INST_A")
            c3_sha = inspect_image_file(root / c3["image_path"])["sha256"]

            split_csv = root / "ml_service" / "data" / "splits" / "btxrd_split_manifest.csv"
            split_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(split_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "split", "sha256_hash"])
                w.writerow(["BTXRD_C3_HOLD", "center_3", c3_sha])

            val = validate_track_d_dataset(root, records=records + [c2, c3])
            self.assertEqual(val["passed_count"], 12)
            self.assertEqual(val["quarantined_count"], 2)

            splits = generate_leakage_safe_splits(val["passed_records"], seed=42)
            leak = verify_cross_split_leakage(splits["records"])
            self.assertTrue(leak["is_clean"])
            self.assertEqual(leak["violations"], [])


if __name__ == "__main__":
    unittest.main()
