"""
OsteoGuard AI - Research Service Implementation (Stage 31).

Integrates the validated engineering components across Stages 1–30:
- Track A (FracAtlas), Track B (Bone Scintigraphy), Track C (BTXRD), Track D (Four-Class Diagnostic Model)
- Model Evidence Registry
- Architecture Readiness & Governance Audits
- Radiomics Demo Engine (81 verified vs 21 unverified features)
- Biomechanics Geometry & Mirels Scorer
- Explainability Grad-CAM Synthetic Demo & SHAP Status
- Uncertainty & Selective Prediction Engine (H_norm > 0.60 candidate threshold)
- End-to-End Synthetic Research Workbench
- Safe Gating: Four-Class Inference strictly BLOCKED
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from osteoguard.constants import (
    CLASS_DISPLAY_NAMES,
    IDX_TO_CLASS,
    TARGET_CLASSES,
    RADIOMICS_FEATURE_DIM,
)
from osteoguard.radiomics import IBSIRadiomicsEngine, SyntheticDigitalPhantomSuite
from osteoguard.biomechanics import (
    CrossSectionalGeometryEngine,
    ConventionalMirelsScorer,
    SyntheticBonePhantomSuite,
)
from osteoguard.explainability import (
    GradCAMExplainer,
    ControlledSpatialCNN,
    ExactShapleyCalculator,
)
from osteoguard.uncertainty import (
    compute_shannon_entropy,
    compute_normalized_entropy,
    compute_top_two_margin,
    compute_distribution_diagnostics,
    SelectivePredictionEngine,
    SelectiveDecision,
    AbstentionReasonCode,
    SyntheticDistributions,
    MSPDetector,
    LogitEnergyDetector,
)
from ml_service.app.schemas.research import (
    TrackID,
    ReadinessStatus,
    ResearchTrackDetail,
    ResearchTracksResponse,
    ModelEvidenceItem,
    EvidenceRegistryResponse,
    ArchitectureComponentStatus,
    ArchitectureReadinessResponse,
    DatasetGovernanceItem,
    DatasetGovernanceResponse,
    RadiomicsDemoRequest,
    RadiomicsDemoResponse,
    BiomechanicsGeometryRequest,
    BiomechanicsGeometryResponse,
    BiomechanicsMirelsRequest,
    BiomechanicsMirelsResponse,
    ExplainabilityDemoRequest,
    ExplainabilityDemoResponse,
    UncertaintyDemoRequest,
    UncertaintyDemoResponse,
    SyntheticWorkbenchRunRequest,
    SyntheticWorkbenchRunResponse,
    BlockedInferenceResponse,
)


class ResearchService:
    """
    Service coordinating research evidence, governance tracking,
    and synthetic engineering workbench demonstrations.
    """

    def __init__(self):
        self.radiomics_engine = IBSIRadiomicsEngine()
        self.geometry_engine = CrossSectionalGeometryEngine()
        self.mirels_scorer = ConventionalMirelsScorer()
        self.uncertainty_engine = SelectivePredictionEngine(
            entropy_threshold=0.60,
            use_normalized_entropy=True,
        )

    def get_research_tracks(self) -> ResearchTracksResponse:
        now_str = datetime.now(timezone.utc).isoformat()
        tracks = [
            ResearchTrackDetail(
                track_id=TrackID.TRACK_A,
                title="FracAtlas Fracture Research",
                modality="Plain Radiography (X-Ray)",
                task_type="Binary Fracture Detection",
                target_classes=["Fractured", "Non-fractured"],
                status="HISTORICAL_RESEARCH_FROZEN",
                historical_metrics={
                    "evaluation_stage": "Stage 10 (Holdout Test N=614)",
                    "roc_auc": 0.9117,
                    "pr_auc": 0.7537,
                    "sensitivity_theta_0_31": 0.9000,
                    "specificity_theta_0_31": 0.7440,
                    "balanced_accuracy_theta_0_31": 0.8220,
                    "accuracy_theta_0_31": 0.7720,
                    "accuracy_theta_0_50": 0.8827,
                    "npv_theta_0_31": 0.9715,
                    "f1_score_theta_0_31": 0.5858,
                    "test_samples": 614,
                },
                limitations=[
                    "Patient IDs absent from source metadata; true patient-level independence cannot be verified across multi-projection views.",
                    "Binary task only; does NOT differentiate bone pathologies, metabolic conditions, or secondary metastases.",
                    "Burned holdout test partition (N=614) is preserved and must not be re-evaluated.",
                ],
                governance_clearance="Academic Research Use (FracAtlas Open Dataset)",
                disclaimer="RESEARCH TRACK ONLY: Binary fracture classification model. Not a four-class differential diagnosis.",
            ),
            ResearchTrackDetail(
                track_id=TrackID.TRACK_B,
                title="Bone Scintigraphy Metastasis Research",
                modality="Tc-99m Whole-Body Bone Scintigraphy",
                task_type="Binary Skeletal Metastasis Assessment",
                target_classes=[
                    "Metastasis-positive",
                    "No detected bone metastasis in breast cancer staging cohort",
                ],
                status="HISTORICAL_RESEARCH_FROZEN",
                historical_metrics={
                    "evaluation_stage": "Stage 14 (Holdout Test N=44 Patients)",
                    "roc_auc": 0.8317,
                    "pr_auc": 0.7257,
                    "accuracy_primary_theta_0_60": 0.8409,
                    "balanced_accuracy_theta_0_60": 0.7762,
                    "sensitivity_primary_theta_0_60": 0.6667,
                    "specificity_primary_theta_0_60": 0.8857,
                    "sensitivity_screening_theta_0_45": 0.7778,
                    "precision_ppv": 0.6000,
                    "npv": 0.9118,
                    "f1_score": 0.6316,
                    "brier_score": 0.1791,
                    "test_patients": 44,
                    "test_scans": 88,
                },
                limitations=[
                    "Negative cohort comprises breast cancer staging patients with no detectable metastasis; NOT healthy normal controls.",
                    "Planar scintigraphy prone to false-positive degenerative joint disease and trauma uptake.",
                    "Burned holdout test partition (44 patients, 88 scans) is preserved and must not be re-evaluated.",
                ],
                governance_clearance="Academic Retrospective Cohort Research",
                disclaimer="RESEARCH TRACK ONLY: Secondary bone metastasis screening in staging breast cancer. Negative scans are NOT physiological healthy controls.",
            ),
            ResearchTrackDetail(
                track_id=TrackID.TRACK_C,
                title="BTXRD Primary Bone Tumour Research",
                modality="Planar Radiography (X-Ray)",
                task_type="3-Class Primary Tumour Classification",
                target_classes=[
                    "Normal/non-tumour bone",
                    "Benign primary bone tumour",
                    "Primary malignant bone tumour",
                ],
                status="HISTORICAL_FROZEN_MODEL (GENERALIZATION LIMITED)",
                historical_metrics={
                    "center1_val_accuracy": 0.8180,
                    "center1_val_balanced_acc": 0.8290,
                    "center1_val_macro_f1": 0.8103,
                    "center1_val_malignant_sens": 0.9024,
                    "center1_val_samples": 588,
                    "center2_test_accuracy": 0.5173,
                    "center2_test_balanced_acc": 0.5374,
                    "center2_test_macro_f1": 0.5069,
                    "center2_test_malignant_sens": 0.5921,
                    "center2_test_samples": 549,
                    "center3_test_accuracy": 0.4710,
                    "center3_test_balanced_acc": 0.4467,
                    "center3_test_macro_f1": 0.3592,
                    "center3_test_malignant_sens": 0.6774,
                    "center3_test_samples": 259,
                },
                limitations=[
                    "Severe cross-center generalization collapse: Accuracy drops from 81.80% (Center 1) to 51.73% (Center 2) and 47.10% (Center 3).",
                    "Macro-F1 collapses from 0.8103 (Center 1) to 0.5069 (Center 2) and 0.3592 (Center 3).",
                    "Balanced accuracy drops from 82.90% (Center 1) to 53.74% (Center 2) and 44.67% (Center 3).",
                    "Primary malignant bone tumours (Osteosarcoma, Ewing) are biologically and radiographically distinct from secondary bone metastases.",
                    "Center 3 referral skew (89.58% tumours) caused normal-class sensitivity collapse to 18.52%.",
                    "Not clinically reliable or deployable; frozen model checkpoint SHA-256 strictly preserved.",
                ],
                governance_clearance="Retrospective Tri-Center Multi-Site Audit Only",
                disclaimer="RESEARCH AUDIT ONLY: Model displays severe domain shift degradation on external hospital centers. Not approved for clinical decision support.",
            ),
            ResearchTrackDetail(
                track_id=TrackID.TRACK_D,
                title="Final Four-Class OsteoGuard AI Diagnostic System",
                modality="Multimodal: Radiography + Bone Scintigraphy + CT",
                task_type="4-Class Differential Diagnosis & Fracture Risk",
                target_classes=[
                    "Secondary bone metastasis",
                    "Paget's disease of bone",
                    "Osteogenesis imperfecta (OI)",
                    "Healthy skeletal control",
                ],
                status="BLOCKED — APPROVED CLINICAL DATA AND GOVERNANCE REQUIRED",
                historical_metrics={
                    "model_readiness": "ZERO MODELS TRAINED",
                    "clinical_training_data": "UNAVAILABLE",
                    "approved_cohorts": "0 / 4 verified clinical cohorts acquired",
                },
                limitations=[
                    "Clinical Paget's disease radiograph cohort is unavailable.",
                    "Human Osteogenesis Imperfecta (OI) data pending multi-center institutional ethics approval and DUA.",
                    "True matched healthy skeletal control cohort requires clinical site harmonization.",
                    "All diagnostic predictions are strictly disabled until full IRB/IEC approval and data acquisition.",
                ],
                governance_clearance="Pending Faculty Review, Institutional Ethics Clearance (IEC), and Data Use Agreements (DUA)",
                disclaimer="PROPOSED KSCST RESEARCH ARCHITECTURE: Active clinical prediction is completely disabled. System operates in synthetic research demonstration mode only.",
            ),
        ]
        return ResearchTracksResponse(
            timestamp=now_str,
            active_tracks_count=len(tracks),
            tracks=tracks,
            disclaimer="All models are experimental research artifacts. No clinical diagnostic claims are made.",
        )

    def get_evidence_registry(self) -> EvidenceRegistryResponse:
        now_str = datetime.now(timezone.utc).isoformat()
        models = [
            ModelEvidenceItem(
                model_id="osteo-trackA-fracatlas-v1",
                track="Track A: FracAtlas Fracture Detection",
                dataset_name="FracAtlas Open Orthopedic Dataset",
                modality="Planar Radiography",
                architecture="ResNet-101 (Pretrained IMAGENET1K_V2)",
                checkpoint_hash="STAGE_09_EPOCH9_HASH_VERIFIED",
                training_status="FROZEN (Stage 09)",
                validation_evidence={
                    "roc_auc": 0.9117,
                    "pr_auc": 0.7537,
                    "sensitivity_theta_0_31": 0.9000,
                    "specificity_theta_0_31": 0.7440,
                    "balanced_accuracy_theta_0_31": 0.8220,
                    "accuracy_theta_0_31": 0.7720,
                    "accuracy_theta_0_50": 0.8827,
                    "npv_theta_0_31": 0.9715,
                    "f1_score_theta_0_31": 0.5858,
                    "holdout_test_samples": 614,
                },
                external_holdout_status="Evaluated in Stage 10 (Burned Holdout)",
                generalization_findings="High discriminative ROC-AUC (0.9117) and PR-AUC (0.7537); patient-level independence unverified due to absent patient IDs in FracAtlas metadata.",
                clinical_deployment_status="RESEARCH_ONLY",
                limitations=["No patient identifiers", "Binary fracture detection only; not differential diagnosis"],
            ),
            ModelEvidenceItem(
                model_id="osteo-trackB-scintigraphy-v1",
                track="Track B: Bone Scintigraphy Metastasis",
                dataset_name="Whole-Body Bone Scintigraphy Staging Cohort",
                modality="Tc-99m Scintigraphy (Anterior/Posterior)",
                architecture="5-Fold ResNet-18 Patient-Aware Ensemble (Clinical Max-Pooling)",
                checkpoint_hash="STAGE_12_5FOLD_ENSEMBLE_VERIFIED",
                training_status="FROZEN (Stage 13)",
                validation_evidence={
                    "roc_auc": 0.8317,
                    "pr_auc": 0.7257,
                    "accuracy_primary_theta_0_60": 0.8409,
                    "balanced_accuracy_theta_0_60": 0.7762,
                    "sensitivity_primary_theta_0_60": 0.6667,
                    "specificity_primary_theta_0_60": 0.8857,
                    "sensitivity_screening_theta_0_45": 0.7778,
                    "brier_score": 0.1791,
                    "holdout_patients": 44,
                    "holdout_scans": 88,
                },
                external_holdout_status="Evaluated in Stage 14 (Burned Holdout)",
                generalization_findings="PR-AUC (0.7257) matches validation (0.7276) with delta -0.0019; negative scans are staging breast cancer, not healthy controls.",
                clinical_deployment_status="RESEARCH_ONLY",
                limitations=["Negative class is not healthy controls", "Hotspots confounded by osteoarthritis"],
            ),
            ModelEvidenceItem(
                model_id="osteo-trackC-btxrd-frozen-v1",
                track="Track C: BTXRD Primary Bone Tumour",
                dataset_name="BTXRD Multi-Center Radiograph Cohort",
                modality="Planar Radiography",
                architecture="ResNet-34 Aspect-Ratio Letterboxed (512x512) + Focal Loss",
                checkpoint_hash="21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD",
                training_status="FROZEN (Stage 18 Cryptographically Locked)",
                validation_evidence={
                    "center1_val_accuracy": 0.8180,
                    "center1_val_balanced_acc": 0.8290,
                    "center1_val_macro_f1": 0.8103,
                    "center1_val_samples": 588,
                    "center2_test_accuracy": 0.5173,
                    "center2_test_balanced_acc": 0.5374,
                    "center2_test_macro_f1": 0.5069,
                    "center2_test_samples": 549,
                    "center3_test_accuracy": 0.4710,
                    "center3_test_balanced_acc": 0.4467,
                    "center3_test_macro_f1": 0.3592,
                    "center3_test_samples": 259,
                },
                external_holdout_status="Center 2 (N=549) & Center 3 (N=259) Burned External Holdouts",
                generalization_findings="Catastrophic cross-center degradation: Accuracy drops from 81.80% (Center 1) to 51.73% (Center 2) and 47.10% (Center 3). Macro-F1 collapses from 0.8103 to 0.5069 and 0.3592.",
                clinical_deployment_status="NOT_SUITABLE_FOR_DEPLOYMENT",
                limitations=[
                    "Severe domain shift across hospital centers",
                    "Primary bone tumours != secondary bone metastases",
                    "Center 3 referral bias collapsed normal sensitivity to 18.52%",
                ],
            ),
            ModelEvidenceItem(
                model_id="osteo-trackD-fourclass-unified",
                track="Track D: Four-Class OsteoGuard AI",
                dataset_name="Multi-Center Harmonized 4-Class Cohort (Proposed)",
                modality="Multimodal (X-Ray + Scintigraphy + CT)",
                architecture="Dual-Stream Swin-Transformer + IBSI Radiomics (512+102 Fusion)",
                checkpoint_hash="NONE (Untrained)",
                training_status="BLOCKED_BY_DATA",
                validation_evidence={"status": "NO_CLINICAL_TRAINING_PERFORMED"},
                external_holdout_status="PENDING_DATA_ACQUISITION",
                generalization_findings="Cannot be evaluated until verified clinical cohorts are legally acquired.",
                clinical_deployment_status="BLOCKED",
                limitations=["Approved clinical Paget, human OI, and matched controls are missing."],
            ),
        ]
        return EvidenceRegistryResponse(
            timestamp=now_str,
            total_models=len(models),
            models=models,
            frozen_checkpoint_verified=True,
            disclaimer="Evidence compiled exclusively from verified historical stage reports (Stages 10, 14, 17, 18, 19, 20).",
        )

    def get_architecture_readiness(self) -> ArchitectureReadinessResponse:
        now_str = datetime.now(timezone.utc).isoformat()
        components = [
            ArchitectureComponentStatus(
                component_name="Modality Input Pathways (Radiography, Scintigraphy, CT)",
                layer_group="Data Intake & Normalization",
                status=ReadinessStatus.PARTIALLY_IMPLEMENTED,
                implemented_features="Planar radiograph and scintigraphy loaders; synthetic digital phantoms.",
                unresolved_gaps="Volumetric CT DICOM pipeline unverified on real bone window series.",
                governance_dependency="Clinical imaging protocol harmonization across centers.",
            ),
            ArchitectureComponentStatus(
                component_name="IBSI Radiomics Feature Engine",
                layer_group="Feature Engineering",
                status=ReadinessStatus.PARTIALLY_IMPLEMENTED,
                implemented_features="81 verified IBSI features (First Order, GLCM, GLRLM, GLSZM, GLDM).",
                unresolved_gaps="21 engineering heuristic features unverified; 102-D spec marked NOT VERIFIED.",
                governance_dependency="IBSI Benchmark 2 compliance review.",
            ),
            ArchitectureComponentStatus(
                component_name="Cross-Sectional Biomechanics Geometry Engine",
                layer_group="Biomechanical Risk",
                status=ReadinessStatus.IMPLEMENTED,
                implemented_features="Area, centroid, principal second moments of area, polar moment of area, retention ratios.",
                unresolved_gaps="Evaluated strictly on 2D synthetic phantoms; does NOT compute clinical fracture probabilities.",
                governance_dependency="Correlation against clinical CT-FEA fracture outcomes required.",
            ),
            ArchitectureComponentStatus(
                component_name="Conventional Mirels Scorer",
                layer_group="Clinical Rule Engine",
                status=ReadinessStatus.IMPLEMENTED,
                implemented_features="Deterministic 4-factor scoring (site, pain, nature, size) with prophylactic fixation thresholds.",
                unresolved_gaps="Requires explicit manual inputs; no automated anatomical segmentation inputs.",
                governance_dependency="Orthopedic surgical guideline verification.",
            ),
            ArchitectureComponentStatus(
                component_name="Grad-CAM Visual Explainability Engine",
                layer_group="Explainable AI",
                status=ReadinessStatus.IMPLEMENTED,
                implemented_features="Production PyTorch hook manager, spatial localization verified, Adebayo sanity check passed.",
                unresolved_gaps="Verified exclusively on synthetic controlled CNNs; does NOT explain real patient predictions.",
                governance_dependency="Radiologist visual utility study.",
            ),
            ArchitectureComponentStatus(
                component_name="SHAP Radiomic Explainability Engine",
                layer_group="Explainable AI",
                status=ReadinessStatus.PARTIALLY_IMPLEMENTED,
                implemented_features="Exact combinatorial Shapley calculator (Efficiency Axiom verified).",
                unresolved_gaps="Production SHAP library is not installed; full 102-D feature SHAP is disabled.",
                governance_dependency="Computational feasibility review on high-dimensional radiomics.",
            ),
            ArchitectureComponentStatus(
                component_name="Uncertainty Quantification & Selective Prediction",
                layer_group="Decision Support & Safety Gate",
                status=ReadinessStatus.IMPLEMENTED,
                implemented_features="Predictive entropy, normalized entropy, top-two margin, epistemic/aleatoric MI decomposition, tri-state triage.",
                unresolved_gaps="Candidate threshold H_norm > 0.60 is an unvalidated engineering heuristic.",
                governance_dependency="Pre-clinical ROC/Cost optimization on patient cohort.",
            ),
            ArchitectureComponentStatus(
                component_name="Multimodal Late Fusion (512 Deep + 102 Radiomics = 614)",
                layer_group="Multimodal Fusion",
                status=ReadinessStatus.ARCHITECTURALLY_SPECIFIED,
                implemented_features="Concatenation and projection tensor schemas defined in architecture documents.",
                unresolved_gaps="Scientifically NOT VERIFIED until 102 radiomics features are resolved and clinical encoders are trained.",
                governance_dependency="Data acquisition for all 4 classes.",
            ),
            ArchitectureComponentStatus(
                component_name="Four-Class Diagnostic Classifier Head",
                layer_group="Inference & Output",
                status=ReadinessStatus.BLOCKED_BY_DATA,
                implemented_features="Output taxonomy defined: metastasis, paget, oi, healthy.",
                unresolved_gaps="Zero trained weights; training blocked due to unavailable clinical datasets.",
                governance_dependency="IEC/IRB approval, DUA execution, and clinical cohort acquisition.",
            ),
        ]
        return ArchitectureReadinessResponse(
            timestamp=now_str,
            system_readiness="ENGINEERING_COMPONENTS_VERIFIED__CLINICAL_MODEL_BLOCKED",
            components=components,
            fusion_contract_status="512 + 102 = 614 Late Fusion Contract is ARCHITECTURALLY_SPECIFIED but NOT_VERIFIED",
            inference_eligibility="STRICTLY_BLOCKED (Diagnostic inference disabled; synthetic demo workbench active)",
            disclaimer="Components are modular engineering fixtures validated on synthetic phantoms only.",
        )

    def get_dataset_governance(self) -> DatasetGovernanceResponse:
        now_str = datetime.now(timezone.utc).isoformat()
        cohorts = [
            DatasetGovernanceItem(
                cohort_name="Secondary Bone Metastasis Cohort",
                target_class="Secondary bone metastasis",
                intended_modality="X-Ray / Bone Scintigraphy / CT",
                source_status="PARTIALLY_IDENTIFIED (Scintigraphy screening cohort available; harmonized plain radiographs required)",
                patient_count_estimate="Target: >= 250 patients",
                acquisition_blockers=[
                    "Need institutional clinical radiographs with confirmed biopsy or PET-CT metastasis ground truth.",
                    "Existing scintigraphy cohort contains breast cancer only; cannot represent all secondary metastases.",
                ],
                icmr_iec_required=True,
                dua_status="Institutional DUA Required",
            ),
            DatasetGovernanceItem(
                cohort_name="Paget's Disease of Bone Cohort",
                target_class="Paget's disease of bone",
                intended_modality="Plain Radiography (Pelvis, Spine, Femur, Skull)",
                source_status="UNAVAILABLE (No open-access human imaging repository exists)",
                patient_count_estimate="Target: >= 100 patients",
                acquisition_blockers=[
                    "Paget's disease is rare in Indian subcontinent; requires collaborative outreach to UK/European registries (PRISM trial, Paget's Association).",
                    "Requires international cross-border data transfer clearance (ICMR-HMSC).",
                ],
                icmr_iec_required=True,
                dua_status="International Collaborative DUA Required",
            ),
            DatasetGovernanceItem(
                cohort_name="Osteogenesis Imperfecta (OI) Cohort",
                target_class="Osteogenesis imperfecta (OI)",
                intended_modality="Plain Radiography (Long bones, Spine)",
                source_status="BLOCKED (Candidate study PMC11324408 investigated: 0 human radiographs available; restricted clinical data)",
                patient_count_estimate="Target: >= 100 pediatric/adult patients",
                acquisition_blockers=[
                    "Rare genetic disorder; necessitates dedicated multi-center pediatric orthopedic network collaboration (OIF, Brittle Bone Society, AIIMS).",
                    "Strict ethical safeguards for vulnerable pediatric populations.",
                ],
                icmr_iec_required=True,
                dua_status="Pediatric Ethics & Clinical DUA Required",
            ),
            DatasetGovernanceItem(
                cohort_name="Healthy Skeletal Control Cohort",
                target_class="Healthy skeletal control",
                intended_modality="Planar Radiography (Matched anatomical sites)",
                source_status="PENDING_HARMONIZATION (FracAtlas negative cohort unverified for metabolic health)",
                patient_count_estimate="Target: >= 250 age/sex-matched subjects",
                acquisition_blockers=[
                    "Existing 'non-fractured' radiographs from trauma centers may contain underlying osteoporosis, osteopenia, or systemic disease.",
                    "Requires verified asymptomatic controls with documented normal bone mineral density.",
                ],
                icmr_iec_required=True,
                dua_status="Institutional DUA Required",
            ),
        ]
        return DatasetGovernanceResponse(
            timestamp=now_str,
            target_classes=TARGET_CLASSES,
            cohorts=cohorts,
            overall_acquisition_gate="BLOCKED — Formal institutional collaboration and ethical clearances required before clinical data ingestion.",
            disclaimer="Research governance prohibits training on surrogate or unverified datasets.",
        )

    def run_radiomics_demo(self, req: RadiomicsDemoRequest) -> RadiomicsDemoResponse:
        # Generate synthetic digital phantom
        shape = (req.grid_size, req.grid_size)
        ptype = req.phantom_type.lower()
        if ptype == "checkerboard":
            image, mask, _ = SyntheticDigitalPhantomSuite.create_checkerboard_phantom(shape=shape, tile_size=max(2, req.grid_size // 4))
        elif ptype == "linear_gradient":
            image, mask, _ = SyntheticDigitalPhantomSuite.create_linear_gradient_phantom(shape=shape)
        elif ptype == "annulus":
            image, mask, _ = SyntheticDigitalPhantomSuite.create_geometric_annulus_phantom(shape=shape)
        else:
            image, mask, _ = SyntheticDigitalPhantomSuite.create_uniform_phantom(shape=shape, intensity=100.0)

        spacing = (req.spacing_mm, req.spacing_mm)
        features_dict = self.radiomics_engine.extract_features_dict(image, mask=mask, spacing=spacing)

        # 81 verified vs 21 unverified features
        # Verified: 18 first-order, 26 glcm, 16 glrlm, 16 glszm, 5 verified gldm
        verified_sample = {k: features_dict[k] for k in list(features_dict.keys())[:8]}
        unverified_sample = {k: features_dict[k] for k in list(features_dict.keys())[-8:]}

        return RadiomicsDemoResponse(
            phantom_type=req.phantom_type,
            grid_size=req.grid_size,
            spacing_mm=req.spacing_mm,
            verified_features_count=81,
            unverified_features_count=21,
            total_features_count=102,
            specification_status="102-D SPECIFICATION NOT VERIFIED (81 Verified / 21 Unverified Heuristic)",
            verified_features_sample=verified_sample,
            unverified_features_sample=unverified_sample,
            disclaimer="DEMONSTRATION ONLY: Extracted from deterministic digital phantom. 21 engineering features remain unverified against IBSI benchmark standards.",
        )

    def run_biomechanics_geometry(self, req: BiomechanicsGeometryRequest) -> BiomechanicsGeometryResponse:
        # Retrieve synthetic bone phantom from 8 available
        phantoms = SyntheticBonePhantomSuite.get_all_phantoms()
        phantom = phantoms.get(req.phantom_type, phantoms["intact_circular_ring"])
        intact_base = self.geometry_engine.compute_geometry(
            phantoms["intact_circular_ring"].mask,
            pixel_spacing_mm=phantoms["intact_circular_ring"].spacing_y_mm,
        )
        res = self.geometry_engine.compute_geometry(
            phantom.mask,
            pixel_spacing_mm=phantom.spacing_y_mm,
            intact_baseline=intact_base,
        )

        return BiomechanicsGeometryResponse(
            phantom_type=req.phantom_type,
            voxel_size_mm=req.voxel_size_mm,
            area_mm2=round(res.cortical_area_mm2, 2),
            centroid_x_mm=round(res.centroid_x_mm, 2),
            centroid_y_mm=round(res.centroid_y_mm, 2),
            I_xx_mm4=round(res.ix_mm4, 2),
            I_yy_mm4=round(res.iy_mm4, 2),
            I_xy_mm4=round(res.ixy_mm4, 2),
            polar_J_mm4=round(res.polar_j_mm4, 2),
            area_retention_ratio=round(res.area_retention_ratio or 1.0, 4),
            bending_retention_ratio_x=round((res.ix_mm4 / intact_base.ix_mm4) if intact_base.ix_mm4 > 0 else 1.0, 4),
            bending_retention_ratio_y=round((res.iy_mm4 / intact_base.iy_mm4) if intact_base.iy_mm4 > 0 else 1.0, 4),
            torsion_retention_ratio=round(res.polar_j_retention_ratio or 1.0, 4),
            warning="MATHEMATICAL DEMONSTRATION ONLY: Evaluated on 2D synthetic bone cross-section. Does NOT compute patient-specific fracture risk or clinical bone strength.",
        )

    def run_biomechanics_mirels(self, req: BiomechanicsMirelsRequest) -> BiomechanicsMirelsResponse:
        res = self.mirels_scorer.calculate_score(
            site=req.site,
            pain=req.pain,
            nature=req.radiographic_nature,
            size=req.lesion_size,
        )
        return BiomechanicsMirelsResponse(
            total_score=res.total_score,
            fracture_risk_percent=res.fracture_risk_estimate_pct,
            recommendation_tier=res.risk_tier.value,
            prophylactic_fixation_indicated=res.prophylactic_fixation_indicated,
            clinical_guidance=res.clinical_interpretation,
            disclaimer="CONVENTIONAL RULE ENGINE: Classical Mirels (1989) scoring based solely on user-supplied parameters. Independent orthopedic consultation required.",
        )

    def run_explainability_demo(self, req: ExplainabilityDemoRequest) -> ExplainabilityDemoResponse:
        # Run Grad-CAM on synthetic controlled spatial CNN fixture
        model = ControlledSpatialCNN()
        explainer = GradCAMExplainer(model, target_layer="conv1")

        # Deterministic synthetic input: 1x1x32x32 tensor with controlled lesion at (16, 16)
        x = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
        center_x = max(4, min(28, 16 + req.shift_horizontal))
        x[0, 0, 14:18, center_x - 2 : center_x + 2] = 2.0  # Bright spot

        heatmap = explainer.generate_cam(x, target_class_idx=req.target_class)
        explainer.remove_hooks()

        # Sample 8x8 coarse grid for JSON serialization
        subsampled = heatmap[::4, ::4].tolist()
        peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        active_ratio = float(np.mean(heatmap > 0.5))

        return ExplainabilityDemoResponse(
            synthetic_input_shape=[1, 1, 32, 32],
            target_class=req.target_class,
            active_region_ratio=round(active_ratio, 4),
            peak_coordinates=[int(peak_idx[0]), int(peak_idx[1])],
            heatmap_preview_base64=None,
            heatmap_grid_sample=subsampled,
            method="Grad-CAM (Gradient-Weighted Class Activation Mapping)",
            shap_status="PARTIAL (Exact low-dimensional Shapley test calculator available; production SHAP dependency not installed)",
            warning="SYNTHETIC DEMO ONLY: Generated on controlled synthetic CNN fixture. Does NOT explain real patient radiographs or clinical predictions.",
        )

    def run_uncertainty_demo(self, req: UncertaintyDemoRequest) -> UncertaintyDemoResponse:
        # Select probability vector
        if req.custom_probabilities is not None and len(req.custom_probabilities) == 4:
            p = np.array(req.custom_probabilities, dtype=np.float64)
        else:
            presets = SyntheticDistributions.get_valid_suite()
            p = presets.get(req.distribution_preset, SyntheticDistributions.highly_confident())

        # Evaluate through SelectivePredictionEngine
        res = self.uncertainty_engine.evaluate(
            probabilities=p,
            ood_flag=req.ood_flag,
            class_labels=TARGET_CLASSES,
        )

        h_nats = compute_shannon_entropy(p, base=math.e)
        h_norm = compute_normalized_entropy(p)
        margin = compute_top_two_margin(p)
        pred_idx = int(np.argmax(p))
        pred_name = CLASS_DISPLAY_NAMES.get(TARGET_CLASSES[pred_idx], TARGET_CLASSES[pred_idx])

        return UncertaintyDemoResponse(
            probabilities=[round(float(v), 4) for v in p],
            predicted_class_idx=pred_idx,
            predicted_class_name=pred_name,
            max_probability=round(float(p[pred_idx]), 4),
            top_two_margin=round(margin, 4),
            shannon_entropy_nats=round(h_nats, 4),
            normalized_entropy=round(h_norm, 4),
            candidate_threshold=0.60,
            decision=res.decision.value,
            human_review_required=res.human_review_required,
            abstention_reason=res.abstention_reason.value,
            reason_description=res.reason_description,
            ood_detected=req.ood_flag,
            clinical_guidance=(
                "Human specialist review mandated. An abstention does NOT indicate absence of disease or healthy bone."
                if res.human_review_required
                else "Acceptable predictive entropy within candidate engineering threshold."
            ),
            disclaimer="RESEARCH ONLY: Threshold H_norm > 0.60 is an unvalidated candidate engineering threshold. Evaluated on synthetic distributions.",
        )

    def run_synthetic_workbench(self, req: SyntheticWorkbenchRunRequest) -> SyntheticWorkbenchRunResponse:
        now_str = datetime.now(timezone.utc).isoformat()

        # 1. Radiomics panel
        rad_req = RadiomicsDemoRequest(phantom_type=req.phantom_type)
        rad_res = self.run_radiomics_demo(rad_req)

        # 2. Biomechanics geometry panel
        geom_req = BiomechanicsGeometryRequest(phantom_type=req.bone_geometry_type)
        geom_res = self.run_biomechanics_geometry(geom_req)

        # 3. Biomechanics Mirels panel
        mir_req = BiomechanicsMirelsRequest(
            site=req.mirels_site,
            pain=req.mirels_pain,
            radiographic_nature=req.mirels_nature,
            lesion_size=req.mirels_size,
        )
        mir_res = self.run_biomechanics_mirels(mir_req)

        # 4. Explainability panel
        exp_req = ExplainabilityDemoRequest(target_class=1)
        exp_res = self.run_explainability_demo(exp_req)

        # 5. Uncertainty panel
        unc_req = UncertaintyDemoRequest(
            distribution_preset=req.uncertainty_preset,
            ood_flag=req.ood_simulation,
        )
        unc_res = self.run_uncertainty_demo(unc_req)

        return SyntheticWorkbenchRunResponse(
            timestamp=now_str,
            status="SYNTHETIC_WORKBENCH_RUN_COMPLETED",
            radiomics_panel=rad_res,
            biomechanics_geometry_panel=geom_res,
            biomechanics_mirels_panel=mir_res,
            explainability_panel=exp_res,
            uncertainty_panel=unc_res,
            four_class_system_status="BLOCKED — APPROVED CLINICAL DATA AND GOVERNANCE REQUIRED",
            outstanding_clinical_requirements=[
                "Acquire verified clinical Paget's disease cohort.",
                "Execute institutional DUA for pediatric/adult human osteogenesis imperfecta.",
                "Harmonize healthy skeletal control acquisition protocols.",
                "Verify remaining 21 unverified radiomics features against IBSI standard.",
                "Obtain institutional ethics (IEC/IRB) approval prior to clinical model training.",
            ],
            disclaimer=(
                "INTEGRATED RESEARCH DEMONSTRATION ONLY: Modules are displayed as separate, transparent analysis panels. "
                "Synthetic values are NOT fed into a clinical model, and no automated patient predictions are generated."
            ),
        )

    def get_blocked_inference(self) -> BlockedInferenceResponse:
        return BlockedInferenceResponse()
