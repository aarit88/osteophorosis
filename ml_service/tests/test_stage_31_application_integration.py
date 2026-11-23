"""
STAGE 31 — End-to-End Application Integration & Research Prototype Verification Suite
====================================================================================
Verifies that all Stage 1-30 engineering engines are correctly mounted into the
application backend and frontend contracts, research tracks are strictly segregated,
historical model evidence and multi-center generalization limits are documented,
and four-class diagnostic inference is strictly BLOCKED pending clinical data.
"""

import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from ml_service.app.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_CHECKPOINT_PATH = PROJECT_ROOT / "ml_artifacts" / "btxrd_frozen_model" / "best_checkpoint.pt"
EXPECTED_CHECKPOINT_HASH = "21064670fa6277b0e96135333708de4692ddb80e428b792f3675405b5dc109cd"


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    app = create_app()
    return TestClient(app)


def test_frozen_checkpoint_immutability():
    """Verify present frozen weights or honestly report the externally missing artifact."""
    if not FROZEN_CHECKPOINT_PATH.exists():
        from ml_service.utils.checkpoint_inventory import build_checkpoint_inventory
        inventory = build_checkpoint_inventory()
        assert next(item for item in inventory["tracks"] if item.get("track") == "C")["status"] == "MISSING"
        return
    h = hashlib.sha256()
    with open(FROZEN_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_hash = h.hexdigest().lower()
    assert actual_hash == EXPECTED_CHECKPOINT_HASH, (
        f"Checkpoint hash mismatch! Expected {EXPECTED_CHECKPOINT_HASH}, got {actual_hash}"
    )


def test_research_status_endpoint(client):
    """GET /api/v1/research/status should return 200 with honest prototype status."""
    res = client.get("/api/v1/research/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "RESEARCH_PROTOTYPE_ONLINE"
    assert "BLOCKED" in data["four_class_inference_status"]
    assert data["active_tracks_count"] == 4
    assert data["frozen_checkpoint_status"] in {"VERIFIED", "MISSING"}


def test_research_tracks_endpoint(client):
    """GET /api/v1/research/tracks should list all 4 discrete research tracks with limitations."""
    res = client.get("/api/v1/research/tracks")
    assert res.status_code == 200
    data = res.json()
    assert data["active_tracks_count"] == 4
    tracks = {t["track_id"]: t for t in data["tracks"]}
    assert "track_a" in tracks
    assert "track_b" in tracks
    assert "track_c" in tracks
    assert "track_d" in tracks

    # Track D must be explicitly blocked
    assert "BLOCKED" in tracks["track_d"]["status"]

    # Sub-tracks must disclose verified limitations
    assert any("patient" in lim.lower() for lim in tracks["track_a"]["limitations"])
    assert any("generalization" in lim.lower() or "external" in lim.lower() for lim in tracks["track_c"]["limitations"])


def test_evidence_registry_endpoint(client):
    """GET /api/v1/evidence/registry should return the historical evidence ledger."""
    res = client.get("/api/v1/evidence/registry")
    assert res.status_code == 200
    data = res.json()
    assert data["total_models"] >= 4
    assert data["frozen_checkpoint_verified"] is FROZEN_CHECKPOINT_PATH.exists()

    # Find the frozen BTXRD model
    btxrd_model = next((m for m in data["models"] if "btxrd" in m["model_id"]), None)
    assert btxrd_model is not None
    assert "51.73" in btxrd_model["generalization_findings"]
    assert "47.10" in btxrd_model["generalization_findings"]
    assert "BURNED" in btxrd_model["external_holdout_status"].upper()


def test_architecture_readiness_endpoint(client):
    """GET /api/v1/architecture/readiness should detail component implementations and gaps."""
    res = client.get("/api/v1/architecture/readiness")
    assert res.status_code == 200
    data = res.json()
    assert "ARCHITECTURALLY_SPECIFIED" in data["fusion_contract_status"]
    assert "BLOCKED" in data["inference_eligibility"]
    assert len(data["components"]) >= 6

    comp_names = [c["component_name"] for c in data["components"]]
    assert any("radiomics" in name.lower() for name in comp_names)
    assert any("biomechanic" in name.lower() for name in comp_names)
    assert any("uncertainty" in name.lower() for name in comp_names)


def test_dataset_governance_endpoint(client):
    """GET /api/v1/datasets/status should report cohort acquisition blockers and DUA status."""
    res = client.get("/api/v1/datasets/status")
    assert res.status_code == 200
    data = res.json()
    assert len(data["cohorts"]) == 4
    assert "BLOCKED" in data["overall_acquisition_gate"]
    for cohort in data["cohorts"]:
        assert len(cohort["acquisition_blockers"]) > 0
        assert cohort["icmr_iec_required"] is True


def test_radiomics_demo_endpoint(client):
    """POST /api/v1/radiomics/demo should extract verified features on synthetic phantoms."""
    payload = {"phantom_type": "checkerboard", "grid_size": 64, "spacing_mm": 1.0}
    res = client.post("/api/v1/radiomics/demo", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["verified_features_count"] == 81
    assert data["unverified_features_count"] == 21
    assert data["total_features_count"] == 102
    assert len(data["verified_features_sample"]) > 0


def test_biomechanics_geometry_endpoint(client):
    """POST /api/v1/biomechanics/geometry should compute 2D moments on bone phantoms."""
    payload = {"phantom_type": "intact_circular_ring", "voxel_size_mm": 0.5}
    res = client.post("/api/v1/biomechanics/geometry", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["area_mm2"] > 0
    assert data["I_xx_mm4"] > 0
    assert data["I_yy_mm4"] > 0
    assert data["polar_J_mm4"] > 0
    assert data["area_retention_ratio"] == 1.0


def test_biomechanics_mirels_endpoint(client):
    """POST /api/v1/biomechanics/mirels should compute standard Mirels score."""
    # Site=2 (lower), Pain=2 (moderate), Nature=3 (lytic), Size=2 (1/3-2/3) -> Score=9
    payload = {"site": 2, "pain": 2, "radiographic_nature": 3, "lesion_size": 2}
    res = client.post("/api/v1/biomechanics/mirels", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] == 9
    assert data["fracture_risk_percent"] == 33.3
    assert data["prophylactic_fixation_indicated"] is True
    assert "HIGH" in data["recommendation_tier"].upper()


def test_explainability_demo_endpoint(client):
    """POST /api/v1/explainability/demo should generate Grad-CAM on synthetic fixture."""
    payload = {"target_class": 0, "shift_horizontal": 0}
    res = client.post("/api/v1/explainability/demo", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "Grad-CAM" in data["method"]
    assert "PARTIAL" in data["shap_status"]
    assert len(data["heatmap_grid_sample"]) == 8
    assert len(data["peak_coordinates"]) == 2


def test_uncertainty_demo_endpoint(client):
    """POST /api/v1/uncertainty/demo should correctly triage confidence."""
    # Test confident preset
    res_conf = client.post("/api/v1/uncertainty/demo", json={"distribution_preset": "highly_confident"})
    assert res_conf.status_code == 200
    data_conf = res_conf.json()
    assert data_conf["decision"] == "ACCEPT_PREDICTION"
    assert data_conf["normalized_entropy"] < 0.35

    # Test high entropy preset
    res_abst = client.post("/api/v1/uncertainty/demo", json={"distribution_preset": "uniform_four_class"})
    assert res_abst.status_code == 200
    data_abst = res_abst.json()
    assert data_abst["decision"] == "ABSTAIN_HUMAN_REVIEW_REQUIRED"
    assert data_abst["human_review_required"] is True
    assert data_abst["normalized_entropy"] >= 0.60


def test_synthetic_workbench_demo_endpoint(client):
    """POST /api/v1/workbench/run-demo should run synchronized multi-engine pipeline."""
    res = client.post("/api/v1/workbench/run-demo", json={})
    assert res.status_code == 200
    data = res.json()
    assert "COMPLETED" in data["status"]
    assert data["radiomics_panel"]["verified_features_count"] == 81
    assert data["biomechanics_mirels_panel"]["total_score"] > 0
    assert "Grad-CAM" in data["explainability_panel"]["method"]
    assert data["uncertainty_panel"]["candidate_threshold"] == 0.60
    assert "BLOCKED" in data["four_class_system_status"]


def test_four_class_inference_strictly_blocked(client):
    """POST /api/v1/diagnostic/four-class must be rejected with HTTP 503 SERVICE UNAVAILABLE."""
    res = client.post("/api/v1/diagnostic/four-class", json={})
    assert res.status_code == 503
    data = res.json()
    assert data["error_code"] == "FOUR_CLASS_MODEL_NOT_READY"
    details = data.get("details", {})
    assert details.get("status") == "BLOCKED"
    assert len(details.get("blockers", [])) >= 4
    assert len(details.get("target_classes", [])) == 4
    assert "Secondary bone metastasis" in details.get("target_classes", [])
    assert "Paget's disease of bone" in details.get("target_classes", [])
    assert any("osteogenesis imperfecta" in c.lower() for c in details.get("target_classes", []))
    assert "Healthy skeletal control" in details.get("target_classes", [])


def test_track_separation_and_zero_conflation(client):
    """Verify that none of the sub-tracks (A, B, C) can be confused with Track D."""
    res = client.get("/api/v1/research/tracks")
    tracks = {t["track_id"]: t for t in res.json()["tracks"]}

    # Track A is binary fracture detection
    assert tracks["track_a"]["target_classes"] == ["Fractured", "Non-fractured"]
    # Track B is bone scintigraphy metastasis
    assert "Metastasis-positive" in tracks["track_b"]["target_classes"]
    # Track C is 3-class primary bone tumour
    assert any("primary malignant" in c.lower() for c in tracks["track_c"]["target_classes"])
    # Track D is the 4 intended research classes
    assert len(tracks["track_d"]["target_classes"]) == 4
