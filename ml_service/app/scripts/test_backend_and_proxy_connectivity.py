"""
OsteoGuard AI — Backend and Frontend Proxy Verification Script
Tests all registered research endpoints according to their exact OpenAPI schemas:
- GET /health
- GET /api/v1/research/status
- GET /api/v1/research/tracks
- GET /api/v1/evidence/registry
- GET /api/v1/architecture/readiness
- GET /api/v1/datasets/status
- POST /api/v1/radiomics/demo
- POST /api/v1/biomechanics/geometry
- POST /api/v1/biomechanics/mirels
- POST /api/v1/explainability/demo
- POST /api/v1/uncertainty/demo
- POST /api/v1/workbench/run-demo
- POST /api/v1/diagnostic/four-class (strictly blocked: 503)
"""

import sys
import httpx

sys.stdout.reconfigure(line_buffering=True)

TARGETS = [
    ("Direct Backend", "http://127.0.0.1:8000"),
    ("Frontend Proxy", "http://127.0.0.1:3000")
]

def run_tests_for_target(name: str, base_url: str) -> bool:
    print("=" * 80)
    print(f"VERIFYING TARGET: {name} ({base_url})")
    print("=" * 80)
    all_ok = True

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        # 1. Health
        try:
            r = client.get("/health")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data.get("status") in ["healthy", "ok"], f"Unexpected status: {data}"
            print(f"  [PASS] GET /health -> 200 OK | status={data.get('status')}")
        except Exception as e:
            print(f"  [FAIL] GET /health -> Error: {e}")
            all_ok = False

        # 2. Research status
        try:
            r = client.get("/api/v1/research/status")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data["status"] == "RESEARCH_PROTOTYPE_ONLINE"
            assert "BLOCKED" in data["four_class_inference_status"]
            print(f"  [PASS] GET /api/v1/research/status -> 200 OK | status={data['status']}, four_class={data['four_class_inference_status'][:30]}...")
        except Exception as e:
            print(f"  [FAIL] GET /api/v1/research/status -> Error: {e}")
            all_ok = False

        # 3. Research tracks
        try:
            r = client.get("/api/v1/research/tracks")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "tracks" in data and len(data["tracks"]) == 4
            t_ids = [t["track_id"] for t in data["tracks"]]
            assert "track_a" in t_ids
            assert "track_b" in t_ids
            assert "track_c" in t_ids
            assert "track_d" in t_ids
            print(f"  [PASS] GET /api/v1/research/tracks -> 200 OK | {len(data['tracks'])} tracks verified ({', '.join(t_ids)})")
        except Exception as e:
            print(f"  [FAIL] GET /api/v1/research/tracks -> Error: {e}")
            all_ok = False

        # 4. Evidence registry
        try:
            r = client.get("/api/v1/evidence/registry")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "models" in data and len(data["models"]) == 4
            evidence = data["models"][2]["validation_evidence"]
            c1_metric = evidence["center1_val_accuracy"]
            c2_metric = evidence["center2_test_accuracy"]
            c3_metric = evidence["center3_test_accuracy"]
            assert c1_metric == 0.8180, f"Expected 0.8180, got {c1_metric}"
            assert c2_metric == 0.5173, f"Expected 0.5173, got {c2_metric}"
            assert c3_metric == 0.4710, f"Expected 0.4710, got {c3_metric}"
            print(f"  [PASS] GET /api/v1/evidence/registry -> 200 OK | Metrics verified: C1={c1_metric:.4f}, C2={c2_metric:.4f}, C3={c3_metric:.4f}")
        except Exception as e:
            print(f"  [FAIL] GET /api/v1/evidence/registry -> Error: {e}")
            all_ok = False

        # 5. Architecture readiness
        try:
            r = client.get("/api/v1/architecture/readiness")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "components" in data and len(data["components"]) >= 5
            assert data["system_readiness"] == "ENGINEERING_COMPONENTS_VERIFIED__CLINICAL_MODEL_BLOCKED"
            print(f"  [PASS] GET /api/v1/architecture/readiness -> 200 OK | readiness={data['system_readiness']}")
        except Exception as e:
            print(f"  [FAIL] GET /api/v1/architecture/readiness -> Error: {e}")
            all_ok = False

        # 6. Datasets status
        try:
            r = client.get("/api/v1/datasets/status")
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "cohorts" in data and len(data["cohorts"]) == 4
            print(f"  [PASS] GET /api/v1/datasets/status -> 200 OK | {len(data['cohorts'])} cohorts verified")
        except Exception as e:
            print(f"  [FAIL] GET /api/v1/datasets/status -> Error: {e}")
            all_ok = False

        # 7. POST Radiomics demo
        try:
            payload = {"phantom_type": "checkerboard", "grid_size": 32, "spacing_mm": 1.0}
            r = client.post("/api/v1/radiomics/demo", json=payload)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data["total_features_count"] == 102
            assert data["verified_features_count"] == 81
            assert data["unverified_features_count"] == 21
            print(f"  [PASS] POST /api/v1/radiomics/demo -> 200 OK | 81 verified / 21 unverified (total 102)")
        except Exception as e:
            print(f"  [FAIL] POST /api/v1/radiomics/demo -> Error: {e}")
            all_ok = False

        # 8. POST Biomechanics geometry
        try:
            payload = {"phantom_type": "intact_circular_ring", "voxel_size_mm": 1.0}
            r = client.post("/api/v1/biomechanics/geometry", json=payload)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "I_xx_mm4" in data and "I_yy_mm4" in data and "polar_J_mm4" in data
            print(f"  [PASS] POST /api/v1/biomechanics/geometry -> 200 OK | I_xx={data['I_xx_mm4']:.2f}, I_yy={data['I_yy_mm4']:.2f}, J={data['polar_J_mm4']:.2f}")
        except Exception as e:
            print(f"  [FAIL] POST /api/v1/biomechanics/geometry -> Error: {e}")
            all_ok = False

        # 9. POST Biomechanics Mirels
        try:
            payload = {"site": 2, "pain": 2, "radiographic_nature": 3, "lesion_size": 2}
            r = client.post("/api/v1/biomechanics/mirels", json=payload)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert data["total_score"] == 9
            assert data["recommendation_tier"] == "HIGH_RISK"
            assert data["prophylactic_fixation_indicated"] is True
            print(f"  [PASS] POST /api/v1/biomechanics/mirels -> 200 OK | Score={data['total_score']}, Tier={data['recommendation_tier']}, Fixation={data['prophylactic_fixation_indicated']}")
        except Exception as e:
            print(f"  [FAIL] POST /api/v1/biomechanics/mirels -> Error: {e}")
            all_ok = False

        # 10. POST Explainability demo
        try:
            payload = {"target_class": 1, "shift_horizontal": 0}
            r = client.post("/api/v1/explainability/demo", json=payload)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert len(data["heatmap_grid_sample"]) == 8
            assert len(data["heatmap_grid_sample"][0]) == 8
            print(f"  [PASS] POST /api/v1/explainability/demo -> 200 OK | Target={data['target_class']}, Heatmap={len(data['heatmap_grid_sample'])}x{len(data['heatmap_grid_sample'][0])}")
        except Exception as e:
            print(f"  [FAIL] POST /api/v1/explainability/demo -> Error: {e}")
            all_ok = False

        # 11. POST Uncertainty demo
        try:
            payload = {"distribution_preset": "moderately_confident", "ood_flag": False}
            r = client.post("/api/v1/uncertainty/demo", json=payload)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "decision" in data
            assert "normalized_entropy" in data
            print(f"  [PASS] POST /api/v1/uncertainty/demo -> 200 OK | Decision={data['decision']}, H_norm={data['normalized_entropy']:.4f}")
        except Exception as e:
            print(f"  [FAIL] POST /api/v1/uncertainty/demo -> Error: {e}")
            all_ok = False

        # 12. POST Workbench run-demo
        try:
            payload = {
                "phantom_type": "checkerboard",
                "bone_geometry_type": "intact_circular_ring",
                "mirels_site": 2,
                "mirels_pain": 2,
                "mirels_nature": 3,
                "mirels_size": 2,
                "uncertainty_preset": "moderately_confident",
                "ood_simulation": False
            }
            r = client.post("/api/v1/workbench/run-demo", json=payload)
            assert r.status_code == 200, f"Expected 200, got {r.status_code}"
            data = r.json()
            assert "radiomics_panel" in data
            assert "biomechanics_geometry_panel" in data
            assert "biomechanics_mirels_panel" in data
            assert "explainability_panel" in data
            assert "uncertainty_panel" in data
            assert "outstanding_clinical_requirements" in data
            print(f"  [PASS] POST /api/v1/workbench/run-demo -> 200 OK | Full synthetic pipeline output assembled successfully")
        except Exception as e:
            print(f"  [FAIL] POST /api/v1/workbench/run-demo -> Error: {e}")
            all_ok = False

        # 13. POST Four-class diagnostic (MUST BE 503 BLOCKED)
        try:
            payload = {"image_id": "test_phantom_001", "modality": "radiograph"}
            r = client.post("/api/v1/diagnostic/four-class", json=payload)
            assert r.status_code == 503, f"Expected 503, got {r.status_code}"
            data = r.json()
            details = data.get("details", {})
            err_code = data.get("error_code") or details.get("error_code")
            assert err_code == "FOUR_CLASS_MODEL_NOT_READY", f"Unexpected error code: {err_code}"
            assert details.get("status") == "BLOCKED"
            assert len(details.get("blockers", [])) == 5
            print(f"  [PASS] POST /api/v1/diagnostic/four-class -> 503 SERVICE UNAVAILABLE | Safety Gate Active: {err_code}")
        except Exception as e:
            print(f"  [FAIL] POST /api/v1/diagnostic/four-class -> Error: {e}")
            all_ok = False

    return all_ok

if __name__ == "__main__":
    print("\nStarting Comprehensive Backend & Proxy Connectivity Audit...\n")
    results = {}
    for name, url in TARGETS:
        results[name] = run_tests_for_target(name, url)
        print()

    print("=" * 80)
    print("FINAL CONNECTIVITY SUMMARY:")
    for name, res in results.items():
        print(f"  {name:<25}: {'PASS' if res else 'FAIL'}")
    print("=" * 80)

    if not all(results.values()):
        sys.exit(1)
    sys.exit(0)
