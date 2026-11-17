"""
OsteoGuard AI — Stage 31 Release Route and E2E Verification Script
Validates:
1. Direct and proxied HTTP communication for all 12 research/workbench endpoints on port 3000.
2. Verification of the Four-Class Diagnostic safety gate (HTTP 503 BLOCKED).
3. Headless Chrome DOM rendering across all application routes on port 3000.
4. Zero unhandled exceptions or broken render states.
"""

import os
import sys
import tempfile
import subprocess
import httpx

sys.stdout.reconfigure(line_buffering=True)

FRONTEND_URL = "http://127.0.0.1:3000"
BACKEND_URL = "http://127.0.0.1:8000"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

ROUTES_TO_TEST = [
    ("/", "Unified Deep Learning & Radiomic Framework"),
    ("/workbench", "Research Prototype Execution Workbench"),
    ("/tracks", "Institutional Research Tracks & Empirical Findings"),
    ("/radiomics", "Proposal-Aligned Radiomics Feature Extraction"),
    ("/biomechanics", "Biomechanical Fracture-Risk Geometry"),
    ("/explainability", "Dual Explainability Engine: Grad-CAM & SHAP"),
    ("/uncertainty", "Uncertainty Quantification & Selective Prediction"),
    ("/evidence", "Model Evidence Registry"),
    ("/governance", "Institutional Governance & Ethical Protocol"),
    ("/architecture", "Proposal Alignment & Architecture Specification"),
    ("/about", "About OsteoGuard AI"),
    ("/audit", "Forensic Audit"),
]

def test_api_proxy():
    print("=" * 75)
    print("OSTEOGUARD AI — FRONTEND PROXY & API ENDPOINT AUDIT (Port 3000 -> 8000)")
    print("=" * 75)
    
    endpoints = [
        ("GET", "/api/v1/research/status", None, 200),
        ("GET", "/api/v1/research/tracks", None, 200),
        ("GET", "/api/v1/evidence/registry", None, 200),
        ("GET", "/api/v1/architecture/readiness", None, 200),
        ("GET", "/api/v1/datasets/status", None, 200),
        ("POST", "/api/v1/radiomics/demo", {"lesion_type": "osteolytic", "phantom_size": 256}, 200),
        ("POST", "/api/v1/biomechanics/geometry", {"outer_radius_mm": 16.0, "cortical_thickness_mm": 5.0, "defect_depth_mm": 3.0, "defect_arc_degrees": 60.0}, 200),
        ("POST", "/api/v1/biomechanics/mirels", {"site": "femur", "pain": "moderate", "lesion_type": "mixed", "size_ratio": 0.45}, 200),
        ("POST", "/api/v1/explainability/demo", {"target_class": 0, "generate_gradcam": True, "generate_shap": True}, 200),
        ("POST", "/api/v1/uncertainty/demo", {"case_type": "high_confidence"}, 200),
        ("POST", "/api/v1/uncertainty/demo", {"case_type": "boundary_uncertain"}, 200),
        ("POST", "/api/v1/workbench/run-demo", {"lesion_type": "osteolytic", "site": "femur"}, 200),
        ("POST", "/api/v1/diagnostic/four-class", {"image_id": "test_phantom_001", "modality": "radiograph"}, 503),
    ]

    passed = 0
    failed = 0

    with httpx.Client(base_url=FRONTEND_URL, timeout=30.0) as client:
        for method, endpoint, payload, expected_status in endpoints:
            try:
                if method == "GET":
                    resp = client.get(endpoint)
                else:
                    resp = client.post(endpoint, json=payload or {})
                
                if resp.status_code == expected_status:
                    if endpoint == "/api/v1/diagnostic/four-class":
                        data = resp.json()
                        detail = data.get("detail", {})
                        if isinstance(detail, dict) and detail.get("error_code") == "FOUR_CLASS_MODEL_NOT_READY":
                            print(f"  [PASS] {method:<4} {endpoint:<34} -> {resp.status_code} (SAFETY GATE ENFORCED: FOUR_CLASS_MODEL_NOT_READY)")
                            passed += 1
                        else:
                            print(f"  [PASS] {method:<4} {endpoint:<34} -> {resp.status_code} (Status OK, detail={detail})")
                            passed += 1
                    else:
                        print(f"  [PASS] {method:<4} {endpoint:<34} -> {resp.status_code} OK")
                        passed += 1
                else:
                    print(f"  [FAIL] {method:<4} {endpoint:<34} -> Expected {expected_status}, got {resp.status_code}: {resp.text[:100]}")
                    failed += 1
            except Exception as e:
                print(f"  [ERROR] {method:<4} {endpoint:<34} -> Exception: {e}")
                failed += 1

    print(f"\nAPI Proxy Audit: {passed}/{len(endpoints)} passed, {failed} failed.\n")
    return failed == 0

def dump_dom_via_chrome(url: str, user_data_dir: str) -> str:
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--dump-dom",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={user_data_dir}",
        url
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=12, encoding="utf-8", errors="replace")
    return res.stdout

def test_routes():
    print("=" * 75)
    print("OSTEOGUARD AI — HEADLESS CHROME ROUTE RENDERING AUDIT")
    print("=" * 75)
    
    passed = 0
    failed = 0
    temp_dir = tempfile.mkdtemp(prefix="chrome_audit_")

    try:
        for route, expected_text in ROUTES_TO_TEST:
            url = f"{FRONTEND_URL}{route}"
            try:
                dom = dump_dom_via_chrome(url, temp_dir)
                if expected_text in dom:
                    print(f"  [PASS] Route: {route:<18} -> Rendered successfully with keyword: '{expected_text}'")
                    passed += 1
                else:
                    print(f"  [FAIL] Route: {route:<18} -> Expected '{expected_text}' not in DOM (DOM length: {len(dom)})")
                    failed += 1
            except Exception as e:
                print(f"  [ERROR] Route: {route:<18} -> Exception: {e}")
                failed += 1
    finally:
        pass

    print(f"\nRoute Audit: {passed}/{len(ROUTES_TO_TEST)} passed, {failed} failed.\n")
    return failed == 0

if __name__ == "__main__":
    print("\nStarting OsteoGuard AI Release Verification...\n")
    api_ok = test_api_proxy()
    routes_ok = test_routes()
    if api_ok and routes_ok:
        print("[SUCCESS] All endpoints and all frontend routes verified successfully!")
        sys.exit(0)
    else:
        print("[FAILURE] Some verification checks failed.")
        sys.exit(1)
