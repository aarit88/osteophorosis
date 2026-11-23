"""Start the real FastAPI and Express processes and check their deployment contract.

Uses temporary, isolated directories and synthetic non-clinical image bytes only for
upload-parser checks. It never creates a model checkpoint or a reviewer identity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(url: str, method: str = "GET", headers: dict[str, str] | None = None, data: bytes | None = None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as response:
        return response.code, response.read()


def wait_for(url: str, process: subprocess.Popen, logs, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            logs.seek(0)
            raise RuntimeError(f"Service exited before becoming healthy (exit={process.returncode}):\n{logs.read()}")
        try:
            status, _ = request(url)
            if status < 400:
                return
        except (OSError, TimeoutError):
            time.sleep(0.25)
    logs.seek(0)
    raise RuntimeError(f"Service did not become healthy at {url}:\n{logs.read()}")


def png_fixture() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x20\x40\x60\xff"))
        + chunk(b"IEND", b"")
    )


def multipart(payload: bytes, media_type: str = "image/png") -> tuple[bytes, str]:
    boundary = "og-smoke-" + secrets.token_hex(12)
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"fixture.png\"\r\n"
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


def stop_process(process: subprocess.Popen | None, name: str, logs) -> bool:
    if process is None or process.poll() is not None:
        return True
    if os.name == "nt":
        # Console control events are not delivered reliably from non-console Codex
        # subprocesses. CI on Linux exercises the graceful signal path.
        process.terminate()
        process.wait(timeout=5)
        return False
    process.terminate()
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        raise RuntimeError(f"{name} did not shut down gracefully.")
    if process.returncode != 0:
        logs.seek(0)
        output = logs.read()
        # Uvicorn handles SIGTERM gracefully but retains the signal-derived
        # negative return code on POSIX. Require its shutdown-complete record
        # so an actual crash or forced termination still fails this smoke test.
        if not (
            os.name != "nt"
            and process.returncode == -signal.SIGTERM
            and "Application shutdown complete." in output
        ):
            raise RuntimeError(f"{name} exited unexpectedly (exit={process.returncode}):\n{output}")
    return True


def main() -> int:
    ml_port, gateway_port = free_port(), free_port()
    gateway_token = secrets.token_urlsafe(36)
    gateway_hmac = secrets.token_hex(32)
    shutdown_results: list[bool] = []

    with tempfile.TemporaryDirectory(prefix="osteoguard-deploy-smoke-") as temp:
        isolated = Path(temp)
        reviewer_registry = isolated / "reviewers.json"
        reviewer_registry.write_text(json.dumps({"registry_version": "1.0", "reviewers": []}), encoding="utf-8")
        environment = os.environ.copy()
        for name in list(environment):
            if name.startswith("OSTEO_") or name in {"NODE_ENV", "BACKEND_URL", "HOST", "PORT"}:
                environment.pop(name, None)
        environment.update({
            "OSTEO_ENV": "development",
            "OSTEO_HOST": "127.0.0.1",
            "OSTEO_PORT": str(ml_port),
            "OSTEO_BASE_DIR": str(isolated),
            "OSTEO_DATA_DIR": str(isolated / "data"),
            "OSTEO_RAW_DATA_DIR": str(isolated / "data" / "raw"),
            "OSTEO_INTERIM_DATA_DIR": str(isolated / "data" / "interim"),
            "OSTEO_PROCESSED_DATA_DIR": str(isolated / "data" / "processed"),
            "OSTEO_SPLITS_DATA_DIR": str(isolated / "data" / "splits"),
            "OSTEO_ARTIFACTS_DIR": str(isolated / "artifacts"),
            "OSTEO_CHECKPOINTS_DIR": str(isolated / "artifacts" / "checkpoints"),
            "OSTEO_LOGS_DIR": str(isolated / "artifacts" / "logs"),
            "OSTEO_GATEWAY_HMAC_SECRET": gateway_hmac,
            "OSTEO_REVIEWER_AUTHORIZATION_FILE": str(reviewer_registry),
            "OSTEO_ONBOARDING_REGISTRY_FILE": str(isolated / "onboarding" / "registry.json"),
            "OSTEO_ONBOARDING_ROOT": str(isolated / "onboarding" / "packages"),
            "OSTEO_STAGING_ROOT": str(isolated / "onboarding" / "staged"),
            "OSTEO_DISABLE_MOCK_FALLBACK": "true",
            "OSTEO_CORS_ORIGINS": f"http://127.0.0.1:{gateway_port}",
        })

        ml_logs = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        gateway_logs = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        ml = gateway = None
        try:
            ml = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "ml_service.app.main:app", "--host", "127.0.0.1", "--port", str(ml_port)],
                cwd=ROOT, env=environment, stdout=ml_logs, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            wait_for(f"http://127.0.0.1:{ml_port}/health/ready", ml, ml_logs)

            gateway_env = environment.copy()
            gateway_env.update({
                "NODE_ENV": "development",
                "HOST": "127.0.0.1",
                "PORT": str(gateway_port),
                "BACKEND_URL": f"http://127.0.0.1:{ml_port}",
                "OSTEO_NODE_DATA_DIR": str(isolated / "node-state"),
                "OSTEO_DISABLE_DEMO_SEED": "true",
                "OSTEO_STRICT_CLINICAL_MODE": "true",
                "OSTEO_CLINICAL_API_TOKEN": gateway_token,
            })
            gateway = subprocess.Popen(
                ["node", "--import", "tsx", "server.ts"],
                cwd=ROOT, env=gateway_env, stdout=gateway_logs, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            gateway_url = f"http://127.0.0.1:{gateway_port}"
            wait_for(f"{gateway_url}/readyz", gateway, gateway_logs)

            status, body = request(f"http://127.0.0.1:{ml_port}/health/live")
            assert status == 200 and json.loads(body)["check"] == "liveness"
            status, body = request(f"http://127.0.0.1:{ml_port}/api/ml/health")
            assert status == 200 and json.loads(body)["status"] == "ok"
            status, body = request(f"{gateway_url}/health")
            assert status == 200 and json.loads(body)["service"] == "osteoguard-frontend-server"
            status, body = request(f"{gateway_url}/readyz")
            assert status == 200 and json.loads(body)["dependencies"]["ml_service"] == "ready"
            status, body = request(f"{gateway_url}/api/health")
            assert status == 200 and json.loads(body)["mode"] == "research"
            assert json.loads(body)["clinical_readiness"] == "blocked"

            status, body = request(f"{gateway_url}/api/v1/research/status")
            assert status == 200, body.decode("utf-8", "replace")
            status, body = request(f"{gateway_url}/api/v1/datasets/governance-status")
            assert status == 200, body.decode("utf-8", "replace")
            status, _ = request(f"{gateway_url}/api/v1/dataset-onboarding/sources")
            assert status == 401, f"Unauthenticated onboarding registry request returned HTTP {status}"
            status, body = request(f"{gateway_url}/api/v1/security/status")
            security_status = json.loads(body)
            assert status == 200
            assert set(security_status["clinicalReadiness"]) == {
                "checkpointCount", "trainingLogsAvailable", "clinicalDatasetAvailable", "isFourClassReady"
            }
            assert "rawClassCounts" not in security_status["clinicalReadiness"]
            assert "checkpointsFound" not in security_status["clinicalReadiness"]
            status, body = request(f"{gateway_url}/api/v1/diagnostic/four-class", "POST", {"Content-Type": "application/json"}, b"{}")
            assert status == 503 and b"FOUR_CLASS_MODEL_NOT_READY" in body

            status, _ = request(f"{gateway_url}/api/v1/feedback/list")
            assert status == 401, f"Missing API credentials returned HTTP {status}"
            status, _ = request(f"{gateway_url}/api/v1/feedback/list", headers={"Authorization": "Bearer invalid"})
            assert status == 401, f"Invalid API credentials returned HTTP {status}"

            png, boundary = multipart(png_fixture())
            auth = {"Authorization": f"Bearer {gateway_token}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
            status, body = request(f"{gateway_url}/api/v1/analyze", "POST", auth, png)
            assert status == 503, f"Track D should remain blocked without clinical evidence; got HTTP {status}: {body[:300]!r}"
            assert b"FOUR_CLASS_MODEL_NOT_READY" in body or b"CLINICAL_INFERENCE_BLOCKED" in body or b"not ready" in body.lower()

            invalid, boundary = multipart(b"not an image")
            invalid_auth = {"Authorization": f"Bearer {gateway_token}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
            status, _ = request(f"{gateway_url}/api/v1/analyze", "POST", invalid_auth, invalid)
            assert status == 415, f"Invalid image bytes returned HTTP {status}"

            invalid_type, boundary = multipart(png_fixture(), "application/pdf")
            invalid_type_auth = {"Authorization": f"Bearer {gateway_token}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
            status, _ = request(f"{gateway_url}/api/v1/analyze", "POST", invalid_type_auth, invalid_type)
            assert status == 415, f"Disallowed upload media type returned HTTP {status}"

            oversized, boundary = multipart(b"x" * (15 * 1024 * 1024 + 1))
            oversize_auth = {"Authorization": f"Bearer {gateway_token}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
            status, _ = request(f"{gateway_url}/api/v1/analyze", "POST", oversize_auth, oversized)
            assert status == 413, f"Oversize upload returned HTTP {status}"

            status, body = request(f"http://127.0.0.1:{ml_port}/health/model")
            assert status == 503 and json.loads(body)["status"] == "not_ready"
            shutdown_results.append(stop_process(ml, "FastAPI service", ml_logs))
            status, body = request(f"{gateway_url}/readyz")
            assert status == 503 and json.loads(body)["status"] == "not_ready"
            status, body = request(f"{gateway_url}/api/health")
            assert status == 503 and json.loads(body)["mode"] == "unavailable"
            print("Deployment smoke passed: startup, health, gateway connectivity, governance, auth, upload guards, dependency failure, Track D block.")
        finally:
            if gateway is not None and gateway.poll() is None:
                shutdown_results.append(stop_process(gateway, "Express gateway", gateway_logs))
            if ml is not None and ml.poll() is None:
                shutdown_results.append(stop_process(ml, "FastAPI service", ml_logs))
            gateway_logs.close()
            ml_logs.close()

    if shutdown_results and all(shutdown_results):
        print("Both API processes shut down gracefully.")
    elif os.name == "nt":
        print("Windows test cleanup used process termination; graceful shutdown is verified by Linux CI.")

    invalid_production_env = os.environ.copy()
    for name in list(invalid_production_env):
        if name.startswith("OSTEO_") or name in {"NODE_ENV", "BACKEND_URL", "HOST", "PORT"}:
            invalid_production_env.pop(name, None)
    invalid_production_env.update({"NODE_ENV": "production", "PORT": str(free_port())})
    production_check = subprocess.run(
        ["node", "--import", "tsx", "server.ts"],
        cwd=ROOT,
        env=invalid_production_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert production_check.returncode != 0
    assert "Production requires explicit OSTEO_CORS_ORIGINS" in production_check.stderr
    assert gateway_token not in production_check.stderr and gateway_hmac not in production_check.stderr
    print("Production startup correctly rejected missing configuration without exposing credentials.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Deployment smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
