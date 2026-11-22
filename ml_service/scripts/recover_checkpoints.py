"""
OsteoGuard AI - Stage 32 Environment-Compatible Checkpoint Recovery & Inventory CLI.

Pure-stdlib Python utility (runs on Python 3.10+ / 3.11+ without PowerShell or third-party
dependencies) supporting:
  1. Recursive workspace scan & 21-target Track A/B/C + Track D checkpoint inventory.
  2. Read-only-source local recovery with SHA-256 / byte-size checks and atomic placement.
  3. Explicitly configured HTTPS remote recovery from an integrity-pinned JSON manifest
     with resumable HTTP Range transfers, temp staging, quarantine on Frozen BTXRD
     mismatch, SSRF protections, and strict credential/query redaction.

Safety & Governance Guarantees:
  - Never assumes a remote storage account exists or invents credentials/URLs.
  - Never fabricates checkpoints or initializes random weights.
  - Never downloads arbitrary files based solely on filename (requires pinned SHA-256 & size).
  - Never overwrites, retrains, modifies, or weakens Frozen BTXRD ResNet-34
    (244,849,703 bytes; SHA-256 21064670FA6277B0E96135333708DE4692DDB80E428B792F3675405B5DC109CD).
  - Quarantines mismatched BTXRD candidates without deleting original source files.
"""

import argparse
import csv
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_service.scripts.run_preliminary_inference import (  # noqa: E402
    CHECKPOINT_INVENTORY,
    FROZEN_BTXRD_EXPECTED_SHA256,
    FROZEN_BTXRD_EXPECTED_SIZE_BYTES,
    compute_sha256,
    inspect_checkpoint_inventory,
)

WEIGHT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".onnx", ".safetensors"}
SKIP_DIR_NAMES = {
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "venvs",
    "site-packages",
    "__pycache__",
    ".cache",
    ".recovery_tmp",
    "recovery_quarantine",
}

ALLOWED_RECOVERY_REL_PATHS = {entry["rel_path"] for entry in CHECKPOINT_INVENTORY}
FROZEN_BTXRD_REL_PATHS = {
    "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt",
    "ml_artifacts/btxrd_full_training/best_checkpoint.pt",
}
HEX64_RE = re.compile(r"^[0-9A-Fa-f]{64}$")
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunks


def redact_url(raw_url: str) -> str:
    """Strips query parameters, userinfo, and fragments so signed tokens are never logged."""
    parsed = urllib.parse.urlsplit(raw_url)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return urllib.parse.urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))


def _is_forbidden_ip(ip_obj: ipaddress._BaseAddress) -> bool:
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def validate_remote_https_url(raw_url: str, resolve_dns: bool = True) -> str:
    """
    Validates that a configured remote checkpoint URL uses strict HTTPS, has no embedded
    userinfo, and does not point to loopback, private, or link-local metadata endpoints.
    """
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("Remote URL must be a non-empty string.")
    parsed = urllib.parse.urlsplit(raw_url.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Only https:// remote checkpoint URLs are permitted (got {parsed.scheme!r}).")
    if parsed.username or parsed.password:
        raise ValueError("Embedded URL credentials (username/password) are forbidden.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Remote URL must contain a valid hostname.")

    lower_host = hostname.lower()
    if lower_host in {"localhost", "localhost.localdomain"} or lower_host.endswith(".local"):
        raise ValueError(f"Local/internal hostnames are forbidden for remote checkpoint downloads: {hostname}")

    # Direct IP literal check
    try:
        ip_literal = ipaddress.ip_address(hostname)
    except ValueError:
        ip_literal = None

    if ip_literal is not None:
        if _is_forbidden_ip(ip_literal):
            raise ValueError(f"Forbidden non-public IP address in remote URL: {hostname}")
        return raw_url.strip()

    if resolve_dns:
        port = parsed.port or 443
        try:
            addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError(f"Unable to resolve remote checkpoint host {hostname}: {exc}") from exc
        if not addr_infos:
            raise ValueError(f"No DNS addresses resolved for remote checkpoint host: {hostname}")
        for info in addr_infos:
            ip_str = info[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if _is_forbidden_ip(ip_obj):
                raise ValueError(
                    f"Remote checkpoint host {hostname} resolved to forbidden non-public IP {ip_str}"
                )

    return raw_url.strip()


class _SafeHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates every redirect target against strict HTTPS + SSRF rules and drops auth across hosts."""

    def __init__(self, resolve_dns: bool = True):
        self.resolve_dns = resolve_dns

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validated_url = validate_remote_https_url(newurl, resolve_dns=self.resolve_dns)
        new_req = super().redirect_request(req, fp, code, msg, headers, validated_url)
        if new_req is not None:
            orig_host = urllib.parse.urlsplit(req.full_url).hostname
            next_host = urllib.parse.urlsplit(validated_url).hostname
            if orig_host != next_host and new_req.has_header("Authorization"):
                new_req.remove_header("Authorization")
        return new_req


def build_safe_https_opener(resolve_dns: bool = True) -> Callable[..., Any]:
    director = urllib.request.build_opener(_SafeHTTPSRedirectHandler(resolve_dns=resolve_dns))
    return director.open


def resolve_confined_target(base_dir: Path, rel_path: str) -> Path:
    """Ensures target checkpoint path stays allowlisted and strictly inside base_dir."""
    normalized_rel = rel_path.replace("\\", "/").lstrip("/")
    if normalized_rel not in ALLOWED_RECOVERY_REL_PATHS:
        raise ValueError(
            f"Refusing arbitrary target path {rel_path!r}: not in allowlisted OsteoGuard checkpoint paths."
        )
    base_resolved = base_dir.resolve()
    dest = (base_resolved / normalized_rel).resolve()
    base_prefix = str(base_resolved) + os.sep
    if not str(dest).startswith(base_prefix):
        raise ValueError(f"Path traversal blocked for target path: {rel_path!r}")
    return dest


def scan_workspace_weight_files(scan_root: Path) -> List[Dict[str, Any]]:
    """
    Recursively scans scan_root (including hidden folders) for weight artifacts
    matching .pt, .pth, .ckpt, .onnx, or .safetensors while skipping dependency trees.
    """
    findings: List[Dict[str, Any]] = []
    if not scan_root.exists():
        return findings

    root_resolved = scan_root.resolve()
    for current_dir, dirnames, filenames in os.walk(root_resolved):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
        for fname in sorted(filenames):
            fpath = Path(current_dir) / fname
            if fpath.suffix.lower() in WEIGHT_EXTENSIONS and fpath.is_file():
                findings.append(
                    {
                        "abs_path": str(fpath),
                        "rel_path": os.path.relpath(fpath, root_resolved).replace("\\", "/"),
                        "size_bytes": fpath.stat().st_size,
                        "sha256": compute_sha256(fpath),
                    }
                )
    return findings


def _quarantine_btxrd_mismatch(src_path: Path, base_dir: Path, sha256: str) -> Path:
    quarantine_dir = base_dir.resolve() / "ml_artifacts" / "recovery_quarantine" / "btxrd_mismatch"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    clean_name = src_path.name[:-5] if src_path.name.endswith(".part") else src_path.name
    dest = quarantine_dir / f"{sha256[:12]}_{clean_name}"
    if not dest.exists():
        tmp = quarantine_dir / f".{dest.name}.tmp.{os.getpid()}"
        shutil.copy2(src_path, tmp)
        os.replace(tmp, dest)
    return dest


def _atomic_place_file(src_temp_path: Path, dest_path: Path, move_source: bool = True) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if move_source:
        tmp_same_fs = dest_path.parent / f".{dest_path.name}.stage.{os.getpid()}"
        try:
            os.replace(src_temp_path, tmp_same_fs)
        except OSError:
            shutil.copyfile(src_temp_path, tmp_same_fs)
            src_temp_path.unlink(missing_ok=True)
        os.replace(tmp_same_fs, dest_path)
    else:
        tmp_same_fs = dest_path.parent / f".{dest_path.name}.stage.{os.getpid()}"
        shutil.copyfile(src_temp_path, tmp_same_fs)
        os.replace(tmp_same_fs, dest_path)


def recover_from_local_directories(
    search_dirs: List[Path],
    base_dir: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """
    Safely inspects local directories for existing OsteoGuard checkpoints.
    Quarantines mismatched BTXRD candidates without deleting source files, verifies
    Frozen BTXRD byte size and SHA-256 before copying, and never overwrites existing targets.
    """
    results: List[Dict[str, Any]] = []
    discovered_total = 0

    for sdir in search_dirs:
        if not sdir.exists():
            results.append({"search_dir": str(sdir), "status": "SEARCH_DIR_NOT_FOUND"})
            continue

        found = scan_workspace_weight_files(sdir)
        discovered_total += len(found)
        for item in found:
            src = Path(item["abs_path"])
            sha = item["sha256"]
            size = item["size_bytes"]
            rel_lower = item["rel_path"].lower()

            matched_rel: Optional[str] = None
            for candidate_rel in sorted(ALLOWED_RECOVERY_REL_PATHS):
                if rel_lower.endswith(candidate_rel.lower()):
                    matched_rel = candidate_rel
                    break

            is_exact_frozen_btxrd = (
                size == FROZEN_BTXRD_EXPECTED_SIZE_BYTES and sha == FROZEN_BTXRD_EXPECTED_SHA256
            )
            looks_like_btxrd = (
                "btxrd" in rel_lower
                and src.name == "best_checkpoint.pt"
                and "btxrd_pilot" not in rel_lower
            )

            if is_exact_frozen_btxrd:
                target_rel = matched_rel if matched_rel in FROZEN_BTXRD_REL_PATHS else "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt"
                dest = resolve_confined_target(base_dir, target_rel)
                if dest.exists():
                    results.append(
                        {
                            "source": str(src),
                            "target": target_rel,
                            "status": "SKIPPED_ALREADY_PRESENT",
                            "sha256": sha,
                            "size_bytes": size,
                        }
                    )
                else:
                    _atomic_place_file(src, dest, move_source=False)
                    results.append(
                        {
                            "source": str(src),
                            "target": target_rel,
                            "status": "RECOVERED_VERIFIED_FROZEN_BTXRD",
                            "sha256": sha,
                            "size_bytes": size,
                        }
                    )
            elif looks_like_btxrd:
                quarantine_dest = _quarantine_btxrd_mismatch(src, base_dir, sha)
                results.append(
                    {
                        "source": str(src),
                        "target": matched_rel or "ml_artifacts/btxrd_frozen_model/best_checkpoint.pt",
                        "status": "QUARANTINED_BTXRD_MISMATCH",
                        "quarantine_path": str(quarantine_dest),
                        "sha256": sha,
                        "size_bytes": size,
                    }
                )
            elif matched_rel is not None:
                dest = resolve_confined_target(base_dir, matched_rel)
                if dest.exists():
                    results.append(
                        {
                            "source": str(src),
                            "target": matched_rel,
                            "status": "SKIPPED_ALREADY_PRESENT",
                            "sha256": sha,
                            "size_bytes": size,
                        }
                    )
                else:
                    _atomic_place_file(src, dest, move_source=False)
                    results.append(
                        {
                            "source": str(src),
                            "target": matched_rel,
                            "status": "RECOVERED_MATCHED_PATH",
                            "sha256": sha,
                            "size_bytes": size,
                        }
                    )
            else:
                results.append(
                    {
                        "source": str(src),
                        "target": None,
                        "status": "UNMAPPED_CANDIDATE_IGNORED",
                        "sha256": sha,
                        "size_bytes": size,
                    }
                )

    return {
        "mode": "local_recovery",
        "discovered_files": discovered_total,
        "actions": results,
    }


def _validate_manifest_entry(
    entry: Dict[str, Any],
    base_dir: Path,
    resolve_dns: bool = True,
) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Manifest entry must be a JSON object.")

    for forbidden_secret_key in ("token", "password", "secret", "authorization", "headers"):
        if forbidden_secret_key in entry:
            raise ValueError(
                f"Inline credentials ({forbidden_secret_key!r}) are forbidden in manifests; "
                "use auth_env_var referencing an environment variable name."
            )

    rel_path = entry.get("rel_path")
    if not isinstance(rel_path, str):
        raise ValueError("Manifest entry missing string 'rel_path'.")
    dest_path = resolve_confined_target(base_dir, rel_path)
    normalized_rel = rel_path.replace("\\", "/").lstrip("/")

    url = validate_remote_https_url(entry.get("url", ""), resolve_dns=resolve_dns)

    expected_sha256 = entry.get("expected_sha256")
    if not isinstance(expected_sha256, str) or not HEX64_RE.match(expected_sha256.strip()):
        raise ValueError(
            f"Remote entry for {normalized_rel} must specify a valid 64-char hex expected_sha256 "
            "(filename-only downloads are strictly forbidden)."
        )
    expected_sha256 = expected_sha256.strip().upper()

    expected_size = entry.get("expected_size_bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
        raise ValueError(
            f"Remote entry for {normalized_rel} must specify a positive integer expected_size_bytes."
        )

    if normalized_rel in FROZEN_BTXRD_REL_PATHS:
        if expected_sha256 != FROZEN_BTXRD_EXPECTED_SHA256:
            raise ValueError(
                f"Manifest SHA-256 for {normalized_rel} conflicts with immutable Frozen BTXRD spec."
            )
        if expected_size != FROZEN_BTXRD_EXPECTED_SIZE_BYTES:
            raise ValueError(
                f"Manifest size for {normalized_rel} conflicts with immutable Frozen BTXRD size."
            )

    auth_env_var = entry.get("auth_env_var")
    if auth_env_var is not None:
        if not isinstance(auth_env_var, str) or not ENV_NAME_RE.match(auth_env_var):
            raise ValueError(f"Invalid auth_env_var name: {auth_env_var!r}")

    return {
        "rel_path": normalized_rel,
        "dest_path": dest_path,
        "url": url,
        "redacted_url": redact_url(url),
        "expected_sha256": expected_sha256,
        "expected_size_bytes": expected_size,
        "auth_env_var": auth_env_var,
    }


def download_checkpoint_resumable(
    validated_entry: Dict[str, Any],
    base_dir: Path,
    timeout_seconds: int = 30,
    resolve_dns: bool = True,
    opener: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """
    Downloads a verified checkpoint using HTTP Range resumption into a staging .part file,
    verifies exact byte size and SHA-256 before atomic placement, and quarantines any
    mismatched Frozen BTXRD file without modifying existing files.
    """
    dest_path: Path = validated_entry["dest_path"]
    rel_path: str = validated_entry["rel_path"]
    expected_size: int = validated_entry["expected_size_bytes"]
    expected_sha256: str = validated_entry["expected_sha256"]
    redacted = validated_entry["redacted_url"]

    # Strict immutability: never overwrite an already present Frozen BTXRD checkpoint
    if dest_path.exists():
        existing_size = dest_path.stat().st_size
        existing_sha = compute_sha256(dest_path)
        if rel_path in FROZEN_BTXRD_REL_PATHS:
            if existing_size == FROZEN_BTXRD_EXPECTED_SIZE_BYTES and existing_sha == FROZEN_BTXRD_EXPECTED_SHA256:
                return {
                    "rel_path": rel_path,
                    "status": "ALREADY_PRESENT_VERIFIED",
                    "size_bytes": existing_size,
                    "sha256": existing_sha,
                    "url": redacted,
                }
            raise FileExistsError(
                f"Existing file at {rel_path} mismatches Frozen BTXRD spec; refusing overwrite."
            )
        return {
            "rel_path": rel_path,
            "status": "SKIPPED_ALREADY_PRESENT",
            "size_bytes": existing_size,
            "sha256": existing_sha,
            "url": redacted,
        }

    headers: Dict[str, str] = {"User-Agent": "OsteoGuard-Recovery/1.0"}
    auth_env_var = validated_entry.get("auth_env_var")
    if auth_env_var:
        token = os.environ.get(auth_env_var, "").strip()
        if not token:
            raise PermissionError(
                f"Required credential environment variable {auth_env_var} is unset or empty."
            )
        headers["Authorization"] = f"Bearer {token}"

    tmp_dir = base_dir.resolve() / "ml_artifacts" / ".recovery_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_key = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]
    part_path = tmp_dir / f"{safe_key}_{dest_path.name}.part"

    resume_offset = 0
    if part_path.is_file():
        cur_size = part_path.stat().st_size
        if 0 < cur_size < expected_size:
            resume_offset = cur_size
            headers["Range"] = f"bytes={resume_offset}-"
        elif cur_size >= expected_size:
            # Oversized stale partial cannot be resumed safely; discard
            part_path.unlink(missing_ok=True)

    req = urllib.request.Request(validated_entry["url"], headers=headers, method="GET")
    open_fn = opener if opener is not None else build_safe_https_opener(resolve_dns=resolve_dns)

    try:
        response = open_fn(req, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and resume_offset > 0:
            part_path.unlink(missing_ok=True)
            headers.pop("Range", None)
            resume_offset = 0
            req = urllib.request.Request(validated_entry["url"], headers=headers, method="GET")
            response = open_fn(req, timeout=timeout_seconds)
        else:
            raise RuntimeError(f"Remote download HTTP error {exc.code} for {rel_path} ({redacted})") from exc

    with response as resp:
        resp_status = getattr(resp, "status", None) or (resp.getcode() if hasattr(resp, "getcode") else 200)
        if resume_offset > 0 and resp_status == 206:
            mode = "ab"
            written_total = resume_offset
        else:
            mode = "wb"
            written_total = 0

        with open(part_path, mode) as out_f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                written_total += len(chunk)
                if written_total > expected_size:
                    out_f.close()
                    if rel_path in FROZEN_BTXRD_REL_PATHS:
                        actual_sha = compute_sha256(part_path)
                        q_path = _quarantine_btxrd_mismatch(part_path, base_dir, actual_sha)
                        part_path.unlink(missing_ok=True)
                        raise ValueError(
                            f"Download exceeded expected size ({written_total} > {expected_size}) for {rel_path}; "
                            f"quarantined partial at {q_path}"
                        )
                    part_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"Download exceeded expected size ({written_total} > {expected_size}) for {rel_path}"
                    )
                out_f.write(chunk)

    actual_size = part_path.stat().st_size
    actual_sha = compute_sha256(part_path)

    if actual_size != expected_size or actual_sha != expected_sha256:
        if rel_path in FROZEN_BTXRD_REL_PATHS:
            q_path = _quarantine_btxrd_mismatch(part_path, base_dir, actual_sha)
            part_path.unlink(missing_ok=True)
            raise ValueError(
                f"Frozen BTXRD integrity mismatch for {rel_path} "
                f"(size={actual_size}/{expected_size}, sha256={actual_sha}); "
                f"quarantined at {q_path}"
            )
        part_path.unlink(missing_ok=True)
        raise ValueError(
            f"Checkpoint integrity mismatch for {rel_path} "
            f"(size={actual_size}/{expected_size}, sha256={actual_sha})"
        )

    _atomic_place_file(part_path, dest_path, move_source=True)
    return {
        "rel_path": rel_path,
        "status": "RECOVERED_RESUMED_AND_VERIFIED" if (resume_offset > 0 and resp_status == 206) else "RECOVERED_AND_VERIFIED",
        "resumed_from_byte": resume_offset if (resume_offset > 0 and resp_status == 206) else 0,
        "size_bytes": actual_size,
        "sha256": actual_sha,
        "url": redacted,
    }


def recover_from_remote_manifest(
    manifest_path: Path,
    base_dir: Path = REPO_ROOT,
    timeout_seconds: int = 30,
    resolve_dns: bool = True,
    opener: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Recovers explicitly pinned checkpoints described in a verified JSON manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Remote checkpoint manifest not found: {manifest_path}")

    raw_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = raw_doc.get("checkpoints") if isinstance(raw_doc, dict) else raw_doc
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a non-empty 'checkpoints' list.")

    validated = [
        _validate_manifest_entry(item, base_dir=base_dir, resolve_dns=resolve_dns)
        for item in entries
    ]

    outcomes = []
    for item in validated:
        outcomes.append(
            download_checkpoint_resumable(
                validated_entry=item,
                base_dir=base_dir,
                timeout_seconds=timeout_seconds,
                resolve_dns=resolve_dns,
                opener=opener,
            )
        )

    return {
        "mode": "remote_manifest_recovery",
        "manifest_path": str(manifest_path),
        "checkpoints_processed": len(outcomes),
        "results": outcomes,
    }


def write_inventory_csv(report: Dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["track", "path", "priority", "status", "size_bytes", "sha256", "integrity"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("inventory", []):
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OsteoGuard AI - Cross-Platform Python Checkpoint Inventory & Recovery CLI"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(REPO_ROOT),
        help="Workspace root directory (default: repository root).",
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Audit workspace weight files (.pt/.pth/.ckpt/.onnx/.safetensors) and expected 21+Track D inventory.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional CSV output path for checkpoint inventory table.",
    )
    parser.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Additional local directory to scan/recover checkpoints from (repeatable, copy-only).",
    )
    parser.add_argument(
        "--local-recover",
        action="store_true",
        help="Copy verified/matched checkpoints from --search-dir into missing workspace targets.",
    )
    parser.add_argument(
        "--remote-manifest",
        type=str,
        default=None,
        help="Path to JSON manifest with explicit HTTPS URLs, expected_sha256, expected_size_bytes, and rel_path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP socket timeout in seconds for remote manifest downloads (default: 30).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir).resolve()

    if args.local_recover:
        if not args.search_dir:
            print(
                json.dumps(
                    {
                        "error": "MISSING_SEARCH_DIR",
                        "detail": "Pass at least one --search-dir path with --local-recover.",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        try:
            payload = recover_from_local_directories(
                [Path(p).resolve() for p in args.search_dir],
                base_dir=base_dir,
            )
            payload["inventory_after"] = inspect_checkpoint_inventory(base_dir)
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, indent=2), file=sys.stderr)
            return 1

    manifest_arg = args.remote_manifest or os.environ.get("OSTEO_CHECKPOINT_MANIFEST_PATH", "").strip() or None
    if manifest_arg:
        try:
            payload = recover_from_remote_manifest(
                manifest_path=Path(manifest_arg).resolve(),
                base_dir=base_dir,
                timeout_seconds=args.timeout,
            )
            payload["inventory_after"] = inspect_checkpoint_inventory(base_dir)
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, indent=2), file=sys.stderr)
            return 1

    # Default / --inventory read-only audit
    workspace_weights = scan_workspace_weight_files(base_dir)
    external_scans = {
        str(Path(p).resolve()): scan_workspace_weight_files(Path(p).resolve())
        for p in args.search_dir
    }
    inventory = inspect_checkpoint_inventory(base_dir)
    if args.csv_out:
        write_inventory_csv(inventory, Path(args.csv_out))

    output = {
        "mode": "inventory_only",
        "remote_storage_configured": False,
        "remote_storage_note": (
            "No remote checkpoint manifest or credentials configured. "
            "Pass --remote-manifest <file.json> (or set OSTEO_CHECKPOINT_MANIFEST_PATH) "
            "with HTTPS URLs and pinned SHA-256 + byte size to enable remote recovery."
        ),
        "workspace_weight_files_found": workspace_weights,
        "additional_search_dir_findings": external_scans,
        "expected_checkpoint_inventory": inventory,
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
