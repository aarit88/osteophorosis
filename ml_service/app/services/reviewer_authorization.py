"""Bearer-token authentication and role authorization for human reviewers.

The registry is provisioned by the institution outside the repository. It stores
SHA-256 digests of reviewer tokens, never bearer tokens themselves.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, FrozenSet, Iterable, Optional


REVIEWER_ID_RE = re.compile(r"^(?:REV|ADJ|RAD|CLIN)_[A-Za-z0-9_-]{2,}$")
ALLOWED_REVIEWER_ROLES = frozenset({
    "CLINICAL_ADJUDICATOR",
    "RADIOLOGIST",
    "PRINCIPAL_INVESTIGATOR",
    "DATA_STEWARD",
    "IRB_OFFICER",
    "DATA_PROTECTION_OFFICER",
    "LEGAL_COUNSEL",
})
CLINICAL_ADJUDICATION_ROLES = frozenset({"CLINICAL_ADJUDICATOR", "RADIOLOGIST"})


class ReviewerAuthorizationError(PermissionError):
    """Authentication or authorization failure without revealing registry data."""


class ReviewerAuthorizationConfigurationError(RuntimeError):
    """Missing or invalid institutional reviewer registry configuration."""


class ReviewerAuthorizationService:
    def __init__(self, registry_path: Optional[Path] = None) -> None:
        configured = registry_path or (Path(os.environ["OSTEO_REVIEWER_AUTHORIZATION_FILE"])
                                        if os.environ.get("OSTEO_REVIEWER_AUTHORIZATION_FILE") else None)
        self.registry_path = configured

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path is None or not self.registry_path.is_file():
            raise ReviewerAuthorizationConfigurationError(
                "Institutional reviewer authorization registry is not configured."
            )
        try:
            registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReviewerAuthorizationConfigurationError(
                "Institutional reviewer authorization registry is unavailable or invalid."
            ) from exc
        if not isinstance(registry, dict) or registry.get("registry_version") != "1.0":
            raise ReviewerAuthorizationConfigurationError("Reviewer authorization registry version is unsupported.")
        reviewers = registry.get("reviewers")
        if not isinstance(reviewers, list):
            raise ReviewerAuthorizationConfigurationError("Reviewer authorization registry is malformed.")
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        for reviewer in reviewers:
            if not isinstance(reviewer, dict):
                raise ReviewerAuthorizationConfigurationError("Reviewer authorization registry is malformed.")
            reviewer_id = str(reviewer.get("reviewer_pseudo_id") or "")
            digest = str(reviewer.get("token_sha256") or "")
            roles = reviewer.get("roles")
            if (not REVIEWER_ID_RE.fullmatch(reviewer_id) or reviewer_id in seen_ids
                    or not re.fullmatch(r"[A-Fa-f0-9]{64}", digest) or digest.lower() in seen_digests
                    or not isinstance(roles, list) or not roles):
                raise ReviewerAuthorizationConfigurationError("Reviewer authorization registry is malformed.")
            normalized = {str(role).strip().upper() for role in roles}
            if not normalized <= ALLOWED_REVIEWER_ROLES:
                raise ReviewerAuthorizationConfigurationError("Reviewer authorization registry contains unsupported roles.")
            seen_ids.add(reviewer_id)
            seen_digests.add(digest.lower())
        return registry

    def authenticate_bearer(self, token: Optional[str]) -> Dict[str, Any]:
        if not token:
            raise ReviewerAuthorizationError("Authenticated reviewer credentials are required.")
        token_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        registry = self._load_registry()
        principal = None
        # Compare each configured digest in constant time. Never disclose the matching row.
        for reviewer in registry["reviewers"]:
            if not isinstance(reviewer, dict) or reviewer.get("active") is not True:
                continue
            expected = reviewer.get("token_sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[A-Fa-f0-9]{64}", expected):
                continue
            if hmac.compare_digest(token_digest, expected.lower()):
                principal = reviewer
        if principal is None:
            raise ReviewerAuthorizationError("Reviewer credentials are invalid or inactive.")
        reviewer_id = str(principal.get("reviewer_pseudo_id") or "")
        roles = principal.get("roles")
        if not REVIEWER_ID_RE.fullmatch(reviewer_id) or not isinstance(roles, list) or not roles:
            raise ReviewerAuthorizationConfigurationError("Authenticated reviewer registry entry is invalid.")
        normalized_roles = frozenset(str(role).strip().upper() for role in roles)
        if not normalized_roles <= ALLOWED_REVIEWER_ROLES:
            raise ReviewerAuthorizationConfigurationError("Reviewer registry contains an unsupported role.")
        return {"reviewer_pseudo_id": reviewer_id, "roles": normalized_roles}

    def authorize_bearer(self, token: Optional[str], allowed_roles: Iterable[str]) -> Dict[str, Any]:
        principal = self.authenticate_bearer(token)
        if not principal["roles"].intersection({str(role).strip().upper() for role in allowed_roles}):
            raise ReviewerAuthorizationError("Authenticated reviewer lacks the required role.")
        return principal
