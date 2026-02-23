import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

settings = get_settings()
bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthContext:
    subject: str
    roles: tuple[str, ...]

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_any_role(self, roles: Iterable[str]) -> bool:
        return any(role in self.roles for role in roles)


@dataclass(frozen=True)
class AuthUser:
    username: str
    password: str
    roles: tuple[str, ...]


def _normalize_roles(raw_roles: Any) -> tuple[str, ...]:
    if not isinstance(raw_roles, list):
        return ("viewer",)
    roles = [str(role).strip() for role in raw_roles if str(role).strip()]
    if not roles:
        return ("viewer",)
    # deterministic order for token payload and tests
    return tuple(sorted(set(roles)))


@lru_cache(maxsize=1)
def configured_auth_users() -> Dict[str, AuthUser]:
    users: Dict[str, AuthUser] = {}
    raw = (settings.auth_users_json or "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("AUTH_USERS_JSON must be valid JSON") from exc

        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list):
            raise RuntimeError("AUTH_USERS_JSON must be a list of user definitions")

        for row in payload:
            if not isinstance(row, Mapping):
                continue
            username = str(row.get("username") or "").strip()
            password = str(row.get("password") or "")
            roles = _normalize_roles(row.get("roles"))
            if not username or not password:
                continue
            users[username] = AuthUser(username=username, password=password, roles=roles)

    if users:
        return users

    # Backward-compatible fallback for local development.
    users[settings.auth_dev_user] = AuthUser(
        username=settings.auth_dev_user,
        password=settings.auth_dev_password,
        roles=("admin", "audit", "viewer"),
    )
    return users


def authenticate_user(username: str, password: str) -> AuthContext | None:
    user = configured_auth_users().get(username)
    if user is None:
        return None
    if user.password != password:
        return None
    return AuthContext(subject=user.username, roles=user.roles)


def issue_access_token(subject: str, roles: Iterable[str]) -> str:
    normalized_roles = tuple(sorted({str(role).strip() for role in roles if str(role).strip()})) or ("viewer",)
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "roles": list(normalized_roles),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=settings.jwt_expires_in)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token") from exc
    return payload


def _context_from_payload(payload: Dict[str, Any]) -> AuthContext:
    subject = str(payload.get("sub") or "")
    if not subject:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token subject")
    roles = _normalize_roles(payload.get("roles"))
    return AuthContext(subject=subject, roles=roles)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> AuthContext:
    if not settings.auth_enabled:
        return AuthContext(subject="anonymous", roles=("admin", "audit", "viewer"))
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    payload = decode_access_token(credentials.credentials)
    return _context_from_payload(payload)
