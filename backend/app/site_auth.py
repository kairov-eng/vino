"""Site password gate: password_admin / password_user from .env."""

from __future__ import annotations

import hmac
from typing import Literal

from starlette.requests import Request

from app.db.config import PASSWORD_ADMIN, PASSWORD_USER, SITE_ACCESS_ENABLED

SITE_PASSWORD_COOKIE = "vino_site_password"
SiteRole = Literal["admin", "user"]


def _eq(a: str, b: str) -> bool:
    if not a or not b:
        return False
    ab = a.encode("utf-8")
    bb = b.encode("utf-8")
    if len(ab) != len(bb):
        # Keep roughly constant work; never raise on length mismatch.
        hmac.compare_digest(ab, ab)
        return False
    return hmac.compare_digest(ab, bb)


def resolve_site_role(password: str | None) -> SiteRole | None:
    """Return role if password matches admin or user; else None."""
    raw = (password or "").strip()
    if not raw or not SITE_ACCESS_ENABLED:
        return None
    if PASSWORD_ADMIN and _eq(raw, PASSWORD_ADMIN):
        return "admin"
    if PASSWORD_USER and _eq(raw, PASSWORD_USER):
        return "user"
    return None


def password_from_request(request: Request) -> str | None:
    """Cookie first, then X-Site-Password header."""
    cookie = (request.cookies.get(SITE_PASSWORD_COOKIE) or "").strip()
    if cookie:
        return cookie
    header = (request.headers.get("x-site-password") or "").strip()
    return header or None


def is_site_auth_exempt(path: str, method: str) -> bool:
    if method.upper() == "OPTIONS":
        return True
    p = path.rstrip("/") or "/"
    if p in {"/api/health", "/api/site-auth"}:
        return True
    # Catalog photos are not secret; skip gate so CDN/nginx can cache freely.
    # (Prod serves /media from vino_frontend nginx, which does not check cookies.)
    if p.startswith("/media/") or p == "/media":
        return True
    return False
