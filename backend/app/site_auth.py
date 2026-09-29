"""Admin password for pipeline settings writes (password_admin from .env)."""

from __future__ import annotations

import hmac

from app.db.config import PASSWORD_ADMIN


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


def admin_password_required() -> bool:
    """True when password_admin is configured in .env."""
    return bool(PASSWORD_ADMIN)


def is_admin_password(password: str | None) -> bool:
    """True if password matches admin, or no admin password is configured."""
    if not PASSWORD_ADMIN:
        return True
    return _eq((password or "").strip(), PASSWORD_ADMIN)
