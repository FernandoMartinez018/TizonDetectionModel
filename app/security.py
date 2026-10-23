"""Autenticación interna Spring → IA mediante el header X-Internal-Token."""
from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


def _expected_token() -> str | None:
    return os.getenv("INTERNAL_API_TOKEN") or None


def check_startup_config() -> None:
    """Fuera de APP_ENV=local el token es obligatorio: sin él el servicio no arranca."""
    if _expected_token() is None:
        if os.getenv("APP_ENV", "local") != "local":
            raise RuntimeError("INTERNAL_API_TOKEN es obligatorio cuando APP_ENV != local")
        logger.warning("INTERNAL_API_TOKEN no definido: autenticación interna DESHABILITADA (solo local).")


async def require_internal_token(
    x_internal_token: str | None = Header(default=None),
) -> None:
    expected = _expected_token()
    if expected is None:
        return  # solo posible con APP_ENV=local (validado al arrancar)
    if x_internal_token is None or not hmac.compare_digest(
        x_internal_token.encode(), expected.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token interno inválido o ausente.",
        )
