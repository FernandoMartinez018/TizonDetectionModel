"""Errores con el cuerpo `Error` del contrato: {status, code, message, timestamp, request_id?, fields?}."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_response(
    request: Request,
    status: int,
    code: str,
    message: str,
    fields: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict = {
        "status": status,
        "code": code,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    request_id = request.headers.get("X-Request-Id")
    if request_id:
        body["request_id"] = request_id
    if fields:
        body["fields"] = fields
    return JSONResponse(status_code=status, content=body)
