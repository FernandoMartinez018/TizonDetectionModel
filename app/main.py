from __future__ import annotations

import base64
import binascii
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from mangum import Mangum
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import ApiError, error_response
from app.model_meta import DEFAULT_META_PATH
from app.predictor import Predictor
from app.schemas import IaAnalyzeRequest, IaAnalyzeResponse, IaHealth
from app.security import check_startup_config, require_internal_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/modelo_tizon_tardio.h5"

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global _predictor

    check_startup_config()

    logger.info("Iniciando servicio de IA (entorno: %s)", os.getenv("APP_ENV", "local"))

    _predictor = Predictor()
    try:
        _predictor.load(
            os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
            os.getenv("MODEL_META_PATH", DEFAULT_META_PATH),
        )
    except Exception:
        # El servicio arranca degradado: /health lo informa y /v1/analyze responde 503.
        logger.exception("No se pudo cargar el modelo o model_meta.json")

    yield

    logger.info("Apagando servicio...")
    _predictor = None


app = FastAPI(
    title="Tizón Tardío — Servicio de IA",
    description=(
        "API interna (solo red Docker). Clasifica regiones de hoja de papa como "
        "SANO o TIZON_TARDIO. Contrato: contracts/tizon-api.yaml (tag interno-ia)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)


# ── Errores con el cuerpo `Error` del contrato ────────────────────────────

_CODE_BY_STATUS = {401: "UNAUTHORIZED", 404: "NOT_FOUND"}


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return error_response(request, exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = {".".join(str(p) for p in e["loc"][1:]) or "body": e["msg"] for e in exc.errors()}
    return error_response(request, 422, "VALIDATION_ERROR", "Datos inválidos.", fields)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    code = _CODE_BY_STATUS.get(exc.status_code, "VALIDATION_ERROR" if exc.status_code < 500 else "INTERNAL_ERROR")
    return error_response(request, exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Error no manejado en %s %s", request.method, request.url.path)
    return error_response(request, 500, "INTERNAL_ERROR", "Error interno del servicio de IA.")


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=IaHealth, tags=["Infraestructura"])
async def health_check() -> IaHealth:
    ready = _predictor is not None and _predictor.is_ready
    meta = _predictor.meta if _predictor else None
    return IaHealth(
        status="ok" if ready else "degraded",
        model_loaded=ready,
        model_version=meta.model_version if meta else "desconocido",
        classes=meta.classes if meta else [],
        input_spec=meta.input_spec if meta else {},
    )


@app.post(
    "/v1/analyze",
    response_model=IaAnalyzeResponse,
    response_model_exclude_none=False,
    tags=["Análisis"],
    dependencies=[Depends(require_internal_token)],
    responses={
        400: {"description": "Base64 inválido o imagen no decodificable (INVALID_IMAGE)."},
        401: {"description": "Falta X-Internal-Token o no coincide."},
        503: {"description": "Modelo no cargado (MODEL_NOT_READY)."},
    },
)
async def analyze_frame(
    body: IaAnalyzeRequest,
    x_request_id: str | None = Header(default=None),
) -> IaAnalyzeResponse:
    request_id = x_request_id or body.request_id

    if _predictor is None or not _predictor.is_ready:
        raise ApiError(503, "MODEL_NOT_READY", "El modelo no está disponible. Reintenta en unos segundos.")

    try:
        image_bytes = base64.b64decode(body.frame_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError(400, "INVALID_IMAGE", "frame_b64 no es Base64 válido (sin prefijo data URI).") from exc

    try:
        result = _predictor.predict(image_bytes)
    except ValueError as exc:
        raise ApiError(400, "INVALID_IMAGE", str(exc)) from exc

    logger.info(
        "analyze request_id=%s status=%s label=%s regions=%d inference_ms=%d",
        request_id, result.status, result.label, len(result.regions), result.inference_ms,
    )
    return result


lambda_handler = Mangum(app, lifespan="off")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
