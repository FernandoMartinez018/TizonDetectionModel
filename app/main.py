from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from mangum import Mangum

from app.predictor import Predictor
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.security import check_startup_config, require_internal_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global _predictor

    check_startup_config()

    logger.info("=" * 60)
    logger.info("Iniciando Tizón Tardío Detection API")
    logger.info("Entorno: %s", os.getenv("APP_ENV", "local"))
    logger.info("=" * 60)

    _predictor = Predictor()
    try:
        _predictor.load_model()
        logger.info("✓ Modelo cargado. Servicio listo.")
    except FileNotFoundError as exc:
        logger.error("✗ No se pudo cargar el modelo: %s", exc)
        logger.warning(
            "El servicio arrancó sin modelo. "
            "/analyze retornará 503 hasta que el modelo esté disponible."
        )

    yield  

    logger.info("Apagando servicio...")
    _predictor = None


app = FastAPI(
    title="Tizón Tardío Detection API",
    description=(
        "Servicio de análisis de imágenes de cultivos de papa.\n\n"
        "Detecta Tizón Temprano y Tizón Tardío usando un modelo TensorFlow.\n\n"
        "**Flujo de uso:**\n"
        "1. Spring Boot recibe el frame del móvil.\n"
        "2. Spring llama a `POST /analyze` con el frame en Base64.\n"
        "3. Este servicio retorna la clasificación con bounding boxes.\n"
        "4. Spring persiste el resultado y lo devuelve al móvil."
    ),
    version="1.0.0",
    docs_url="/docs",        
    redoc_url="/redoc",      
    lifespan=lifespan,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Error no manejado en %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Error interno del servidor.",
            "type": type(exc).__name__,
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar estado del servicio",
    tags=["Infraestructura"],
)
async def health_check() -> HealthResponse:
    
    logger.info("invocaron a /health")
    return HealthResponse(
        status="ok",
        model_loaded=_predictor is not None and _predictor.is_ready,
        model_path=_predictor.model_path if _predictor else "no cargado",
        environment=os.getenv("APP_ENV", "local"),
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analizar un frame de cámara",
    tags=["Análisis"],
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "Frame analizado correctamente."},
        400: {"description": "Frame inválido o Base64 incorrecto."},
        401: {"description": "Falta X-Internal-Token o no coincide."},
        503: {"description": "Modelo no disponible aún (arranque del servidor)."},
    },
)
async def analyze_frame(request: AnalyzeRequest) -> AnalyzeResponse:
    
    if _predictor is None or not _predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo no está disponible. "
                "El servidor puede estar iniciando. Reintenta en unos segundos."
            ),
        )

    try:
        image_bytes = base64.b64decode(request.frame_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error decodificando Base64: {exc}",
        ) from exc

    try:
        result = _predictor.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Error durante el análisis del frame.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno durante el análisis de la imagen.",
        ) from exc

    result.session_id = request.session_id
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