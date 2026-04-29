from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
import base64




class AnalyzeRequest(BaseModel):

    frame_b64: str = Field(
        ...,
        description=(
            "Imagen del frame codificada en Base64 estándar (RFC 4648). "
            "Formatos soportados: JPEG, PNG. "
            "Resolución recomendada: 224×224 px para óptimo rendimiento."
        ),
        min_length=100,  
    )
    session_id: str | None = Field(
        default=None,
        description="ID de sesión del usuario para agrupar análisis. Opcional.",
    )

    @field_validator("frame_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError(
                "frame_b64 no es un string Base64 válido. "
                "Asegúrate de no incluir el prefijo 'data:image/...;base64,'."
            )
        return v




class BoundingBox(BaseModel):

    x: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Coordenada X del borde izquierdo (relativa, 0=izquierda, 1=derecha).",
    )
    y: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Coordenada Y del borde superior (relativa, 0=arriba, 1=abajo).",
    )
    width: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Ancho relativo de la caja de detección.",
    )
    height: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Alto relativo de la caja de detección.",
    )
    label: str = Field(
        ...,
        description="Clase predicha: 'Sano', 'Tizon Temprano' o 'Tizon Tardio'.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidad de la clase predicha [0.0 – 1.0].",
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Probabilidad de cada clase: {'Sano': 0.1, 'Tizon Temprano': 0.2, 'Tizon Tardio': 0.7}.",
    )


class AnalyzeResponse(BaseModel):

    label: str = Field(
        ...,
        description="Clase dominante del frame: 'Sano', 'Tizon Temprano', 'Tizon Tardio' o 'Sin deteccion'.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confianza de la predicción dominante [0.0 – 1.0].",
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Probabilidades de la predicción dominante por clase.",
    )
    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list,
        description="Lista de todas las detecciones en el frame con sus coordenadas relativas.",
    )
    is_vegetation: bool = Field(
        ...,
        description="True si se detectó vegetación válida en el frame.",
    )
    frame_width: int = Field(
        ...,
        description="Ancho original del frame en píxeles.",
    )
    frame_height: int = Field(
        ...,
        description="Alto original del frame en píxeles.",
    )
    session_id: str | None = Field(
        default=None,
        description="Eco del session_id recibido en el request.",
    )


class HealthResponse(BaseModel):

    status: str = Field(..., description="'ok' si el servicio está operativo.")
    model_loaded: bool = Field(
        ...,
        description="True si el modelo TensorFlow está cargado en memoria.",
    )
    model_path: str = Field(
        ...,
        description="Ruta desde donde se cargó el modelo (local o S3).",
    )
    environment: str = Field(
        ...,
        description="Entorno actual: 'local' o 'aws'.",
    )