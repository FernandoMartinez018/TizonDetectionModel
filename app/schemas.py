"""Esquemas de la API interna de IA. Idénticos a `contracts/tizon-api.yaml` (tag interno-ia)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.model_meta import ClassId, InputSpec, ModelClass


class Probabilities(BaseModel):
    """Probabilidades de las clases de papa. Con el modelo de 4 salidas la suma puede ser < 1 (el resto es NO_PAPA)."""

    SANO: float = Field(ge=0, le=1)
    TIZON_TEMPRANO: float | None = Field(default=None, ge=0, le=1, description="null con el modelo anterior de 2 clases")
    TIZON_TARDIO: float = Field(ge=0, le=1)


class Region(BaseModel):
    """Región de hoja analizada (no necesariamente la lesión). Coordenadas relativas al frame."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    label: ClassId
    confidence: float = Field(ge=0, le=1)
    probabilities: Probabilities


class IaAnalyzeRequest(BaseModel):
    frame_b64: str = Field(min_length=100, description="JPEG/PNG en Base64 estándar, sin prefijo data URI")
    request_id: str | None = None


class IaAnalyzeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["DETECTED", "NO_VEGETATION", "NOT_POTATO"]
    label: ClassId | None = None
    confidence: float | None = None
    probabilities: Probabilities | None = None
    regions: list[Region] = Field(default_factory=list)
    frame_width: int
    frame_height: int
    model_version: str
    inference_ms: int


class IaHealth(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    classes: list[ModelClass]   # salidas del modelo en orden, incluida NO_PAPA
    input_spec: InputSpec | dict  # {} cuando no hay model_meta.json válido
