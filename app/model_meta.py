"""Metadatos del modelo (`model_meta.json`): única fuente de clases, preprocesamiento y versión.

RNF-MAN-03 / RNF-MAN-02: cambiar el modelo = reemplazar el .h5 y este JSON, sin recompilar.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ClassId = Literal["SANO", "TIZON_TEMPRANO", "TIZON_TARDIO"]   # clases del contrato, en orden de severidad
NOT_POTATO = "NO_PAPA"   # salida del modelo para hojas de otras plantas: no se analiza el tizón (status NOT_POTATO)
ModelClass = Literal["SANO", "TIZON_TEMPRANO", "TIZON_TARDIO", "NO_PAPA"]

DEFAULT_META_PATH = "models/model_meta.json"
DEFAULT_MODEL_PATH = "models/modelo_tizon_4c.keras"


class InputSpec(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    color_order: Literal["RGB", "BGR"]
    scaling: Literal["unit", "mobilenet_v2", "none"]


class ModelMeta(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    file: str
    sha256: str | None = None
    classes: list[ModelClass] = Field(description="Orden = índice de salida del modelo")
    input_spec: InputSpec
    min_confidence: float = Field(default=0.30, ge=0, le=1)

    @field_validator("classes")
    @classmethod
    def _clases_validas(cls, v: list[str]) -> list[str]:
        # TIZON_TEMPRANO y NO_PAPA son opcionales: el modelo anterior de 2 clases sigue sirviendo para volver atrás.
        if len(set(v)) != len(v) or not {"SANO", "TIZON_TARDIO"} <= set(v):
            raise ValueError("classes debe incluir SANO y TIZON_TARDIO, sin repetir "
                             "(TIZON_TEMPRANO y NO_PAPA son opcionales)")
        return v


def load_meta(path: str | None = None) -> ModelMeta:
    meta_path = Path(path or os.getenv("MODEL_META_PATH", DEFAULT_META_PATH))
    return ModelMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
