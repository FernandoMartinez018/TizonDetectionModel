"""Metadatos del modelo (`model_meta.json`): única fuente de clases, preprocesamiento y versión.

RNF-MAN-03 / RNF-MAN-02: cambiar el modelo = reemplazar el .h5 y este JSON, sin recompilar.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ClassId = Literal["SANO", "TIZON_TARDIO"]

DEFAULT_META_PATH = "models/model_meta.json"
DEFAULT_MODEL_PATH = "models/modelo_tizon_tardio.h5"


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
    classes: list[ClassId] = Field(description="Orden = índice de salida del modelo")
    input_spec: InputSpec
    min_confidence: float = Field(default=0.30, ge=0, le=1)

    @field_validator("classes")
    @classmethod
    def _dos_clases_distintas(cls, v: list[str]) -> list[str]:
        if sorted(v) != ["SANO", "TIZON_TARDIO"]:
            raise ValueError("classes debe contener exactamente SANO y TIZON_TARDIO")
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
