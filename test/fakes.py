"""Datos simulados compartidos por las pruebas (respuestas válidas según el contrato)."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.model_meta import ModelMeta
from app.schemas import IaAnalyzeResponse, Probabilities, Region

META = ModelMeta(
    model_version="test-v1",
    file="modelo.h5",
    classes=["SANO", "TIZON_TARDIO"],
    input_spec={"width": 224, "height": 224, "color_order": "BGR", "scaling": "unit"},
)

DETECTED = IaAnalyzeResponse(
    status="DETECTED",
    label="TIZON_TARDIO",
    confidence=0.92,
    probabilities=Probabilities(SANO=0.08, TIZON_TARDIO=0.92),
    regions=[Region(
        x=0.1, y=0.2, width=0.3, height=0.4, label="TIZON_TARDIO", confidence=0.92,
        probabilities=Probabilities(SANO=0.08, TIZON_TARDIO=0.92),
    )],
    frame_width=224, frame_height=224, model_version="test-v1", inference_ms=12,
)

NO_VEGETATION = IaAnalyzeResponse(
    status="NO_VEGETATION", regions=[], frame_width=224, frame_height=224,
    model_version="test-v1", inference_ms=5,
)

NOT_POTATO = IaAnalyzeResponse(
    status="NOT_POTATO", regions=[], frame_width=224, frame_height=224,
    model_version="test-v1", inference_ms=6,
)

# Modelo de 4 salidas: las probabilidades de papa suman < 1 (el resto es NO_PAPA).
EARLY_BLIGHT_PROBS = Probabilities(SANO=0.05, TIZON_TEMPRANO=0.88, TIZON_TARDIO=0.04)
DETECTED_EARLY = IaAnalyzeResponse(
    status="DETECTED", label="TIZON_TEMPRANO", confidence=0.88, probabilities=EARLY_BLIGHT_PROBS,
    regions=[Region(x=0.1, y=0.2, width=0.3, height=0.4, label="TIZON_TEMPRANO", confidence=0.88,
                    probabilities=EARLY_BLIGHT_PROBS)],
    frame_width=224, frame_height=224, model_version="test-v1", inference_ms=12,
)


META_4C = META.model_copy(update={"classes": ["SANO", "TIZON_TEMPRANO", "TIZON_TARDIO", "NO_PAPA"]})


def fake_predictor(result: IaAnalyzeResponse = DETECTED, meta: ModelMeta = META) -> MagicMock:
    predictor = MagicMock(is_ready=True, meta=meta)
    predictor.predict.return_value = result
    return predictor
