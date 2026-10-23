"""F1-IA-01: /analyze exige X-Internal-Token (comparación en tiempo constante)."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import app.main as main_module
from app.main import app
from app.schemas import AnalyzeResponse

TOKEN = "token-interno-de-prueba"


@pytest.fixture
def frame_b64() -> str:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 1] = 120
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", TOKEN)
    with TestClient(app) as c:
        predictor = MagicMock(is_ready=True, model_path="modelo.h5")
        predictor.predict.return_value = AnalyzeResponse(
            label="Sano", confidence=0.9, probabilities={"Sano": 0.9, "Tizon Tardio": 0.1},
            bounding_boxes=[], is_vegetation=True, frame_width=224, frame_height=224,
        )
        main_module._predictor = predictor
        yield c


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Token": "incorrecto"}])
def test_sin_token_o_token_incorrecto_responde_401(client, frame_b64, headers):
    response = client.post("/analyze", json={"frame_b64": frame_b64}, headers=headers)
    assert response.status_code == 401


def test_token_correcto_responde_200(client, frame_b64):
    response = client.post(
        "/analyze", json={"frame_b64": frame_b64}, headers={"X-Internal-Token": TOKEN}
    )
    assert response.status_code == 200


def test_health_no_exige_token(client):
    assert client.get("/health").status_code == 200


def test_no_arranca_sin_token_fuera_de_local(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="INTERNAL_API_TOKEN"):
        with TestClient(app):
            pass
