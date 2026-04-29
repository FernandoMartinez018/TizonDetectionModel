from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import AnalyzeResponse, BoundingBox


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Cliente de prueba de FastAPI (síncrono)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_jpeg_b64() -> str:
    """
    Genera una imagen JPEG mínima en Base64 para tests.
    Es una imagen de 10×10 píxeles negra válida.
    """
    import io
    import cv2

    # Crear imagen verde 224×224 (simula una planta)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 1] = 120  # Canal verde

    # Comprimir a JPEG
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@pytest.fixture
def mock_predictor():
    """Mock del predictor que simula una detección de Tizón Tardío."""
    mock = MagicMock()
    mock.is_ready = True
    mock.model_path = "model/test_model.h5"
    mock.environment = "local"
    mock.predict.return_value = AnalyzeResponse(
        label="Tizon Tardio",
        confidence=0.92,
        probabilities={
            "Sano": 0.03,
            "Tizon Temprano": 0.05,
            "Tizon Tardio": 0.92,
        },
        bounding_boxes=[
            BoundingBox(
                x=0.1,
                y=0.2,
                width=0.3,
                height=0.4,
                label="Tizon Tardio",
                confidence=0.92,
                probabilities={
                    "Sano": 0.03,
                    "Tizon Temprano": 0.05,
                    "Tizon Tardio": 0.92,
                },
            )
        ],
        is_vegetation=True,
        frame_width=224,
        frame_height=224,
    )
    return mock


# ── Tests de /health ───────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """El endpoint /health siempre debe retornar 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """La respuesta de /health debe tener los campos requeridos."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "environment" in data
        assert "model_path" in data
        assert data["status"] == "ok"


# ── Tests de /analyze ──────────────────────────────────────────────────────

class TestAnalyzeEndpoint:

    def test_analyze_sin_modelo_retorna_503(self, client):
        """Sin modelo cargado, /analyze debe retornar 503."""
        import app.main as main_module
        original = main_module._predictor
        main_module._predictor = None

        response = client.post("/analyze", json={"frame_b64": "a" * 100})
        assert response.status_code == 503

        main_module._predictor = original

    def test_analyze_base64_invalido_retorna_400(self, client, mock_predictor):
        """Base64 inválido debe retornar 400."""
        import app.main as main_module
        main_module._predictor = mock_predictor

        response = client.post(
            "/analyze",
            json={"frame_b64": "esto_no_es_base64_!!!"},
        )
        # El validador Pydantic puede retornar 422 si falla la validación del campo
        assert response.status_code in (400, 422)

    def test_analyze_con_imagen_valida(self, client, mock_predictor, sample_jpeg_b64):
        """Con imagen válida y mock del predictor, debe retornar 200."""
        import app.main as main_module
        main_module._predictor = mock_predictor

        response = client.post(
            "/analyze",
            json={
                "frame_b64": sample_jpeg_b64,
                "session_id": "test-session-001",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["label"] == "Tizon Tardio"
        assert data["confidence"] == 0.92
        assert data["is_vegetation"] is True
        assert len(data["bounding_boxes"]) == 1
        assert data["session_id"] == "test-session-001"

    def test_analyze_bbox_coordenadas_relativas(self, client, mock_predictor, sample_jpeg_b64):
        """Las coordenadas de bounding boxes deben ser relativas [0-1]."""
        import app.main as main_module
        main_module._predictor = mock_predictor

        response = client.post("/analyze", json={"frame_b64": sample_jpeg_b64})
        data = response.json()

        for bbox in data["bounding_boxes"]:
            assert 0.0 <= bbox["x"] <= 1.0
            assert 0.0 <= bbox["y"] <= 1.0
            assert 0.0 < bbox["width"] <= 1.0
            assert 0.0 < bbox["height"] <= 1.0
            assert 0.0 <= bbox["confidence"] <= 1.0

    def test_analyze_sin_vegetacion(self, client, sample_jpeg_b64):
        """Frame sin vegetación debe retornar label='Sin deteccion'."""
        import app.main as main_module

        mock = MagicMock()
        mock.is_ready = True
        mock.predict.return_value = AnalyzeResponse(
            label="Sin deteccion",
            confidence=0.0,
            probabilities={"Sano": 0.0, "Tizon Temprano": 0.0, "Tizon Tardio": 0.0},
            bounding_boxes=[],
            is_vegetation=False,
            frame_width=224,
            frame_height=224,
        )
        main_module._predictor = mock

        response = client.post("/analyze", json={"frame_b64": sample_jpeg_b64})
        data = response.json()

        assert data["label"] == "Sin deteccion"
        assert data["is_vegetation"] is False
        assert data["bounding_boxes"] == []