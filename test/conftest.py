from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from fakes import fake_predictor


@pytest.fixture
def frame_b64() -> str:
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 1] = 120
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def client(monkeypatch):
    """Cliente con el modelo simulado (cargado). Sin INTERNAL_API_TOKEN: modo local."""
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setenv("APP_ENV", "local")
    with TestClient(main_module.app) as c:
        main_module._predictor = fake_predictor()
        yield c
