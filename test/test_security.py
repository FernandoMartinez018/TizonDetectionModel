"""F1-IA-01: /v1/analyze exige X-Internal-Token (comparación en tiempo constante)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from fakes import fake_predictor

TOKEN = "token-interno-de-prueba"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", TOKEN)
    with TestClient(app) as c:
        main_module._predictor = fake_predictor()
        yield c


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Token": "incorrecto"}])
def test_sin_token_o_token_incorrecto_responde_401(client, frame_b64, headers):
    response = client.post("/v1/analyze", json={"frame_b64": frame_b64}, headers=headers)
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_token_correcto_responde_200(client, frame_b64):
    response = client.post(
        "/v1/analyze", json={"frame_b64": frame_b64}, headers={"X-Internal-Token": TOKEN}
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
