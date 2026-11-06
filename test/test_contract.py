"""F2-IA-01: las respuestas del servicio cumplen contracts/tizon-api.yaml (tag interno-ia).

contracts/tizon-api.yaml es una copia de la fuente de verdad de la carpeta raíz del proyecto.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openapi_core import OpenAPI
from openapi_core.testing import MockRequest, MockResponse

import app.main as main_module
from fakes import DETECTED_EARLY, META_4C, NO_VEGETATION, NOT_POTATO, fake_predictor

SPEC = OpenAPI.from_file_path(str(Path(__file__).parents[1] / "contracts" / "tizon-api.yaml"))


def assert_cumple_contrato(method: str, path: str, response) -> None:
    SPEC.validate_response(
        MockRequest("http://ia:8000", method, path),
        MockResponse(response.content, status_code=response.status_code,
                     content_type=response.headers["content-type"]),
    )


def test_health(client):
    assert_cumple_contrato("get", "/health", client.get("/health"))


def test_health_con_cuatro_salidas(client):
    main_module._predictor = fake_predictor(meta=META_4C)
    response = client.get("/health")
    assert response.json()["classes"] == ["SANO", "TIZON_TEMPRANO", "TIZON_TARDIO", "NO_PAPA"]
    assert_cumple_contrato("get", "/health", response)


def test_health_degradado(client):
    main_module._predictor = None
    assert_cumple_contrato("get", "/health", client.get("/health"))


def test_analyze_detected(client, frame_b64):
    response = client.post("/v1/analyze", json={"frame_b64": frame_b64})
    assert response.status_code == 200
    assert_cumple_contrato("post", "/v1/analyze", response)


@pytest.mark.parametrize("result", [NO_VEGETATION, NOT_POTATO, DETECTED_EARLY], ids=lambda r: r.label or r.status)
def test_analyze_otros_resultados(client, frame_b64, result):
    main_module._predictor = fake_predictor(result)
    response = client.post("/v1/analyze", json={"frame_b64": frame_b64})
    assert response.json()["status"] == result.status
    assert_cumple_contrato("post", "/v1/analyze", response)


@pytest.mark.parametrize("body, status", [({"frame_b64": "@" * 120}, 400), ({"frame_b64": "abc"}, 422)])
def test_analyze_errores(client, body, status):
    response = client.post("/v1/analyze", json=body)
    assert response.status_code == status
    assert_cumple_contrato("post", "/v1/analyze", response)


def test_analyze_sin_modelo(client, frame_b64):
    main_module._predictor = None
    response = client.post("/v1/analyze", json={"frame_b64": frame_b64})
    assert response.status_code == 503
    assert_cumple_contrato("post", "/v1/analyze", response)


def test_analyze_sin_token(monkeypatch, frame_b64):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secreto-de-prueba")
    with TestClient(main_module.app) as c:
        main_module._predictor = fake_predictor()
        response = c.post("/v1/analyze", json={"frame_b64": frame_b64})
    assert response.status_code == 401
    assert_cumple_contrato("post", "/v1/analyze", response)
