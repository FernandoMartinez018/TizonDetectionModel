from __future__ import annotations

import app.main as main_module
from fakes import NO_VEGETATION, fake_predictor


class TestHealth:

    def test_health_informa_metadatos_del_modelo(self, client):
        data = client.get("/health").json()
        assert data == {
            "status": "ok",
            "model_loaded": True,
            "model_version": "test-v1",
            "classes": ["SANO", "TIZON_TARDIO"],
            "input_spec": {"width": 224, "height": 224, "color_order": "BGR", "scaling": "unit"},
        }

    def test_health_degradado_sin_modelo(self, client):
        main_module._predictor = None
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["model_loaded"] is False


class TestAnalyze:

    def test_frame_valido_devuelve_resultado_principal(self, client, frame_b64):
        response = client.post("/v1/analyze", json={"frame_b64": frame_b64})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "DETECTED"
        assert data["label"] == "TIZON_TARDIO"
        assert data["probabilities"] == {"SANO": 0.08, "TIZON_TARDIO": 0.92}
        assert len(data["regions"]) == 1

    def test_sin_hoja_es_resultado_valido_con_label_null(self, client, frame_b64):
        main_module._predictor = fake_predictor(NO_VEGETATION)
        response = client.post("/v1/analyze", json={"frame_b64": frame_b64})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NO_VEGETATION"
        assert data["label"] is None
        assert data["confidence"] is None
        assert data["regions"] == []

    def test_base64_invalido_responde_400_invalid_image(self, client):
        response = client.post("/v1/analyze", json={"frame_b64": "@" * 120})
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_IMAGE"

    def test_imagen_corrupta_responde_400_invalid_image(self, client, frame_b64):
        main_module._predictor.predict.side_effect = ValueError("No se pudo decodificar la imagen.")
        response = client.post("/v1/analyze", json={"frame_b64": frame_b64})
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_IMAGE"

    def test_frame_corto_responde_422_con_campo(self, client):
        response = client.post("/v1/analyze", json={"frame_b64": "abc"})
        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "VALIDATION_ERROR"
        assert "frame_b64" in body["fields"]

    def test_sin_modelo_responde_503_model_not_ready(self, client, frame_b64):
        main_module._predictor = None
        response = client.post("/v1/analyze", json={"frame_b64": frame_b64})
        assert response.status_code == 503
        assert response.json()["code"] == "MODEL_NOT_READY"

    def test_request_id_se_devuelve_en_errores(self, client):
        response = client.post(
            "/v1/analyze", json={"frame_b64": "abc"}, headers={"X-Request-Id": "req-123"}
        )
        assert response.json()["request_id"] == "req-123"

    def test_ruta_v1_antigua_ya_no_existe(self, client, frame_b64):
        assert client.post("/analyze", json={"frame_b64": frame_b64}).status_code == 404
