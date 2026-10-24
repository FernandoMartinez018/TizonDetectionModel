from __future__ import annotations


import json
import os

import cv2
import numpy as np
import pytest

from app.model_meta import InputSpec
from app.predictor import Predictor, pick_main_region, preprocess
from app.schemas import Probabilities, Region


# ── Helpers de test ────────────────────────────────────────────────────────

def create_green_image(width: int = 224, height: int = 224) -> np.ndarray:
    """Crea una imagen verde con textura (simula una hoja de planta)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Verde natural (HSV ~50°, alta saturación)
    img[:, :, 0] = 30    # B
    img[:, :, 1] = 120   # G — canal verde dominante
    img[:, :, 2] = 40    # R
    # Agregar textura aleatoria para superar el umbral de varianza
    noise = np.random.randint(0, 80, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def create_uniform_green_image(width: int = 224, height: int = 224) -> np.ndarray:
    """Crea una imagen verde uniforme (simula un objeto artificial, sin textura)."""
    img = np.full((height, width, 3), fill_value=[30, 120, 40], dtype=np.uint8)
    return img


def image_to_bytes(img: np.ndarray, quality: int = 70) -> bytes:
    """Convierte un array NumPy a bytes JPEG."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


# ── Tests de _decode_image ─────────────────────────────────────────────────

class TestDecodeImage:

    def test_decodifica_jpeg_valido(self):
        """Debe decodificar un JPEG válido sin error."""
        img = create_green_image()
        img_bytes = image_to_bytes(img)
        result = Predictor._decode_image(img_bytes)
        assert result is not None
        assert result.shape[2] == 3  # Tres canales BGR

    def test_error_con_bytes_invalidos(self):
        """Debe lanzar ValueError con bytes que no son imagen."""
        with pytest.raises(ValueError, match="No se pudo decodificar"):
            Predictor._decode_image(b"esto_no_es_una_imagen")

    def test_preserva_dimensiones(self):
        """Las dimensiones del frame deben preservarse tras la decodificación."""
        img = create_green_image(width=640, height=480)
        img_bytes = image_to_bytes(img)
        result = Predictor._decode_image(img_bytes)
        assert result.shape[0] == 480  # height
        assert result.shape[1] == 640  # width


# ── Preprocesamiento y resultado principal ────────────────────────────────

class TestPreprocess:

    def _spec(self, color_order="BGR", scaling="unit"):
        return InputSpec(width=224, height=224, color_order=color_order, scaling=scaling)

    def test_escala_unit_queda_en_0_1(self):
        roi = np.full((50, 50, 3), 255, dtype=np.uint8)
        out = preprocess(roi, self._spec(scaling="unit"))
        assert out.shape == (224, 224, 3)
        assert out.max() == pytest.approx(1.0)

    def test_escala_mobilenet_v2_queda_en_menos1_1(self):
        roi = np.zeros((50, 50, 3), dtype=np.uint8)
        assert preprocess(roi, self._spec(scaling="mobilenet_v2")).min() == pytest.approx(-1.0)

    def test_rgb_invierte_canales_bgr(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        roi[:, :, 0] = 200  # azul en BGR
        out = preprocess(roi, self._spec(color_order="RGB", scaling="none"))
        assert out[0, 0, 2] == 200 and out[0, 0, 0] == 0


def _region(label, confidence):
    probs = {"SANO": 1 - confidence, "TIZON_TARDIO": confidence} if label == "TIZON_TARDIO"         else {"SANO": confidence, "TIZON_TARDIO": 1 - confidence}
    return Region(x=0, y=0, width=0.5, height=0.5, label=label, confidence=confidence,
                  probabilities=Probabilities(**probs))


class TestPickMainRegion:

    def test_sin_regiones_no_hay_resultado(self):
        assert pick_main_region([]) is None

    def test_gana_la_clase_mas_severa_aunque_tenga_menor_confianza(self):
        main = pick_main_region([_region("SANO", 0.99), _region("TIZON_TARDIO", 0.60)])
        assert main.label == "TIZON_TARDIO"

    def test_en_empate_de_clase_gana_la_mayor_confianza(self):
        main = pick_main_region([_region("SANO", 0.70), _region("SANO", 0.95)])
        assert main.confidence == 0.95


# ── Tests de instanciación ─────────────────────────────────────────────────

class TestPredictorInit:

    def test_predictor_inicia_no_listo(self):
        p = Predictor()
        assert p.is_ready is False
        assert p.meta is None

    def test_predict_sin_modelo_lanza_error(self):
        with pytest.raises(RuntimeError, match="Modelo no inicializado"):
            Predictor().predict(b"cualquier_bytes")

    def test_load_falla_si_el_sha256_no_coincide(self, tmp_path):
        model = tmp_path / "modelo.h5"
        model.write_bytes(b"no es el modelo esperado")
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps({
            "model_version": "x", "file": "modelo.h5", "sha256": "0" * 64,
            "classes": ["SANO", "TIZON_TARDIO"],
            "input_spec": {"width": 224, "height": 224, "color_order": "BGR", "scaling": "unit"},
        }))
        p = Predictor()
        with pytest.raises(ValueError, match="sha256"):
            p.load(str(model), str(meta))
        assert p.is_ready is False


# ── Tests de integración (requieren el modelo real) ────────────────────────

@pytest.mark.integration
class TestPredictorIntegration:
    """
    Requieren el .h5 real y su model_meta.json:
        MODEL_PATH=../modelo_tizon_tardio.h5 pytest test/ -m integration
    """

    @pytest.fixture(scope="class")
    def predictor_loaded(self):
        p = Predictor()
        p.load(os.environ.get("MODEL_PATH", "models/modelo_tizon_tardio.h5"),
               os.environ.get("MODEL_META_PATH", "models/model_meta.json"))
        return p

    def test_load_model(self, predictor_loaded):
        assert predictor_loaded.is_ready is True

    def test_predict_imagen_verde_cumple_el_contrato(self, predictor_loaded):
        result = predictor_loaded.predict(image_to_bytes(create_green_image()))
        assert result.status in ("DETECTED", "NO_VEGETATION")
        assert result.frame_width == 224 and result.frame_height == 224
        assert result.inference_ms >= 0
        for region in result.regions:
            total = region.probabilities.SANO + region.probabilities.TIZON_TARDIO
            assert abs(total - 1.0) < 0.01
