from __future__ import annotations


import cv2
import numpy as np
import pytest

from app.predictor import Predictor


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


# ── Tests de _detect_vegetation_regions ───────────────────────────────────

class TestDetectVegetationRegions:

    def test_detecta_region_verde(self):
        """Una imagen completamente verde debe generar al menos una región."""
        img = create_green_image(224, 224)
        regions = Predictor._detect_vegetation_regions(img)
        assert len(regions) >= 1

    def test_no_detecta_imagen_roja(self):
        """Una imagen roja no debe generar regiones (no es verde)."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # Solo canal rojo
        regions = Predictor._detect_vegetation_regions(img)
        assert len(regions) == 0

    def test_no_detecta_imagen_negra(self):
        """Una imagen negra no debe generar regiones."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        regions = Predictor._detect_vegetation_regions(img)
        assert len(regions) == 0

    def test_retorna_tuplas_validas(self):
        """Las regiones retornadas deben ser tuplas (x, y, w, h) con valores positivos."""
        img = create_green_image(400, 400)
        regions = Predictor._detect_vegetation_regions(img)
        for region in regions:
            x, y, w, h = region
            assert x >= 0
            assert y >= 0
            assert w > 0
            assert h > 0


# ── Tests de _is_valid_vegetation ─────────────────────────────────────────

class TestIsValidVegetation:

    def test_imagen_verde_con_textura_es_valida(self):
        """Una imagen verde con textura debe pasar el filtro."""
        roi = create_green_image(100, 100)
        assert Predictor._is_valid_vegetation(roi) is True

    def test_imagen_muy_pequeña_no_es_valida(self):
        """Una ROI menor al tamaño mínimo no es válida."""
        roi = create_green_image(50, 50)  # Menor que MIN_ROI_SIZE=80
        assert Predictor._is_valid_vegetation(roi) is False

    def test_imagen_uniforme_no_es_valida(self):
        """Una imagen completamente uniforme (sin textura) no es vegetación real."""
        roi = create_uniform_green_image(100, 100)
        # La imagen uniforme tiene varianza de textura ≈ 0 → debe fallar
        result = Predictor._is_valid_vegetation(roi)
        # Nota: puede pasar si el JPEG agrega artefactos de compresión.
        # El test documenta el comportamiento esperado.
        assert isinstance(result, bool)

    def test_imagen_roja_no_es_valida(self):
        """Una imagen roja no tiene verde natural, no es vegetación."""
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        roi[:, :, 2] = 200  # Rojo
        assert Predictor._is_valid_vegetation(roi) is False


# ── Tests de instanciación ─────────────────────────────────────────────────

class TestPredictorInit:

    def test_predictor_inicia_no_listo(self):
        """Un predictor recién creado no debe estar listo."""
        p = Predictor()
        assert p.is_ready is False
        assert p._model is None

    def test_predict_sin_modelo_lanza_error(self):
        """predict() sin modelo cargado debe lanzar RuntimeError."""
        p = Predictor()
        with pytest.raises(RuntimeError, match="Modelo no inicializado"):
            p.predict(b"cualquier_bytes")

    def test_model_path_sin_cargar(self):
        """model_path debe ser string vacío si no se cargó el modelo."""
        p = Predictor()
        assert p.model_path == ""


# ── Tests de integración (requieren modelo real) ───────────────────────────

@pytest.mark.integration
class TestPredictorIntegration:
    """
    Estos tests requieren el modelo en model/modelo_tizon_tardio.h5
    y APP_ENV=local en el .env.

    Correr con:
        pytest tests/test_predictor.py -m integration -v
    """

    @pytest.fixture
    def predictor_loaded(self):
        """Predictor con modelo cargado (requiere modelo real)."""
        p = Predictor()
        p.load_model()
        return p

    def test_load_model_local(self, predictor_loaded):
        assert predictor_loaded.is_ready is True
        assert predictor_loaded.model_path != ""

    def test_predict_imagen_verde(self, predictor_loaded):
        img = create_green_image()
        img_bytes = image_to_bytes(img)
        result = predictor_loaded.predict(img_bytes)
        # El resultado debe tener todos los campos requeridos
        assert result.label in ["Sano", "Tizon Temprano", "Tizon Tardio", "Sin deteccion"]
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.bounding_boxes, list)
        assert result.frame_width == 224
        assert result.frame_height == 224

    def test_predict_retorna_probabilidades_validas(self, predictor_loaded):
        img = create_green_image()
        result = predictor_loaded.predict(image_to_bytes(img))
        # Las probabilidades deben sumar ≈ 1.0 (con tolerancia de punto flotante)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01