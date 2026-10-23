from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from app.model_meta import InputSpec, ModelMeta, load_meta, sha256_of
from app.schemas import IaAnalyzeResponse, Probabilities, Region

logger = logging.getLogger(__name__)

# Resultado principal (contrato, doc 03): gana la clase más severa; en empate, la de mayor confianza.
SEVERITY = {"SANO": 0, "TIZON_TARDIO": 1}

# ── Filtro de vegetación (se rediseña en F4-IA-03) ─────────────────────────
MIN_CONTOUR_AREA = 3000           # Área mínima (px²) para considerar una región
MIN_ROI_SIZE = 80                 # Lado mínimo (px) de una ROI
GREEN_RATIO_THRESHOLD = 0.40      # Al menos 40% de píxeles verdes naturales
TEXTURE_VAR_THRESHOLD = 200.0     # Varianza mínima de textura para descartar objetos planos
COLOR_VAR_THRESHOLD = 15.0        # Variación mínima de color para descartar fondos uniformes
ASPECT_MIN = 0.3                  # Proporción mínima w/h de un contorno válido
ASPECT_MAX = 3.0                  # Proporción máxima w/h de un contorno válido

# Rango HSV para verde natural de plantas
# Evita verdes artificiales muy brillantes (objetos plásticos, ropa verde, etc.)
PLANT_GREEN_LOWER = np.array([35, 50, 50])
PLANT_GREEN_UPPER = np.array([75, 255, 200])


class _DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
    """El .h5 se guardó con Keras 2.14, que serializa `groups`; Keras 3 (TF 2.16) no lo acepta.

    En DepthwiseConv2D `groups` siempre vale 1, así que se descarta. Verificado contra tf_keras:
    diferencia máxima de salida < 1e-5 (F4-IA-01).
    """

    def __init__(self, *args, groups: int | None = None, **kwargs):
        if groups not in (None, 1):
            raise ValueError(f"DepthwiseConv2D con groups={groups} no es compatible")
        super().__init__(*args, **kwargs)


def preprocess(roi_bgr: np.ndarray, spec: InputSpec) -> np.ndarray:
    """Aplica el preprocesamiento declarado en model_meta.json (orden de color y escala)."""
    img = cv2.resize(roi_bgr, (spec.width, spec.height))
    if spec.color_order == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    if spec.scaling == "unit":
        img /= 255.0
    elif spec.scaling == "mobilenet_v2":
        img = img / 127.5 - 1.0
    return img


def pick_main_region(regions: list[Region]) -> Region | None:
    return max(regions, key=lambda r: (SEVERITY[r.label], r.confidence), default=None)


class Predictor:

    def __init__(self) -> None:
        self._model: tf.keras.Model | None = None
        self.meta: ModelMeta | None = None
        self.is_ready: bool = False

    def load(self, model_path: str, meta_path: str | None = None) -> None:
        """Carga metadatos y modelo. Si algo no cuadra, lanza excepción y el servicio queda degradado."""
        self.is_ready = False
        self.meta = load_meta(meta_path)
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en '{path.resolve()}' (MODEL_PATH).")
        if self.meta.sha256 and sha256_of(path) != self.meta.sha256:
            raise ValueError("El sha256 del modelo no coincide con model_meta.json.")

        logger.info("Cargando modelo %s desde %s", self.meta.model_version, path)
        self._model = tf.keras.models.load_model(
            str(path), custom_objects={"DepthwiseConv2D": _DepthwiseConv2DCompat}, compile=False
        )

        spec = self.meta.input_spec
        # El .h5 declara la entrada como lista [input]; pasarla así evita la advertencia de Keras 3.
        self._model([np.zeros((1, spec.height, spec.width, 3), dtype=np.float32)], training=False)
        self.is_ready = True
        logger.info("Modelo cargado. Clases: %s, entrada: %s", self.meta.classes, spec)

    def predict(self, image_bytes: bytes) -> IaAnalyzeResponse:
        if not self.is_ready or self._model is None or self.meta is None:
            raise RuntimeError("Modelo no inicializado. Llama a load() primero.")

        start = time.perf_counter()
        frame = self._decode_image(image_bytes)
        frame_h, frame_w = frame.shape[:2]

        boxes = [
            (x, y, w, h)
            for (x, y, w, h) in self._detect_vegetation_regions(frame)
            if self._is_valid_vegetation(frame[y : y + h, x : x + w])
        ]

        regions: list[Region] = []
        if boxes:
            spec = self.meta.input_spec
            batch = np.stack([preprocess(frame[y : y + h, x : x + w], spec) for (x, y, w, h) in boxes])
            preds = self._model([batch], training=False).numpy()
            for (x, y, w, h), pred in zip(boxes, preds):
                class_id = int(np.argmax(pred))
                confidence = float(pred[class_id])
                if confidence < self.meta.min_confidence:
                    continue
                probs = {c: round(float(pred[i]), 4) for i, c in enumerate(self.meta.classes)}
                regions.append(Region(
                    x=round(x / frame_w, 4),
                    y=round(y / frame_h, 4),
                    width=round(w / frame_w, 4),
                    height=round(h / frame_h, 4),
                    label=self.meta.classes[class_id],
                    confidence=round(confidence, 4),
                    probabilities=Probabilities(**probs),
                ))

        main = pick_main_region(regions)
        inference_ms = int((time.perf_counter() - start) * 1000)
        common = dict(
            frame_width=frame_w,
            frame_height=frame_h,
            model_version=self.meta.model_version,
            inference_ms=inference_ms,
        )
        if main is None:
            # Sin hoja analizable: resultado válido, no es un error (contrato, regla 3).
            return IaAnalyzeResponse(status="NO_VEGETATION", regions=[], **common)
        return IaAnalyzeResponse(
            status="DETECTED",
            label=main.label,
            confidence=main.confidence,
            probabilities=main.probabilities,
            regions=regions,
            **common,
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(
                "No se pudo decodificar la imagen. "
                "Verifica que el Base64 corresponda a un JPEG o PNG válido."
            )
        return frame

    @staticmethod
    def _detect_vegetation_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, PLANT_GREEN_LOWER, PLANT_GREEN_UPPER)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue  

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)

            if ASPECT_MIN <= aspect_ratio <= ASPECT_MAX:
                regions.append((x, y, w, h))

        return regions

    @staticmethod
    def _is_valid_vegetation(roi: np.ndarray) -> bool:
        
        h, w = roi.shape[:2]

        if h < MIN_ROI_SIZE or w < MIN_ROI_SIZE:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, PLANT_GREEN_LOWER, PLANT_GREEN_UPPER)
        green_ratio = float(np.sum(green_mask > 0)) / (h * w)
        if green_ratio < GREEN_RATIO_THRESHOLD:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_variance = float(np.var(gray))
        if texture_variance < TEXTURE_VAR_THRESHOLD:
            return False

        color_variation = float(
            np.mean([np.std(roi[:, :, c]) for c in range(3)])
        )
        if color_variation < COLOR_VAR_THRESHOLD:
            return False

        return True
