from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import boto3
import cv2
import numpy as np
import tensorflow as tf

from app.schemas import AnalyzeResponse, BoundingBox

logger = logging.getLogger(__name__)


LABELS = ["Sano", "Tizon Tardio"]
TARGET_SIZE = (224, 224)          # Tamaño de entrada del modelo
MIN_CONTOUR_AREA = 3000           # Área mínima (px²) para considerar una región
MIN_ROI_SIZE = 80                 # Lado mínimo (px) de una ROI
GREEN_RATIO_THRESHOLD = 0.40      # Al menos 40% de píxeles verdes naturales
TEXTURE_VAR_THRESHOLD = 200.0     # Varianza mínima de textura para descartar objetos planos
COLOR_VAR_THRESHOLD = 15.0        # Variación mínima de color para descartar fondos uniformes
ASPECT_MIN = 0.3                  # Proporción mínima w/h de un contorno válido
ASPECT_MAX = 3.0                  # Proporción máxima w/h de un contorno válido
MODEL_MIN_CONFIDENCE = 0.30       # Confianza mínima para considerar una detección válida

# Rango HSV para verde natural de plantas
# Evita verdes artificiales muy brillantes (objetos plásticos, ropa verde, etc.)
PLANT_GREEN_LOWER = np.array([35, 50, 50])
PLANT_GREEN_UPPER = np.array([75, 255, 200])

# Ruta local del modelo cuando se descarga desde S3 en Lambda
# /tmp es el único directorio escribible en Lambda
LAMBDA_TMP_MODEL = "/tmp/modelo_tizon_tardio.h5"


class Predictor:

    def __init__(self) -> None:
        self._model: Optional[tf.keras.Model] = None
        self.is_ready: bool = False
        self._model_path_used: str = ""
        self._environment: str = os.getenv("APP_ENV", "local")

    def load_model(self) -> None:
        try:
            if self._environment == "aws":
                model_path = self._download_from_s3()
            else:
                model_path = self._resolve_local_path()

            logger.info("Cargando modelo desde: %s", model_path)

            self._model = tf.keras.models.load_model(str(model_path))

            dummy_input = np.zeros((1, *TARGET_SIZE, 3), dtype=np.float32)
            self._model.predict(dummy_input, verbose=0)

            self._model_path_used = str(model_path)
            self.is_ready = True
            logger.info("Modelo cargado OK")

        except Exception:
            logger.exception("🔥 ERROR REAL cargando modelo")
            self.is_ready = False
            raise



    def _resolve_local_path(self) -> Path:
        local_path = os.getenv("LOCAL_MODEL_PATH", "model/modelo_tizon_tardio.h5")
        path = Path(local_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en '{path.resolve()}'.\n"
                f"Coloca tu archivo modelo_tizon_tardio.h5 en la carpeta 'model/'.\n"
                f"También puedes configurar LOCAL_MODEL_PATH en tu .env."
            )
        return path

    def _download_from_s3(self) -> Path:
        
        tmp_path = Path(LAMBDA_TMP_MODEL)

        if tmp_path.exists():
            logger.info("Modelo ya en /tmp/ (Lambda warm). Omitiendo descarga.")
            return tmp_path

        bucket = os.environ["MODEL_S3_BUCKET"]
        key = os.environ["MODEL_S3_KEY"]

        logger.info("Descargando modelo desde s3://%s/%s ...", bucket, key)
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(tmp_path))
        logger.info("Modelo descargado correctamente en %s", tmp_path)

        return tmp_path

    @property
    def model_path(self) -> str:
        return self._model_path_used

    @property
    def environment(self) -> str:
        return self._environment


    def predict(self, image_bytes: bytes) -> AnalyzeResponse:
        
        if not self.is_ready:
            raise RuntimeError("Modelo no inicializado. Llama a load_model() primero.")

        # 1. Decodificar imagen
        frame = self._decode_image(image_bytes)
        frame_h, frame_w = frame.shape[:2]

        # 2. Detectar regiones de vegetación
        regions = self._detect_vegetation_regions(frame)

        bounding_boxes: list[BoundingBox] = []
        max_area = 0
        best_pred: Optional[np.ndarray] = None

        # 3. Clasificar cada región
        for (x, y, w, h) in regions:
            roi = frame[y : y + h, x : x + w]

            # Filtrar si no es vegetación real
            if not self._is_valid_vegetation(roi):
                continue

            # Clasificar la ROI
            pred = self._classify_roi(roi)
            class_id = int(np.argmax(pred))
            confidence = float(pred[class_id])

            # Ignorar detecciones con confianza muy baja
            if confidence < MODEL_MIN_CONFIDENCE:
                continue

            area = w * h

            # Construir bounding box con coordenadas relativas
            bbox = BoundingBox(
                x=round(x / frame_w, 4),
                y=round(y / frame_h, 4),
                width=round(w / frame_w, 4),
                height=round(h / frame_h, 4),
                label=LABELS[class_id],
                confidence=round(confidence, 4),
                probabilities={
                    LABELS[i]: round(float(pred[i]), 4) for i in range(len(LABELS))
                },
            )
            bounding_boxes.append(bbox)

            # Guardar la detección de mayor área como resultado principal
            if area > max_area:
                max_area = area
                best_pred = pred

        # 4. Construir respuesta
        if best_pred is not None:
            main_class_id = int(np.argmax(best_pred))
            return AnalyzeResponse(
                label=LABELS[main_class_id],
                confidence=round(float(best_pred[main_class_id]), 4),
                probabilities={
                    LABELS[i]: round(float(best_pred[i]), 4) for i in range(len(LABELS))
                },
                bounding_boxes=bounding_boxes,
                is_vegetation=True,
                frame_width=frame_w,
                frame_height=frame_h,
            )

        # Sin detecciones válidas
        return AnalyzeResponse(
            label="Sin deteccion",
            confidence=0.0,
            probabilities={label: 0.0 for label in LABELS},
            bounding_boxes=[],
            is_vegetation=False,
            frame_width=frame_w,
            frame_height=frame_h,
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

    def _classify_roi(self, roi: np.ndarray) -> np.ndarray:
        
        img = cv2.resize(roi, TARGET_SIZE)

        img = img.astype(np.float32) / 255.0

        img = np.expand_dims(img, axis=0)

        predictions: np.ndarray = self._model.predict(img, verbose=0)[0]

        return predictions