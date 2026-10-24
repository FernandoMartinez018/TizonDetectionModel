"""Filtro de vegetación (F4-IA-03, 08-modelo-ia §3).

La máscara de "hoja" parte del tejido vegetal (verde **y** el amarillo de la clorosis del tizón temprano conectado
al verde) y suma los tonos marrones u oliva de la lesión conectados a ese tejido (dentro de su envolvente convexa o
junto al tejido dilatado). Así una hoja con necrosis o amarillenta por tizón ya no se descarta como "sin vegetación",
y el amarillo o el marrón sueltos (suelo, madera, paja) siguen sin contar. Todos los umbrales están aquí.
"""
from __future__ import annotations

import cv2
import numpy as np

# ── Color (HSV de OpenCV: H 0–180, S y V 0–255) ────────────────────────────
GREEN_LOWER = np.array([35, 50, 50])     # verde natural de planta
GREEN_UPPER = np.array([75, 255, 200])   # excluye verdes artificiales muy brillantes
BROWN_LOWER = np.array([5, 40, 30])      # marrón u oliva de la lesión
BROWN_UPPER = np.array([30, 255, 200])
# Amarillo de la clorosis (tizón temprano). Medido en hojas de PlantVillage que el filtro anterior descartaba:
# H 18–34 cubre el 88 % de sus píxeles no verdes; S ≥ 74 y V ≥ 82 en el percentil 5.
YELLOW_LOWER = np.array([18, 70, 80])
YELLOW_UPPER = np.array([34, 255, 255])

# ── Geometría de las regiones ─────────────────────────────────────────────
MIN_CONTOUR_AREA = 3000      # px²
MIN_ROI_SIZE = 80            # lado mínimo en px
ASPECT_MIN = 0.3             # proporción w/h válida
ASPECT_MAX = 3.0
LESION_LINK_PX = 15          # el marrón cuenta si está a esta distancia del tejido
FULL_FRAME_RATIO = 0.60      # región ≥ 60 % del frame → se clasifica el frame completo

# ── Validación de la región ───────────────────────────────────────────────
MIN_TISSUE_RATIO = 0.15      # verde + amarillo; antes era 0,40 de verde y descartaba las hojas enfermas
TEXTURE_VAR_THRESHOLD = 200.0   # varianza de gris mínima (descarta objetos planos)
COLOR_VAR_THRESHOLD = 15.0      # variación de color mínima (descarta fondos uniformes)

_KERNEL = np.ones((5, 5), np.uint8)

Box = tuple[int, int, int, int]


def green_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)


def _linked(base: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Píxeles de `extra` conectados a `base`: dentro de su envolvente convexa o a LESION_LINK_PX de ella."""
    near = cv2.dilate(base, np.ones((2 * LESION_LINK_PX + 1,) * 2, np.uint8))
    contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA / 4:
            cv2.drawContours(near, [cv2.convexHull(cnt)], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(extra, near)


def tissue_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Tejido vegetal: verde ∪ amarillo de clorosis conectado al verde (el amarillo solo no basta)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.morphologyEx(cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER), cv2.MORPH_CLOSE, _KERNEL)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, _KERNEL)
    green = green_mask(frame_bgr)
    return cv2.bitwise_or(green, _linked(green, yellow))


def leaf_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Tejido ∪ (marrón conectado al tejido)."""
    tissue = tissue_mask(frame_bgr)
    brown = cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), BROWN_LOWER, BROWN_UPPER)
    leaf = cv2.bitwise_or(tissue, _linked(tissue, brown))
    return cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, _KERNEL)


def detect_regions(frame_bgr: np.ndarray) -> list[Box]:
    """Regiones candidatas (x, y, w, h). Una hoja en primer plano devuelve el frame completo."""
    frame_h, frame_w = frame_bgr.shape[:2]
    contours, _ = cv2.findContours(leaf_mask(frame_bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[Box] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if ASPECT_MIN <= w / max(h, 1) <= ASPECT_MAX:
            regions.append((x, y, w, h))

    if any(w * h >= FULL_FRAME_RATIO * frame_w * frame_h for (_, _, w, h) in regions):
        return [(0, 0, frame_w, frame_h)]
    return regions


def green_ratio(roi_bgr: np.ndarray) -> float:
    h, w = roi_bgr.shape[:2]
    return float(np.count_nonzero(green_mask(roi_bgr))) / (h * w)


def tissue_ratio(roi_bgr: np.ndarray) -> float:
    h, w = roi_bgr.shape[:2]
    return float(np.count_nonzero(tissue_mask(roi_bgr))) / (h * w)


def is_valid_leaf(roi_bgr: np.ndarray) -> bool:
    """Tamaño mínimo, algo de tejido verde o amarillo (≥ 15 %) y textura/color de hoja real (no plástico ni fondo)."""
    h, w = roi_bgr.shape[:2]
    if h < MIN_ROI_SIZE or w < MIN_ROI_SIZE:
        return False
    if tissue_ratio(roi_bgr) < MIN_TISSUE_RATIO:
        return False
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if float(np.var(gray)) < TEXTURE_VAR_THRESHOLD:
        return False
    return float(np.mean([np.std(roi_bgr[:, :, c]) for c in range(3)])) >= COLOR_VAR_THRESHOLD
