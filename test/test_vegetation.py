"""F4-IA-03: el filtro acepta hojas con necrosis y sigue rechazando lo que no es hoja."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app import vegetation

RNG = np.random.default_rng(42)


def _textured(img: np.ndarray, amplitude: int = 60) -> np.ndarray:
    noise = RNG.integers(0, amplitude, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def leaf_on_background(size=(480, 640), lesion_fraction=0.0, background=(200, 200, 200)) -> np.ndarray:
    """Hoja elíptica verde con textura sobre fondo claro; `lesion_fraction` de su área en marrón."""
    h, w = size
    img = np.full((h, w, 3), background, np.uint8)
    leaf = np.zeros((h, w), np.uint8)
    cv2.ellipse(leaf, (w // 2, h // 2), (w // 5, h // 4), 20, 0, 360, 255, cv2.FILLED)
    img[leaf > 0] = (40, 130, 50)                              # verde (BGR)
    if lesion_fraction:
        ys, xs = np.nonzero(leaf)
        target = int(len(xs) * lesion_fraction)
        painted = 0
        while painted < target:                                 # manchas marrones dentro de la hoja
            i = RNG.integers(len(xs))
            r = int(RNG.integers(10, 25))
            spot = np.zeros_like(leaf)
            cv2.circle(spot, (int(xs[i]), int(ys[i])), r, 255, cv2.FILLED)
            spot &= leaf
            new = (spot > 0) & ~np.all(img == (30, 70, 110), axis=2)
            img[spot > 0] = (30, 70, 110)                       # marrón (BGR)
            painted += int(np.count_nonzero(new))
    return _textured(img)


def _roi(img, box):
    x, y, w, h = box
    return img[y : y + h, x : x + w]


def test_hoja_sana_se_detecta_y_es_valida():
    img = leaf_on_background()
    regions = vegetation.detect_regions(img)
    assert len(regions) == 1
    assert vegetation.is_valid_leaf(_roi(img, regions[0]))


def test_hoja_con_necrosis_se_detecta_completa_y_pasa_el_filtro():
    img = leaf_on_background(lesion_fraction=0.6)
    regions = vegetation.detect_regions(img)
    assert len(regions) == 1, "la lesión no debe partir la hoja en pedazos"
    roi = _roi(img, regions[0])
    ratio = vegetation.green_ratio(roi)
    assert ratio < 0.40, "con el umbral anterior (40 %) esta hoja se descartaba"
    assert vegetation.is_valid_leaf(roi)


def test_la_region_incluye_la_lesion_conectada():
    healthy = vegetation.detect_regions(leaf_on_background())[0]
    sick = vegetation.detect_regions(leaf_on_background(lesion_fraction=0.6))[0]
    # La caja de la hoja enferma cubre casi la misma área que la sana (la lesión cuenta como hoja).
    assert sick[2] * sick[3] >= 0.8 * healthy[2] * healthy[3]


@pytest.mark.parametrize("size", [(480, 640), (960, 1280)])
def test_suelo_marron_sin_verde_no_es_hoja(size):
    # El ruido puede dejar regiones candidatas, pero ninguna pasa la validación: el suelo no se analiza.
    soil = _textured(np.full((*size, 3), (30, 70, 110), np.uint8))
    assert not any(vegetation.is_valid_leaf(_roi(soil, box)) for box in vegetation.detect_regions(soil))


def test_hoja_amarillenta_por_clorosis_se_acepta():
    """Tizón temprano avanzado: la hoja está amarilla y solo las nervaduras siguen verdes."""
    h, w = 480, 640
    img = np.full((h, w, 3), (200, 200, 200), np.uint8)
    leaf = np.zeros((h, w), np.uint8)
    cv2.ellipse(leaf, (w // 2, h // 2), (w // 5, h // 4), 20, 0, 360, 255, cv2.FILLED)
    veins = np.zeros_like(leaf)
    axis, normal = np.array([0.94, 0.34]), np.array([-0.34, 0.94])   # eje mayor de la elipse (20°) y su normal
    center = np.array([w // 2, h // 2])
    cv2.line(veins, tuple((center - 120 * axis).astype(int)), tuple((center + 120 * axis).astype(int)), 255, 6)
    for t in (-80, -40, 0, 40, 80):
        p = center + t * axis
        cv2.line(veins, tuple((p - 110 * normal).astype(int)), tuple((p + 110 * normal).astype(int)), 255, 6)
    img[leaf > 0] = (40, 190, 210)                     # amarillo (BGR)
    img[(veins > 0) & (leaf > 0)] = (40, 130, 50)      # nervaduras verdes
    img = _textured(img)

    regions = vegetation.detect_regions(img)
    assert len(regions) == 1
    roi = _roi(img, regions[0])
    assert vegetation.green_ratio(roi) < vegetation.MIN_TISSUE_RATIO, "con el filtro anterior se descartaba"
    assert vegetation.is_valid_leaf(roi)


@pytest.mark.parametrize("color", [(0, 0, 200), (0, 0, 0), (255, 255, 255)])
def test_rojo_negro_o_blanco_no_tienen_regiones(color):
    assert vegetation.detect_regions(np.full((240, 320, 3), color, np.uint8)) == []


def test_verde_uniforme_artificial_se_rechaza_por_textura():
    plastic = np.full((200, 200, 3), (40, 130, 50), np.uint8)
    assert not vegetation.is_valid_leaf(plastic)


def test_hoja_en_primer_plano_usa_el_frame_completo():
    img = _textured(np.full((300, 400, 3), (40, 130, 50), np.uint8))
    assert vegetation.detect_regions(img) == [(0, 0, 400, 300)]


def test_roi_pequena_no_es_valida():
    assert not vegetation.is_valid_leaf(leaf_on_background(size=(60, 60)))
