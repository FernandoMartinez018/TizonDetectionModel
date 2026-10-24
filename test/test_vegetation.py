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


def test_suelo_marron_sin_verde_no_es_hoja():
    soil = _textured(np.full((480, 640, 3), (30, 70, 110), np.uint8))
    assert vegetation.detect_regions(soil) == []


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
