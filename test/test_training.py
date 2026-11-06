"""F4-IA-05: al agregar datos, la partición anterior se conserva (los modelos se comparan con la misma prueba)."""
from __future__ import annotations

import cv2
import numpy as np

from evaluation.common import CLASSES
from training.train import assign_splits


def _write(root, label, name, rng):
    img = cv2.resize(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
    path = root / label / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())
    return path, label


def test_los_anteriores_conservan_su_particion_y_los_nuevos_no_van_a_prueba(tmp_path):
    rng = np.random.default_rng(0)
    old = [_write(tmp_path, label, i, rng) for label in CLASSES for i in range(10)]
    first = assign_splits(old, seed=1)
    existing = {f"{label}/{p.name}": first[p] for p, label in old}
    new = [_write(tmp_path, label, f"n{i}", rng) for label in CLASSES for i in range(6)]

    second = assign_splits(old + new, seed=7, existing=existing, new_shares=(("train", 0.85), ("val", 1.0)))

    assert "test" in set(first.values())
    assert all(second[p] == first[p] for p, _ in old)
    assert {second[p] for p, _ in new} <= {"train", "val"}


def test_un_duplicado_nuevo_hereda_la_particion_de_su_original(tmp_path):
    rng = np.random.default_rng(1)
    old = [_write(tmp_path, "SANO", i, rng) for i in range(5)]
    existing = {f"SANO/{p.name}": "test" for p, _ in old}
    copy = tmp_path / "SANO" / "copia.png"
    copy.write_bytes(old[0][0].read_bytes())

    split = assign_splits(old + [(copy, "SANO")], seed=3, existing=existing, new_shares=(("train", 1.0), ("val", 1.0)))

    assert split[copy] == "test"
