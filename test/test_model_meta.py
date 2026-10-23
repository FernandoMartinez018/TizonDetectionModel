from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.model_meta import load_meta, sha256_of

VALID = {
    "model_version": "v-test",
    "file": "modelo.h5",
    "classes": ["TIZON_TARDIO", "SANO"],
    "input_spec": {"width": 224, "height": 224, "color_order": "RGB", "scaling": "mobilenet_v2"},
}


def _write(tmp_path, data):
    path = tmp_path / "model_meta.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_carga_clases_en_el_orden_del_archivo(tmp_path):
    meta = load_meta(_write(tmp_path, VALID))
    assert meta.classes == ["TIZON_TARDIO", "SANO"]
    assert meta.input_spec.scaling == "mobilenet_v2"
    assert meta.min_confidence == 0.30


@pytest.mark.parametrize("classes", [["SANO", "SANO"], ["SANO", "TIZON_TEMPRANO"], ["SANO"]])
def test_rechaza_clases_fuera_del_alcance(tmp_path, classes):
    with pytest.raises(ValidationError):
        load_meta(_write(tmp_path, {**VALID, "classes": classes}))


def test_lee_la_ruta_de_model_meta_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_META_PATH", _write(tmp_path, VALID))
    assert load_meta().model_version == "v-test"


def test_el_meta_del_repo_es_valido():
    meta = load_meta("models/model_meta.json")
    assert sorted(meta.classes) == ["SANO", "TIZON_TARDIO"]


def test_sha256(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
