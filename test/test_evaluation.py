"""F4-IA-02 / F4-IA-04: calibración, métricas, pHash y auditoría con imágenes sintéticas (sin el .h5)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.model_meta import ModelMeta, sha256_of
from evaluation import audit, calibrate, common, evaluate

GREEN, RED = (0, 255, 0), (0, 0, 255)   # BGR: hoja sana y "lesión"


def write_set(root: Path, n: int = 3) -> Path:
    for label, color in (("SANO", GREEN), ("TIZON_TARDIO", RED)):
        (root / label).mkdir(parents=True)
        for i in range(n):
            img = np.full((60, 80, 3), color, np.uint8)
            img[i, i] = (i, i, i)   # archivos distintos (sha256)
            (root / label / f"{i}.png").write_bytes(cv2.imencode(".png", img)[1].tobytes())
    return root


class FakeModel:
    """Solo entiende RGB + mobilenet_v2 con salida [TIZON_TARDIO, SANO]: canal 0 alto y valores negativos."""

    def __call__(self, inputs, training=False):
        x = inputs[0]
        tizon = (x[..., 0].mean(axis=(1, 2)) > 0.5) & (x.min(axis=(1, 2, 3)) < 0)
        return np.stack([tizon, ~tizon], axis=1).astype(np.float32)


@pytest.fixture
def meta_file(tmp_path) -> Path:
    path = tmp_path / "model_meta.json"
    shutil.copy(Path(__file__).parent.parent / "models" / "model_meta.json", path)
    return path


@pytest.fixture
def fake_model_file(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(calibrate, "load_keras_model", lambda _: FakeModel())
    monkeypatch.setattr(evaluate, "load_keras_model", lambda _: FakeModel())
    path = tmp_path / "modelo.h5"
    path.write_bytes(b"modelo de prueba")
    return path


class TestMetricas:
    def test_positivo_es_tizon(self):
        y_true = ["TIZON_TARDIO"] * 4 + ["SANO"] * 6
        y_pred = ["TIZON_TARDIO"] * 3 + ["SANO"] * 6 + ["TIZON_TARDIO"]
        m = common.classification_metrics(y_true, y_pred)
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
        assert m["accuracy"] == pytest.approx(0.8)
        assert m["precision"] == pytest.approx(0.75)
        assert m["recall"] == pytest.approx(0.75)
        assert m["f1"] == pytest.approx(0.75)
        assert m["specificity"] == pytest.approx(5 / 6)
        assert m["error_rate"] == pytest.approx(0.2)

    def test_sin_predicciones_positivas_no_divide_por_cero(self):
        m = common.classification_metrics(["TIZON_TARDIO", "SANO"], ["SANO", "SANO"])
        assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0

    def test_cuatro_clases_accuracy_global_y_tardio_contra_el_resto(self):
        y_true = ["SANO", "TIZON_TEMPRANO", "TIZON_TEMPRANO", "TIZON_TARDIO", "TIZON_TARDIO", "NO_PAPA"]
        y_pred = ["SANO", "TIZON_TEMPRANO", "TIZON_TARDIO", "TIZON_TARDIO", "TIZON_TARDIO", "SANO"]
        m = common.classification_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(4 / 6)              # global, no binaria
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 0, 1, 3)   # TIZON_TARDIO contra el resto
        assert m["precision"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(3 / 4)
        assert m["confusion"]["NO_PAPA"]["SANO"] == 1
        assert m["per_class"]["TIZON_TEMPRANO"]["recall"] == pytest.approx(0.5)

    def test_formato_del_documento(self):
        assert common.num(1500, 0) == "1.500"
        assert common.num(1234.56) == "1.234,6"
        assert common.pct(0.9) == "90,0 %"

    def test_metas_del_pdf(self):
        rows = {r["metric"]: r["ok"] for r in common.check_targets(
            {"accuracy": 0.9, "precision": 0.95, "recall": 0.89, "f1": 0.92, "specificity": 0.85, "error_rate": 0.1})}
        assert rows == {"accuracy": True, "precision": True, "recall": False, "f1": True,
                        "specificity": True, "error_rate": True}


class TestCalibracion:
    def test_elige_la_unica_combinacion_correcta(self, tmp_path):
        samples = common.load_dataset(write_set(tmp_path / "cal"))
        images = [common.read_bgr(p) for p, _ in samples]
        results = calibrate.score_combinations(FakeModel(), images, [label for _, label in samples], 224, 224)
        best = results[0]
        assert (best["color_order"], best["scaling"], best["classes"]) == ("RGB", "mobilenet_v2", ["TIZON_TARDIO", "SANO"])
        assert best["accuracy"] == 1.0 and results[1]["accuracy"] == 0.5
        assert len(results) == 8

    def test_write_guarda_la_combinacion_y_el_meta_es_valido(self, tmp_path, meta_file, fake_model_file, monkeypatch):
        monkeypatch.setattr(calibrate, "MIN_PER_CLASS", 3)
        code = calibrate.main(["--data", str(write_set(tmp_path / "cal")), "--model", str(fake_model_file),
                               "--meta", str(meta_file), "--reports", str(tmp_path / "rep"), "--write"])
        assert code == 0
        meta = ModelMeta.model_validate_json(meta_file.read_text(encoding="utf-8"))
        assert meta.classes == ["TIZON_TARDIO", "SANO"]
        assert (meta.input_spec.color_order, meta.input_spec.scaling) == ("RGB", "mobilenet_v2")
        assert meta.sha256 == sha256_of(fake_model_file)
        assert list((tmp_path / "rep").glob("*_calibracion.md"))

    def test_no_escribe_con_pocas_imagenes(self, tmp_path, meta_file, fake_model_file):
        before = meta_file.read_text(encoding="utf-8")
        code = calibrate.main(["--data", str(write_set(tmp_path / "cal")), "--model", str(fake_model_file),
                               "--meta", str(meta_file), "--reports", str(tmp_path / "rep"), "--write"])
        assert code == 1   # 3 por clase < 30
        assert meta_file.read_text(encoding="utf-8") == before

    def test_con_mas_clases_prueba_todas_las_permutaciones(self, tmp_path):
        class ThreeOutputs:
            def __call__(self, inputs, training=False):
                return np.tile([0.2, 0.3, 0.5], (len(inputs[0]), 1))

        root = write_set(tmp_path / "cal")
        (root / "TIZON_TEMPRANO").mkdir()
        shutil.copy(root / "SANO" / "0.png", root / "TIZON_TEMPRANO" / "0.png")
        samples = common.load_dataset(root)
        images = [common.read_bgr(p) for p, _ in samples]
        assert len(calibrate.score_combinations(ThreeOutputs(), images, [lab for _, lab in samples], 224, 224)) == 24
        two = [(img, lab) for img, (_, lab) in zip(images, samples) if lab != "TIZON_TEMPRANO"]
        with pytest.raises(SystemExit, match="3 salidas y el set 2 clases"):
            calibrate.score_combinations(ThreeOutputs(), [i for i, _ in two], [lab for _, lab in two], 224, 224)

    def test_margen_menor_a_5_puntos_pide_ampliar_el_set(self):
        results = [{"accuracy": 0.80}, {"accuracy": 0.77}]
        problems = calibrate.problems_of({"SANO": 30, "TIZON_TARDIO": 30}, results)
        assert len(problems) == 1 and "ampliar" in problems[0]


class TestEvaluacion:
    def test_no_vegetation_cuenta_como_sano_y_se_reporta_aparte(self):
        samples = [(Path("t1"), "TIZON_TARDIO"), (Path("t2"), "TIZON_TARDIO"), (Path("s1"), "SANO")]
        answers = {"t1": "TIZON_TARDIO", "t2": "NO_VEGETATION", "s1": "NO_VEGETATION"}
        report = evaluate.evaluate(samples, lambda p: (answers[p.name], 10.0))
        m = report["metrics"]
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 0, 1)
        assert report["no_vegetation"] == {"SANO": 1, "TIZON_TARDIO": 1}
        assert report["no_vegetation_tizon_ratio"] == 0.5
        assert report["errors"] == [{"file": "t2", "true": "TIZON_TARDIO", "pred": "NO_VEGETATION"}]

    def test_hojas_que_no_son_de_papa(self):
        samples = [(Path("p1"), "NO_PAPA"), (Path("p2"), "NO_PAPA"), (Path("p3"), "NO_PAPA"), (Path("s1"), "SANO"),
                   (Path("e1"), "TIZON_TEMPRANO")]
        answers = {"p1": "NOT_POTATO", "p2": "NO_VEGETATION", "p3": "TIZON_TARDIO", "s1": "NOT_POTATO",
                   "e1": "TIZON_TEMPRANO"}
        report = evaluate.evaluate(samples, lambda p: (answers[p.name], 10.0))
        c = report["metrics"]["confusion"]
        assert c["NO_PAPA"]["NO_PAPA"] == 2      # NOT_POTATO y NO_VEGETATION: no se analiza, como se espera
        assert c["NO_PAPA"]["TIZON_TARDIO"] == 1 and c["SANO"]["NO_PAPA"] == 1
        assert report["potato_discrimination"]["potato_rejected"] == pytest.approx(1 / 2)
        assert report["potato_discrimination"]["not_potato_accepted"] == pytest.approx(1 / 3)

    def test_modo_classifier_genera_el_reporte(self, tmp_path, meta_file, fake_model_file):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta.update(classes=["TIZON_TARDIO", "SANO"], input_spec={**meta["input_spec"], "color_order": "RGB",
                                                                  "scaling": "mobilenet_v2"})
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
        data = write_set(tmp_path / "prueba")
        code = evaluate.main(["--data", str(data), "--mode", "classifier", "--model", str(fake_model_file),
                              "--meta", str(meta_file), "--train", str(data), "--reports", str(tmp_path / "rep")])
        assert code == 0
        md = next((tmp_path / "rep").glob("*_classifier.md")).read_text(encoding="utf-8")
        assert "| Accuracy | ≥ 90,0 % | 100,0 % | ✔ |" in md
        assert "Duplicados exactos con el entrenamiento: 6." in md   # el mismo set como entrenamiento


class TestPhash:
    def test_distingue_casi_duplicados(self):
        rng = np.random.default_rng(0)

        def pattern():
            return cv2.resize(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), (128, 128),
                              interpolation=cv2.INTER_CUBIC)

        img, other = pattern(), pattern()
        brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=10)
        assert common.hamming(common.phash(img), common.phash(brighter)) <= common.PHASH_MAX_DISTANCE
        assert common.hamming(common.phash(img), common.phash(other)) > common.PHASH_MAX_DISTANCE


class TestAuditoria:
    def test_duplicados_conflictos_y_muestra(self, tmp_path):
        root = write_set(tmp_path / "train", n=3)
        shutil.copy(root / "SANO" / "0.png", root / "SANO" / "copia.png")
        shutil.copy(root / "SANO" / "1.png", root / "TIZON_TARDIO" / "mal_etiquetada.png")
        report = audit.audit(common.load_dataset(root), sample_size=2, seed=1)
        assert report["counts"] == {"SANO": 4, "TIZON_TARDIO": 4}
        assert len(report["exact_duplicates"]) == 2
        assert len(report["label_conflicts"]) == 1
        assert {c: len(v) for c, v in report["review_sample"].items()} == {"SANO": 2, "TIZON_TARDIO": 2}
        assert report["resolution"]["SANO"] == {"width": [80, 80, 80], "height": [60, 60, 60]}
