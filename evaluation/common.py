"""Utilidades compartidas por calibrate, evaluate y audit: dataset, métricas, pHash y reportes."""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import platform
from pathlib import Path

import cv2
import numpy as np

from app.model_meta import NOT_POTATO, InputSpec
from app.predictor import preprocess

# Carpetas <dir>/<CLASE>: las de papa en orden de severidad y NO_PAPA (hojas de otras plantas, que la IA rechaza sin
# analizar el tizón). Solo TIZON_TARDIO es obligatoria (clase positiva de las metas del PDF): así se evalúa también
# el modelo anterior de 2 clases y sets de campo sin hojas sanas, como PlantDoc.
POTATO_CLASSES = ("SANO", "TIZON_TEMPRANO", "TIZON_TARDIO")
CLASSES = (*POTATO_CLASSES, NOT_POTATO)
REQUIRED_CLASSES = ("TIZON_TARDIO",)
POSITIVE = "TIZON_TARDIO"   # clase positiva de las métricas del PDF (uno contra el resto)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
REPORTS_DIR = Path(__file__).parent / "reports"

# Metas del PDF (05-requerimientos-no-funcionales §Calidad del modelo): (comparación, meta).
TARGETS = {
    "accuracy": (">=", 0.90),
    "precision": (">=", 0.90),
    "recall": (">=", 0.90),
    "f1": (">=", 0.90),
    "specificity": (">=", 0.85),
    "error_rate": ("<=", 0.10),
}
METRIC_NAMES = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
    "specificity": "Especificidad", "error_rate": "Tasa de error",
}

# Distancia de Hamming máxima entre pHash de 64 bits para considerar dos imágenes casi duplicadas.
# Heurística: copias redimensionadas o recomprimidas quedan en 0–4; los pares encontrados se revisan a mano.
PHASH_MAX_DISTANCE = 6


def load_dataset(root: Path, manifest: Path | None = None, split: str | None = None) -> list[tuple[Path, str]]:
    """Pares (imagen, clase) de <root>/<CLASE>. Falla si falta TIZON_TARDIO.

    Con `manifest` (training/split.csv) solo se toman las imágenes de la partición `split`.
    """
    wanted = None
    if manifest:
        with manifest.open(encoding="utf-8", newline="") as f:
            wanted = {row["file"] for row in csv.DictReader(f) if row["split"] == split}
    samples: list[tuple[Path, str]] = []
    for label in CLASSES:
        folder = root / label
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS) if folder.is_dir() else []
        if wanted is not None:
            files = [p for p in files if f"{label}/{p.name}" in wanted]
        if not files and label in REQUIRED_CLASSES:
            raise SystemExit(f"No hay imágenes {'/'.join(IMAGE_EXTS)} en {folder}" + (f" (partición {split})" if split else ""))
        samples += [(p, label) for p in files]
    return samples


def decode_bgr(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def read_bgr(path: Path) -> np.ndarray:
    # np.fromfile + imdecode: cv2.imread falla con rutas con tildes en Windows.
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen {path}")
    return img


def predict_probs(model, images: list[np.ndarray], spec: InputSpec, batch: int = 32) -> np.ndarray:
    """Salida del modelo para cada imagen completa, con el preprocesamiento de `spec`."""
    out = []
    for i in range(0, len(images), batch):
        x = np.stack([preprocess(img, spec) for img in images[i : i + batch]])
        out.append(np.asarray(model([x], training=False)))
    return np.concatenate(out)


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Matriz de confusión, accuracy global y métricas de cada clase contra el resto.

    Metas del PDF: accuracy y tasa de error globales; precision, recall, F1 y especificidad de TIZON_TARDIO (con 2
    clases equivale al caso binario). Una razón sin denominador vale 0 (no alcanza la meta).
    """
    def ratio(a: float, b: float) -> float:
        return a / b if b else 0.0

    present = set(y_true) | set(y_pred)
    labels = [c for c in CLASSES if c in present or c in REQUIRED_CLASSES]
    confusion = {t: dict.fromkeys(labels, 0) for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t][p] += 1
    n = len(y_true)
    per_class = {}
    for c in labels:
        tp = confusion[c][c]
        fn = sum(confusion[c].values()) - tp
        fp = sum(confusion[t][c] for t in labels) - tp
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        per_class[c] = {"support": tp + fn, "tp": tp, "fn": fn, "fp": fp, "tn": n - tp - fn - fp,
                        "precision": precision, "recall": recall,
                        "f1": ratio(2 * precision * recall, precision + recall),
                        "specificity": ratio(n - tp - fn - fp, n - tp - fn)}
    accuracy = ratio(sum(confusion[c][c] for c in labels), n)
    pos = per_class[POSITIVE]
    return {"n": n, "accuracy": accuracy, "error_rate": 1 - accuracy,
            **{k: pos[k] for k in ("tp", "fn", "fp", "tn", "precision", "recall", "f1", "specificity")},
            "confusion": confusion, "per_class": per_class}


def check_targets(metrics: dict) -> list[dict]:
    rows = []
    for name, (op, goal) in TARGETS.items():
        value = metrics[name]
        rows.append({"metric": name, "op": op, "target": goal, "value": value,
                     "ok": value >= goal if op == ">=" else value <= goal})
    return rows


def latency_stats(ms: list[float]) -> dict:
    return {"mean_ms": round(float(np.mean(ms)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1)}


def phash(img_bgr: np.ndarray) -> int:
    """Hash perceptual de 64 bits: DCT de la imagen en gris a 32×32 y bloque 8×8 de baja frecuencia vs. su mediana."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()
    return sum(1 << i for i, bit in enumerate(low > np.median(low)) if bit)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicates(a: dict[str, int], b: dict[str, int] | None = None,
                    max_distance: int = PHASH_MAX_DISTANCE) -> list[tuple[str, str, int]]:
    """Pares (archivo_a, archivo_b, distancia) con distancia ≤ max_distance. Sin `b`, compara `a` consigo mismo."""
    items_a = list(a.items())
    pairs = []
    for i, (name_a, hash_a) in enumerate(items_a):
        others = items_a[i + 1 :] if b is None else b.items()
        pairs += [(name_a, name_b, d) for name_b, hash_b in others if (d := hamming(hash_a, hash_b)) <= max_distance]
    return pairs


def num(value: float, decimals: int = 1) -> str:
    """Formato del documento: coma decimal y punto de miles (1.500,0)."""
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def pct(value: float) -> str:
    return f"{num(value * 100)} %"


def machine() -> str:
    """Dónde se midió la latencia (la meta RNF-REN-02 aplica en la EC2), sin el nombre del equipo."""
    return f"{platform.platform()} · {os.cpu_count()} CPU"


def relative(path: str | Path, root: Path) -> str:
    """Ruta dentro del set de datos, para no publicar rutas locales en los reportes."""
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def write_report(stem: str, data: dict, markdown: str, out_dir: Path) -> Path:
    """Escribe <out_dir>/<stem>.md y .json; devuelve la ruta del .md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
                                          encoding="utf-8")
    md = out_dir / f"{stem}.md"
    md.write_text(markdown, encoding="utf-8")
    return md


def today() -> str:
    return dt.date.today().isoformat()
