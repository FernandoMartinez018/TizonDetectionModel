"""F4-IA-04 (y el reporte de F4-IA-03): métricas del modelo frente a las metas del PDF (08-modelo-ia §4).

Uso, desde la raíz de TizonDetectionModel:
    python -m evaluation.evaluate --data <dir_prueba> --mode classifier --model ../modelo_tizon_tardio.h5
    python -m evaluation.evaluate --data <dir_prueba> --mode pipeline --model ../modelo_tizon_tardio.h5 \
        --train <dir_entrenamiento> --source "PlantVillage (papa) + fotos de campo propias"

classifier: imagen completa al modelo (calidad pura del modelo).
pipeline:   el mismo Predictor.predict de /v1/analyze (filtro de vegetación + modelo + resultado principal).
            NO_VEGETATION cuenta como predicción SANO (falso negativo si era tizón) y además se reporta aparte.
Positivo = TIZON_TARDIO. Las métricas se reportan aunque no alcancen la meta (05-RNF, regla de calidad).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.model_meta import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, ModelMeta, load_meta, sha256_of
from app.predictor import Predictor, load_keras_model, preprocess
from evaluation.common import (
    CLASSES, METRIC_NAMES, PHASH_MAX_DISTANCE, POSITIVE, REPORTS_DIR, binary_metrics, check_targets,
    latency_stats, load_dataset, machine, near_duplicates, num, pct, phash, read_bgr, today, write_report,
)

NO_VEGETATION = "NO_VEGETATION"
MAX_NO_VEGETATION_TIZON = 0.05   # F4-IA-03: imágenes con tizón que terminan en NO_VEGETATION
P95_INFERENCE_MS = 1500          # RNF-REN-02, en la EC2

Predict = Callable[[Path], tuple[str, float]]   # imagen → (SANO | TIZON_TARDIO | NO_VEGETATION, ms)


def evaluate(samples: list[tuple[Path, str]], predict: Predict) -> dict:
    y_true, y_pred, latencies, errors = [], [], [], []
    no_vegetation: Counter = Counter()
    for path, label in samples:
        raw, ms = predict(path)
        pred = "SANO" if raw == NO_VEGETATION else raw
        if raw == NO_VEGETATION:
            no_vegetation[label] += 1
        if pred != label:
            errors.append({"file": str(path), "true": label, "pred": raw})
        y_true.append(label)
        y_pred.append(pred)
        latencies.append(ms)
    counts = Counter(y_true)
    metrics = binary_metrics(y_true, y_pred)
    return {
        "counts": {c: counts[c] for c in CLASSES},
        "metrics": metrics,
        "targets": check_targets(metrics),
        "latency": latency_stats(latencies),
        "no_vegetation": {c: no_vegetation[c] for c in CLASSES},
        "no_vegetation_tizon_ratio": no_vegetation[POSITIVE] / counts[POSITIVE] if counts[POSITIVE] else 0.0,
        "errors": errors,
    }


def classifier_predict(model, meta: ModelMeta) -> Predict:
    def predict(path: Path) -> tuple[str, float]:
        img = read_bgr(path)
        start = time.perf_counter()
        probs = np.asarray(model([preprocess(img, meta.input_spec)[None]], training=False))[0]
        return meta.classes[int(probs.argmax())], (time.perf_counter() - start) * 1000
    return predict


def pipeline_predict(predictor: Predictor) -> Predict:
    def predict(path: Path) -> tuple[str, float]:
        result = predictor.predict(path.read_bytes())
        return (result.label if result.status == "DETECTED" else NO_VEGETATION), float(result.inference_ms)
    return predict


def overlap(test: list[tuple[Path, str]], train: list[tuple[Path, str]]) -> dict:
    """Integridad del set de prueba: duplicados exactos (sha256) y perceptuales (pHash) contra el entrenamiento."""
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    train_sha = {sha(p): p for p, _ in train}
    exact = [{"test": str(p), "train": str(train_sha[s])} for p, _ in test if (s := sha(p)) in train_sha]
    near = near_duplicates({str(p): phash(read_bgr(p)) for p, _ in test},
                           {str(p): phash(read_bgr(p)) for p, _ in train})
    return {"exact": exact, "near": [{"test": a, "train": b, "distance": d} for a, b, d in near]}


def render(report: dict) -> str:
    m, mode = report["metrics"], report["mode"]
    lines = [
        f"# Evaluación del modelo — modo `{mode}` — {report['date']}",
        "",
        f"- Modelo: `{report['model_version']}` (`{report['model_file']}`, sha256 `{report['model_sha256']}`)",
        f"- Set de prueba: `{report['data']}` — origen: {report['source'] or 'sin documentar (usa --source)'}",
        "- Imágenes: " + ", ".join(f"{k} {v}" for k, v in report["counts"].items()) + f" (total {m['n']})",
        f"- Máquina: {report['machine']}",
        "- Positivo = TIZON_TARDIO."
        + (" NO_VEGETATION cuenta como predicción SANO." if mode == "pipeline" else " Imagen completa, sin filtro."),
        "",
        "## Métricas frente a las metas del PDF",
        "",
        "| Métrica | Meta | Medido | Cumple |",
        "|---|---|---|---|",
    ]
    for row in report["targets"]:
        op = "≥" if row["op"] == ">=" else "≤"
        lines.append(f"| {METRIC_NAMES[row['metric']]} | {op} {pct(row['target'])} | {pct(row['value'])} | "
                     f"{'✔' if row['ok'] else '✘'} |")
    lat = report["latency"]
    lines.append(f"| Latencia p95 ({'inferencia' if mode == 'classifier' else 'IA completa'}) | "
                 f"< {num(P95_INFERENCE_MS, 0)} ms en la EC2 | {num(lat['p95_ms'])} ms (media {num(lat['mean_ms'])} ms) | "
                 f"{'✔' if lat['p95_ms'] < P95_INFERENCE_MS else '✘'} |")
    lines += [
        "",
        "## Matriz de confusión",
        "",
        "| | Predicho SANO | Predicho TIZON_TARDIO |",
        "|---|---|---|",
        f"| Real SANO | {m['tn']} | {m['fp']} |",
        f"| Real TIZON_TARDIO | {m['fn']} | {m['tp']} |",
    ]
    if mode == "pipeline":
        nv, ratio = report["no_vegetation"], report["no_vegetation_tizon_ratio"]
        lines += [
            "",
            "## NO_VEGETATION (filtro de vegetación)",
            "",
            f"- SANO: {nv['SANO']} imágenes. TIZON_TARDIO: {nv['TIZON_TARDIO']} ({pct(ratio)}; meta de F4-IA-03 ≤ "
            f"{pct(MAX_NO_VEGETATION_TIZON)}: {'✔' if ratio <= MAX_NO_VEGETATION_TIZON else '✘'}).",
        ]
    lines += ["", "## Integridad del set de prueba", ""]
    if report["overlap"] is None:
        lines.append("- No se comparó con el set de entrenamiento (usa `--train`).")
    else:
        ov = report["overlap"]
        lines.append(f"- Duplicados exactos con el entrenamiento: {len(ov['exact'])}.")
        lines.append(f"- Casi duplicados (pHash, distancia ≤ {PHASH_MAX_DISTANCE}): {len(ov['near'])}. "
                     "Se revisan a mano y se retiran del set de prueba si son la misma hoja.")
        lines += [f"  - `{n['test']}` ↔ `{n['train']}` ({n['distance']})" for n in ov["near"][:20]]
    lines += ["", f"## Errores ({len(report['errors'])}; primeros 20)", "", "| Archivo | Real | Predicho |",
              "|---|---|---|"]
    lines += [f"| `{e['file']}` | {e['true']} | {e['pred']} |" for e in report["errors"][:20]]
    lines += ["", "Se reporta el valor medido aunque no alcance la meta; el análisis de causas va en el documento "
                  "(05-requerimientos-no-funcionales)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Set de prueba (SANO/ y TIZON_TARDIO/)")
    parser.add_argument("--mode", choices=("classifier", "pipeline"), required=True)
    parser.add_argument("--model", type=Path, default=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--meta", type=Path, default=Path(os.getenv("MODEL_META_PATH", DEFAULT_META_PATH)))
    parser.add_argument("--train", type=Path, help="Set de entrenamiento, para buscar solapamiento")
    parser.add_argument("--source", default="", help="Origen del set de prueba (se documenta en el reporte)")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    samples = load_dataset(args.data)
    meta = load_meta(str(args.meta))
    if args.mode == "classifier":
        predict = classifier_predict(load_keras_model(args.model), meta)
    else:
        predictor = Predictor()
        predictor.load(str(args.model), str(args.meta))
        predict = pipeline_predict(predictor)

    report = {
        "date": today(), "mode": args.mode, "data": str(args.data), "source": args.source,
        "model_version": meta.model_version, "model_file": args.model.name, "model_sha256": sha256_of(args.model),
        "machine": machine(),
        **evaluate(samples, predict),
        "overlap": overlap(samples, load_dataset(args.train)) if args.train else None,
    }
    md = write_report(f"{report['date']}_{args.mode}", report, render(report), args.reports)
    print(f"Reporte: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
