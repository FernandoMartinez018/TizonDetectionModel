"""F4-IA-02: verifica el orden de clases y el preprocesamiento del modelo (08-modelo-ia §2).

Uso, desde la raíz de TizonDetectionModel:
    python -m evaluation.calibrate --data <dir_calibracion> --model ../modelo_tizon_tardio.h5 [--write]

<dir_calibracion>/SANO y <dir_calibracion>/TIZON_TARDIO: 30 o más imágenes por clase, separadas del set de prueba.
Prueba las 8 combinaciones {RGB, BGR} × {unit, mobilenet_v2} × {orden de clases} con la imagen completa (sin el
filtro de vegetación). Con --write guarda la mejor en model_meta.json, solo si hay 30 imágenes por clase y supera a
la segunda por 5 puntos o más; si no, hay que ampliar el set de calibración.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

import cv2

from app.model_meta import DEFAULT_META_PATH, DEFAULT_MODEL_PATH, InputSpec, ModelMeta, load_meta, sha256_of
from app.predictor import load_keras_model
from evaluation.common import (
    CLASSES, REPORTS_DIR, load_dataset, num, pct, predict_probs, read_bgr, today, write_report,
)

MIN_PER_CLASS = 30
MIN_MARGIN = 0.05
COLOR_ORDERS = ("RGB", "BGR")
SCALINGS = ("unit", "mobilenet_v2")


def score_combinations(model, images, labels: list[str], width: int, height: int) -> list[dict]:
    """Accuracy de cada combinación de color, escala y orden de clases, de mayor a menor (en empate se conserva el
    orden de prueba). Con 2 clases son las 8 de 08 §2; con más, todas las permutaciones del orden de clases."""
    classes = [c for c in CLASSES if c in set(labels)]
    results = []
    for color_order in COLOR_ORDERS:
        for scaling in SCALINGS:
            spec = InputSpec(width=width, height=height, color_order=color_order, scaling=scaling)
            probs = predict_probs(model, images, spec)
            if probs.shape[1] != len(classes):
                raise SystemExit(f"El modelo tiene {probs.shape[1]} salidas y el set {len(classes)} clases "
                                 f"({', '.join(classes)}): usa un set con las mismas clases del modelo.")
            indices = probs.argmax(axis=1)
            for order in permutations(classes):
                correct = sum(order[i] == label for i, label in zip(indices, labels, strict=True))
                results.append({"color_order": color_order, "scaling": scaling, "classes": list(order),
                                "correct": correct, "accuracy": correct / len(labels)})
    return sorted(results, key=lambda r: r["accuracy"], reverse=True)


def problems_of(counts: Counter, results: list[dict]) -> list[str]:
    problems = []
    if min(counts.values()) < MIN_PER_CLASS:
        detail = ", ".join(f"{label} {n}" for label, n in sorted(counts.items()))
        problems.append(f"Hay menos de {MIN_PER_CLASS} imágenes en alguna clase ({detail}).")
    margin = results[0]["accuracy"] - results[1]["accuracy"]
    if margin < MIN_MARGIN:
        problems.append(f"La mejor combinación supera a la segunda por {num(margin * 100)} puntos (< 5): "
                        "hay que ampliar el set de calibración.")
    return problems


def updated_meta(meta_path: Path, model_path: Path, results: list[dict], n_images: int) -> dict:
    best, second = results[0], results[1]
    sha = sha256_of(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({
        "model_version": f"mobilenetv2-2c-{sha[:8]}-{today().replace('-', '')}",
        "file": model_path.name,
        "sha256": sha,
        "classes": best["classes"],
        "input_spec": {**meta["input_spec"], "color_order": best["color_order"], "scaling": best["scaling"]},
        "calibration": {
            "date": today(), "n_images": n_images, "accuracy": round(best["accuracy"], 4),
            "notes": f"Mejor de 8 combinaciones (evaluation/calibrate.py); la segunda obtuvo {pct(second['accuracy'])}.",
        },
    })
    ModelMeta.model_validate(meta)   # el servicio debe poder cargarlo tal cual
    return meta


def render(report: dict) -> str:
    lines = [
        f"# Calibración del modelo (F4-IA-02) — {report['date']}",
        "",
        f"- Modelo: `{report['model_file']}` (sha256 `{report['model_sha256']}`)",
        f"- Set de calibración: `{report['data']}` — " + ", ".join(f"{k}: {v}" for k, v in report["counts"].items()),
        "- Imagen completa, sin filtro de vegetación (08-modelo-ia §2).",
        "",
        "| # | Orden de color | Escala | Orden de clases (por índice de salida) | Aciertos | Accuracy |",
        "|---|---|---|---|---|---|",
    ]
    n = sum(report["counts"].values())
    for i, r in enumerate(report["results"], 1):
        lines.append(f"| {i} | {r['color_order']} | {r['scaling']} | {', '.join(r['classes'])} | "
                     f"{r['correct']}/{n} | {pct(r['accuracy'])} |")
    best = report["results"][0]
    lines += ["", f"**Mejor combinación:** {best['color_order']}, {best['scaling']}, clases "
                  f"[{', '.join(best['classes'])}], con {pct(best['accuracy'])}."]
    if report["problems"]:
        lines += ["", "**No se puede adoptar todavía:**"] + [f"- {p}" for p in report["problems"]]
    else:
        lines += ["", "Sin problemas: se puede guardar en `models/model_meta.json` con `--write`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Set de calibración (SANO/ y TIZON_TARDIO/)")
    parser.add_argument("--model", type=Path, default=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--meta", type=Path, default=Path(os.getenv("MODEL_META_PATH", DEFAULT_META_PATH)))
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--write", action="store_true", help="Guardar la mejor combinación en --meta")
    args = parser.parse_args(argv)

    samples = load_dataset(args.data)
    spec = load_meta(str(args.meta)).input_spec
    # Se reduce al tamaño de entrada al leer: el preprocess del servicio hace el mismo resize (idempotente).
    images = [cv2.resize(read_bgr(path), (spec.width, spec.height)) for path, _ in samples]
    labels = [label for _, label in samples]
    results = score_combinations(load_keras_model(args.model), images, labels, spec.width, spec.height)

    counts = Counter(labels)
    report = {
        "date": today(), "data": str(args.data), "model_file": args.model.name,
        "model_sha256": sha256_of(args.model), "counts": dict(sorted(counts.items())),
        "results": results, "problems": problems_of(counts, results),
    }
    md = write_report(f"{report['date']}_calibracion", report, render(report), args.reports)
    print(f"Reporte: {md}")

    if args.write:
        if report["problems"]:
            print("No se escribe model_meta.json:\n- " + "\n- ".join(report["problems"]), file=sys.stderr)
            return 1
        meta = updated_meta(args.meta, args.model, results, len(labels))
        args.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"model_meta.json actualizado: {meta['model_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
