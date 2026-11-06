"""F4-IA-04: auditoría del dataset de entrenamiento (08-modelo-ia §4) → evaluation/reports/dataset_audit.md.

Uso, desde la raíz de TizonDetectionModel:
    python -m evaluation.audit --data <dir_entrenamiento>

Conteo por clase y desbalance, duplicados exactos (sha256) y perceptuales (pHash), la misma imagen en las dos
clases, resoluciones, archivos ilegibles y una muestra fija de 20 imágenes por clase para revisar etiquetas a mano.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.common import (
    CLASSES, PHASH_MAX_DISTANCE, REPORTS_DIR, decode_bgr, load_dataset, near_duplicates, num, phash, relative, today,
    write_report,
)


def audit(samples: list[tuple[Path, str]], sample_size: int = 20, seed: int = 42, root: Path | None = None) -> dict:
    """`root`: carpeta del set, para que el reporte muestre rutas relativas a ella."""
    def name(p) -> str:
        return relative(p, root) if root else str(p)

    by_sha: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    sizes: dict[str, list[tuple[int, int]]] = defaultdict(list)
    hashes: dict[str, int] = {}
    unreadable = []
    for path, label in samples:
        data = path.read_bytes()
        img = decode_bgr(data)
        if img is None:
            unreadable.append(name(path))
            continue
        by_sha[hashlib.sha256(data).hexdigest()].append((path, label))
        sizes[label].append((img.shape[1], img.shape[0]))
        hashes[name(path)] = phash(img)

    exact = [[name(p) for p, _ in group] for group in by_sha.values() if len(group) > 1]
    conflicts = [[f"{name(p)} ({label})" for p, label in group] for group in by_sha.values()
                 if len({label for _, label in group}) > 1]
    exact_pairs = {frozenset(map(str, (a, b))) for group in exact for a in group for b in group if a != b}
    near = [(a, b, d) for a, b, d in near_duplicates(hashes) if frozenset((a, b)) not in exact_pairs]

    present = [c for c in CLASSES if any(label == c for _, label in samples)]
    counts = {c: sum(1 for _, label in samples if label == c) for c in present}
    rng = random.Random(seed)
    per_class = {c: [name(p) for p, label in samples if label == c] for c in present}
    return {
        "counts": counts,
        "imbalance": round(max(counts.values()) / min(counts.values()), 2),
        "unreadable": unreadable,
        "exact_duplicates": exact,
        "label_conflicts": conflicts,
        "near_duplicates": [{"a": a, "b": b, "distance": d} for a, b, d in near],
        "resolution": {c: _resolution(sizes[c]) for c in CLASSES if sizes[c]},
        "review_sample": {c: sorted(rng.sample(files, min(sample_size, len(files)))) for c, files in per_class.items()},
        "seed": seed,
    }


def _resolution(sizes: list[tuple[int, int]]) -> dict:
    w, h = np.array(sizes).T
    return {"width": [int(w.min()), int(np.median(w)), int(w.max())],
            "height": [int(h.min()), int(np.median(h)), int(h.max())]}


def render(report: dict, data: str) -> str:
    lines = [
        f"# Auditoría del dataset de entrenamiento — {report['date']}",
        "",
        f"- Dataset: `{data}`",
        "- Imágenes por clase: " + ", ".join(f"{k} {v}" for k, v in report["counts"].items())
        + f". Desbalance (mayor/menor): {num(report['imbalance'], 2)}.",
        f"- Archivos ilegibles: {len(report['unreadable'])}.",
        f"- Duplicados exactos (sha256): {len(report['exact_duplicates'])} grupos.",
        f"- **Misma imagen en las dos clases** (conflicto de etiqueta): {len(report['label_conflicts'])}.",
        f"- Casi duplicados (pHash, distancia ≤ {PHASH_MAX_DISTANCE}): {len(report['near_duplicates'])} pares.",
        "",
        "## Resolución (mínimo, mediana, máximo)",
        "",
        "| Clase | Ancho (px) | Alto (px) |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {' / '.join(map(str, r['width']))} | {' / '.join(map(str, r['height']))} |"
              for c, r in report["resolution"].items()]
    for title, key in (("Conflictos de etiqueta", "label_conflicts"), ("Duplicados exactos", "exact_duplicates")):
        if report[key]:
            lines += ["", f"## {title}", ""] + [f"- {' · '.join(group)}" for group in report[key][:50]]
    if report["near_duplicates"]:
        lines += ["", "## Casi duplicados (primeros 50)", ""]
        lines += [f"- `{n['a']}` ↔ `{n['b']}` ({n['distance']})" for n in report["near_duplicates"][:50]]
    lines += ["", f"## Revisión manual de etiquetas (muestra fija, semilla {report['seed']})", "",
              "Marca las imágenes que no correspondan a su clase y anota el total por clase.", ""]
    for c, files in report["review_sample"].items():
        lines += [f"### {c}", ""] + [f"- [ ] `{f}`" for f in files] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Dataset de entrenamiento (SANO/ y TIZON_TARDIO/)")
    parser.add_argument("--sample", type=int, default=20, help="Imágenes por clase para la revisión manual")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)

    report = {"date": today(), **audit(load_dataset(args.data), args.sample, args.seed, args.data)}
    md = write_report("dataset_audit", report, render(report, args.data.name), args.reports)
    print(f"Reporte: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
