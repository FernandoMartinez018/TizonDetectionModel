"""F4-IA-05: entrena MobileNetV2 de forma reproducible (08 §5) con 4 salidas: SANO, TIZON_TEMPRANO, TIZON_TARDIO y
NO_PAPA (hojas de otras plantas, para no analizar el tizón si la hoja no es de papa).

Uso, desde la raíz de TizonDetectionModel:
    python -m training.train --data <dir> --model-out ../modelo_tizon_3c.keras \
        --source "PlantVillage (papa), github.com/spMohanty/PlantVillage-Dataset, raw/color"

<dir>/SANO, <dir>/TIZON_TEMPRANO, <dir>/TIZON_TARDIO y <dir>/NO_PAPA (por ejemplo, PlantVillage: las carpetas de
papa y una muestra de los otros cultivos).
- Partición estratificada 70/15/15 con semilla fija; los casi duplicados (pHash) quedan en la misma partición.
  Se guarda en training/split.csv (versionado) para repetir la evaluación con la misma prueba.
- El preprocesamiento va DENTRO del modelo (Rescaling a [-1, 1], igual que preprocess_input de MobileNetV2): la IA
  solo redimensiona y pasa RGB (input_spec scaling "none"). Las imágenes se leen con el mismo preprocess() del servicio.
- Aumento de datos solo en entrenamiento y pesos por clase (hay muchas menos hojas sanas).
- Fase 1 con la base congelada y fase 2 de ajuste fino de los últimos bloques; parada temprana por val_loss.
- Guarda el modelo (.keras), actualiza model_meta.json y deja el reporte en evaluation/reports/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.model_meta import DEFAULT_META_PATH, InputSpec, ModelMeta, sha256_of
from app.predictor import preprocess
from evaluation.common import (
    CLASSES, REPORTS_DIR, classification_metrics, load_dataset, near_duplicates, num, pct, phash, read_bgr, today,
    write_report,
)

SPLIT_FILE = Path(__file__).parent / "split.csv"
SHARES = (("train", 0.70), ("val", 0.85))   # cortes acumulados; el resto es test
SPEC = InputSpec(width=224, height=224, color_order="RGB", scaling="none")
FINE_TUNE_FROM = "block_13_expand"          # se ajustan los bloques 13 a 16 y Conv_1 de MobileNetV2


def near_duplicate_groups(files: list[Path]) -> list[list[Path]]:
    """Grupos de casi duplicados (pHash), para no repartir la misma hoja entre particiones."""
    hashes = {str(p): phash(read_bgr(p)) for p in files}
    parent = {name: name for name in hashes}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for a, b, _ in near_duplicates(hashes):
        parent[find(a)] = find(b)
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        groups[find(str(p))].append(p)
    return sorted(groups.values(), key=lambda g: g[0].name)


def assign_splits(samples: list[tuple[Path, str]], seed: int, existing: dict[str, str] | None = None,
                  new_shares: tuple = SHARES) -> dict[Path, str]:
    """Partición estratificada por clase; los casi duplicados quedan juntos.

    `existing` ("CLASE/archivo" → partición, de un split.csv anterior): esos archivos conservan su partición y los
    nuevos que son casi duplicados de ellos la heredan, para comparar modelos con la misma prueba. El resto de los
    nuevos se reparte con `new_shares` (cortes acumulados; lo que queda después es test).
    """
    existing = existing or {}
    rng = random.Random(seed)
    split_of: dict[Path, str] = {}
    for label in CLASSES:
        files = [p for p, lab in samples if lab == label]
        groups = near_duplicate_groups(files)
        rng.shuffle(groups)
        fresh = []
        for group in groups:
            known = [existing[f"{label}/{p.name}"] for p in group if f"{label}/{p.name}" in existing]
            if known:
                split_of.update({p: existing.get(f"{label}/{p.name}", known[0]) for p in group})
            else:
                fresh.append(group)
        total, done = sum(len(g) for g in fresh), 0
        for group in fresh:
            split = next((name for name, cut in new_shares if done / total < cut), "test")
            split_of.update(dict.fromkeys(group, split))
            done += len(group)
    return split_of


def write_split(samples: list[tuple[Path, str]], split_of: dict[Path, str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["split", "class", "file", "sha256"])
        for p, label in sorted(samples, key=lambda s: (CLASSES.index(s[1]), s[0].name)):
            writer.writerow([split_of[p], label, f"{label}/{p.name}", hashlib.sha256(p.read_bytes()).hexdigest()])


def build_model(n_classes: int):
    import keras

    inputs = keras.Input((SPEC.height, SPEC.width, 3))
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)   # = mobilenet_v2.preprocess_input
    base = keras.applications.MobileNetV2(input_shape=(SPEC.height, SPEC.width, 3), include_top=False,
                                          weights="imagenet", pooling="avg")
    base.trainable = False
    x = base(x, training=False)   # BatchNormalization siempre en modo inferencia, también al ajustar
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def exportable(model):
    """Mismo grafo con la entrada como lista [input], igual que el .h5 anterior: el servicio llama model([batch])."""
    import keras

    return keras.Model([model.inputs[0]], model.outputs[0])


def datasets(x_train, y_train, x_val, y_val, seed: int, batch: int = 32):
    import keras
    import tensorflow as tf

    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        keras.layers.RandomRotation(0.1, seed=seed),
        keras.layers.RandomZoom(0.1, seed=seed),
        keras.layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
        keras.layers.RandomContrast(0.15, value_range=(0, 255), seed=seed),
    ])

    def as_float(x, y):
        return tf.cast(x, tf.float32), y

    train = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train), seed=seed)
             .batch(batch).map(as_float).map(lambda x, y: (augment(x, training=True), y)).prefetch(2))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch).map(as_float).prefetch(2)
    return train, val


def train(x, y, splits, seed: int, epochs: int, fine_tune_epochs: int):
    import keras

    model, base = build_model(len(CLASSES))
    counts = Counter(y[splits == "train"])
    class_weight = {i: len(y[splits == "train"]) / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))}
    train_ds, val_ds = datasets(x[splits == "train"], y[splits == "train"], x[splits == "val"], y[splits == "val"], seed)

    def fit(lr: float, n_epochs: int):
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        return model.fit(train_ds, validation_data=val_ds, epochs=n_epochs, class_weight=class_weight,
                         callbacks=[stop], verbose=2).history

    history = {"frozen": fit(1e-3, epochs)}
    base.trainable = True
    start = [layer.name for layer in base.layers].index(FINE_TUNE_FROM)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, keras.layers.BatchNormalization)
    history["fine_tune"] = fit(1e-5, fine_tune_epochs)
    return model, history, class_weight


def predict_classes(model, x: np.ndarray, batch: int = 64) -> list[str]:
    out = [np.asarray(model([x[i : i + batch].astype(np.float32)], training=False)) for i in range(0, len(x), batch)]
    return [CLASSES[i] for i in np.concatenate(out).argmax(axis=1)]


def render(report: dict) -> str:
    m = report["test_metrics"]
    lines = [
        f"# Entrenamiento del modelo (F4-IA-05) — {report['date']}",
        "",
        f"- Modelo: `{report['model_version']}` (`{report['model_file']}`, sha256 `{report['model_sha256']}`)",
        f"- Datos: {report['source'] or 'sin documentar (usa --source)'}",
        f"- Partición (semilla {report['seed']}, `training/split.csv`; {report['kept']} imágenes conservan la "
        f"partición anterior y {report['added']} se agregaron): "
        + "; ".join(f"{s}: " + ", ".join(f"{c} {n}" for c, n in report["split_counts"][s].items())
                    for s in ("train", "val", "test")),
        "- MobileNetV2 (ImageNet) con el preprocesamiento dentro del modelo; aumento de datos y pesos por clase "
        + "(" + ", ".join(f"{c} {num(w, 2)}" for c, w in report["class_weight"].items()) + ").",
        f"- Épocas: {report['epochs']['frozen']} con la base congelada y {report['epochs']['fine_tune']} de ajuste fino.",
        "",
        f"## Prueba ({m['n']} imágenes que el modelo no vio)",
        "",
        f"Accuracy global: **{pct(m['accuracy'])}**. Tasa de error: {pct(m['error_rate'])}.",
        "",
        "| Clase | Soporte | Precision | Recall | F1 | Especificidad |",
        "|---|---|---|---|---|---|",
    ]
    for c, r in m["per_class"].items():
        lines.append(f"| {c} | {r['support']} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | "
                     f"{pct(r['specificity'])} |")
    labels = list(m["confusion"])
    lines += ["", "Matriz de confusión (filas: real, columnas: predicho):", "",
              "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    lines += [f"| {t} | " + " | ".join(str(m["confusion"][t][p]) for p in labels) + " |" for t in labels]
    lines += ["", "Limitación: esta prueba es la partición test de training/split.csv (fotos de laboratorio de "
                  "PlantVillage). El desempeño con fotos de campo se mide aparte (reportes campo-*, P-02).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True, help="Ruta del .keras (fuera de git)")
    parser.add_argument("--meta", type=Path, default=Path(DEFAULT_META_PATH))
    parser.add_argument("--source", default="", help="Origen de las imágenes (se documenta en el reporte y el meta)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--keep-split", type=Path, help="split.csv anterior: sus archivos conservan la partición")
    parser.add_argument("--new-shares", default="0.70,0.85",
                        help="Cortes acumulados train,val para los archivos nuevos (p. ej. 0.85,1.0: ninguno a test)")
    args = parser.parse_args(argv)

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    samples = load_dataset(args.data)
    if {label for _, label in samples} != set(CLASSES):
        raise SystemExit(f"El entrenamiento necesita las carpetas {', '.join(CLASSES)}")
    existing = {}
    if args.keep_split:
        with args.keep_split.open(encoding="utf-8", newline="") as f:
            existing = {row["file"]: row["split"] for row in csv.DictReader(f)}
    train_cut, val_cut = (float(v) for v in args.new_shares.split(","))
    split_of = assign_splits(samples, args.seed, existing, (("train", train_cut), ("val", val_cut)))
    write_split(samples, split_of, SPLIT_FILE)

    x = np.stack([preprocess(read_bgr(p), SPEC).astype(np.uint8) for p, _ in samples])   # RGB 0–255, sin escalar
    y = np.array([CLASSES.index(label) for _, label in samples])
    splits = np.array([split_of[p] for p, _ in samples])

    trained, history, class_weight = train(x, y, splits, args.seed, args.epochs, args.fine_tune_epochs)
    model = exportable(trained)
    test = splits == "test"
    metrics = classification_metrics([CLASSES[i] for i in y[test]], predict_classes(model, x[test]))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    sha = sha256_of(args.model_out)
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    meta.pop("calibration", None)   # el preprocesamiento se conoce por construcción
    meta.update({
        "model_version": f"mobilenetv2-{len(CLASSES)}c-{sha[:8]}-{today().replace('-', '')}",
        "file": args.model_out.name,
        "sha256": sha,
        "classes": list(CLASSES),
        "input_spec": SPEC.model_dump(),
        "training": {"date": today(), "source": args.source, "seed": args.seed, "split": "training/split.csv",
                     "test_accuracy": round(metrics["accuracy"], 4)},
    })
    ModelMeta.model_validate(meta)
    args.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    split_counts = {s: {c: int(((splits == s) & (y == i)).sum()) for i, c in enumerate(CLASSES)}
                    for s in ("train", "val", "test")}
    report = {
        "date": today(), "source": args.source, "seed": args.seed, "model_version": meta["model_version"],
        "model_file": args.model_out.name, "model_sha256": sha, "split_counts": split_counts,
        "kept": sum(f"{label}/{p.name}" in existing for p, label in samples),
        "added": sum(f"{label}/{p.name}" not in existing for p, label in samples),
        "class_weight": {CLASSES[i]: w for i, w in class_weight.items()},
        "epochs": {phase: len(h["loss"]) for phase, h in history.items()},
        "history": history, "test_metrics": metrics,
    }
    md = write_report(f"{report['date']}_entrenamiento", report, render(report), args.reports)
    print(f"Modelo: {args.model_out} ({meta['model_version']})\nReporte: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
