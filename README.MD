# TizonDetectionModel — Servicio de IA

Servicio interno (FastAPI + TensorFlow CPU) que clasifica regiones de hoja de papa como `SANO` o
`TIZON_TARDIO`. Lo consume **solo** la API de Spring por la red interna de Docker; no se expone a
Internet ni usa CORS. Contrato: `contracts/tizon-api.yaml` (tag `interno-ia`), copia de la fuente de
verdad de la carpeta raíz del proyecto.

## Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del modelo: `status` (`ok`/`degraded`), `model_loaded`, `model_version`, `classes` (orden de salida del modelo) e `input_spec`. Siempre responde 200. |
| `POST` | `/v1/analyze` | Analiza un frame (`frame_b64`, JPEG/PNG en Base64 sin prefijo). Exige `X-Internal-Token`. Responde `DETECTED` con regiones o `NO_VEGETATION` (resultado válido, `label=null`). |

Errores con el cuerpo `Error` del contrato: 400 `INVALID_IMAGE`, 401 `UNAUTHORIZED`, 422 `VALIDATION_ERROR`
y 503 `MODEL_NOT_READY`. El header `X-Request-Id` se registra en los logs y se devuelve en los errores.

## Modelo y metadatos
- `MODEL_PATH`: el `.h5` (MobileNetV2, 2 clases, Keras 2.14). **No va a git.**
- `MODEL_META_PATH`: `model_meta.json`, única fuente de clases, preprocesamiento (`color_order`, `scaling`),
  `min_confidence` y `model_version`. Si falta, no es válido o su `sha256` no coincide con el `.h5`, el servicio
  arranca **degradado** (`/health` lo informa y `/v1/analyze` responde 503).
- `models/model_meta.json` es **provisional**: replica el preprocesamiento del código v1. El orden de clases
  y la escala reales se verifican en F4-IA-02 (`evaluation/calibrate.py`).
- Para actualizar el modelo se reemplazan el `.h5` y el `model_meta.json` y se reinicia el contenedor, sin
  reconstruir la imagen.
- TensorFlow 2.16 (Keras 3) no deserializa el `DepthwiseConv2D` de Keras 2.14; `app/predictor.py` descarta
  el argumento `groups` (siempre 1). Se verificó contra `tf_keras`: diferencia máxima de salida 4×10⁻⁶.

## Configuración
| Variable | Obligatoria | Descripción |
|---|---|---|
| `APP_ENV` | No | `local` (por defecto fuera de Docker) o `production` (por defecto en la imagen) |
| `INTERNAL_API_TOKEN` | Sí, salvo `APP_ENV=local` | Secreto compartido con la API. Sin él, fuera de `local`, el servicio no arranca |
| `MODEL_PATH` | No | Por defecto `models/modelo_tizon_tardio.h5` (en Docker `/models/...`) |
| `MODEL_META_PATH` | No | Por defecto `models/model_meta.json` (en Docker `/models/...`) |

Ver `.env.example`.

## Ejecución local
```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp ../modelo_tizon_tardio.h5 models/              # models/*.h5 está en .gitignore
uvicorn app.main:app --reload
```

## Pruebas y linter
```bash
pytest test/ -m "not integration"                       # unitarias + contrato (openapi-core)
MODEL_PATH=../modelo_tizon_tardio.h5 pytest test/ -m integration
ruff check app test evaluation
```

## Calibración y evaluación del modelo (F4, 08-modelo-ia)
Los sets van **fuera del repositorio** (o en `evaluation/data/`, ignorada por git), con una carpeta por clase:
`<dir>/SANO/*.jpg` y `<dir>/TIZON_TARDIO/*.jpg`. Los reportes quedan en `evaluation/reports/` (`.md` y `.json`)
y sí se versionan como evidencia. Las imágenes nunca se versionan.

```bash
# 1. Orden de clases y preprocesamiento (F4-IA-02): 8 combinaciones con la imagen completa.
#    --write guarda la mejor en models/model_meta.json solo con ≥ 30 imágenes por clase y ≥ 5 puntos de ventaja.
python -m evaluation.calibrate --data <set_calibracion> --model ../modelo_tizon_tardio.h5 --write

# 2. Métricas frente a las metas del PDF (F4-IA-04). classifier = modelo puro; pipeline = el flujo de /v1/analyze,
#    con el filtro de vegetación (también da el % de tizón en NO_VEGETATION de F4-IA-03).
python -m evaluation.evaluate --data <set_prueba> --mode classifier --model ../modelo_tizon_tardio.h5
python -m evaluation.evaluate --data <set_prueba> --mode pipeline --model ../modelo_tizon_tardio.h5 \
    --train <set_entrenamiento> --source "origen de cada subconjunto"

# 3. Auditoría del dataset de entrenamiento → evaluation/reports/dataset_audit.md
python -m evaluation.audit --data <set_entrenamiento>
```

- El set de calibración y el de prueba deben ser distintos. Con `--train`, `evaluate` busca duplicados exactos y
  perceptuales (pHash) entre la prueba y el entrenamiento, y los lista para revisarlos a mano.
- La latencia del reporte es la de la máquina donde corre (queda anotada). La meta de RNF-REN-02 aplica en la EC2.
- Las métricas se reportan aunque no alcancen la meta. No se ajusta el set de prueba para mejorar los números.

## Docker
```bash
docker build -t tizon-ia .
docker run --rm -e INTERNAL_API_TOKEN=... -v "$PWD/models:/models:ro" -p 8000:8000 tizon-ia
```
La imagen corre con un usuario sin privilegios y su `HEALTHCHECK` solo pasa si el modelo cargó. En el sistema
completo se levanta con `../deploy/docker-compose.yml`, sin publicar el puerto 8000.

## CI/CD (GitHub Actions)
`.github/workflows/ci.yml` corre en cada push y PR a `dev`, `qa` y `main`: ruff, pytest (sin las pruebas de
integración, que requieren el `.h5`) y gitleaks. En cada push a `qa` o `main` publica la imagen
`ghcr.io/<owner>/tizon-ia` con las etiquetas `:qa` o `:main` y `:sha-<commit>`. La EC2 la descarga por *pull*
(`deploy/scripts/auto-update.sh`): GitHub nunca entra a la EC2. Si el paquete de GHCR es privado, la EC2
necesita un token de solo lectura (`read:packages`).
