# TizonDetectionModel — Servicio de IA

Servicio interno (FastAPI + TensorFlow CPU) que clasifica regiones de hoja de papa como `SANO`, `TIZON_TEMPRANO`
o `TIZON_TARDIO`. Si la hoja es de otra planta, responde `NOT_POTATO` y no analiza el tizón. Lo consume **solo** la
API de Spring por la red interna de Docker; no se expone a Internet ni usa CORS. Contrato: `contracts/tizon-api.yaml`
(tag `interno-ia`), copia de la fuente de verdad de la carpeta raíz del proyecto.

## Endpoints
| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del modelo: `status` (`ok`/`degraded`), `model_loaded`, `model_version`, `classes` (salidas del modelo en orden, incluida `NO_PAPA`) e `input_spec`. Siempre responde 200. |
| `POST` | `/v1/analyze` | Analiza un frame (`frame_b64`, JPEG/PNG en Base64 sin prefijo). Exige `X-Internal-Token`. Responde `DETECTED` con regiones, `NO_VEGETATION` (no hay hoja) o `NOT_POTATO` (la hoja es de otra planta); los dos últimos son resultados válidos con `label=null`. |

Errores con el cuerpo `Error` del contrato: 400 `INVALID_IMAGE`, 401 `UNAUTHORIZED`, 422 `VALIDATION_ERROR`
y 503 `MODEL_NOT_READY`. El header `X-Request-Id` se registra en los logs y se devuelve en los errores.

## Modelo y metadatos
- Modelo en uso: `modelo_tizon_4c.keras` (MobileNetV2 con el preprocesamiento dentro del modelo), con 4 salidas:
  `SANO`, `TIZON_TEMPRANO`, `TIZON_TARDIO` y `NO_PAPA`. **No va a git**: está en la carpeta raíz del proyecto y en
  `deploy/models/`. Se entrena con `training/train.py` (F4-IA-05).
- `MODEL_META_PATH`: `model_meta.json`, única fuente de clases, preprocesamiento (`color_order`, `scaling`),
  `min_confidence` y `model_version`. Si falta, no es válido o su `sha256` no coincide con el modelo, el servicio
  arranca **degradado** (`/health` lo informa y `/v1/analyze` responde 503).
- Resultado principal: gana la clase más severa (`TIZON_TARDIO` > `TIZON_TEMPRANO` > `SANO`) entre las regiones con
  confianza ≥ `min_confidence`. Una región `NO_PAPA` no se analiza; si ninguna región es de papa, la respuesta es
  `NOT_POTATO`. Las probabilidades solo traen las clases de papa, así que pueden sumar menos de 1.
- El filtro de vegetación (`app/vegetation.py`) acepta el verde, el amarillo de la clorosis conectado al verde y
  las lesiones marrones conectadas al tejido (F4-IA-03).
- Para actualizar el modelo se reemplazan el archivo del modelo y el `model_meta.json` y se reinicia el contenedor,
  sin reconstruir la imagen. **Volver al modelo anterior** (`modelo_tizon_tardio.h5`, 2 clases, Keras 2.14): usar su
  `model_meta.json` del historial de git (`git log -- models/model_meta.json`); el servicio sigue aceptándolo.
- TensorFlow 2.16 (Keras 3) no deserializa el `DepthwiseConv2D` del `.h5` de Keras 2.14; `app/predictor.py` descarta
  el argumento `groups` (siempre 1). Keras queda fijado en `requirements.txt` en la versión con que se guardó el modelo.

## Configuración
| Variable | Obligatoria | Descripción |
|---|---|---|
| `APP_ENV` | No | `local` (por defecto fuera de Docker) o `production` (por defecto en la imagen) |
| `INTERNAL_API_TOKEN` | Sí, salvo `APP_ENV=local` | Secreto compartido con la API. Sin él, fuera de `local`, el servicio no arranca |
| `MODEL_PATH` | No | Por defecto `models/modelo_tizon_4c.keras` (en Docker `/models/...`) |
| `MODEL_META_PATH` | No | Por defecto `models/model_meta.json` (en Docker `/models/...`) |

Ver `.env.example`.

## Ejecución local
```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp ../modelo_tizon_4c.keras models/               # models/* está en .gitignore (salvo model_meta.json)
uvicorn app.main:app --reload
```

## Pruebas y linter
```bash
pytest test/ -m "not integration"                       # unitarias + contrato (openapi-core)
MODEL_PATH=../modelo_tizon_4c.keras pytest test/ -m integration
ruff check app test evaluation training
```

## Entrenamiento y evaluación (F4, 08-modelo-ia)
Los datos van **fuera del repositorio** (o en `evaluation/data/`, ignorada por git), con una carpeta por clase:
`SANO/`, `TIZON_TEMPRANO/`, `TIZON_TARDIO/` y `NO_PAPA/`. Los reportes quedan en `evaluation/reports/` (`.md` y
`.json`) y sí se versionan como evidencia. Las imágenes nunca se versionan.

```bash
# 1. Entrenamiento reproducible (semilla 42): partición 70/15/15 sin casi duplicados entre particiones,
#    guardada en training/split.csv. Actualiza models/model_meta.json y deja el reporte de entrenamiento.
python -m training.train --data <dir> --model-out ../modelo_tizon_4c.keras --source "origen de las imágenes"

# 2. Métricas frente a las metas del PDF sobre la partición de prueba. classifier = modelo puro;
#    pipeline = el flujo de /v1/analyze (filtro de vegetación + modelo + resultado principal).
python -m evaluation.evaluate --data <dir> --mode classifier --model ../modelo_tizon_4c.keras \
    --manifest training/split.csv --split test --train-split train --source "origen"
python -m evaluation.evaluate --data <dir> --mode pipeline --model ../modelo_tizon_4c.keras \
    --manifest training/split.csv --split test --train-split train --source "origen"

# 3. Auditoría del dataset → evaluation/reports/dataset_audit.md
python -m evaluation.audit --data <dir>

# Para un modelo de origen desconocido: orden de clases y preprocesamiento (F4-IA-02).
python -m evaluation.calibrate --data <set_calibracion> --model <modelo> --write
```

Resultados del 24-oct (PlantVillage, partición de prueba de 479 imágenes que el modelo no vio; reportes en
`evaluation/reports/2026-10-24_*`):

| Métrica (TIZON_TARDIO contra el resto; accuracy global) | Meta del PDF | `classifier` | `pipeline` |
|---|---|---|---|
| Accuracy | ≥ 90 % | 95,6 % | 94,4 % |
| Precision | ≥ 90 % | 94,2 % | 94,8 % |
| Recall | ≥ 90 % | 97,3 % | 97,3 % |
| F1 | ≥ 90 % | 95,7 % | 96,1 % |
| Especificidad | ≥ 85 % | 97,3 % | 97,6 % |
| Tasa de error | ≤ 10 % | 4,4 % | 5,6 % |

- Tizón temprano: F1 98,0 % (`classifier`) y 95,6 % (`pipeline`); el 4,0 % de sus hojas termina en `NO_VEGETATION`.
- ¿Es papa?: el 1,6 % de las hojas de papa se rechaza como `NO_PAPA` y el 7,6 % de las de otras plantas se acepta
  como papa (`pipeline`).
- **Limitación:** PlantVillage son fotos de laboratorio (una hoja sobre fondo uniforme). Estas métricas no miden el
  desempeño en campo; para eso hacen falta fotos propias (P-02). `NO_PAPA` solo conoce 13 cultivos de PlantVillage.
- La latencia del reporte es la de la máquina donde corre (queda anotada). La meta de RNF-REN-02 aplica en la EC2.
- Las métricas se reportan aunque no alcancen la meta. No se ajusta el set de prueba para mejorar los números.

## Docker
```bash
docker build -t tizon-ia .
docker run --rm -e INTERNAL_API_TOKEN=... -v "$PWD/models:/models:ro" -p 8000:8000 tizon-ia
```
La imagen corre con un usuario sin privilegios y su `HEALTHCHECK` solo pasa si el modelo cargó. No incluye
`training/` ni `evaluation/`. En el sistema completo se levanta con `../deploy/docker-compose.yml`, sin publicar el
puerto 8000.

## CI/CD (GitHub Actions)
`.github/workflows/ci.yml` corre en cada push y PR a `dev`, `qa` y `main`: ruff, pytest (sin las pruebas de
integración, que requieren el modelo) y gitleaks. En cada push a `qa` o `main` publica la imagen
`ghcr.io/<owner>/tizon-ia` con las etiquetas `:qa` o `:main` y `:sha-<commit>`. La EC2 la descarga por *pull*
(`deploy/scripts/auto-update.sh`): GitHub nunca entra a la EC2. Si el paquete de GHCR es privado, la EC2
necesita un token de solo lectura (`read:packages`).
