# 🌿 Tizón Tardío — Servicio Python (FastAPI)

> **Autenticación interna (F1-IA-01):** `POST /analyze` exige el header `X-Internal-Token`
> igual a la variable `INTERNAL_API_TOKEN`. Fuera de `APP_ENV=local`, el servicio no arranca
> sin esa variable. Este servicio no se expone a Internet ni usa CORS.

Servicio de detección de Tizón Temprano y Tardío en cultivos de papa.
Recibe frames en Base64 desde Spring Boot, los analiza con TensorFlow
y retorna etiquetas de clasificación con bounding boxes en coordenadas relativas.

---

## Estructura de carpetas

```
tizon-python/
│
├── app/                          ← Código fuente principal
│   ├── __init__.py
│   ├── main.py                   ← FastAPI app + handler Lambda (Mangum)
│   ├── predictor.py              ← Lógica ML: carga modelo, detección, inferencia
│   └── schemas.py                ← Modelos Pydantic (request / response)
│
├── model/
│   └── .gitkeep                  ← Coloca aquí modelo_tizon_tardio.h5 para local
│                                   (el .h5 está en .gitignore — va a S3 en AWS)
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py               ← Tests de integración de endpoints
│   └── test_predictor.py         ← Tests unitarios de la lógica ML
│
├── scripts/
│   ├── upload_model.sh           ← Sube el modelo .h5 a S3
│   ├── deploy_aws.sh             ← Deploy completo a AWS Lambda
│   └── test_local.sh             ← Prueba rápida del servicio local
│
├── .env.example                  ← Plantilla de variables de entorno
├── .env                          ← Tu config local (NO commitear)
├── .gitignore
├── Dockerfile                    ← Multistage: "lambda" (prod) y "dev" (local)
├── docker-compose.yml            ← Levanta el servicio local con hot-reload
├── requirements.txt              ← Dependencias de producción
├── requirements-dev.txt          ← Dependencias de desarrollo y tests
├── template.yaml                 ← SAM template para despliegue en AWS Lambda
└── README.md
```

---

## ¿Cómo funciona?

```
Móvil (Kotlin)
    │  Frame JPEG → Base64
    ▼
Spring Boot  ──POST /analyze──▶  Este servicio (Python / FastAPI)
                                       │
                                       ├─ Detecta regiones verdes (OpenCV + HSV)
                                       ├─ Filtra vegetación real (textura + color)
                                       ├─ Clasifica con TensorFlow (.h5)
                                       └─ Retorna JSON con label + bboxes
```

**Respuesta de ejemplo:**
```json
{
  "label": "Tizon Tardio",
  "confidence": 0.92,
  "probabilities": {
    "Sano": 0.03,
    "Tizon Temprano": 0.05,
    "Tizon Tardio": 0.92
  },
  "bounding_boxes": [
    {
      "x": 0.12, "y": 0.20, "width": 0.35, "height": 0.40,
      "label": "Tizon Tardio",
      "confidence": 0.92,
      "probabilities": { "Sano": 0.03, "Tizon Temprano": 0.05, "Tizon Tardio": 0.92 }
    }
  ],
  "is_vegetation": true,
  "frame_width": 224,
  "frame_height": 224,
  "session_id": "uuid-de-la-sesion"
}
```

---

## OPCIÓN A — Local SIN Docker

### Requisitos previos
- Python 3.10, 3.11 o 3.12 instalado
- pip actualizado
- Tu archivo `modelo_tizon_tardio.h5` entrenado

### Paso 1 — Entrar al proyecto
```bash
cd tizon-python
ls   # debe ver: app/  model/  tests/  scripts/  Dockerfile  ...
```

### Paso 2 — Crear entorno virtual
```bash
python3 -m venv .venv

# Activar (Linux / Mac):
source .venv/bin/activate

# Activar (Windows PowerShell):
# .venv\Scripts\Activate.ps1

# Verificar: debe aparecer (.venv) al inicio del prompt
which python
```

### Paso 3 — Instalar dependencias
```bash
pip install -r requirements-dev.txt
# Tarda 3-5 minutos la primera vez (TensorFlow es grande)
```

### Paso 4 — Colocar el modelo
```bash
cp /ruta/a/tu/modelo_tizon_tardio.h5 model/

# Verificar:
ls -lh model/
# -rw-r--r-- ... 87M modelo_tizon_tardio.h5
```

### Paso 5 — Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env:
nano .env
```

Contenido mínimo del `.env` para local:
```bash
APP_ENV=local
LOCAL_MODEL_PATH=model/modelo_tizon_tardio.h5
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info
```

### Paso 6 — Arrancar el servidor
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Salida esperada:
```
INFO  | Iniciando Tizón Tardío Detection API
INFO  | Entorno: local
INFO  | Cargando modelo desde: model/modelo_tizon_tardio.h5
INFO  | Modelo cargado y listo. Servicio listo.
INFO  | Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Paso 7 — Verificar
```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"environment":"local",...}

# Documentación interactiva (abrir en navegador):
# http://localhost:8000/docs
```

---

## OPCIÓN B — Local CON Docker (recomendado)

No necesitas instalar Python ni dependencias en tu máquina.

### Requisitos
- Docker Desktop instalado y corriendo
- Tu archivo `modelo_tizon_tardio.h5`

### Pasos

```bash
# 1. Colocar el modelo
cp /ruta/a/tu/modelo_tizon_tardio.h5 model/

# 2. Copiar variables de entorno (valores por defecto ya son correctos para local)
cp .env.example .env

# 3. Levantar el servicio
docker-compose up

# Para correr en segundo plano:
docker-compose up -d

# Ver logs:
docker-compose logs -f tizon-python
```

Salida esperada (igual que sin Docker):
```
tizon-python-dev  | INFO  | Modelo cargado y listo. Servicio listo.
tizon-python-dev  | INFO  | Uvicorn running on http://0.0.0.0:8000
```

```bash
# Verificar:
curl http://localhost:8000/health

# Swagger UI: http://localhost:8000/docs
```

**Hot-reload activo:** cualquier cambio en `app/*.py` recarga el servidor automáticamente sin reiniciar Docker.

### Comandos Docker útiles
```bash
docker-compose down          # Detener
docker-compose build         # Reconstruir (si cambias requirements.txt)
docker-compose logs -f       # Ver logs en tiempo real
docker-compose exec tizon-python bash   # Entrar al contenedor
```

---

## Probar el endpoint /analyze

### Con curl (genera imagen de prueba)
```bash
# Generar frame de prueba en Base64
FRAME=$(python3 -c "
import cv2, numpy as np, base64
img = np.zeros((224, 224, 3), dtype=np.uint8)
img[:, :, 1] = 120  # Verde
noise = (np.random.rand(224,224,3) * 50).astype(np.uint8)
img = cv2.add(img, noise)
_, buf = cv2.imencode('.jpg', img)
print(base64.b64encode(buf.tobytes()).decode())
")

# Enviar al endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"frame_b64\": \"$FRAME\", \"session_id\": \"test-001\"}" \
  | python3 -m json.tool
```

### Con una imagen real del disco
```bash
FRAME=$(base64 -i /ruta/a/foto_planta.jpg)

curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"frame_b64\": \"$FRAME\"}" \
  | python3 -m json.tool
```

### Script automático de tests
```bash
chmod +x scripts/test_local.sh
./scripts/test_local.sh
```

---

## Correr los tests

```bash
# Activar entorno virtual primero (si no usas Docker)
source .venv/bin/activate

# Tests unitarios — NO requieren el modelo .h5
pytest tests/ -v -m "not integration"

# Tests de integración — SÍ requieren modelo en model/
pytest tests/ -v -m integration

# Todos los tests
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ -v --cov=app --cov-report=term-missing

# Dentro de Docker:
docker-compose exec tizon-python pytest tests/ -v -m "not integration"
```

---

## DESPLIEGUE EN AWS

### Arquitectura

```
Móvil ──▶ Spring Boot (EC2/ECS)
                │
                │  POST /analyze  (JSON + Base64)
                ▼
         API Gateway (REST)
                │
                ▼
         Lambda Function
         (imagen Docker en ECR)
                │
                ├──▶ S3: descarga modelo .h5 (solo en cold start)
                └──▶ Retorna JSON con label + bboxes
```

### Requisitos previos AWS

```bash
# 1. Instalar AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
aws --version   # aws-cli/2.x.x

# 2. Instalar AWS SAM CLI
pip install aws-sam-cli
sam --version   # SAM CLI, version 1.x.x

# 3. Docker instalado y corriendo
docker --version

# 4. Configurar credenciales AWS
aws configure
#   AWS Access Key ID:     [tu key]
#   AWS Secret Access Key: [tu secret]
#   Default region name:   us-east-1
#   Default output format: json

# Verificar que funciona:
aws sts get-caller-identity
# Debe mostrar Account ID, UserId y Arn
```

### Paso 1 — Primer despliegue completo

```bash
chmod +x scripts/deploy_aws.sh scripts/upload_model.sh scripts/test_local.sh

# Este script hace todo:
#   1. Crea repositorio ECR (si no existe)
#   2. Login en ECR
#   3. Build imagen Docker (target: lambda)
#   4. Push imagen a ECR
#   5. sam deploy → crea Lambda + API Gateway + S3 en CloudFormation
./scripts/deploy_aws.sh
```

Al terminar verás:
```
══════════════════════════════════════════
 ✅ Despliegue completado
══════════════════════════════════════════
  API URL:      https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod
  Model Bucket: tizon-tardio-models-123456789012-prod

  → Prueba el health check:
    curl https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod/health

  → Sube el modelo:
    ./scripts/upload_model.sh model/modelo_tizon_tardio.h5

  → Configura en Spring Boot:
    PYTHON_SERVICE_URL=https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod
```

### Paso 2 — Subir el modelo a S3

```bash
# Solo necesario la primera vez, o cuando re-entrenes el modelo
./scripts/upload_model.sh model/modelo_tizon_tardio.h5
```

### Paso 3 — Verificar en AWS

```bash
API_URL="https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod"

# Health check (primera llamada tarda 10-15s por cold start del modelo)
curl $API_URL/health
# {"status":"ok","model_loaded":true,"environment":"aws",...}

# Test completo contra AWS
API_URL=$API_URL bash scripts/test_local.sh
```

### Paso 4 — Configurar Spring Boot

En `application.yml` de Spring Boot:
```yaml
python:
  service:
    base-url: https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod
    timeout-seconds: 30
```

O como variable de entorno en EC2/ECS:
```bash
PYTHON_SERVICE_URL=https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod
```

### Re-desplegar después de cambios en el código

```bash
# Cambios en código Python → rebuild + redeploy
./scripts/deploy_aws.sh

# Solo cambio de modelo (re-entrenamiento)
./scripts/upload_model.sh /ruta/nuevo/modelo.h5
# La próxima invocación Lambda descarga el modelo nuevo automáticamente
```

---

## Variables de entorno — referencia completa

| Variable | Local | AWS | Descripción |
|---|---|---|---|
| `APP_ENV` | `local` | `aws` | Modo de carga del modelo |
| `LOCAL_MODEL_PATH` | `model/modelo_tizon_tardio.h5` | — | Ruta local del modelo |
| `MODEL_S3_BUCKET` | — | `tizon-tardio-models-...` | Bucket S3 del modelo |
| `MODEL_S3_KEY` | — | `models/modelo_tizon_tardio.h5` | Clave S3 del modelo |
| `AWS_DEFAULT_REGION` | `us-east-1` | `us-east-1` | Región AWS |
| `HOST` | `0.0.0.0` | — | Host uvicorn (solo local) |
| `PORT` | `8000` | — | Puerto uvicorn (solo local) |
| `RELOAD` | `true` | — | Hot-reload (solo local) |
| `LOG_LEVEL` | `info` | `info` | Nivel de logging |

---

## Problemas frecuentes

**"Modelo no encontrado en model/"**
```bash
cp /ruta/modelo.h5 model/modelo_tizon_tardio.h5
```

**"No se pudo decodificar la imagen"**
El Base64 enviado incluye el prefijo `data:image/jpeg;base64,`. Enviar solo el Base64 puro.

**Lambda OOM (Out of Memory)**
Subir `LambdaMemoryMB` a `3008` en `template.yaml` y re-desplegar.

**Cold start muy lento (>15s)**
Activar Provisioned Concurrency:
```bash
aws lambda put-provisioned-concurrency-config \
  --function-name tizon-detection-prod \
  --qualifier LATEST \
  --provisioned-concurrent-executions 1
```
Costo extra: ~$15 USD/mes.

**"Access Denied" descargando modelo desde S3**
Verificar que el bucket en S3 es el mismo que creó SAM:
```bash
aws cloudformation describe-stacks \
  --stack-name tizon-tardio-stack \
  --query "Stacks[0].Outputs[?OutputKey=='ModelBucketName'].OutputValue" \
  --output text
```

---

## Costos AWS estimados

| Servicio | Uso mensual estimado | Costo |
|---|---|---|
| Lambda 2048 MB | 500K invocaciones × 2s | ~$8 USD |
| API Gateway | 500K requests | ~$1.75 USD |
| S3 modelo | ~100 MB almacenamiento | ~$0.01 USD |
| ECR imagen | ~2 GB almacenamiento | ~$0.20 USD |
| **Total** | | **~$10 USD/mes** |

## CI/CD (GitHub Actions)
`.github/workflows/ci.yml` corre en cada push y PR a `dev`, `qa` y `main`: ruff, pytest (sin las pruebas de integración, que requieren el `.h5`) y gitleaks.
En cada push a `qa` o `main` publica la imagen `ghcr.io/<owner>/tizon-ia` con las etiquetas `:qa` o `:main`
y `:sha-<commit>`. La EC2 la descarga por *pull* (`deploy/scripts/auto-update.sh`): GitHub nunca entra a la EC2.
Si el paquete de GHCR es privado, la EC2 necesita un token de solo lectura (`read:packages`).

