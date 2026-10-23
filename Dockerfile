# Servicio de IA (FastAPI + TensorFlow CPU). Una sola etapa, usuario sin privilegios.
# El modelo NO va en la imagen: se monta en /models (.h5 + model_meta.json), así se actualiza
# reemplazando los archivos y reiniciando el contenedor, sin reconstruir (RNF-MAN-02).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    APP_ENV=production \
    MODEL_PATH=/models/modelo_tizon_tardio.h5 \
    MODEL_META_PATH=/models/model_meta.json

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/

RUN useradd --system --uid 10001 --no-create-home tizon
USER tizon

EXPOSE 8000

# Sano solo si el modelo cargó: así docker compose y el auto-update detectan un modelo roto.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import json,urllib.request,sys; h=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=4)); sys.exit(0 if h['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
