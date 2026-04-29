#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# test_local.sh — Prueba rápida del servicio corriendo en local
#
# Uso:
#   ./scripts/test_local.sh
#
# El servicio debe estar corriendo en http://localhost:8000
# (via docker-compose up o uvicorn app.main:app)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

BASE_URL="${API_URL:-http://localhost:8000}"

echo "══════════════════════════════════════════"
echo " Test rápido: Tizón Tardío API"
echo " URL: $BASE_URL"
echo "══════════════════════════════════════════"

# ── Test 1: Health check ──────────────────────────────────────────────────
echo ""
echo "[ 1/3 ] GET /health"
HEALTH=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "${BASE_URL}/health")
HTTP_STATUS=$(echo "$HEALTH" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$HEALTH" | grep -v "HTTP_STATUS")

echo "  Status: $HTTP_STATUS"
echo "  Body:   $BODY"

if [ "$HTTP_STATUS" != "200" ]; then
    echo "  ❌ Health check falló (esperado 200)"
    exit 1
fi
echo "  ✅ Health check OK"

# ── Test 2: Analyze con imagen de prueba ─────────────────────────────────
echo ""
echo "[ 2/3 ] POST /analyze (imagen verde de prueba)"

# Generar imagen de prueba en Python y encodear a Base64
FRAME_B64=$(python3 -c "
import cv2, numpy as np, base64
img = np.zeros((224, 224, 3), dtype=np.uint8)
img[:, :, 0] = 30   # B
img[:, :, 1] = 120  # G — verde dominante
img[:, :, 2] = 40   # R
# Agregar ruido para simular textura de hoja
noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
img = cv2.add(img, noise)
_, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
print(base64.b64encode(buf.tobytes()).decode('utf-8'))
")

ANALYZE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -X POST "${BASE_URL}/analyze" \
    -H "Content-Type: application/json" \
    -d "{\"frame_b64\": \"${FRAME_B64}\", \"session_id\": \"test-local-001\"}")

HTTP_STATUS=$(echo "$ANALYZE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$ANALYZE" | grep -v "HTTP_STATUS")

echo "  Status: $HTTP_STATUS"
echo "  Body:   $(echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY")"

if [ "$HTTP_STATUS" != "200" ]; then
    echo "  ❌ Analyze falló (esperado 200)"
    exit 1
fi

LABEL=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['label'])")
echo "  → Label detectado: $LABEL"
echo "  ✅ Analyze OK"

# ── Test 3: Base64 inválido → debe retornar error ─────────────────────────
echo ""
echo "[ 3/3 ] POST /analyze (Base64 inválido — debe fallar)"
INVALID=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -X POST "${BASE_URL}/analyze" \
    -H "Content-Type: application/json" \
    -d '{"frame_b64": "esto_no_es_base64_valido!!!"}')

HTTP_STATUS=$(echo "$INVALID" | grep "HTTP_STATUS" | cut -d: -f2)
echo "  Status: $HTTP_STATUS"

if [ "$HTTP_STATUS" == "200" ]; then
    echo "  ❌ Debería haber fallado con 400 o 422"
    exit 1
fi
echo "  ✅ Validación de Base64 funciona correctamente"

# ── Resumen ───────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " ✅ Todos los tests pasaron"
echo "══════════════════════════════════════════"