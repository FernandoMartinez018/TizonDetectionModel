# Entrenamiento del modelo (F4-IA-05) — 2026-10-24

- Modelo: `mobilenetv2-4c-02b1cb27-20261024` (`modelo_tizon_4c.keras`, sha256 `02b1cb276fed2ab35e709dfe86ddc5d39563a3d920119bf90d1f753d006f0416`)
- Datos: PlantVillage (github.com/spMohanty/PlantVillage-Dataset, raw/color): papa completa (sana, tizón temprano, tizón tardío) y, como NO_PAPA, 30 hojas por cada tipo de otros 13 cultivos
- Partición (semilla 42, `training/split.csv`): train: SANO 107, TIZON_TEMPRANO 700, TIZON_TARDIO 700, NO_PAPA 735; val: SANO 23, TIZON_TEMPRANO 150, TIZON_TARDIO 150, NO_PAPA 158; test: SANO 22, TIZON_TEMPRANO 150, TIZON_TARDIO 150, NO_PAPA 157
- MobileNetV2 (ImageNet) con el preprocesamiento dentro del modelo; aumento de datos y pesos por clase (SANO 5,24, TIZON_TEMPRANO 0,80, TIZON_TARDIO 0,80, NO_PAPA 0,76).
- Épocas: 13 con la base congelada y 4 de ajuste fino.

## Prueba (479 imágenes que el modelo no vio)

Accuracy global: **95,6 %**. Tasa de error: 4,4 %.

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| SANO | 22 | 77,8 % | 95,5 % | 85,7 % | 98,7 % |
| TIZON_TEMPRANO | 150 | 98,6 % | 97,3 % | 98,0 % | 99,4 % |
| TIZON_TARDIO | 150 | 94,2 % | 97,3 % | 95,7 % | 97,3 % |
| NO_PAPA | 157 | 97,3 % | 92,4 % | 94,8 % | 98,8 % |

Matriz de confusión (filas: real, columnas: predicho):

| | SANO | TIZON_TEMPRANO | TIZON_TARDIO | NO_PAPA |
|---|---|---|---|---|
| SANO | 21 | 0 | 0 | 1 |
| TIZON_TEMPRANO | 0 | 146 | 4 | 0 |
| TIZON_TARDIO | 1 | 0 | 146 | 3 |
| NO_PAPA | 5 | 2 | 5 | 145 |

Limitación: PlantVillage son fotos de laboratorio (una hoja sobre fondo uniforme). Estas métricas no miden el desempeño en campo; para eso hace falta el set de fotos propias (P-02).
