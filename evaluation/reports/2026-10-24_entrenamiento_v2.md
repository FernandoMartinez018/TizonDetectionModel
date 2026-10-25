# Entrenamiento del modelo (F4-IA-05) — 2026-10-24

- Modelo: `mobilenetv2-4c-a7c205d2-20261024` (`modelo_tizon_4c_v2.keras`, sha256 `a7c205d2c8a85dc9b00d435ef270b5d3e9b3f232b9e73e59cf2fb399e7a42da6`)
- Datos: PlantVillage (github.com/spMohanty/PlantVillage-Dataset, raw/color): papa completa y 1.490 hojas de otros 13 cultivos como NO_PAPA; PlantDoc (github.com/pratikkayal/PlantDoc-Dataset, CC-BY-4.0, fotos de campo): 137 hojas de papa con tizón (repetidas 3 veces) y 349 de otras especies como NO_PAPA. Todo PlantDoc test y el 30 % de la papa de PlantDoc train quedan fuera para la prueba de campo
- Partición (semilla 42, `training/split.csv`; 3202 imágenes conservan la partición anterior y 1200 se agregaron): train: SANO 107, TIZON_TEMPRANO 880, TIZON_TARDIO 871, NO_PAPA 1406; val: SANO 23, TIZON_TEMPRANO 180, TIZON_TARDIO 180, NO_PAPA 276; test: SANO 22, TIZON_TEMPRANO 150, TIZON_TARDIO 150, NO_PAPA 157
- MobileNetV2 (ImageNet) con el preprocesamiento dentro del modelo; aumento de datos y pesos por clase (SANO 7,63, TIZON_TEMPRANO 0,93, TIZON_TARDIO 0,94, NO_PAPA 0,58).
- Épocas: 12 con la base congelada y 8 de ajuste fino.

## Prueba (479 imágenes que el modelo no vio)

Accuracy global: **96,0 %**. Tasa de error: 4,0 %.

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| SANO | 22 | 75,0 % | 95,5 % | 84,0 % | 98,5 % |
| TIZON_TEMPRANO | 150 | 97,4 % | 98,7 % | 98,0 % | 98,8 % |
| TIZON_TARDIO | 150 | 96,7 % | 96,7 % | 96,7 % | 98,5 % |
| NO_PAPA | 157 | 98,0 % | 93,0 % | 95,4 % | 99,1 % |

Matriz de confusión (filas: real, columnas: predicho):

| | SANO | TIZON_TEMPRANO | TIZON_TARDIO | NO_PAPA |
|---|---|---|---|---|
| SANO | 21 | 0 | 0 | 1 |
| TIZON_TEMPRANO | 0 | 148 | 2 | 0 |
| TIZON_TARDIO | 3 | 0 | 145 | 2 |
| NO_PAPA | 4 | 4 | 3 | 146 |

Limitación: esta prueba es la partición test de training/split.csv (fotos de laboratorio de PlantVillage). El desempeño con fotos de campo se mide aparte (reportes campo-*, P-02).
