# Entrenamiento del modelo de 3 clases (F4-IA-05) — 2026-10-24

- Modelo: `mobilenetv2-3c-46f03ce2-20261024` (`modelo_tizon_3c.keras`, sha256 `46f03ce28e9a8a13f2d2605ce38911fc145f2f63c88ca3042f529f9193e53348`)
- Datos: PlantVillage, papa (Potato___healthy, Potato___Early_blight, Potato___Late_blight), github.com/spMohanty/PlantVillage-Dataset raw/color
- Partición (semilla 42, `training/split.csv`): train: SANO 107, TIZON_TEMPRANO 700, TIZON_TARDIO 700; val: SANO 23, TIZON_TEMPRANO 150, TIZON_TARDIO 150; test: SANO 22, TIZON_TEMPRANO 150, TIZON_TARDIO 150
- MobileNetV2 (ImageNet) con el preprocesamiento dentro del modelo; aumento de datos y pesos por clase (SANO 4,69, TIZON_TEMPRANO 0,72, TIZON_TARDIO 0,72).
- Épocas: 15 con la base congelada y 6 de ajuste fino.

## Prueba (322 imágenes que el modelo no vio)

Accuracy global: **98,1 %**. Tasa de error: 1,9 %.

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| SANO | 22 | 95,5 % | 95,5 % | 95,5 % | 99,7 % |
| TIZON_TEMPRANO | 150 | 99,3 % | 98,0 % | 98,7 % | 99,4 % |
| TIZON_TARDIO | 150 | 97,4 % | 98,7 % | 98,0 % | 97,7 % |

Matriz de confusión (filas: real, columnas: predicho):

| | SANO | TIZON_TEMPRANO | TIZON_TARDIO |
|---|---|---|---|
| SANO | 21 | 0 | 1 |
| TIZON_TEMPRANO | 0 | 147 | 3 |
| TIZON_TARDIO | 1 | 1 | 148 |

Limitación: PlantVillage son fotos de laboratorio (una hoja sobre fondo uniforme). Estas métricas no miden el desempeño en campo; para eso hace falta el set de fotos propias (P-02).
