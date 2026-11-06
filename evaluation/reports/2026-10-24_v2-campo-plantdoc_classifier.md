# Evaluación del modelo — modo `classifier` — v2-campo-plantdoc — 2026-10-24

- Modelo: `mobilenetv2-4c-a7c205d2-20261024` (`modelo_tizon_4c_v2.keras`, sha256 `a7c205d2c8a85dc9b00d435ef270b5d3e9b3f232b9e73e59cf2fb399e7a42da6`)
- Set de prueba: `eval_field` — origen: PlantDoc (fotos de campo, CC-BY-4.0): toda su partición test (papa y otras especies) y el 30 % de las hojas de papa de su partición train, apartado por grupos de casi duplicados. No hay papa sana. Ninguna foto se usó para entrenar. Se retiraron 6 fotos que PlantDoc repite en su partición train (5 con otra etiqueta: la misma imagen como tizón tardío y como temprano, o como tomate)
- Imágenes: TIZON_TEMPRANO 38, TIZON_TARDIO 32, NO_PAPA 204 (total 274)
- Máquina: Windows-11-10.0.26200-SP0 · 14 CPU
- Accuracy y tasa de error: globales. Precision, Recall, F1 y Especificidad: TIZON_TARDIO contra el resto. Imagen completa, sin filtro.

## Métricas frente a las metas del PDF

| Métrica | Meta | Medido | Cumple |
|---|---|---|---|
| Accuracy | ≥ 90,0 % | 73,7 % | ✘ |
| Precision | ≥ 90,0 % | 47,5 % | ✘ |
| Recall | ≥ 90,0 % | 59,4 % | ✘ |
| F1 | ≥ 90,0 % | 52,8 % | ✘ |
| Especificidad | ≥ 85,0 % | 91,3 % | ✔ |
| Tasa de error | ≤ 10,0 % | 26,3 % | ✘ |
| Latencia p95 (inferencia) | < 1.500 ms en la EC2 | 305,5 ms (media 231,5 ms) | ✔ |

## Por clase

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| TIZON_TEMPRANO | 38 | 32,1 % | 47,4 % | 38,3 % | 83,9 % |
| TIZON_TARDIO | 32 | 47,5 % | 59,4 % | 52,8 % | 91,3 % |
| NO_PAPA | 204 | 92,7 % | 80,9 % | 86,4 % | 81,4 % |

## Matriz de confusión (filas: real, columnas: predicho)

| | TIZON_TEMPRANO | TIZON_TARDIO | NO_PAPA |
|---|---|---|---|
| TIZON_TEMPRANO | 18 | 12 | 8 |
| TIZON_TARDIO | 8 | 19 | 5 |
| NO_PAPA | 30 | 9 | 165 |

## ¿Es papa?

- Hojas de papa rechazadas como NO_PAPA (no se analizan): 18,6 %.
- Hojas de otras plantas aceptadas como papa (se analizarían por error): 19,1 %.

## Integridad del set de prueba

- Duplicados exactos con el entrenamiento: 0.
- Casi duplicados (pHash, distancia ≤ 6): 0. Se revisan a mano y se retiran del set de prueba si son la misma hoja.

## Errores (72; primeros 20)

| Archivo | Real | Predicho |
|---|---|---|
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__000_3023.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__001_backus_056_potato_blight.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__004_earlyblightpotato.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__005_potato_blight_phytophtho.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__001_0042241.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__010_18028_1.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__019_3785483636_d888419c98_o.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__025_87A_Pot_Early.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__038_IMG_0014.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__039_IMG_0020.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__047_Potato_blight_on_leaf.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__048_Potato_earlyblight2.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__056_blight_leaf_top.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__057_blighted_potatoe_leave.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__064_early_blight_courtesy_Me.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__074_fac66s01a.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__079_maxresdefault.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__081_potato_blight_damage_on_.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__082_potato_blight_leaf.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__087_potato_blight_phytophtho.jpg` | TIZON_TEMPRANO | NO_PAPA |

Se reporta el valor medido aunque no alcance la meta; el análisis de causas va en el documento (05-requerimientos-no-funcionales).
