# Evaluación del modelo — modo `classifier` — campo-plantdoc — 2026-10-24

- Modelo: `mobilenetv2-4c-02b1cb27-20261024` (`modelo_tizon_4c.keras`, sha256 `02b1cb276fed2ab35e709dfe86ddc5d39563a3d920119bf90d1f753d006f0416`)
- Set de prueba: `eval_field` — origen: PlantDoc (fotos de campo, CC-BY-4.0): toda su partición test (papa y otras especies) y el 30 % de las hojas de papa de su partición train, apartado por grupos de casi duplicados. No hay papa sana. Ninguna foto se usó para entrenar. Se retiraron 6 fotos que PlantDoc repite en su partición train (5 con otra etiqueta: la misma imagen como tizón tardío y como temprano, o como tomate)
- Imágenes: TIZON_TEMPRANO 38, TIZON_TARDIO 32, NO_PAPA 204 (total 274)
- Máquina: Windows-11-10.0.26200-SP0 · 14 CPU
- Accuracy y tasa de error: globales. Precision, Recall, F1 y Especificidad: TIZON_TARDIO contra el resto. Imagen completa, sin filtro.

## Métricas frente a las metas del PDF

| Métrica | Meta | Medido | Cumple |
|---|---|---|---|
| Accuracy | ≥ 90,0 % | 74,5 % | ✘ |
| Precision | ≥ 90,0 % | 20,0 % | ✘ |
| Recall | ≥ 90,0 % | 9,4 % | ✘ |
| F1 | ≥ 90,0 % | 12,8 % | ✘ |
| Especificidad | ≥ 85,0 % | 95,0 % | ✔ |
| Tasa de error | ≤ 10,0 % | 25,5 % | ✘ |
| Latencia p95 (inferencia) | < 1.500 ms en la EC2 | 323,7 ms (media 243,9 ms) | ✔ |

## Por clase

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| TIZON_TEMPRANO | 38 | 55,6 % | 13,2 % | 21,3 % | 98,3 % |
| TIZON_TARDIO | 32 | 20,0 % | 9,4 % | 12,8 % | 95,0 % |
| NO_PAPA | 204 | 78,4 % | 96,1 % | 86,3 % | 22,9 % |

## Matriz de confusión (filas: real, columnas: predicho)

| | TIZON_TEMPRANO | TIZON_TARDIO | NO_PAPA |
|---|---|---|---|
| TIZON_TEMPRANO | 5 | 7 | 26 |
| TIZON_TARDIO | 1 | 3 | 28 |
| NO_PAPA | 3 | 5 | 196 |

## ¿Es papa?

- Hojas de papa rechazadas como NO_PAPA (no se analizan): 77,1 %.
- Hojas de otras plantas aceptadas como papa (se analizarían por error): 3,9 %.

## Integridad del set de prueba

- No se comparó con el set de entrenamiento (usa `--train`).

## Errores (70; primeros 20)

| Archivo | Real | Predicho |
|---|---|---|
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__000_3023.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__001_backus_056_potato_blight.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__003_early_blight_or_target_s.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__004_earlyblightpotato.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__005_potato_blight_phytophtho.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__007_potatobd001.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__001_0042241.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__010_18028_1.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__012_20100816earlyblight.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__019_3785483636_d888419c98_o.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__025_87A_Pot_Early.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__036_Early_Blight_of_Tomato17.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__038_IMG_0014.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__039_IMG_0020.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__043_PHYTIN23_img.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__047_Potato_blight_on_leaf.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__048_Potato_earlyblight2.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__052_agriculture_potato_early.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__056_blight_leaf_top.jpg` | TIZON_TEMPRANO | NO_PAPA |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__057_blighted_potatoe_leave.jpg` | TIZON_TEMPRANO | NO_PAPA |

Se reporta el valor medido aunque no alcance la meta; el análisis de causas va en el documento (05-requerimientos-no-funcionales).
