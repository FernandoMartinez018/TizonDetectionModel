# Evaluación del modelo — modo `pipeline` — campo-plantdoc — 2026-10-24

- Modelo: `mobilenetv2-4c-02b1cb27-20261024` (`modelo_tizon_4c.keras`, sha256 `02b1cb276fed2ab35e709dfe86ddc5d39563a3d920119bf90d1f753d006f0416`)
- Set de prueba: `eval_field` — origen: PlantDoc (fotos de campo, CC-BY-4.0): toda su partición test (papa y otras especies) y el 30 % de las hojas de papa de su partición train, apartado por grupos de casi duplicados. No hay papa sana. Ninguna foto se usó para entrenar. Se retiraron 6 fotos que PlantDoc repite en su partición train (5 con otra etiqueta: la misma imagen como tizón tardío y como temprano, o como tomate)
- Imágenes: TIZON_TEMPRANO 38, TIZON_TARDIO 32, NO_PAPA 204 (total 274)
- Máquina: Windows-11-10.0.26200-SP0 · 14 CPU
- Accuracy y tasa de error: globales. Precision, Recall, F1 y Especificidad: TIZON_TARDIO contra el resto. NOT_POTATO cuenta como NO_PAPA; NO_VEGETATION, como SANO en hojas de papa.

## Métricas frente a las metas del PDF

| Métrica | Meta | Medido | Cumple |
|---|---|---|---|
| Accuracy | ≥ 90,0 % | 73,7 % | ✘ |
| Precision | ≥ 90,0 % | 11,8 % | ✘ |
| Recall | ≥ 90,0 % | 6,2 % | ✘ |
| F1 | ≥ 90,0 % | 8,2 % | ✘ |
| Especificidad | ≥ 85,0 % | 93,8 % | ✔ |
| Tasa de error | ≤ 10,0 % | 26,3 % | ✘ |
| Latencia p95 (IA completa) | < 1.500 ms en la EC2 | 336,7 ms (media 253,6 ms) | ✔ |

## Por clase

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| SANO | 0 | 0,0 % | 0,0 % | 0,0 % | 98,9 % |
| TIZON_TEMPRANO | 38 | 62,5 % | 13,2 % | 21,7 % | 98,7 % |
| TIZON_TARDIO | 32 | 11,8 % | 6,2 % | 8,2 % | 93,8 % |
| NO_PAPA | 204 | 79,3 % | 95,6 % | 86,7 % | 27,1 % |

## Matriz de confusión (filas: real, columnas: predicho)

| | SANO | TIZON_TEMPRANO | TIZON_TARDIO | NO_PAPA |
|---|---|---|---|---|
| SANO | 0 | 0 | 0 | 0 |
| TIZON_TEMPRANO | 1 | 5 | 9 | 23 |
| TIZON_TARDIO | 2 | 0 | 2 | 28 |
| NO_PAPA | 0 | 3 | 6 | 195 |

## ¿Es papa?

- Hojas de papa rechazadas como NO_PAPA (no se analizan): 72,9 %.
- Hojas de otras plantas aceptadas como papa (se analizarían por error): 4,4 %.

## NO_VEGETATION (filtro de vegetación)

- TIZON_TEMPRANO: 1, TIZON_TARDIO: 2, NO_PAPA: 8.
- TIZON_TARDIO en NO_VEGETATION: 6,2 % (meta de F4-IA-03 ≤ 5,0 %: ✘).

## Integridad del set de prueba

- No se comparó con el set de entrenamiento (usa `--train`).

## Errores (72; primeros 20)

| Archivo | Real | Predicho |
|---|---|---|
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__000_3023.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__001_backus_056_potato_blight.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__003_early_blight_or_target_s.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__004_earlyblightpotato.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__005_potato_blight_phytophtho.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_test_Potato_leaf_early_blight__007_potatobd001.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__001_0042241.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__010_18028_1.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__012_20100816earlyblight.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__019_3785483636_d888419c98_o.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__025_87A_Pot_Early.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__036_Early_Blight_of_Tomato17.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__038_IMG_0014.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__039_IMG_0020.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__043_PHYTIN23_img.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__047_Potato_blight_on_leaf.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__048_Potato_earlyblight2.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__052_agriculture_potato_early.jpg` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__056_blight_leaf_top.jpg` | TIZON_TEMPRANO | NOT_POTATO |
| `TIZON_TEMPRANO/pd_train_Potato_leaf_early_blight__057_blighted_potatoe_leave.jpg` | TIZON_TEMPRANO | NOT_POTATO |

Se reporta el valor medido aunque no alcance la meta; el análisis de causas va en el documento (05-requerimientos-no-funcionales).
