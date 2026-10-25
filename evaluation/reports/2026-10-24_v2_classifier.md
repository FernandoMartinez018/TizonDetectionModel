# Evaluación del modelo — modo `classifier` — v2 — 2026-10-24

- Modelo: `mobilenetv2-4c-a7c205d2-20261024` (`modelo_tizon_4c_v2.keras`, sha256 `a7c205d2c8a85dc9b00d435ef270b5d3e9b3f232b9e73e59cf2fb399e7a42da6`)
- Set de prueba: `train_v2` (partición `test` de `training/split.csv`) — origen: PlantVillage (github.com/spMohanty/PlantVillage-Dataset, raw/color), partición test de training/split.csv: las mismas 479 imágenes de la prueba del modelo 4c
- Imágenes: SANO 22, TIZON_TEMPRANO 150, TIZON_TARDIO 150, NO_PAPA 157 (total 479)
- Máquina: Windows-11-10.0.26200-SP0 · 14 CPU
- Accuracy y tasa de error: globales. Precision, Recall, F1 y Especificidad: TIZON_TARDIO contra el resto. Imagen completa, sin filtro.

## Métricas frente a las metas del PDF

| Métrica | Meta | Medido | Cumple |
|---|---|---|---|
| Accuracy | ≥ 90,0 % | 96,0 % | ✔ |
| Precision | ≥ 90,0 % | 96,7 % | ✔ |
| Recall | ≥ 90,0 % | 96,7 % | ✔ |
| F1 | ≥ 90,0 % | 96,7 % | ✔ |
| Especificidad | ≥ 85,0 % | 98,5 % | ✔ |
| Tasa de error | ≤ 10,0 % | 4,0 % | ✔ |
| Latencia p95 (inferencia) | < 1.500 ms en la EC2 | 262,1 ms (media 213,2 ms) | ✔ |

## Por clase

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| SANO | 22 | 75,0 % | 95,5 % | 84,0 % | 98,5 % |
| TIZON_TEMPRANO | 150 | 97,4 % | 98,7 % | 98,0 % | 98,8 % |
| TIZON_TARDIO | 150 | 96,7 % | 96,7 % | 96,7 % | 98,5 % |
| NO_PAPA | 157 | 98,0 % | 93,0 % | 95,4 % | 99,1 % |

## Matriz de confusión (filas: real, columnas: predicho)

| | SANO | TIZON_TEMPRANO | TIZON_TARDIO | NO_PAPA |
|---|---|---|---|---|
| SANO | 21 | 0 | 0 | 1 |
| TIZON_TEMPRANO | 0 | 148 | 2 | 0 |
| TIZON_TARDIO | 3 | 0 | 145 | 2 |
| NO_PAPA | 4 | 4 | 3 | 146 |

## ¿Es papa?

- Hojas de papa rechazadas como NO_PAPA (no se analizan): 0,9 %.
- Hojas de otras plantas aceptadas como papa (se analizarían por error): 7,0 %.

## Integridad del set de prueba

- Duplicados exactos con el entrenamiento: 0.
- Casi duplicados (pHash, distancia ≤ 6): 5. Se revisan a mano y se retiran del set de prueba si son la misma hoja.
  - `SANO/57e262fc-e0ba-44db-a967-5ba6f1826e49___RS_HL 1897.JPG` ↔ `TIZON_TARDIO/5759c134-1965-4dd2-b71a-4eef4048f962___RS_LB 3975.JPG` (6)
  - `SANO/5e19b1f4-fd65-4c4d-bcd2-18c8c519eddb___RS_HL 1965.JPG` ↔ `NO_PAPA/pv_Cherry_including_sour_health__ea1daecd_c98f_4c73_8c21_6b9aaf757b67.jpg` (6)
  - `TIZON_TARDIO/0f243024-b1fa-4f96-ac7e-ecaf6dc5bc37___RS_LB 4925.JPG` ↔ `SANO/1dcfeaa9-006d-470c-b3e5-d67609d07d4e___RS_HL 1808.JPG` (6)
  - `TIZON_TARDIO/da16cb94-4f1d-4390-b446-19eac2110115___RS_LB 2912.JPG` ↔ `NO_PAPA/pv_Soybean_healthy__2d011336_97ff_4a75_ad31_f7ed3591eb61.jpg` (6)
  - `TIZON_TARDIO/ea06da1e-81c9-4e2e-8266-1f3d56914ff3___RS_LB 2701.JPG` ↔ `NO_PAPA/Soybean_healthy__114e4945-280e-4acd-821c-73d01552fbf7.jpg` (6)

## Errores (19; primeros 20)

| Archivo | Real | Predicho |
|---|---|---|
| `SANO/ec6e43d3-512d-4908-99f2-c9e2a0be54be___RS_HL 4202.JPG` | SANO | NO_PAPA |
| `TIZON_TEMPRANO/31d0ecc8-045d-4a4a-ac33-431556c83e4c___RS_Early.B 8692.JPG` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/d998c614-017c-4f90-8e39-f8c500d21218___RS_Early.B 8292.JPG` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TARDIO/57a137da-7424-4a9d-9e28-bce63decbdcf___RS_LB 2761.JPG` | TIZON_TARDIO | SANO |
| `TIZON_TARDIO/5f6e341d-3a02-4bc5-ae02-c0a343a835ea___RS_LB 4882.JPG` | TIZON_TARDIO | SANO |
| `TIZON_TARDIO/84e85005-c647-4381-8762-c487074c9571___RS_LB 5186.JPG` | TIZON_TARDIO | NO_PAPA |
| `TIZON_TARDIO/d9817cc3-ef03-41e5-8892-1fddd27b7bcc___RS_LB 4878.JPG` | TIZON_TARDIO | SANO |
| `TIZON_TARDIO/dbbd22b5-a565-470a-9e72-2c7d7c26dbe5___RS_LB 4932.JPG` | TIZON_TARDIO | NO_PAPA |
| `NO_PAPA/Blueberry_healthy__811c4a45-2217-418f-834d-e782bd8f9f57.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Corn_maize_Common_rust__RS_Rust 1660.JPG.jpg` | NO_PAPA | TIZON_TEMPRANO |
| `NO_PAPA/Soybean_healthy__045dba20-7120-47d2-b7dc-86cdbe77580b.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Soybean_healthy__1b957666-ac0a-4f3d-a7f7-627a53972c9b.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Soybean_healthy__29ca525d-7049-4262-b65a-dcfe1a063035.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Strawberry_Leaf_scorch__c5aa66d1-22cf-48af-a622-a9fbbd0d2c00.jpg` | NO_PAPA | TIZON_TEMPRANO |
| `NO_PAPA/Tomato_Early_blight__7b32caa4-0d67-4b84-a1bf-cf9dffd7ffe5.jpg` | NO_PAPA | TIZON_TEMPRANO |
| `NO_PAPA/Tomato_Early_blight__b5c94157-7110-420f-99fc-040594d6ff12.jpg` | NO_PAPA | TIZON_TEMPRANO |
| `NO_PAPA/Tomato_Late_blight__15e1e8d7-e05a-4dc1-b872-364b15f004f0.jpg` | NO_PAPA | TIZON_TARDIO |
| `NO_PAPA/Tomato_Late_blight__34ce281a-ca92-4f89-a074-157fcedabf4c.jpg` | NO_PAPA | TIZON_TARDIO |
| `NO_PAPA/Tomato_Late_blight__4780302b-3329-4f1f-a697-67698af1d261.jpg` | NO_PAPA | TIZON_TARDIO |

Se reporta el valor medido aunque no alcance la meta; el análisis de causas va en el documento (05-requerimientos-no-funcionales).
