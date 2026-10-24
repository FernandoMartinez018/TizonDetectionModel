# Evaluación del modelo — modo `classifier` — 2026-10-24

- Modelo: `mobilenetv2-4c-02b1cb27-20261024` (`modelo_tizon_4c.keras`, sha256 `02b1cb276fed2ab35e709dfe86ddc5d39563a3d920119bf90d1f753d006f0416`)
- Set de prueba: `plantvillage` (partición `test` de `training/split.csv`) — origen: PlantVillage (github.com/spMohanty/PlantVillage-Dataset, raw/color), partición de prueba de training/split.csv: papa (sana, tizón temprano, tizón tardío) y otros 13 cultivos como NO_PAPA
- Imágenes: SANO 22, TIZON_TEMPRANO 150, TIZON_TARDIO 150, NO_PAPA 157 (total 479)
- Máquina: Windows-11-10.0.26200-SP0 · 14 CPU
- Accuracy y tasa de error: globales. Precision, Recall, F1 y Especificidad: TIZON_TARDIO contra el resto. Imagen completa, sin filtro.

## Métricas frente a las metas del PDF

| Métrica | Meta | Medido | Cumple |
|---|---|---|---|
| Accuracy | ≥ 90,0 % | 95,6 % | ✔ |
| Precision | ≥ 90,0 % | 94,2 % | ✔ |
| Recall | ≥ 90,0 % | 97,3 % | ✔ |
| F1 | ≥ 90,0 % | 95,7 % | ✔ |
| Especificidad | ≥ 85,0 % | 97,3 % | ✔ |
| Tasa de error | ≤ 10,0 % | 4,4 % | ✔ |
| Latencia p95 (inferencia) | < 1.500 ms en la EC2 | 109,4 ms (media 96,9 ms) | ✔ |

## Por clase

| Clase | Soporte | Precision | Recall | F1 | Especificidad |
|---|---|---|---|---|---|
| SANO | 22 | 77,8 % | 95,5 % | 85,7 % | 98,7 % |
| TIZON_TEMPRANO | 150 | 98,6 % | 97,3 % | 98,0 % | 99,4 % |
| TIZON_TARDIO | 150 | 94,2 % | 97,3 % | 95,7 % | 97,3 % |
| NO_PAPA | 157 | 97,3 % | 92,4 % | 94,8 % | 98,8 % |

## Matriz de confusión (filas: real, columnas: predicho)

| | SANO | TIZON_TEMPRANO | TIZON_TARDIO | NO_PAPA |
|---|---|---|---|---|
| SANO | 21 | 0 | 0 | 1 |
| TIZON_TEMPRANO | 0 | 146 | 4 | 0 |
| TIZON_TARDIO | 1 | 0 | 146 | 3 |
| NO_PAPA | 5 | 2 | 5 | 145 |

## ¿Es papa?

- Hojas de papa rechazadas como NO_PAPA (no se analizan): 1,2 %.
- Hojas de otras plantas aceptadas como papa (se analizarían por error): 7,6 %.

## Integridad del set de prueba

- Duplicados exactos con el entrenamiento: 0.
- Casi duplicados (pHash, distancia ≤ 6): 3. Se revisan a mano y se retiran del set de prueba si son la misma hoja.
  - `SANO/57e262fc-e0ba-44db-a967-5ba6f1826e49___RS_HL 1897.JPG` ↔ `TIZON_TARDIO/5759c134-1965-4dd2-b71a-4eef4048f962___RS_LB 3975.JPG` (6)
  - `TIZON_TARDIO/0f243024-b1fa-4f96-ac7e-ecaf6dc5bc37___RS_LB 4925.JPG` ↔ `SANO/1dcfeaa9-006d-470c-b3e5-d67609d07d4e___RS_HL 1808.JPG` (6)
  - `TIZON_TARDIO/ea06da1e-81c9-4e2e-8266-1f3d56914ff3___RS_LB 2701.JPG` ↔ `NO_PAPA/Soybean_healthy__114e4945-280e-4acd-821c-73d01552fbf7.jpg` (6)

## Errores (21; primeros 20)

| Archivo | Real | Predicho |
|---|---|---|
| `SANO/ec6e43d3-512d-4908-99f2-c9e2a0be54be___RS_HL 4202.JPG` | SANO | NO_PAPA |
| `TIZON_TEMPRANO/31d0ecc8-045d-4a4a-ac33-431556c83e4c___RS_Early.B 8692.JPG` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/357b61ef-a56b-4207-aa26-161b0ef3e635___RS_Early.B 8492.JPG` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/8f3f9eed-1a3d-4f0f-adcc-0fe3a832d95f___RS_Early.B 8518.JPG` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TEMPRANO/f686133a-e89a-4242-a52d-02f32ffd5275___RS_Early.B 8295.JPG` | TIZON_TEMPRANO | TIZON_TARDIO |
| `TIZON_TARDIO/84e85005-c647-4381-8762-c487074c9571___RS_LB 5186.JPG` | TIZON_TARDIO | NO_PAPA |
| `TIZON_TARDIO/87681326-2843-4e63-9042-602b08a98a66___RS_LB 4145.JPG` | TIZON_TARDIO | NO_PAPA |
| `TIZON_TARDIO/bd632f59-ff1c-452f-b504-053fa95ab000___RS_LB 4356.JPG` | TIZON_TARDIO | NO_PAPA |
| `TIZON_TARDIO/d9817cc3-ef03-41e5-8892-1fddd27b7bcc___RS_LB 4878.JPG` | TIZON_TARDIO | SANO |
| `NO_PAPA/Blueberry_healthy__16d7d729-30a4-426a-b4d9-c615e33dabe2.jpg` | NO_PAPA | TIZON_TARDIO |
| `NO_PAPA/Blueberry_healthy__4c7b7f86-9816-403a-98bc-1123dba9478e.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Blueberry_healthy__811c4a45-2217-418f-834d-e782bd8f9f57.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Cherry_including_sour_Powdery_mildew__c82e8bfd-a4d6-448f-9c6c-1da54df1e12f.jpg` | NO_PAPA | TIZON_TEMPRANO |
| `NO_PAPA/Cherry_including_sour_Powdery_mildew__df3c80e0-e479-4b11-b488-f527fe8f8777.jpg` | NO_PAPA | TIZON_TARDIO |
| `NO_PAPA/Orange_Haunglongbing_Citrus_greening__2c1e3cc5-9bc3-4555-85af-e0b90b2dedd6.jpg` | NO_PAPA | TIZON_TARDIO |
| `NO_PAPA/Soybean_healthy__045dba20-7120-47d2-b7dc-86cdbe77580b.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Soybean_healthy__1b957666-ac0a-4f3d-a7f7-627a53972c9b.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Soybean_healthy__29ca525d-7049-4262-b65a-dcfe1a063035.jpg` | NO_PAPA | SANO |
| `NO_PAPA/Strawberry_Leaf_scorch__c5aa66d1-22cf-48af-a622-a9fbbd0d2c00.jpg` | NO_PAPA | TIZON_TEMPRANO |
| `NO_PAPA/Tomato_Late_blight__34ce281a-ca92-4f89-a074-157fcedabf4c.jpg` | NO_PAPA | TIZON_TARDIO |

Se reporta el valor medido aunque no alcance la meta; el análisis de causas va en el documento (05-requerimientos-no-funcionales).
