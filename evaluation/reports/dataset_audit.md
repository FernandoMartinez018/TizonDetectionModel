# Auditoría del dataset de entrenamiento — 2026-10-24

- Dataset: `plantvillage`
- Imágenes por clase: SANO 152, TIZON_TEMPRANO 1000, TIZON_TARDIO 1000, NO_PAPA 1050. Desbalance (mayor/menor): 6,91.
- Archivos ilegibles: 0.
- Duplicados exactos (sha256): 0 grupos.
- **Misma imagen en las dos clases** (conflicto de etiqueta): 0.
- Casi duplicados (pHash, distancia ≤ 6): 44 pares.

## Resolución (mínimo, mediana, máximo)

| Clase | Ancho (px) | Alto (px) |
|---|---|---|
| SANO | 256 / 256 / 256 | 256 / 256 / 256 |
| TIZON_TEMPRANO | 256 / 256 / 256 | 256 / 256 / 256 |
| TIZON_TARDIO | 256 / 256 / 256 | 256 / 256 / 256 |
| NO_PAPA | 256 / 256 / 256 | 256 / 256 / 256 |

## Casi duplicados (primeros 50)

- `SANO/1a1184f8-c414-4ead-a4c4-41ae78e29a82___RS_HL 1971.JPG` ↔ `TIZON_TARDIO/5e8203e5-b699-4d5a-acb5-08a5f06ee984___RS_LB 5291.JPG` (6)
- `SANO/1dcfeaa9-006d-470c-b3e5-d67609d07d4e___RS_HL 1808.JPG` ↔ `TIZON_TARDIO/0f243024-b1fa-4f96-ac7e-ecaf6dc5bc37___RS_LB 4925.JPG` (6)
- `SANO/414f6249-9f78-4af5-9593-9d5a7e7d979f___RS_HL 1918.JPG` ↔ `TIZON_TARDIO/f964b423-2146-4523-9f71-d7937c475bf0___RS_LB 3842.JPG` (6)
- `SANO/4a08efaa-28cc-4356-8769-4002f4b299ef___RS_HL 1850.JPG` ↔ `SANO/9a6eb7c4-6b43-477a-89e0-69f62ef67991___RS_HL 1846.JPG` (6)
- `SANO/57e262fc-e0ba-44db-a967-5ba6f1826e49___RS_HL 1897.JPG` ↔ `TIZON_TARDIO/5759c134-1965-4dd2-b71a-4eef4048f962___RS_LB 3975.JPG` (6)
- `SANO/5a3fc4bb-a78c-4276-8934-f0b037ffb860___RS_HL 5412.JPG` ↔ `TIZON_TARDIO/53367bb0-3f38-4c90-a71d-05592341e4aa___RS_LB 4149.JPG` (6)
- `SANO/5a3fc4bb-a78c-4276-8934-f0b037ffb860___RS_HL 5412.JPG` ↔ `TIZON_TARDIO/ff7160f9-f8d7-4576-82b8-009bf1b63b3b___RS_LB 4974.JPG` (6)
- `SANO/a5050bde-febc-4931-b31d-45c9652df318___RS_HL 5373.JPG` ↔ `TIZON_TARDIO/d8e3b97f-d362-4e05-8600-a103580aace4___RS_LB 4773.JPG` (6)
- `SANO/a5050bde-febc-4931-b31d-45c9652df318___RS_HL 5373.JPG` ↔ `TIZON_TARDIO/fd81a3e2-1d3e-4e52-a010-ad1b36a5220d___RS_LB 4067.JPG` (6)
- `SANO/aa04db6d-645f-4e8a-88dc-c9f9396f0800___RS_HL 1975.JPG` ↔ `NO_PAPA/Apple_healthy__38c039b6-23f0-4fc2-a0df-e5ec3f523a09.jpg` (6)
- `SANO/c9ad5c9a-c9b0-4e07-9e03-9e9edfd58a19___RS_HL 1755.JPG` ↔ `TIZON_TARDIO/1338edcb-9e0e-458c-845e-206b869f9199___RS_LB 3205.JPG` (6)
- `SANO/e34e563f-7a4a-4b44-bdf2-b6f96cbcd929___RS_HL 5370.JPG` ↔ `TIZON_TARDIO/f964b423-2146-4523-9f71-d7937c475bf0___RS_LB 3842.JPG` (6)
- `SANO/f4b5ec24-d318-4309-8294-9126450d5d7f___RS_HL 1824.JPG` ↔ `TIZON_TARDIO/be013aee-1311-4679-ae86-1c6fba5f55e5___RS_LB 2799.JPG` (6)
- `TIZON_TEMPRANO/0ad3ba53-f01b-403b-a99d-5991eed85045___RS_Early.B 7600.JPG` ↔ `TIZON_TEMPRANO/c5cb3e89-f00f-4032-80ca-eb39bead0796___RS_Early.B 8645.JPG` (6)
- `TIZON_TEMPRANO/0c5b14d9-8b1c-4c39-bb23-1835b5760caa___RS_Early.B 7937.JPG` ↔ `TIZON_TARDIO/3f624c56-6bd1-4e7e-8fcd-d6b4480abe12___RS_LB 5215.JPG` (6)
- `TIZON_TEMPRANO/1994cd3e-4000-4c56-bede-d35a1eecc287___RS_Early.B 6867.JPG` ↔ `TIZON_TEMPRANO/c28107c6-ea57-4044-bb6e-508d6d360cd7___RS_Early.B 7057.JPG` (6)
- `TIZON_TEMPRANO/1c4e0445-375a-4c5d-b6fd-6123d22f009f___RS_Early.B 7972.JPG` ↔ `TIZON_TEMPRANO/49b24f9e-12d4-4f32-8bc8-482d66514a17___RS_Early.B 8037.JPG` (4)
- `TIZON_TEMPRANO/1ee04a42-fa1c-4f6b-a88b-4cf583b188ea___RS_Early.B 8494.JPG` ↔ `TIZON_TEMPRANO/4138df1f-b8f7-429f-9e95-a627f6aba912___RS_Early.B 7667.JPG` (6)
- `TIZON_TEMPRANO/2a4fd894-0c4f-471b-b80b-d362a7b513f4___RS_Early.B 8038.JPG` ↔ `TIZON_TEMPRANO/37d94308-8783-4657-9ca6-eaca86e1015a___RS_Early.B 7825.JPG` (6)
- `TIZON_TEMPRANO/46819687-0705-4628-8748-9cb356587f30___RS_Early.B 8421.JPG` ↔ `TIZON_TARDIO/c89d5a2a-2c1e-496e-84b8-b7a844a97ac9___RS_LB 4920.JPG` (6)
- `TIZON_TEMPRANO/495e73a1-0f61-43a6-b539-d7f74e3fbf71___RS_Early.B 8245.JPG` ↔ `TIZON_TARDIO/d80e59e3-a88d-4c32-8dd1-a23a6fed45ae___RS_LB 2846.JPG` (6)
- `TIZON_TEMPRANO/49b24f9e-12d4-4f32-8bc8-482d66514a17___RS_Early.B 8037.JPG` ↔ `TIZON_TEMPRANO/b8803558-9e0f-404e-80b3-c78d0bc7f6d0___RS_Early.B 7548.JPG` (6)
- `TIZON_TEMPRANO/5ada96db-43de-4510-8c10-eaaebb106c0c___RS_Early.B 7216.JPG` ↔ `TIZON_TEMPRANO/fbc7567e-ce08-4ad0-b8fe-ffd98296ce02___RS_Early.B 8607.JPG` (6)
- `TIZON_TEMPRANO/5aee3e9f-0469-42e7-8ae7-56f7a1900fd5___RS_Early.B 9091.JPG` ↔ `TIZON_TEMPRANO/bc378ba0-533d-42db-a8b2-82decc73f4d5___RS_Early.B 9092.JPG` (2)
- `TIZON_TEMPRANO/68a9df4a-f803-4cf1-bfce-1a637a2befcc___RS_Early.B 7654.JPG` ↔ `TIZON_TEMPRANO/d1072068-911c-4656-8bf2-e662982199e1___RS_Early.B 8141.JPG` (6)
- `TIZON_TEMPRANO/a3ed964c-36f1-403c-8146-b2a3b79a352d___RS_Early.B 8310.JPG` ↔ `TIZON_TARDIO/f528b00d-99a6-41db-8d96-9eed58b519c1___RS_LB 4544.JPG` (6)
- `TIZON_TARDIO/031c39d3-9a91-4e81-85ac-b36512133b0a___RS_LB 5220.JPG` ↔ `TIZON_TARDIO/625cf166-4cb5-4e6f-ac94-1193698388f2___RS_LB 5427.JPG` (6)
- `TIZON_TARDIO/09a30eff-d386-456d-a6ce-ead7c0d6235b___RS_LB 5068.JPG` ↔ `TIZON_TARDIO/83f821e6-0924-4bbe-8977-0e11c220aed0___RS_LB 5011.JPG` (6)
- `TIZON_TARDIO/09fa42e4-1736-494e-8c46-50c6900586d6___RS_LB 5212.JPG` ↔ `TIZON_TARDIO/53367bb0-3f38-4c90-a71d-05592341e4aa___RS_LB 4149.JPG` (6)
- `TIZON_TARDIO/0f6eac3b-d674-4c4d-ab3a-88689feec07f___RS_LB 5232.JPG` ↔ `TIZON_TARDIO/b74d0fd1-9af7-4053-8538-ee937236a64d___RS_LB 4918.JPG` (6)
- `TIZON_TARDIO/36d3502f-32cd-4ef9-b63d-575db1c18cf8___RS_LB 4908.JPG` ↔ `TIZON_TARDIO/be013aee-1311-4679-ae86-1c6fba5f55e5___RS_LB 2799.JPG` (6)
- `TIZON_TARDIO/37ac120b-240a-42fc-839c-90c6dea13d85___RS_LB 5214.JPG` ↔ `TIZON_TARDIO/551f74c9-3212-4a08-8f5e-a101b4195e1e___RS_LB 2593.JPG` (6)
- `TIZON_TARDIO/3f30c40f-105b-4ce7-a376-d2e6f4681e3b___RS_LB 4921.JPG` ↔ `TIZON_TARDIO/74ee080b-cad8-4090-9f2a-07411eb8481a___RS_LB 4956.JPG` (6)
- `TIZON_TARDIO/4cc2dba7-6871-481d-8989-07ea683e58b5___RS_LB 4979.JPG` ↔ `TIZON_TARDIO/da16cb94-4f1d-4390-b446-19eac2110115___RS_LB 2912.JPG` (6)
- `TIZON_TARDIO/74277267-6a9f-44d0-95fe-7f52f89630a0___RS_LB 5074.JPG` ↔ `NO_PAPA/Raspberry_healthy__c7d0d034-94b5-4b25-b899-d571649786b5.jpg` (6)
- `TIZON_TARDIO/c84a1cfc-5e19-46df-a315-c7fb6a61e60f___RS_LB 4103.JPG` ↔ `TIZON_TARDIO/df919a28-9f11-42f6-823c-15f70bf1325e___RS_LB 4936.JPG` (6)
- `TIZON_TARDIO/c89d5a2a-2c1e-496e-84b8-b7a844a97ac9___RS_LB 4920.JPG` ↔ `TIZON_TARDIO/f528b00d-99a6-41db-8d96-9eed58b519c1___RS_LB 4544.JPG` (4)
- `TIZON_TARDIO/e47b3a9a-24ea-460a-8a8f-4fc9814a072e___RS_LB 3314.JPG` ↔ `NO_PAPA/Strawberry_Leaf_scorch__1f2420de-a55f-4538-934a-ed289da7a953.jpg` (4)
- `TIZON_TARDIO/ea06da1e-81c9-4e2e-8266-1f3d56914ff3___RS_LB 2701.JPG` ↔ `NO_PAPA/Soybean_healthy__114e4945-280e-4acd-821c-73d01552fbf7.jpg` (6)
- `NO_PAPA/Grape_healthy__1e126702-2613-4265-b2df-03648d947627.jpg` ↔ `NO_PAPA/Soybean_healthy__1b701ad3-6b86-4ffc-89f7-68bd179aaa40.jpg` (6)
- `NO_PAPA/Raspberry_healthy__27ef7a65-771c-4eab-9169-03a5fa121c71.jpg` ↔ `NO_PAPA/Raspberry_healthy__7fdeb60a-15db-4759-858a-e1118bad583a.jpg` (4)
- `NO_PAPA/Raspberry_healthy__7fdeb60a-15db-4759-858a-e1118bad583a.jpg` ↔ `NO_PAPA/Soybean_healthy__106959ce-ef15-4725-ad2f-d4bb93d4858e.jpg` (6)
- `NO_PAPA/Raspberry_healthy__e2bc5cd9-a590-4b06-a610-dec52f74ae1e.jpg` ↔ `NO_PAPA/Tomato_Tomato_Yellow_Leaf_Curl_Virus__1c3779b9-cb13-43b0-9a4a-01f96e7d039f.jpg` (6)
- `NO_PAPA/Soybean_healthy__0cebd395-f291-4225-b174-fa02dcada2b1.jpg` ↔ `NO_PAPA/Soybean_healthy__193860b3-c9cd-40b9-ae1a-bd42277fca38.jpg` (6)

## Revisión manual de etiquetas (muestra fija, semilla 42)

Marca las imágenes que no correspondan a su clase y anota el total por clase.

### SANO

- [ ] `SANO/0be9d721-82f5-42c3-b535-7494afe01dbe___RS_HL 1814.JPG`
- [ ] `SANO/0f4ebc5a-d646-436a-919d-961342997cde___RS_HL 4183.JPG`
- [ ] `SANO/1106c3fc-92cb-41a6-a6c6-8f08b9b45108___RS_HL 1914.JPG`
- [ ] `SANO/23f1ab77-d664-49c4-92eb-41eb66c6d101___RS_HL 4169.JPG`
- [ ] `SANO/23f4d3b1-23f4-43fb-96c2-dae37224ec74___RS_HL 1803.JPG`
- [ ] `SANO/2dee1571-ef6b-40ef-8c46-334e89aad3f1___RS_HL 1950.JPG`
- [ ] `SANO/30126310-39de-4c02-b10a-23409417a4f6___RS_HL 1826.JPG`
- [ ] `SANO/3a1dbeee-089c-43f0-8f51-a92d3687a515___RS_HL 1754.JPG`
- [ ] `SANO/52dd2fdd-6966-450c-a42b-ee14428d4271___RS_HL 1955.JPG`
- [ ] `SANO/57e24d88-fd01-4422-b8b5-d688d2296184___RS_HL 5401.JPG`
- [ ] `SANO/58a98860-86d8-41e7-9f8c-cc2ca5e90012___RS_HL 1758.JPG`
- [ ] `SANO/5cb360dd-5c17-4c57-b31d-97cd8dd8d0bd___RS_HL 1797.JPG`
- [ ] `SANO/5e19b1f4-fd65-4c4d-bcd2-18c8c519eddb___RS_HL 1965.JPG`
- [ ] `SANO/71c06efe-089f-49b3-beed-095bd7640e32___RS_HL 1947.JPG`
- [ ] `SANO/b35b9f8b-daca-4d12-a5be-defe68d0acc5___RS_HL 1903.JPG`
- [ ] `SANO/b4031970-60ea-4739-9fed-6c81be1bad1c___RS_HL 1752.JPG`
- [ ] `SANO/d287be3d-cac6-4485-8c31-674fb8b459be___RS_HL 1973.JPG`
- [ ] `SANO/e693354a-053c-4093-8642-cc8f5f9eb2cc___RS_HL 1746.JPG`
- [ ] `SANO/f0036dfd-9f46-4f2f-9001-5973495f2171___RS_HL 1849.JPG`
- [ ] `SANO/ff700844-68ad-4e99-8427-58a39c07f817___RS_HL 1860.JPG`

### TIZON_TEMPRANO

- [ ] `TIZON_TEMPRANO/0267d4ca-522e-4ca0-b1a2-ce925e5b54a2___RS_Early.B 7020.JPG`
- [ ] `TIZON_TEMPRANO/1994cd3e-4000-4c56-bede-d35a1eecc287___RS_Early.B 6867.JPG`
- [ ] `TIZON_TEMPRANO/1cb05d87-027d-4bf0-b5c4-4c1a57896ac6___RS_Early.B 7936.JPG`
- [ ] `TIZON_TEMPRANO/29f79006-adf2-4d71-a6a8-bfc02c020b71___RS_Early.B 6833.JPG`
- [ ] `TIZON_TEMPRANO/2b3e46d9-09dd-4966-99c6-fe44c741b94c___RS_Early.B 8290.JPG`
- [ ] `TIZON_TEMPRANO/3a8429cf-f8a7-4f64-a435-ec4f1ad692c7___RS_Early.B 8990.JPG`
- [ ] `TIZON_TEMPRANO/3b090cee-ba51-44c2-9cc6-4581db95070e___RS_Early.B 7789.JPG`
- [ ] `TIZON_TEMPRANO/4c246dbf-ad6c-4563-ad59-d5ea84453817___RS_Early.B 7662.JPG`
- [ ] `TIZON_TEMPRANO/5aee3e9f-0469-42e7-8ae7-56f7a1900fd5___RS_Early.B 9091.JPG`
- [ ] `TIZON_TEMPRANO/5c3363d5-5873-40f5-b179-37ab6df0328c___RS_Early.B 8828.JPG`
- [ ] `TIZON_TEMPRANO/71ab124e-93ba-4d4e-ae53-68ac12fcb858___RS_Early.B 8129.JPG`
- [ ] `TIZON_TEMPRANO/78313ae7-3589-4d23-a5b6-0457e500bd1a___RS_Early.B 6768.JPG`
- [ ] `TIZON_TEMPRANO/9ce9c6dc-b20c-47f8-91e0-31886945c4bb___RS_Early.B 8094.JPG`
- [ ] `TIZON_TEMPRANO/bb569e8c-4f57-481f-86d9-291ebd67336e___RS_Early.B 7543.JPG`
- [ ] `TIZON_TEMPRANO/c9d5a9b6-4e5c-42df-8f9b-5f57e13000df___RS_Early.B 6933.JPG`
- [ ] `TIZON_TEMPRANO/caef0735-6517-4194-b124-5916815e4a71___RS_Early.B 7956.JPG`
- [ ] `TIZON_TEMPRANO/d50dde4d-fe2f-4e8c-b336-69be71e34ce8___RS_Early.B 6905.JPG`
- [ ] `TIZON_TEMPRANO/d59c5aa9-8d45-45db-be3e-0276e24641f5___RS_Early.B 7897.JPG`
- [ ] `TIZON_TEMPRANO/e24cb505-5756-4269-aa4f-57d27da31b20___RS_Early.B 8973.JPG`
- [ ] `TIZON_TEMPRANO/faac446b-34ac-414a-9f0e-148d6ee87bfa___RS_Early.B 8934.JPG`

### TIZON_TARDIO

- [ ] `TIZON_TARDIO/09a30eff-d386-456d-a6ce-ead7c0d6235b___RS_LB 5068.JPG`
- [ ] `TIZON_TARDIO/1736a728-bae9-4133-bfca-60a7c3b6382c___RS_LB 4855.JPG`
- [ ] `TIZON_TARDIO/1b8ae46e-1202-4715-8ad1-fd18046c42d3___RS_LB 3880.JPG`
- [ ] `TIZON_TARDIO/227f2f04-4d59-48d9-958d-5cccc70d3f2b___RS_LB 2772.JPG`
- [ ] `TIZON_TARDIO/447c4811-7ed2-4576-81b4-2f67c8993db7___RS_LB 4137.JPG`
- [ ] `TIZON_TARDIO/4d96b41e-8e2e-4a0d-b9a2-0ab8456e4074___RS_LB 3893.JPG`
- [ ] `TIZON_TARDIO/5b12fde1-fe3a-49fe-8430-479775018322___RS_LB 3071.JPG`
- [ ] `TIZON_TARDIO/5ec337d4-aff8-4174-bae6-72c246db3ce0___RS_LB 3175.JPG`
- [ ] `TIZON_TARDIO/65bab626-51a1-4d16-ab9c-ddd25f4e2e33___RS_LB 4927.JPG`
- [ ] `TIZON_TARDIO/6637f5da-44b1-43bc-9678-1f1188f1ed7c___RS_LB 4796.JPG`
- [ ] `TIZON_TARDIO/7a0b26a9-4abe-4e84-ad20-699b0c0cd27e___RS_LB 4376.JPG`
- [ ] `TIZON_TARDIO/90596f7b-1efa-4595-8d57-bcbd4c48253f___RS_LB 4753.JPG`
- [ ] `TIZON_TARDIO/9570095c-02e9-497b-bec1-4e526f3462e5___RS_LB 2815.JPG`
- [ ] `TIZON_TARDIO/a34d3777-22c7-4723-8b85-f766e042c121___RS_LB 4375.JPG`
- [ ] `TIZON_TARDIO/c3cd2ad3-174e-48b4-a7a5-ce63be745fd5___RS_LB 3243.JPG`
- [ ] `TIZON_TARDIO/d752ca70-320b-4d33-970f-8f3eead9141e___RS_LB 3166.JPG`
- [ ] `TIZON_TARDIO/dc0ad08c-a096-4ad2-9a69-597c10e12f71___RS_LB 4869.JPG`
- [ ] `TIZON_TARDIO/df60df5b-a619-4b38-926d-6372c0f6329a___RS_LB 5308.JPG`
- [ ] `TIZON_TARDIO/f27d1c83-25e9-4ab8-8970-3ddfd2a4c6fc___RS_LB 3230.JPG`
- [ ] `TIZON_TARDIO/feefc118-4434-4ffb-afbb-02fb292f72b6___RS_LB 2874.JPG`

### NO_PAPA

- [ ] `NO_PAPA/Apple_healthy__13dba956-2c74-400e-a2ac-b06baeee5e38.jpg`
- [ ] `NO_PAPA/Blueberry_healthy__811c4a45-2217-418f-834d-e782bd8f9f57.jpg`
- [ ] `NO_PAPA/Blueberry_healthy__990015b1-d83b-40ed-83dd-5e155dda7b19.jpg`
- [ ] `NO_PAPA/Cherry_including_sour_Powdery_mildew__e7fe2094-95a1-40fe-ba1d-687ab0756edb.jpg`
- [ ] `NO_PAPA/Cherry_including_sour_healthy__6383ba62-2f58-4bf0-89f0-9a9159c25aad.jpg`
- [ ] `NO_PAPA/Grape_Black_rot__0d907173-6ef8-4fbd-b682-f0a7a5263de0.jpg`
- [ ] `NO_PAPA/Grape_Black_rot__881ea69c-736d-4ea9-8820-17f37d5ecfd4.jpg`
- [ ] `NO_PAPA/Grape_Leaf_blight_Isariopsis_Leaf_Spot__402e809f-6fed-46a4-bf4d-3a78431c1555.jpg`
- [ ] `NO_PAPA/Grape_healthy__1c453594-a071-422c-a733-49aa167bac99.jpg`
- [ ] `NO_PAPA/Orange_Haunglongbing_Citrus_greening__1a5d2d4c-076f-4d05-9a80-3f429db489b3.jpg`
- [ ] `NO_PAPA/Orange_Haunglongbing_Citrus_greening__299a004e-6b6a-498f-a8f1-3be100478f40.jpg`
- [ ] `NO_PAPA/Pepper_bell_Bacterial_spot__42268507-28e9-4fb2-bd74-a2afdb3eebfa.jpg`
- [ ] `NO_PAPA/Pepper_bell_Bacterial_spot__fe0f9f93-fb79-45fa-a195-6b85141ad2aa.jpg`
- [ ] `NO_PAPA/Pepper_bell_healthy__87a97810-bd20-4f2f-8a20-1953664f8576.jpg`
- [ ] `NO_PAPA/Strawberry_Leaf_scorch__4da9a515-440d-495b-8969-320b27b460be.jpg`
- [ ] `NO_PAPA/Strawberry_Leaf_scorch__b39b1fbc-038c-4bc8-8154-3c6ea168af62.jpg`
- [ ] `NO_PAPA/Strawberry_Leaf_scorch__d89256de-8d57-49d0-9fa3-3be4a3bb3c20.jpg`
- [ ] `NO_PAPA/Tomato_Bacterial_spot__1d2bcd85-fdb7-4294-ba11-eb58373b4474.jpg`
- [ ] `NO_PAPA/Tomato_Bacterial_spot__754dfec2-60c1-4848-803c-21ab8471014e.jpg`
- [ ] `NO_PAPA/Tomato_Septoria_leaf_spot__93d905ab-5def-4105-80b8-d85f54258a4a.jpg`
