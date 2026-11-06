# Despliegue del sistema Tizón Tardío en la nube (ejemplo: AWS)

Esta guía describe, paso a paso, cómo llevar el sistema completo a una nube. El ejemplo usa **AWS EC2**, que es donde
se desplegó el sistema el 5-nov-2026, pero los pasos 6 en adelante sirven igual en cualquier máquina virtual con Ubuntu
y Docker (ver la sección 12).

> Esta misma guía está copiada en los tres repositorios (`TizonDetectionModel`, `TizonApiIntegration` y `TizonMovile`)
> para que no dependa de ninguno de ellos. Todos los valores de ejemplo (IP, hosts, identificadores) son ficticios:
> reemplázalos por los tuyos. **Nunca** pegues contraseñas, llaves ni tokens reales en un repositorio.

## 1. Qué se despliega

```
 Celular Android (APK)
        │ HTTPS 443
        ▼
 ┌─────────────────────── EC2 (Ubuntu 24.04, docker compose) ────────────────────────┐
 │  caddy  ──►  api (Spring Boot, JWT)  ──►  ia (FastAPI + TensorFlow, modelo .keras)  │
 │  80/443        │                                                                    │
 │                └──►  db (PostgreSQL 16, volumen pgdata)                             │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

| Servicio | Qué hace | ¿Se expone a Internet? |
|---|---|---|
| `caddy` | Termina HTTPS (certificado gratuito de Let's Encrypt) y reenvía `/api/*` a la API | Sí, puertos 80 y 443 |
| `api` | Lógica de negocio, usuarios, sesiones, alertas | No directamente (solo a través de Caddy) |
| `ia` | Inferencia: clasifica la hoja (Sano, Tizón temprano, Tizón tardío o No es papa) | **No** (red interna) |
| `db` | PostgreSQL 16 con volumen persistente | **No** (red interna) |

Piezas que **no** van en AWS: la APK (se instala a mano en el celular) y el código fuente (está en GitHub).

### Dónde está cada cosa
| Pieza | Dónde vive |
|---|---|
| Código de la IA | repo `TizonDetectionModel` |
| Código de la API | repo `TizonApiIntegration` |
| Código de la app | repo `TizonMovile` |
| `deploy/` (compose, Caddyfile, scripts), `contracts/`, `docs/` | rama **`sistema`** de `TizonApiIntegration` |
| Imágenes Docker | GHCR: `ghcr.io/<usuario>/tizon-ia` y `ghcr.io/<usuario>/tizon-api`, con etiquetas `main`, `qa` y `sha-<commit>` |
| Modelo de IA (`.keras`) | **No está en git.** Se copia al servidor a mano |

## 2. Requisitos previos

En tu computador:
- Cuenta de AWS con un **usuario IAM** que pueda crear EC2 (evita usar el usuario *root* para el trabajo diario).
- **AWS CLI v2.32 o superior** (`aws --version`). Para iniciar sesión: `aws login` (abre el navegador y entrega credenciales
  temporales). La política IAM `SignInLocalDevelopmentAccess` debe estar asociada a tu usuario; el usuario *root* no la necesita.
- **Git** y **GitHub CLI** (`gh`) con sesión iniciada.
- **Docker Desktop** (para compilar la APK sin instalar el SDK de Android).
- Un cliente **SSH** (`ssh`, `scp`).
- El archivo del modelo (`modelo_tizon_4c_v2.keras`) y su `model_meta.json`.

En GitHub:
- Los tres repositorios con las ramas `dev`, `qa` y `main`.
- Los flujos de GitHub Actions habilitados.

## 3. Preparar el código y las imágenes (CI/CD)

El servidor no construye nada: **descarga imágenes ya publicadas** en GHCR. Esas imágenes se publican solas cuando algo
llega a `qa` o `main`.

1. Trabaja y haz commit en `dev`.
2. Abre un *pull request* `dev → qa` en cada repositorio y espera que el CI pase (pruebas, linter y *gitleaks*).
   ```bash
   gh pr create --repo <usuario>/TizonApiIntegration --base qa --head dev --fill
   ```
3. Fusiona. La publicación crea `tizon-api:qa` y `tizon-ia:qa`.
4. Repite `qa → main` para obtener `tizon-api:main` y `tizon-ia:main`.
5. Revisa en GitHub → *Packages* que las imágenes existan. Si el paquete es **privado**, el servidor necesitará
   `docker login ghcr.io` con un *token* de solo lectura (`read:packages`). Si es público, no hace falta.

La APK **no** se publica en GHCR: se compila aparte (sección 9).

## 4. Elegir y preparar el modelo de IA

- El modelo no va en git. Se copian dos archivos al servidor, en `deploy/models/`:
  `modelo_tizon_4c_v2.keras` y `model_meta.json`.
- La IA calcula el **sha256** del `.keras` al arrancar y lo compara con el de `model_meta.json`. Si no coinciden, no
  carga el modelo. Verifica antes de subirlo:
  ```bash
  sha256sum modelo_tizon_4c_v2.keras      # debe coincidir con "sha256" en model_meta.json
  ```
- El nombre del archivo se indica con `MODEL_FILE` en el `.env` (sección 7).
- Para volver al modelo anterior: sube su `.keras` y su meta, cambia `MODEL_FILE` y reinicia solo la IA.

## 5. Crear la infraestructura en AWS

Elige una región (por ejemplo `us-east-1`). Todos los recursos deben estar en la misma.

```bash
export AWS_REGION=us-east-1
aws sts get-caller-identity          # confirma la cuenta con la que operas
```

> Si el CLI responde `NoRegion` o `Could not connect to the endpoint URL`, tu configuración tiene una región inválida:
> define `AWS_REGION` como arriba o ejecuta `aws configure set region us-east-1`.

### 5.1 Par de llaves SSH
```bash
aws ec2 create-key-pair --key-name tizon-prod --key-type ed25519 \
  --query KeyMaterial --output text > ~/.ssh/tizon-prod.pem
chmod 600 ~/.ssh/tizon-prod.pem
```
- Guarda la llave **fuera de cualquier repositorio** y haz una copia de respaldo. AWS no puede volver a entregarla.
- En Windows, la redirección puede dejar saltos de línea `\r`; corrige con `sed -i 's/\r$//' ~/.ssh/tizon-prod.pem`
  y valida con `ssh-keygen -y -f ~/.ssh/tizon-prod.pem`.

### 5.2 Grupo de seguridad (firewall)
Reglas de entrada:

| Puerto | Origen | Para qué |
|---|---|---|
| 80 | `0.0.0.0/0` | Let's Encrypt (validación) y redirección a HTTPS |
| 443 | `0.0.0.0/0` | La API para la app |
| 22 | **solo tu IP** (`<tu-ip>/32`) | Administración por SSH |

```bash
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 create-security-group --group-name tizon-sg --description "Tizon: 80/443 publico, 22 solo admin" \
      --vpc-id $VPC --query GroupId --output text)
MI_IP=$(curl -s -4 https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id $SG --ip-permissions \
  'IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}]' \
  'IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0}]' \
  "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MI_IP/32,Description=admin}]"
```
> Si tu proveedor de Internet usa varias IP de salida (se nota si dos servicios distintos de "mi IP" devuelven
> valores diferentes), el SSH fallará de forma intermitente con *Connection timed out*. Abre un rango pequeño
> (por ejemplo un `/29`) en lugar de una sola IP.

### 5.3 Instancia
- Imagen: Ubuntu Server 24.04 LTS (AMI oficial de Canonical, propietario `099720109477`).
- Tipo: `t3.medium` (2 vCPU, 4 GB) o mayor. TensorFlow necesita memoria; con menos de 4 GB el modelo puede no cargar.
- Disco: gp3 de 30 GB.
- IMDSv2 obligatorio (`HttpTokens=required`).

```bash
AMI=$(aws ec2 describe-images --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

# En Git Bash de Windows, anteponer MSYS_NO_PATHCONV=1: si no, "/dev/sda1" se convierte en una ruta de Windows.
MSYS_NO_PATHCONV=1 aws ec2 run-instances --image-id $AMI --instance-type t3.medium \
  --key-name tizon-prod --security-group-ids $SG \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
  --metadata-options HttpTokens=required \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tizon}]' \
  --query 'Instances[0].InstanceId' --output text
```
Espera a que esté lista: `aws ec2 wait instance-status-ok --instance-ids <id>`.

### 5.4 IP elástica (dirección fija)
Sin IP elástica, la IP pública cambia cada vez que se detiene la instancia, y la app dejaría de encontrar el servidor.
```bash
ALLOC=$(aws ec2 allocate-address --domain vpc --query AllocationId --output text)
aws ec2 associate-address --instance-id <id> --allocation-id $ALLOC
aws ec2 describe-addresses --allocation-ids $ALLOC --query 'Addresses[0].PublicIp' --output text
```
Anota la IP, por ejemplo `203.0.113.10`.

### 5.5 Nombre de host con HTTPS
Let's Encrypt necesita un nombre, no una IP. Sin comprar dominio puedes usar **sslip.io**: convierte la IP en un nombre.

| IP | Nombre de host |
|---|---|
| `203.0.113.10` | `203-0-113-10.sslip.io` |

Si tienes un dominio propio (más robusto, ver sección 11), crea un registro `A` que apunte a la IP elástica y usa ese nombre.

## 6. Preparar el servidor

Conéctate:
```bash
ssh -i ~/.ssh/tizon-prod.pem ubuntu@203.0.113.10
```

### 6.1 Instalar Docker y preparar el sistema
El script `deploy/scripts/bootstrap-ec2.sh` instala Docker y Compose, crea 2 GB de *swap*, fija la zona horaria de Bogotá y
crea `/opt/tizon`, `/var/log/tizon` y `/var/backups/tizon`. Desde tu computador:
```bash
git clone -b sistema --single-branch https://github.com/<usuario>/TizonApiIntegration.git tizon-sistema
scp -i ~/.ssh/tizon-prod.pem tizon-sistema/deploy/scripts/bootstrap-ec2.sh ubuntu@203.0.113.10:/tmp/
ssh -i ~/.ssh/tizon-prod.pem ubuntu@203.0.113.10 "sudo bash /tmp/bootstrap-ec2.sh"
```
Cierra y vuelve a abrir la sesión SSH para usar `docker` sin `sudo`.

### 6.2 Copiar `deploy/` y el modelo
Copia `deploy/` **sin** el `.env` local ni los modelos:
```bash
tar -C tizon-sistema/deploy --exclude=.env --exclude='models/*' -cf - . | \
  ssh -i ~/.ssh/tizon-prod.pem ubuntu@203.0.113.10 "mkdir -p /opt/tizon/deploy/models && tar -C /opt/tizon/deploy -xf -"
scp -i ~/.ssh/tizon-prod.pem modelo_tizon_4c_v2.keras model_meta.json ubuntu@203.0.113.10:/opt/tizon/deploy/models/
```
Comprueba en el servidor que el sha256 coincide (sección 4).

## 7. Configurar el `.env` en el servidor

En el servidor, en `/opt/tizon/deploy`:
```bash
cp .env.example .env && chmod 600 .env
```
Edita los valores. **Los secretos se generan en el propio servidor y no se copian a ningún otro lado.**

| Variable | Valor |
|---|---|
| `IMAGE_TAG` | `main` (producción) o `qa` |
| `COMPOSE_PROFILES` | `tls` (activa Caddy y HTTPS) |
| `PUBLIC_HOST` | `203-0-113-10.sslip.io` (o tu dominio) |
| `POSTGRES_PASSWORD` | `openssl rand -hex 24` |
| `JWT_SECRET` | `openssl rand -hex 48` (mínimo 32 bytes) |
| `INTERNAL_API_TOKEN` | `openssl rand -hex 32` (secreto compartido API ↔ IA) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | usuario administrador inicial; se crea en el primer arranque |
| `MODEL_FILE` | `modelo_tizon_4c_v2.keras` |
| `API_BIND` | déjalo en `127.0.0.1` (solo Caddy llega a la API) |

La API se niega a arrancar si faltan `JWT_SECRET` o `INTERNAL_API_TOKEN`: es intencional.

## 8. Desplegar y verificar

```bash
cd /opt/tizon/deploy
chmod +x scripts/*.sh
scripts/deploy.sh                       # descarga las imágenes de GHCR y levanta los 4 servicios
docker compose --env-file .env ps       # db, ia, api y caddy deben quedar "healthy"
```

Verificaciones:
```bash
curl https://203-0-113-10.sslip.io/api/actuator/health          # {"status":"UP"}
set -a; . ./.env; set +a
SMOKE_USER=$ADMIN_USERNAME SMOKE_PASS=$ADMIN_PASSWORD scripts/smoke-test.sh https://$PUBLIC_HOST/api
```
La prueba de humo valida: salud, login, configuración, creación de sesión, análisis de un frame y cierre de sesión.
Para ver la versión del modelo cargado:
```bash
docker compose --env-file .env exec -T ia python -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```
Debe mostrar `"model_loaded": true` y el `model_version` esperado.

### 8.1 Despliegue continuo y tareas programadas
```bash
sudo cp scripts/systemd/tizon-auto-update.* /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now tizon-auto-update.timer
sudo mkdir -p /var/log/tizon /var/backups/tizon && sudo chown ubuntu /var/log/tizon /var/backups/tizon
crontab -e
# agregar estas dos líneas:
# * * * * * /opt/tizon/deploy/scripts/uptime.sh
# 30 2 * * * /opt/tizon/deploy/scripts/backup-db.sh >> /var/log/tizon/backup.log 2>&1
```
- **Auto-actualización:** cada 5 minutos el servidor consulta si hay una imagen nueva para su `IMAGE_TAG`. Si la hay, recrea
  `ia` y luego `api`, espera a que estén sanas y, si no lo logran en 4 minutos, **vuelve a la imagen anterior**.
  GitHub nunca entra al servidor: no se abre el puerto 22 a GitHub ni se guardan llaves SSH allí.
- **Disponibilidad:** `uptime.sh` registra una línea por minuto en `/var/log/tizon/uptime.log`; `scripts/uptime.sh --summary` resume el día.
- **Respaldo:** `backup-db.sh` guarda un `.sql.gz` diario en `/var/backups/tizon` y conserva 7 días.

## 9. Compilar y distribuir la APK

La dirección del servidor queda **fija dentro de la APK al compilarla**. Por eso la APK de entrega se compila después de
tener la URL. En esta guía, la URL es `https://203-0-113-10.sslip.io/api/` (con `/api/` y la barra final).

### 9.1 Keystore (una sola vez)
Sin firma no se puede instalar una APK *release*. Crea un keystore y guárdalo **fuera de los repositorios**, con copia de respaldo:
```bash
keytool -genkeypair -keystore tizon-release.jks -storetype PKCS12 -alias tizon \
  -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Tizon Tardio, O=Proyecto academico, C=CO"
```
Si pierdes el keystore, las actualizaciones futuras no se podrán instalar sobre la APK ya instalada.

### 9.2 Compilar
Desde la carpeta `TizonMovile`, con las variables de firma definidas **solo en tu sesión**:
```bash
export TIZON_KEYSTORE_PATH=/ruta/segura/tizon-release.jks
export TIZON_KEYSTORE_PASSWORD='<contraseña>'
export TIZON_KEY_ALIAS=tizon
export TIZON_KEY_PASSWORD='<contraseña>'
scripts/gradlew-docker.sh :composeApp:assembleRelease "-Ptizon.apiBaseUrl=https://203-0-113-10.sslip.io/api/"
scripts/apk-sha256.sh composeApp/build/outputs/apk/release/TizonMovile-1.1.0-release.apk
```
- La APK queda en `composeApp/build/outputs/apk/release/`.
- `apk-sha256.sh` muestra el certificado de firma y el SHA-256 del archivo (para verificar la integridad).
- La compilación *release* **falla a propósito** si no se indica `-Ptizon.apiBaseUrl`.
- La alternativa por CI: define la variable `TIZON_API_BASE_URL` y los secretos `TIZON_KEYSTORE_BASE64`,
  `TIZON_KEYSTORE_PASSWORD`, `TIZON_KEY_ALIAS` y `TIZON_KEY_PASSWORD` en GitHub; cada *push* a `main` genera la APK como artefacto.

### 9.3 Instalar en el celular
Pasa el archivo al celular (cable USB, Drive o correo), ábrelo y permite "instalar apps desconocidas". El proyecto no usa Play Store.
Los usuarios se registran desde la propia app (rol AGRICULTOR); el administrador gestiona usuarios desde la sección de usuarios.

## 10. Operación diaria

Todos los comandos, en `/opt/tizon/deploy`:

| Tarea | Comando |
|---|---|
| Estado | `docker compose --env-file .env ps` |
| Logs | `docker compose --env-file .env logs -f api ia` |
| Registro del auto-update | `journalctl -u tizon-auto-update -n 50` |
| Disponibilidad del día | `scripts/uptime.sh --summary` |
| Respaldo manual | `scripts/backup-db.sh` |
| Restaurar un respaldo | `gunzip -c /var/backups/tizon/<archivo>.sql.gz \| docker compose --env-file .env exec -T db psql -U tizon -d tizon` |
| Fijar o volver a una versión | `IMAGE_TAG=sha-<commit>` en `.env` y `scripts/deploy.sh` |
| Cambiar el modelo | subir `.keras` y `model_meta.json`, cambiar `MODEL_FILE`, `docker compose --env-file .env restart ia` |
| Prueba de recuperación | `docker kill tizon-ia-1` → vuelve sola en menos de 60 s |

### 10.1 Apagar y volver a encender
| Acción | IP | ¿Recompilar la APK? | Datos |
|---|---|---|---|
| **Detener** la instancia y volver a iniciarla | se conserva (IP elástica) | No | se conservan |
| **Terminar** la instancia conservando la IP elástica | se conserva | No, al asociar la misma IP a una instancia nueva | **se pierden** (el disco se borra) |
| **Liberar** la IP elástica | cambia | **Sí** (nuevo nombre de host) | depende |

- Al reiniciar, Docker, los contenedores, el auto-update y el cron vuelven solos; el servicio tarda unos minutos en quedar sano.
- Una IP elástica reservada **se cobra aunque la instancia esté detenida**. Consulta la calculadora de precios y la
  facturación de AWS: las tarifas cambian.
- **Antes de terminar o liberar nada**, copia a otro lugar el último respaldo de la base y el keystore.

## 11. Seguridad y buenas prácticas

- **Cero secretos en git:** nada de `.pem`, `.jks`, `.env`, contraseñas ni tokens en los repositorios (los tres son públicos).
  El CI ejecuta *gitleaks* en cada cambio.
- Usa un **usuario IAM** con permisos acotados en lugar de *root*, y activa MFA.
- SSH solo desde tu IP (o rango); nunca `0.0.0.0/0` en el puerto 22.
- La IA y la base de datos están en una red interna sin salida a Internet: no publiques sus puertos.
- **Respaldos fuera de la instancia:** el respaldo diario vive en el mismo disco; si se pierde la instancia, se pierde con ella.
  Copia periódicamente los `.sql.gz` a otro lugar (otro equipo o un almacenamiento externo).
- **Dominio propio:** con un nombre como `tizon.midominio.com` apuntando a la IP, la APK se compila una vez; si la IP cambia
  basta editar el DNS, sin recompilar. Ajusta `PUBLIC_HOST` y vuelve a ejecutar `scripts/deploy.sh`.
- Revisa la facturación y detén la instancia cuando no se use.

## 12. Otras nubes

Los pasos 6 en adelante no dependen de AWS. En cualquier proveedor necesitas lo mismo:

| Necesidad | AWS | Azure | Google Cloud | DigitalOcean |
|---|---|---|---|---|
| Máquina virtual (2 vCPU, 4 GB, 30 GB, Ubuntu 24.04) | EC2 `t3.medium` | VM `B2s` | `e2-medium` | Droplet 4 GB |
| IP fija | Elastic IP | IP pública estática | IP externa reservada | Reserved IP |
| Firewall (80 y 443 abiertos, 22 restringido) | Security Group | Network Security Group | Reglas de firewall de VPC | Cloud Firewall |
| Llave SSH | Key pair | Llave SSH de la VM | Metadatos SSH | Llave SSH del Droplet |

Después: ejecutar `bootstrap-ec2.sh`, copiar `deploy/` y el modelo, crear el `.env`, `scripts/deploy.sh` y compilar la
APK con la URL nueva. Los nombres de tamaños de VM son referenciales; confírmalos en el proveedor.

## 13. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `aws`: `NoRegion` o *Could not connect to the endpoint URL* | Región inválida en la configuración. Usa `export AWS_REGION=us-east-1` |
| `InvalidBlockDeviceMapping: Invalid device name C:/.../dev/sda1` | Git Bash de Windows convirtió la ruta. Anteponer `MSYS_NO_PATHCONV=1` |
| SSH: *Connection timed out* | Tu IP de salida no está en la regla del puerto 22 (o rota entre varias). Abre un rango pequeño |
| SSH: *Load key … invalid format* | La llave quedó con saltos de línea de Windows. `sed -i 's/\r$//' llave.pem` |
| Una IP elástica quedó sin asociar tras un intento fallido | Se cobra aunque no se use. Revisa `aws ec2 describe-addresses` y libera la sobrante |
| La IA no carga el modelo | El `sha256` del `.keras` no coincide con `model_meta.json`, o `MODEL_FILE` apunta a otro archivo |
| La API no arranca | Faltan `JWT_SECRET` o `INTERNAL_API_TOKEN` en el `.env` |
| `docker pull` de GHCR pide credenciales | El paquete es privado: `docker login ghcr.io` con un token `read:packages`, o hazlo público |
| Caddy no obtiene el certificado | Los puertos 80 y 443 deben estar abiertos a todo Internet y `PUBLIC_HOST` debe resolver a la IP elástica |
| La app dice "Servicio no disponible" y el servidor no recibe frames | Revisa en los logs de Caddy qué respondió la API. Se vio una vez porque `POST /sessions` omitía `started_at`; ya está corregido en `main`. Una respuesta que no cumple el contrato produce ese mensaje |
| `curl` desde Windows se queda en el *handshake* TLS pero el servidor responde desde otra red | Es un problema de la red local del equipo, no del servidor. Prueba desde WSL u otra red |
| La app no conecta tras cambiar de IP | La URL está fija en la APK: recompilar con la nueva URL (sección 9) |

## 14. Lista de verificación

- [ ] Los tres repositorios están en `main` y el CI pasó.
- [ ] Las imágenes `tizon-ia:main` y `tizon-api:main` existen en GHCR.
- [ ] Usuario IAM con MFA; llave SSH guardada fuera de los repositorios, con copia.
- [ ] Security Group: 80 y 443 abiertos, 22 solo desde el administrador.
- [ ] Instancia `t3.medium`, disco de 30 GB, IP elástica asociada.
- [ ] `.env` creado en el servidor con secretos generados allí, permisos `600`.
- [ ] Modelo copiado y sha256 verificado.
- [ ] `docker compose ps`: los 4 servicios `healthy`.
- [ ] `scripts/smoke-test.sh` termina con *Smoke test OK*.
- [ ] Auto-update, uptime y respaldo activos.
- [ ] Keystore creado y con copia de respaldo; APK compilada con la URL del servidor y firma verificada.
- [ ] Prueba real en el celular: registro, inicio de escaneo y fin de sesión.
- [ ] Respaldo de la base copiado fuera de la instancia antes de apagar o terminar.
