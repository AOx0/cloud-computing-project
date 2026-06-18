# MLOps Platform: Jigsaw Toxic Comment Classifier

Documentacion operativa completa de la plataforma MLOps desplegada en Google Cloud Platform. Incluye todos los comandos necesarios para replicar el entorno desde cero.

---

## Tabla de contenidos

1. [Arquitectura general](#1-arquitectura-general)
2. [Requisitos previos](#2-requisitos-previos)
3. [Configuracion del proyecto GCP](#3-configuracion-del-proyecto-gcp)
4. [Infraestructura paso a paso](#4-infraestructura-paso-a-paso)
5. [Almacenamiento y datos en GCS](#5-almacenamiento-y-datos-en-gcs)
6. [Imagen Docker y Artifact Registry](#6-imagen-docker-y-artifact-registry)
7. [Cache de embeddings](#7-cache-de-embeddings)
8. [Despliegue en Cloud Run (serving)](#8-despliegue-en-cloud-run-serving)
9. [Entrenamiento via Vertex AI Custom Job](#9-entrenamiento-via-vertex-ai-custom-job)
10. [Pipeline de Vertex AI (KFP)](#10-pipeline-de-vertex-ai-kfp)
11. [API endpoints](#11-api-endpoints)
12. [Cuentas de servicio e IAM](#12-cuentas-de-servicio-e-iam)
13. [Secret Manager](#13-secret-manager)
14. [Reentrenamiento automatico (implementado)](#14-reentrenamiento-automatico-implementado)
15. [CI/CD automatico con Cloud Build (implementado)](#15-cicd-automatico-con-cloud-build-implementado)
16. [Monitoreo](#16-monitoreo)
17. [Estructura del repositorio](#17-estructura-del-repositorio)
18. [Modelo final](#18-modelo-final)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. Arquitectura general

```
                       ┌──────────────────────────────────────────┐
                       │           GCP Project                    │
                       │      mlops-toxic-classifier              │
                       │      (943214853579)                      │
                       └──────────────────────────────────────────┘
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           │                             │                              │
   ┌───────▼───────┐          ┌─────────▼─────────┐         ┌─────────▼─────────┐
   │   Cloud Build  │          │    Vertex AI      │         │    Cloud Run       │
   │   (CI/CD)      │          │  Custom Training  │         │  (Serving API)     │
   │                │          │    Job             │         │                    │
   │  build+push    │          │  --mode train     │         │  --mode serve      │
   │  Docker image  │          │                   │         │                    │
   └───────┬───────┘          └─────────┬─────────┘         └─────────┬─────────┘
           │                             │                              │
           ▼                             ▼                              ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                        Artifact Registry                                  │
   │     us-central1-docker.pkg.dev/.../mlops-containers/toxic-classifier     │
   └───────────────────────────────────────────────────────────────────────────┘
           │                             │                              │
           ▼                             ▼                              ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                         Cloud Storage (GCS)                               │
   │                   gs://mlops-toxic-classifier-ml/                        │
   │                                                                          │
   │  train.csv          ← datos de entrenamiento (65 MB)                     │
   │  model/             ← artefactos del modelo (8 archivos, ~30 MB)         │
   │  cache/             ← embeddings cacheados (272 MB npz)                  │
   │  pipeline_templates/ ← pipeline KFP compilado                           │
   │  pipeline_root/     ← artefactos de ejecucion del pipeline              │
   └───────────────────────────────────────────────────────────────────────────┘
```

Principio clave de diseno: **GCS es el contrato**. La imagen Docker es generica (no contiene el modelo). El pipeline de entrenamiento escribe artefactos en GCS. Cloud Run los descarga al arrancar. Esto desacopla entrenamiento de serving.

La misma imagen Docker sirve para ambos modos:
- `--mode train` en Vertex AI Custom Training Job
- `--mode serve` en Cloud Run

---

## 2. Requisitos previos

- Cuenta de GCP con billing habilitado
- `gcloud` CLI instalado y autenticado
- Docker instalado (para builds locales o Cloud Build)
- Dataset Jigsaw Toxic Comment Classification Challenge (`train.csv`)

### Autenticacion

```bash
gcloud auth login CUENTA@gmail.com
gcloud config set project mlops-toxic-classifier
```

---

## 3. Configuracion del proyecto GCP

### Crear el proyecto

```bash
gcloud projects create mlops-toxic-classifier \
  --name="MLOps Toxic Classifier"

PROJECT_NUM=$(gcloud projects describe mlops-toxic-classifier --format="value(projectNumber)")
# Resultado: 943214853579
```

### Vincular billing

```bash
gcloud billing accounts list
# ACCOUNT_ID: 017496-4917BA-727421

gcloud billing projects link mlops-toxic-classifier \
  --billing-account=017496-4917BA-727421
```

### Habilitar APIs necesarias

```bash
for API in \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  cloudfunctions.googleapis.com \
  pubsub.googleapis.com \
  cloudscheduler.googleapis.com \
  bigquery.googleapis.com; do
  gcloud services enable $API --project=mlops-toxic-classifier
done
```

---

## 4. Infraestructura paso a paso

### 4.1 Cloud Storage

```bash
gsutil mb -p mlops-toxic-classifier -l us-central1 gs://mlops-toxic-classifier-ml/
```

Estructura del bucket:

```
gs://mlops-toxic-classifier-ml/
  train.csv                           # Dataset completo (159,571 filas)
  model/
    svc_toxic.joblib                  # LinearSVC calibrado por etiqueta
    svc_severe_toxic.joblib
    svc_obscene.joblib
    svc_threat.joblib
    svc_insult.joblib
    svc_identity_hate.joblib
    tfidf_charwb_2_5.joblib           # Vectorizador TF-IDF
    metadata.json                     # Umbrales, metricas, config
  cache/
    nomic_embeddings_full.npz         # 768-d embeddings para 159k textos (272 MB)
  pipeline_templates/
    mlops_pipeline.json               # Pipeline KFP compilado
  pipeline_root/                      # Artefactos de ejecucion
```

### 4.2 Artifact Registry

```bash
gcloud artifacts repositories create mlops-containers \
  --repository-format=docker \
  --location=us-central1 \
  --project=mlops-toxic-classifier
```

### 4.3 Secret Manager

```bash
# Crear secreto con la API key de Synthetic (para nomic-embed)
echo -n "syn_aa37e9b92fa823a7b7a9eab01f24ad06" | \
  gcloud secrets create synthetic-api-key \
    --data-file=- \
    --project=mlops-toxic-classifier
```

---

## 5. Almacenamiento y datos en GCS

### Subir datos

```bash
gsutil cp raw/juegos/train.csv gs://mlops-toxic-classifier-ml/train.csv
```

### Subir artefactos del modelo (si se pre-entreno localmente)

```bash
gsutil -m cp reports/training/model_final/*.joblib gs://mlops-toxic-classifier-ml/model/
gsutil cp reports/training/model_final/metadata.json gs://mlops-toxic-classifier-ml/model/
```

### Subir cache de embeddings

```bash
gsutil cp data/nomic_embeddings_full.npz gs://mlops-toxic-classifier-ml/cache/nomic_embeddings_full.npz
```

El cache permite que el Custom Training Job salte la fase de computar embeddings via API (~67 minutos) y en su lugar descargue el archivo npz desde GCS (~30 segundos).

---

## 6. Imagen Docker y Artifact Registry

### Dockerfile

La imagen es generica: no contiene el modelo. Soporta dos modos via el entrypoint.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    scikit-learn==1.6.1 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    joblib==1.4.2 \
    requests==2.32.3 \
    pydantic==2.9.2 \
    pandas>=2.2.0 \
    google-cloud-storage>=2.14.0 \
    google-cloud-secret-manager>=2.20.0

COPY src/ /app/src/
RUN mkdir -p /app/model
COPY src/serving/train.py /app/entrypoint.py

ENV MODEL_DIR=/app/model
ENV PORT=8080
ENV PYTHONPATH=/app
ENV GCS_MODEL_URI=""
ENV PROJECT_ID=""

EXPOSE 8080

ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["--mode", "serve"]
```

### .dockerignore

Excluye datos, modelos locales, reportes y pipeline code para mantener la imagen pequena.

```
*.pyc
__pycache__
.git
.venv
data/
raw/
reports/
docs/
pipeline/
*.zip
*.pdf
*.typ
```

### Construir y subir (manual)

Con Cloud Build manual:

```bash
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/mlops-toxic-classifier/mlops-containers/toxic-classifier:latest
```

Imagen resultante: `us-central1-docker.pkg.dev/mlops-toxic-classifier/mlops-containers/toxic-classifier:latest`

Tamano aprox: ~550 MB (Python 3.11 + sklearn + deps).

### CI/CD automatico con Cloud Build trigger

Cada push a la rama `main` del repo `AOx0/cloud-computing-project` dispara automaticamente un build que construye la imagen, la sube a Artifact Registry y despliega en Cloud Run.

#### Configuracion de la conexion GitHub

Cloud Build usa una conexion GitHub v2 para recibir eventos de push via la GitHub App.

```bash
# Crear la conexion (requiere autorizacion OAuth en el navegador)
gcloud builds connections create github github-conn \
  --project=mlops-toxic-classifier \
  --region=us-central1

# Despues de autorizar, vincular el repo
gcloud builds repositories create cloud-computing-project \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --connection=github-conn \
  --remote-uri="https://github.com/AOx0/cloud-computing-project.git"
```

La autorizacion OAuth requiere abrir un enlace en el navegador, iniciar sesion con la cuenta de Google vinculada al proyecto y autorizar Cloud Build en GitHub. Despues, instalar la Cloud Build GitHub App en el repo desde https://github.com/settings/installations.

#### Crear el trigger

```bash
gcloud builds triggers create github \
  --name="build-on-push" \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --repository="projects/mlops-toxic-classifier/locations/us-central1/connections/github-conn/repositories/cloud-computing-project" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml" \
  --service-account="projects/mlops-toxic-classifier/serviceAccounts/943214853579-compute@developer.gserviceaccount.com"
```

#### cloudbuild.yaml

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      [
        'build',
        '-t',
        'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:$COMMIT_SHA',
        '-t',
        'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:latest',
        '.',
      ]

  - name: 'gcr.io/cloud-builders/docker'
    args:
      [
        'push', '--all-tags',
        'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier',
      ]

  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      [
        'run', 'deploy', 'toxic-comment-classifier',
        '--image=us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:$COMMIT_SHA',
        '--region=us-central1',
        '--platform=managed',
        '--allow-unauthenticated',
      ]

options:
  logging: CLOUD_LOGING_ONLY

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:$COMMIT_SHA'
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:latest'
```

La opcion `CLOUD_LOGING_ONLY` es necesaria cuando se usa un service account propio (BYOSA). Sin ella, Cloud Build intenta crear un bucket de logs y falla con error de permisos.

#### Permisos necesarios para CI/CD

```bash
# Cloud Build SA necesita actuar como compute SA para deploy a Cloud Run
gcloud iam service-accounts add-iam-policy-binding \
  943214853579-compute@developer.gserviceaccount.com \
  --member="serviceAccount:943214853579@cloudbuild.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --project=mlops-toxic-classifier

# Compute SA necesita rol de developer en Cloud Run
gcloud projects add-iam-policy-binding mlops-toxic-classifier \
  --member="serviceAccount:943214853579-compute@developer.gserviceaccount.com" \
  --role="roles/run.developer"

# Cloud Build P4SA necesita acceso a Secret Manager (para la conexion GitHub)
gcloud projects add-iam-policy-binding mlops-toxic-classifier \
  --member="serviceAccount:service-943214853579@gcp-sa-cloudbuild.iam.gserviceaccount.com" \
  --role="roles/secretmanager.admin"
```

#### Verificar el build

```bash
# Listar builds recientes
gcloud builds list --project=mlops-toxic-classifier --limit=5

# Disparar manualmente (para probar sin hacer push)
gcloud builds triggers run build-on-push \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --branch=main

# Ver revisiones desplegadas en Cloud Run
gcloud run revisions list \
  --service=toxic-comment-classifier \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --limit=3
```

#### Recursos desplegados

| Recurso | Nombre | Detalle |
|---|---|---|
| Conexion GitHub | `github-conn` | region us-central1, usuario AOx0 |
| Repositorio vinculado | `cloud-computing-project` | `AOx0/cloud-computing-project.git` |
| Trigger | `build-on-push` | branch `^main$`, cloudbuild.yaml |
| Service account (trigger) | compute SA | BYOSA con CLOUD_LOGING_ONLY |

---

## 7. Cache de embeddings

Los embeddings de nomic-embed-text-v1.5 (768 dimensiones) son costosos de computar para 159k textos (~67 minutos via API). El cache evita recomputarlos en cada reentrenamiento.

### Como funciona

El entrypoint `train.py` acepta `--gcs-embeddings-cache-uri`:

1. Si el URI existe en GCS y el hash de los IDs coincide, descarga el npz y lo usa directamente.
2. Si no existe o el hash no coincide, computa via API y sube el nuevo cache a GCS.
3. Si no hay API key ni cache, entrena sin embeddings (solo TF-IDF).

### Generar el cache localmente

```bash
uv run python src/trainer/embeddings_experiment.py
# Genera data/nomic_embeddings_full.npz (272 MB)
```

### Subir a GCS

```bash
gsutil cp data/nomic_embeddings_full.npz \
  gs://mlops-toxic-classifier-ml/cache/nomic_embeddings_full.npz
```

### Formato del npz

```python
# Contenido del archivo npz:
# embeddings: np.float32, shape (159571, 768)
# data_hash:  hash de los IDs para validacion de integridad
# n_rows:     159571
```

---

## 8. Despliegue en Cloud Run (serving)

```bash
gcloud run deploy toxic-comment-classifier \
  --image=us-central1-docker.pkg.dev/mlops-toxic-classifier/mlops-containers/toxic-classifier:latest \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="\
SYNTHETIC_API_KEY=syn_aa37e9b92fa823a7b7a9eab01f24ad06,\
GCS_MODEL_URI=gs://mlops-toxic-classifier-ml/model,\
PROJECT_ID=mlops-toxic-classifier" \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=3 \
  --port=8080 \
  --timeout=120
```

### URL del servicio

```
https://toxic-comment-classifier-943214853579.us-central1.run.app
```

### Comportamiento al arrancar

1. Cloud Run inicia el contenedor con `--mode serve`.
2. `predictor.py` detecta que no hay modelo local en `/app/model`.
3. Si `GCS_MODEL_URI` esta configurado, descarga los 8 archivos desde GCS.
4. Carga los 6 LinearSVC calibrados + TF-IDF en memoria.
5. El endpoint `/health` responde `healthy` con `model_source=gcs:gs://...`.

### Verificar despliegue

```bash
# Health check
curl https://toxic-comment-classifier-943214853579.us-central1.run.app/health

# Prediccion
curl -X POST https://toxic-comment-classifier-943214853579.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["You are a stupid idiot", "Thank you for your help"]}'
```

Respuesta esperada:

```json
{
  "predictions": [
    {
      "text": "You are a stupid idiot",
      "probabilities": {
        "toxic": 1.0,
        "severe_toxic": 0.04,
        "obscene": 0.9985,
        "threat": 0.001,
        "insult": 1.0,
        "identity_hate": 0.03
      },
      "labels": {
        "toxic": true,
        "severe_toxic": false,
        "obscene": true,
        "threat": false,
        "insult": true,
        "identity_hate": false
      }
    }
  ],
  "model_info": {
    "algorithm": "LinearSVC + CalibratedClassifierCV (sigmoid)",
    "features": "TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d)",
    "auc_macro": 0.9903
  },
  "latency_ms": 450
}
```

---

## 9. Entrenamiento via Vertex AI Custom Job

### Lanzar un Custom Training Job

```bash
TOKEN=$(gcloud auth print-access-token)

curl -X POST \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/mlops-toxic-classifier/locations/us-central1/customJobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "jigsaw-toxic-training",
    "jobSpec": {
      "workerPoolSpecs": [{
        "machineSpec": {
          "machineType": "n1-highmem-4"
        },
        "replicaCount": 1,
        "containerSpec": {
          "imageUri": "us-central1-docker.pkg.dev/mlops-toxic-classifier/mlops-containers/toxic-classifier:latest",
          "command": ["python", "/app/entrypoint.py"],
          "args": [
            "--mode", "train",
            "--project-id", "mlops-toxic-classifier",
            "--gcs-data-uri", "gs://mlops-toxic-classifier-ml/train.csv",
            "--gcs-output-uri", "gs://mlops-toxic-classifier-ml/model",
            "--model-dir", "/tmp/model",
            "--gcs-embeddings-cache-uri", "gs://mlops-toxic-classifier-ml/cache/nomic_embeddings_full.npz"
          ],
          "env": [
            {"name": "SYNTHETIC_API_KEY", "value": "syn_aa37e9b92fa823a7b7a9eab01f24ad06"}
          ]
        }
      }],
      "serviceAccount": "mlops-vertex-pipeline@mlops-toxic-classifier.iam.gserviceaccount.com"
    }
  }'
```

### Tipo de maquina

| Tipo | CPU | RAM | Notas |
|---|---|---|---|
| n1-standard-4 | 4 | 15 GB | OOM con TF-IDF + embeddings |
| **n1-highmem-4** | 4 | **26 GB** | Funciona correctamente |

El primer intento con `n1-standard-4` fallo por OOM. El TF-IDF char_wb (194k features) concatenado con 768-d embeddings para 159k filas requiere mas de 15 GB en la fase de fit.

### Monitorear el job

```bash
JOB_ID=<job_id_devuelto>

# Estado
TOKEN=$(gcloud auth print-access-token)
curl -s \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/mlops-toxic-classifier/locations/us-central1/customJobs/$JOB_ID" \
  -H "Authorization: Bearer $TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'State: {data.get(\"state\",\"\")}')
print(f'End: {data.get(\"endTime\",\"\")}')
if 'error' in data:
    print(f'Error: {data[\"error\"][\"message\"][:300]}')
"

# Verificar artefactos en GCS
gsutil ls -l gs://mlops-toxic-classifier-ml/model/
```

### Resultado del entrenamiento exitoso

El job escribe en GCS los artefactos actualizados. El archivo `metadata.json` contiene:

```json
{
  "model_type": "LinearSVC + CalibratedClassifierCV (sigmoid, cv=3)",
  "features": "TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d)",
  "tfidf_features": 194794,
  "embedding_features": 768,
  "label_cols": ["toxic","severe_toxic","obscene","threat","insult","identity_hate"],
  "metrics": {
    "toxic":          {"auc_train": 0.9927},
    "severe_toxic":   {"auc_train": 0.9964},
    "obscene":        {"auc_train": 0.9971},
    "threat":         {"auc_train": 0.9995},
    "insult":         {"auc_train": 0.9937},
    "identity_hate":  {"auc_train": 0.9983}
  },
  "f2_optimal_thresholds": {
    "toxic": 0.15, "severe_toxic": 0.10, "obscene": 0.10,
    "threat": 0.15, "insult": 0.15, "identity_hate": 0.10
  }
}
```

---

## 10. Pipeline de Vertex AI (KFP)

### Compilar el pipeline

```bash
uv run python pipeline/compile_pipeline.py
# Genera pipeline/compiled/mlops_pipeline.json
```

### Subir la plantilla a GCS

```bash
gsutil cp pipeline/compiled/mlops_pipeline.json \
  gs://mlops-toxic-classifier-ml/pipeline_templates/mlops_pipeline.json
```

### Nota sobre compatibilidad de esquemas

Los componentes de `google_cloud_pipeline_components` v1 usan `schemaVersion: 0.0.1` en los tipos de artefactos, que Vertex AI Pipelines ya no acepta (requiere >= 2.0.0). Por ello, el pipeline se ejecuta como Custom Training Job directo en vez de usar GCPC v1 components. Si se migra a componentes nativos v2 en el futuro, se podra usar `templateUri` en Vertex AI Pipeline Jobs.

### Ejecutar pipeline via REST API

```bash
TOKEN=$(gcloud auth print-access-token)

curl -X POST \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/mlops-toxic-classifier/locations/us-central1/pipelineJobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "jigsaw-toxic-pipeline",
    "templateUri": "gs://mlops-toxic-classifier-ml/pipeline_templates/mlops_pipeline.json",
    "runtimeConfig": {
      "parameters": {
        "project_id": {"stringValue": "mlops-toxic-classifier"},
        "region": {"stringValue": "us-central1"},
        "gcs_data_uri": {"stringValue": "gs://mlops-toxic-classifier-ml/train.csv"},
        "pipeline_root": {"stringValue": "gs://mlops-toxic-classifier-ml/pipeline_root"},
        "model_display_name": {"stringValue": "jigsaw-toxic-svc-embed"},
        "endpoint_display_name": {"stringValue": "jigsaw-toxic-endpoint"},
        "deploy_threshold": {"doubleValue": 0.95}
      },
      "gcsOutputDirectory": "gs://mlops-toxic-classifier-ml/pipeline_root"
    },
    "serviceAccount": "mlops-vertex-pipeline@mlops-toxic-classifier.iam.gserviceaccount.com"
  }'
```

### Pipeline actual (componentes custom)

El pipeline compilado (`pipeline/pipeline.py`) define estos pasos:

1. **validate_data** — verifica filas, columnas de etiqueta, textos no vacios, balance de clases
2. **CustomTrainingJobOp** — ejecuta la imagen Docker con `--mode train`
3. **evaluate_model** — calcula AUC por etiqueta y AUC macro
4. **condicion de despliegue** — si AUC macro >= umbral, despliega a Vertex AI Endpoint

Los componentes custom estan en `pipeline/components/`:

- `data_validation.py` — validacion de esquema del dataset Jigsaw
- `evaluate.py` — evaluacion de metricas por etiqueta
- `preprocessing.py` — limpieza de texto
- `train.py` — entrenamiento del modelo
- `deploy.py` — despliegue a endpoint

---

## 11. API endpoints

### GET /health

Verifica que los modelos estan cargados.

```bash
curl https://toxic-comment-classifier-943214853579.us-central1.run.app/health
```

Respuesta:

```json
{
  "status": "healthy",
  "models_loaded": true,
  "n_labels": 6,
  "model_source": "gcs:gs://mlops-toxic-classifier-ml/model"
}
```

### POST /predict

Clasifica una lista de textos (1-128).

```bash
curl -X POST https://toxic-comment-classifier-943214853579.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["texto a clasificar"]}'
```

Cada prediccion incluye probabilidades por etiqueta, etiquetas binarias (aplicando umbrales F2-optimal) y los umbrales usados.

### GET /model_info

Devuelve metadatos del modelo cargado.

```bash
curl https://toxic-comment-classifier-943214853579.us-central1.run.app/model_info
```

---

## 12. Cuentas de servicio e IAM

### Cuenta de servicio para pipelines

```bash
gcloud iam service-accounts create mlops-vertex-pipeline \
  --display-name="MLOps Vertex AI Pipeline SA" \
  --project=mlops-toxic-classifier
```

Email: `mlops-vertex-pipeline@mlops-toxic-classifier.iam.gserviceaccount.com`

Roles asignados:

```bash
SA="mlops-vertex-pipeline@mlops-toxic-classifier.iam.gserviceaccount.com"

for ROLE in \
  aiplatform.user \
  storage.objectAdmin \
  storage.objectCreator \
  artifactregistry.reader \
  run.admin \
  logging.logWriter \
  secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding mlops-toxic-classifier \
    --member="serviceAccount:$SA" \
    --role="roles/$ROLE"
done
```

### Cuenta de servicio por defecto (compute)

Email: `943214853579-compute@developer.gserviceaccount.com`

Roles:

```bash
CSA="943214853579-compute@developer.gserviceaccount.com"

for ROLE in storage.objectAdmin artifactregistry.writer logging.logWriter; do
  gcloud projects add-iam-policy-binding mlops-toxic-classifier \
    --member="serviceAccount:$CSA" \
    --role="roles/$ROLE"
done
```

### Resumen de roles

| Cuenta de servicio | Roles |
|---|---|
| `mlops-vertex-pipeline@...` | aiplatform.user, storage.objectAdmin, storage.objectCreator, artifactregistry.reader, run.admin, logging.logWriter, secretmanager.secretAccessor, run.developer (Cloud Build deploy) |
| `943214853579-compute@...` | storage.objectAdmin, artifactregistry.writer, logging.logWriter, editor (por defecto), run.developer, iam.serviceAccountUser (Cloud Build impersona) |
| `943214853579@cloudbuild.gserviceaccount.com` | cloudbuild.builds.builder, storage.objectAdmin, artifactregistry.writer, iam.serviceAccountUser sobre compute SA |
| `service-943214853579@gcp-sa-cloudbuild.iam...` | secretmanager.admin (para conexion GitHub OAuth) |

---

## 13. Secret Manager

### Secreto: synthetic-api-key

Almacena la API key para el servicio de embeddings (Synthetic API, modelo nomic-embed-text-v1.5).

```bash
# Crear
echo -n "syn_aa37e9b92fa823a7b7a9eab01f24ad06" | \
  gcloud secrets create synthetic-api-key --data-file=-

# Verificar
gcloud secrets describe synthetic-api-key --project=mlops-toxic-classifier

# Acceder desde codigo (Python)
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
name = f"projects/mlops-toxic-classifier/secrets/synthetic-api-key/versions/latest"
response = client.access_secret_version(request={"name": name})
api_key = response.payload.data.decode("UTF-8")
```

---

## 14. Reentrenamiento automatico (implementado)

El reentrenamiento automatico esta implementado y verificado end-to-end. Cloud Scheduler publica un mensaje cada lunes a las 2 AM (CDMX) en un topico de Pub/Sub. La Cloud Function `trigger-retraining` recibe el evento y lanza un Vertex AI Custom Training Job via el SDK de aiplatform. El job sobreescribe los artefactos en GCS. La siguiente instancia de Cloud Run carga el modelo actualizado.

```
Cloud Scheduler (cron: 0 2 * * 1)
    |
    v
Pub/Sub topic "retrain-trigger"
    |
    v
Cloud Function "trigger-retraining" (gen 2, 512 MiB)
    |
    v
Vertex AI Custom Job (n1-highmem-4, ~13 min)
    |
    v
GCS model/ actualizado (8 archivos, ~30 MB)
    |
    v
Cloud Run carga modelo nuevo en siguiente cold start
```

### 14.1 Pub/Sub

```bash
gcloud pubsub topics create retrain-trigger --project=mlops-toxic-classifier
```

Topico: `projects/mlops-toxic-classifier/topics/retrain-trigger`

### 14.2 Cloud Function

Ubicacion del codigo: `src/cloud_function/main.py`

La funcion recibe el evento de Pub/Sub, genera un nombre de job con timestamp y lanza un Custom Training Job con `job.submit()` (fire-and-forget, no bloquea hasta completar). Usa el SDK `google-cloud-aiplatform` para crear el job.

Parametros del container spec:
- `--mode train`
- `--project-id mlops-toxic-classifier`
- `--gcs-data-uri gs://mlops-toxic-classifier-ml/train.csv`
- `--gcs-output-uri gs://mlops-toxic-classifier-ml/model`
- `--gcs-embeddings-cache-uri gs://mlops-toxic-classifier-ml/cache/nomic_embeddings_full.npz`
- `--synthetic-api-key-secret synthetic-api-key` (el job lee la key desde Secret Manager)

La API key no se pasa como env var. El job la lee desde Secret Manager usando `--project-id` y `--synthetic-api-key-secret`.

#### Desplegar la Cloud Function

```bash
gcloud functions deploy trigger-retraining \
  --gen2 \
  --region=us-central1 \
  --project=mlops-toxic-classifier \
  --runtime=python312 \
  --source=src/cloud_function/ \
  --entry-point=trigger_retraining \
  --trigger-topic=retrain-trigger \
  --service-account=mlops-vertex-pipeline@mlops-toxic-classifier.iam.gserviceaccount.com \
  --timeout=60 \
  --memory=512Mi \
  --min-instances=0 \
  --max-instances=1 \
  --no-allow-unauthenticated
```

Notas sobre la configuracion:
- **512 MiB** de memoria (256 MiB produce OOM con google-cloud-aiplatform)
- **timeout 60s** es suficiente porque `job.submit()` solo lanza el job y retorna (no bloquea)
- **service-account mlops-vertex-pipeline** es la identidad de runtime. Tiene `aiplatform.user`, `storage.objectAdmin`, `secretmanager.secretAccessor`
- **gen 2** usa Cloud Run como backend, necesita Eventarc habilitado

#### Dependencias

`src/cloud_function/requirements.txt`:
```
google-cloud-aiplatform>=1.68.0
```

### 14.3 Cloud Scheduler

```bash
gcloud scheduler jobs create pubsub retrain-monday-2am \
  --project=mlops-toxic-classifier \
  --location=us-central1 \
  --schedule="0 2 * * 1" \
  --time-zone="America/Mexico_City" \
  --topic=retrain-trigger \
  --message-body='{"trigger": "scheduled", "action": "retrain"}' \
  --attributes="action=retrain" \
  --description="Retrain toxic classifier every Monday at 2 AM CDMX"
```

Expresion cron `0 2 * * 1`: cada lunes a las 2:00 AM en horario de Ciudad de Mexico.

### 14.4 Permisos necesarios

```bash
# Habilitar Eventarc (requerido por Cloud Functions gen 2)
gcloud services enable eventarc.googleapis.com --project=mlops-toxic-classifier

# El SA de Cloud Build P4SA necesita acceso a Secret Manager
# (para almacenar el token OAuth de la conexion GitHub)
gcloud projects add-iam-policy-binding mlops-toxic-classifier \
  --member="serviceAccount:service-943214853579@gcp-sa-cloudbuild.iam.gserviceaccount.com" \
  --role="roles/secretmanager.admin"
```

### 14.5 Lanzar reentrenamiento manual

Para lanzar un reentrenamiento fuera del schedule semanal:

```bash
gcloud pubsub topics publish retrain-trigger \
  --project=mlops-toxic-classifier \
  --message='{"action": "retrain"}'
```

Esto publica un mensaje en el topico, la Cloud Function lo recibe y lanza el Custom Training Job.

### 14.6 Verificar el resultado

```bash
# Listar Custom Training Jobs recientes
gcloud ai custom-jobs list \
  --region=us-central1 \
  --project=mlops-toxic-classifier \
  --limit=5 \
  --sort-by=~create_time

# Verificar artefactos actualizados en GCS
gsutil ls -l gs://mlops-toxic-classifier-ml/model/

# Verificar que Cloud Run sirve el modelo actualizado
curl -s https://toxic-comment-classifier-943214853579.us-central1.run.app/health | python3 -m json.tool
```

### 14.7 Job de ejemplo (verificado)

| Campo | Valor |
|---|---|
| Job ID | `2196835210232856576` |
| Display name | `toxic-classifier-retrain-20260618-014006` |
| Maquina | `n1-highmem-4` (26 GB RAM) |
| Duracion | ~13 minutos |
| Estado | SUCCEEDED |
| Artefactos | 8 archivos en GCS, timestamps 2026-06-18T01:52:45-46Z |

---

## 15. CI/CD automatico con Cloud Build (implementado)

El CI/CD esta implementado y verificado end-to-end. Cada push a la rama `main` del fork `AOx0/cloud-computing-project` dispara automaticamente un build que construye la imagen Docker, la sube a Artifact Registry con dos tags (`$COMMIT_SHA` y `latest`) y despliega una nueva revision en Cloud Run.

```
Push a AOx0/cloud-computing-project (main)
    |
    v
GitHub App webhook -> Cloud Build
    |
    v
cloudbuild.yaml (3 steps):
  1. docker build  (tags: $COMMIT_SHA, latest)
  2. docker push --all-tags
  3. gcloud run deploy (nueva revision)
    |
    v
Cloud Run serving API actualizada (~2 min)
```

### 15.1 Conexion GitHub v2

Cloud Build usa una conexion GitHub v2 para recibir eventos de push via la GitHub App de Google Cloud Build.

```bash
# Crear la conexion (requiere autorizacion OAuth en el navegador)
gcloud builds connections create github github-conn \
  --project=mlops-toxic-classifier \
  --region=us-central1

# Despues de autorizar, vincular el repo
gcloud builds repositories create cloud-computing-project \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --connection=github-conn \
  --remote-uri="https://github.com/AOx0/cloud-computing-project.git"
```

La autorizacion OAuth abre un enlace en el navegador. Despues de autorizar con la cuenta de Google, hay que instalar la Cloud Build GitHub App en el repo desde https://github.com/settings/installations. La App se encarga de enviar los webhooks de push a Cloud Build.

### 15.2 Trigger

```bash
gcloud builds triggers create github \
  --name="build-on-push" \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --repository="projects/mlops-toxic-classifier/locations/us-central1/connections/github-conn/repositories/cloud-computing-project" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml" \
  --service-account="projects/mlops-toxic-classifier/serviceAccounts/943214853579-compute@developer.gserviceaccount.com"
```

### 15.3 cloudbuild.yaml

El archivo define tres pasos secuenciales. Las variables `$PROJECT_ID` y `$COMMIT_SHA` se sustituyen automaticamente por Cloud Build.

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      [
        'build',
        '-t', 'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:$COMMIT_SHA',
        '-t', 'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:latest',
        '.',
      ]

  - name: 'gcr.io/cloud-builders/docker'
    args:
      [
        'push', '--all-tags',
        'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier',
      ]

  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      [
        'run', 'deploy', 'toxic-comment-classifier',
        '--image=us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:$COMMIT_SHA',
        '--region=us-central1',
        '--platform=managed',
        '--allow-unauthenticated',
      ]

options:
  logging: CLOUD_LOGING_ONLY

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:$COMMIT_SHA'
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/mlops-containers/toxic-classifier:latest'
```

La opcion `CLOUD_LOGING_ONLY` es obligatoria cuando se usa un service account propio (BYOSA). Sin ella, Cloud Build intenta crear un bucket de logs y falla con error de permisos.

### 15.4 Permisos necesarios

```bash
# Cloud Build SA necesita actuar como compute SA para deploy en Cloud Run
gcloud iam service-accounts add-iam-policy-binding \
  943214853579-compute@developer.gserviceaccount.com \
  --member="serviceAccount:943214853579@cloudbuild.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --project=mlops-toxic-classifier

# Compute SA necesita rol de developer en Cloud Run
gcloud projects add-iam-policy-binding mlops-toxic-classifier \
  --member="serviceAccount:943214853579-compute@developer.gserviceaccount.com" \
  --role="roles/run.developer"

# Cloud Build P4SA necesita acceso a Secret Manager
# (para almacenar el token OAuth de la conexion GitHub)
gcloud projects add-iam-policy-binding mlops-toxic-classifier \
  --member="serviceAccount:service-943214853579@gcp-sa-cloudbuild.iam.gserviceaccount.com" \
  --role="roles/secretmanager.admin"
```

### 15.5 Disparar manualmente

Para probar el build sin hacer push:

```bash
gcloud builds triggers run build-on-push \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --branch=main
```

### 15.6 Verificar

```bash
# Listar builds recientes
gcloud builds list --project=mlops-toxic-classifier --limit=5

# Ver revisiones desplegadas
gcloud run revisions list \
  --service=toxic-comment-classifier \
  --project=mlops-toxic-classifier \
  --region=us-central1 \
  --limit=3

# Verificar que la API sirve el modelo actualizado
curl -s https://toxic-comment-classifier-943214853579.us-central1.run.app/health
```

### 15.7 Recursos desplegados

| Recurso | Nombre | Detalle |
|---|---|---|
| Conexion GitHub | `github-conn` | region us-central1, usuario AOx0 |
| Repositorio vinculado | `cloud-computing-project` | `AOx0/cloud-computing-project.git` |
| Trigger | `build-on-push` | branch `^main$`, cloudbuild.yaml |
| Service account | compute SA | BYOSA con CLOUD_LOGING_ONLY |
| Duracion tipica | ~2 min | build + push + deploy |
| Builds verificados | `ec5e6a47...` | SUCCESS, disparado por push automatico |

## 16. Monitoreo

### Alertas recomendadas en Cloud Monitoring

| Alerta | Condicion | Severidad |
|---|---|---|
| Pipeline failure | Custom Job state = FAILED | Critical |
| Cloud Run 5xx | rate > 1% por 5 min | Warning |
| Cloud Run latency | p95 > 2s por 5 min | Warning |
| Embedding API errors | error rate > 5% | Warning |
| Sin trafico | 0 requests en 1 hora | Info |

### Logs relevantes

```bash
# Logs de Cloud Run
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="toxic-comment-classifier"' \
  --project=mlops-toxic-classifier --limit=50

# Logs de Custom Training Job
gcloud logging read 'resource.type="aiplatform_custom_job"' \
  --project=mlops-toxic-classifier --limit=50

# Logs de Cloud Build
gcloud logging read 'resource.type="cloud_build"' \
  --project=mlops-toxic-classifier --limit=20
```

### Drift detection

Para produccion, se recomienda configurar Vertex AI Model Monitoring con:
- Baseline: datos de entrenamiento
- Feature drift: distribucion de longitud de texto, frecuencia de TF-IDF features
- Prediction drift: distribucion de probabilidades por etiqueta
- Alerting: cuando drift excede threshold configurable

---

## 17. Estructura del repositorio

```
cloud-computing-project/
  Dockerfile                    # Imagen generica (train + serve)
  .dockerignore
  cloudbuild.yaml               # CI/CD pipeline: build + push + deploy
  cloudbuild-analysis.yaml      # Cloud Build config para EDA
  docs/
    analysis_report.md          # Reporte completo del analisis
    architecture.md             # Diagramas de arquitectura original
    mlops_workflow.md            # Flujo MLOps general
    monitoring_strategy.md      # Estrategia de monitoreo
    pitch.md                    # Sales pitch
    mlops_platform.md           # Este documento
  pipeline/
    pipeline.py                 # Definicion KFP del pipeline
    compile_pipeline.py         # Script para compilar
    compiled/
      mlops_pipeline.json       # Pipeline compilado
    components/
      data_validation.py        # Validacion de datos Jigsaw
      evaluate.py               # Evaluacion de metricas
      preprocessing.py          # Preprocesamiento
      train.py                  # Componente de entrenamiento
      deploy.py                 # Componente de despliegue
  raw/
    juegos/                     # Dataset original
      train.csv                 # 159,571 filas, 8 columnas
      test.csv
      test_labels.csv
      sample_submission.csv
  reports/
    eda/
      main.typ / main.pdf       # Reporte typst compilado
      imgs/                     # 44+ figuras del EDA
    training/
      model_final/              # Artefactos del modelo final
      model_svc/                # Artefactos del modelo sin embeddings
      imgs/                     # Figuras de comparacion de modelos
      *.json                    # Metricas de cada experimento
  src/
    analysis_toxic_comments.py  # EDA principal
    analysis_sentiment.py       # Analisis VADER
    analysis_empath.py          # Analisis EMPATH
    serving/
      predictor.py              # FastAPI con GCS-first model loading
      train.py                  # Entrypoint dual (train/serve)
    cloud_function/
      main.py                   # Cloud Function: lanza retraining job
      requirements.txt           # google-cloud-aiplatform
    trainer/
      features.py               # FeaturePipeline (TF-IDF + VADER + EMPATH)
      model.py                  # ClassifierChainLGBM
      evaluation.py             # Evaluacion multi-label con bootstrap
      train.py                  # Script de entrenamiento local
      compare_models.py         # Comparacion de 6 modelos CPU
      embeddings_experiment.py  # Experimento TF-IDF vs embeddings
      nb_variants.py            # Variantes de Naive Bayes
      charwb_ridge.py           # Experimento char_wb + Ridge
      validation.py             # Validacion estadistica rigurosa
  statistical_toolbelt/         # Libreria de analisis estadistico
  data/                         # Caches locales (no en git)
    sentiment_scores.csv
    empath_scores.parquet
    nomic_embeddings_full.npz   # 272 MB
```

---

## 18. Modelo final

### Algoritmo

6 clasificadores binarios independientes, uno por etiqueta de toxicidad.

- **Base**: LinearSVC (C=0.1, class_weight=balanced, max_iter=5000)
- **Calibracion**: CalibratedClassifierCV (method=sigmoid, cv=3)
- **Features**: TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d)

### Metricas (validacion cruzada local)

| Etiqueta | AUC | F1 (F2-optimal) | Umbral |
|---|---|---|---|
| toxic | 0.9927 | 0.79 | 0.15 |
| severe_toxic | 0.9964 | 0.42 | 0.10 |
| obscene | 0.9971 | 0.83 | 0.10 |
| threat | 0.9995 | 0.45 | 0.15 |
| insult | 0.9937 | 0.73 | 0.15 |
| identity_hate | 0.9983 | 0.46 | 0.10 |
| **Macro** | **0.9903** | **0.6388** | — |

### Por que LinearSVC + char_wb

La comparacion de modelos demostro que:
- LinearSVC supera a LogisticRegression, NaiveBayes, Ridge y LightGBM en AUC macro.
- char_wb (2,5) supera a word (1,2) en todas las etiquetas, especialmente en identity_hate (+0.011 AUC) porque captura ofuscacion de insultos.
- Los embeddings nomic-embed aportan ganancia complementaria en threat (+0.006) e identity_hate (+0.006) donde la dimension semantica importa mas que la ortografica.
- La combinacion TF-IDF + embeddings (AUC 0.9903) supera a cada uno individualmente en las 6 etiquetas.

### Umbrales F2-optimal

Los umbrales de decision se optimizaron para el F2-score (beta=2), que prioriza recall sobre precision. Esto es apropiado para moderacion de contenido donde es mas costoso dejar pasar un comentario toxico que falsamente marcar uno innocuo.

---

## 19. Troubleshooting

### OOM en Custom Training Job

Si el job falla con "Replicas low on memory":

```
Error: Replicas low on memory: workerpool0. Specify a machine with larger memory and try again.
```

Solucion: usar `n1-highmem-4` (26 GB) en vez de `n1-standard-4` (15 GB). El TF-IDF char_wb con 194k features + 768-d embeddings para 159k filas supera 15 GB en la fase de fit.

### Cloud Run cold start lento

El primer request despues de inactividad puede tardar 30-60 segundos porque:
1. Cloud Run levanta una nueva instancia.
2. El contenedor descarga los 8 archivos del modelo desde GCS (~30 MB).
3. Carga los modelos joblib en memoria.

Para reducir latencia de cold start, se puede configurar `--min-instances=1` (mantiene una instancia caliente, pero genera costo continuo).

### Pipeline KFP con error de schema version

```
Error: SchemaVersion < 2.0.0 are not longer supported.
```

Los componentes `google_cloud_pipeline_components` v1 usan `schemaVersion: 0.0.1` en los artifact types, incompatible con Vertex AI Pipelines actual. Solucion: usar Custom Training Jobs directos via REST API en vez de GCPC v1 components, o migrar a componentes v2 nativos.

### API key de Synthetic no funciona

Si la API de embeddings devuelve 403 o timeout:
1. Verificar que la key sea valida: `curl -H "Authorization: Bearer syn_..." https://api.synthetic.new/openai/v1/models`
2. Si se uso Secret Manager, verificar que el secreto tenga el valor correcto: `gcloud secrets versions access latest --secret=synthetic-api-key`
3. El entrypoint `train.py` busca la key primero en `SYNTHETIC_API_KEY` env var, luego en Secret Manager.

### Modelo no se actualiza en Cloud Run

Cloud Run carga el modelo desde GCS al arrancar. Si se reentrena y los artefactos en GCS cambian, las instancias existentes siguen usando el modelo viejo en memoria. Para forzar recarga:

1. Desplegar una nueva revision: `gcloud run services update toxic-comment-classifier --region=us-central1`
2. O esperar a que las instancias se reciclen (scale-to-zero y cold start).

---

## Referencia rapida de comandos

```bash
# === Configuracion inicial ===
gcloud auth login danielosorniolopez@gmail.com
gcloud config set project mlops-toxic-classifier

# === GCS ===
gsutil mb -p mlops-toxic-classifier -l us-central1 gs://mlops-toxic-classifier-ml/
gsutil cp raw/juegos/train.csv gs://mlops-toxic-classifier-ml/train.csv
gsutil cp data/nomic_embeddings_full.npz gs://mlops-toxic-classifier-ml/cache/

# === Artifact Registry ===
gcloud artifacts repositories create mlops-containers --repository-format=docker --location=us-central1

# === Docker build + push (manual) ===
gcloud builds submit --tag us-central1-docker.pkg.dev/mlops-toxic-classifier/mlops-containers/toxic-classifier:latest

# === CI/CD automatico ===
# Push a AOx0/cloud-computing-project (main) dispara build automatico
# Ver: seccion 15 de este documento

# === Reentrenamiento manual ===
gcloud pubsub topics publish retrain-trigger --project=mlops-toxic-classifier --message='{"action": "retrain"}'

# === Cloud Run deploy ===
gcloud run deploy toxic-comment-classifier \
  --image=us-central1-docker.pkg.dev/mlops-toxic-classifier/mlops-containers/toxic-classifier:latest \
  --region=us-central1 --platform=managed --allow-unauthenticated \
  --set-env-vars="SYNTHETIC_API_KEY=syn_...,GCS_MODEL_URI=gs://mlops-toxic-classifier-ml/model,PROJECT_ID=mlops-toxic-classifier" \
  --memory=2Gi --cpu=2 --min-instances=0 --max-instances=3 --port=8080 --timeout=120

# === Vertex AI Custom Training Job ===
# (ver seccion 9 para el comando curl completo)

# === Service Account ===
gcloud iam service-accounts create mlops-vertex-pipeline --display-name="MLOps Vertex AI Pipeline SA"
# (ver seccion 12 para roles completos)

# === Secret Manager ===
echo -n "syn_aa37e9b92fa823a7b7a9eab01f24ad06" | gcloud secrets create synthetic-api-key --data-file=-

# === Verificar API ===
curl https://toxic-comment-classifier-943214853579.us-central1.run.app/health
curl -X POST https://toxic-comment-classifier-943214853579.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test comment"]}'
```
