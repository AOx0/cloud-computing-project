# Arquitectura

## Principio de diseno: GCS como contrato

La imagen Docker es generica. No contiene el modelo ni los datos. Esto desacopla entrenamiento de serving:

- El **Custom Training Job** lee datos de GCS, entrena el modelo y escribe artefactos a GCS.
- **Cloud Run** descarga los artefactos de GCS al arrancar (cold start) y los carga en memoria.
- Cuando el pipeline reentrena, los artefactos en GCS se actualizan. La siguiente instancia de Cloud Run carga el modelo nuevo.

La misma imagen sirve para ambos modos (`--mode train` y `--mode serve`) via el entrypoint `src/serving/train.py`.

## Componentes GCP

### Cloud Build + CI/CD automatico

Cloud Build ejecuta el CI/CD de forma automatica. Cada push a la rama `main` del fork `AOx0/cloud-computing-project` dispara un build que construye la imagen Docker, la sube a Artifact Registry (con tag `$COMMIT_SHA` y `latest`) y despliega la nueva revision en Cloud Run.

El trigger esta configurado via una conexion GitHub v2 (`github-conn`) que vincula el repo con Cloud Build. El archivo `cloudbuild.yaml` define los tres pasos del pipeline:

1. `docker build` con dos tags (`$COMMIT_SHA` + `latest`)
2. `docker push --all-tags` a Artifact Registry
3. `gcloud run deploy` para actualizar el servicio

- **Trigger:** `build-on-push` (v2, region us-central1)
- **Conexion GitHub:** `github-conn` (usuario autorizado: AOx0)
- **Repo:** `AOx0/cloud-computing-project`, rama `main`
- **Build config:** `cloudbuild.yaml` (3 steps, CLOUD_LOGGING_ONLY)
- **Duracion tipica:** ~2 minutos
- **Service account:** `943214853579-compute@developer.gserviceaccount.com`

### Cloud Storage

Almacena todos los artefactos que no pertenecen al codigo fuente:

| Ruta | Contenido |
|---|---|
| `train.csv` | Dataset de entrenamiento (159,571 filas, 65 MB) |
| `model/*.joblib` | 6 LinearSVC calibrados + TF-IDF vectorizador |
| `model/metadata.json` | Umbrales F2-optimal, metricas, configuracion |
| `cache/nomic_embeddings_full.npz` | Embeddings cacheados (272 MB) |
| `pipeline_templates/mlops_pipeline.json` | Pipeline KFP compilado |

### Artifact Registry

Almacena la imagen Docker `toxic-classifier:latest`. La imagen contiene Python 3.11, scikit-learn, FastAPI y el codigo fuente. No contiene el modelo.

### Vertex AI Custom Training Job

Ejecuta el entrenamiento en una VM efimera. Lee datos y embeddings de GCS, entrena 6 LinearSVC calibrados, sube artefactos a GCS. Tipo de maquina: `n1-highmem-4` (26 GB RAM). El primer intento con `n1-standard-4` (15 GB) fallo por OOM.

### Cloud Run

Sirve la API de prediccion. Lee el modelo de GCS al arrancar. Expone `/health`, `/predict`, `/model_info`. Escala a cero cuando no hay trafico. Cada prediccion computa TF-IDF localmente y llama la Synthetic API para embeddings.

### Cloud Scheduler + Pub/Sub + Cloud Function

Implementa el reentrenamiento automatico. Cloud Scheduler publica un mensaje cada lunes a las 2 AM (CDMX) en el topico `retrain-trigger` de Pub/Sub. La Cloud Function `trigger-retraining` recibe el evento y lanza un Vertex AI Custom Training Job via el SDK de aiplatform. El job sobreescribe los artefactos en GCS. La siguiente instancia de Cloud Run carga el modelo actualizado.

- **Topico Pub/Sub:** `projects/mlops-toxic-classifier/topics/retrain-trigger`
- **Cloud Function:** `trigger-retraining` (gen 2, Python 312, 512 MiB)
- **Scheduler:** `retrain-monday-2am`, cron `0 2 * * 1`, timezone `America/Mexico_City`
- **Training job:** `n1-highmem-4`, misma imagen Docker que Cloud Run

Verificado end-to-end: Scheduler -> Pub/Sub -> Cloud Function -> Custom Job -> GCS -> Cloud Run.

### Secret Manager

Almacena la API key de Synthetic (para nomic-embed-text-v1.5). El Custom Training Job puede leer la key desde Secret Manager si no se pasa como env var.

## Flujo de datos

1. El dataset (`train.csv`) se carga a GCS manualmente.
2. El Custom Training Job lee el dataset y el cache de embeddings desde GCS.
3. Si no hay cache de embeddings, computa via Synthetic API y sube el cache a GCS.
4. Entrena 6 LinearSVC + CalibratedClassifierCV sobre TF-IDF + embeddings.
5. Sube los 7 joblib + metadata.json a GCS.
6. Cloud Run lee los artefactos de GCS al arrancar y carga el modelo en memoria.
7. Cada request a `/predict` limpia el texto, computa TF-IDF localmente, obtiene embeddings via API, concatena y predice.

## Flujo CI/CD (automatico)

1. Push a `AOx0/cloud-computing-project` (rama `main`).
2. Cloud Build trigger `build-on-push` detecta el push via GitHub App webhook.
3. `cloudbuild.yaml` ejecuta 3 pasos:
   - `docker build` con tags `$COMMIT_SHA` y `latest`
   - `docker push --all-tags` a Artifact Registry
   - `gcloud run deploy toxic-comment-classifier` con la imagen `$COMMIT_SHA`
4. Cloud Run despliega una nueva revision con la imagen actualizada.
5. La siguiente solicitud a la API usa el codigo nuevo (cold start si escala a cero).

El flujo es completamente automatico. No requiere intervencion manual tras el push.

## Ciclo de vida MLOps

- **Ingesta de datos:** carga manual a GCS.
- **Validacion de datos:** componente KFP que verifica filas, etiquetas, textos no vacios, balance.
- **Feature engineering:** TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d).
- **Entrenamiento:** Vertex AI Custom Training Job.
- **Evaluacion:** AUC por etiqueta, AUC macro, gate de despliegue (>= 0.95).
- **Despliegue condicional:** si la metrica pasa, los artefactos ya estan en GCS. Cloud Run los carga al arrancar.
- **Serving:** Cloud Run con GCS-first model loading.
- **Monitoreo:** Cloud Logging, Cloud Monitoring, drift detection.
- **Reentrenamiento:** automatico via Cloud Scheduler (lunes 2 AM CDMX) -> Pub/Sub -> Cloud Function -> Custom Training Job. Tambien se puede lanzar manualmente publicando un mensaje al topico o via REST API.
- **Rollback:** versiones anteriores de artefactos en GCS, desplegar revision anterior en Cloud Run.
