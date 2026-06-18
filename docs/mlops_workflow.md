# Flujo MLOps

## 1. Ingesta de datos

El dataset Jigsaw Toxic Comment Classification Challenge (159,571 comentarios de Wikipedia talk pages) se carga a GCS como `train.csv`. No se usa BigQuery ni datos sinteticos. Los datos incluyen 6 etiquetas binarias: toxic, severe_toxic, obscene, threat, insult, identity_hate.

## 2. Validacion de datos

El componente `validate_data` (`pipeline/components/data_validation.py`) verifica:

- El CSV tiene filas (count > 0).
- Las 6 columnas de etiqueta existen.
- Ningun `comment_text` esta vacio.
- La prevalencia de cada etiqueta esta dentro del rango esperado (0.3% - 10%).

Para produccion, agregar reglas de negocio (longitud maxima, idioma, contenido prohibido) o herramientas como TensorFlow Data Validation.

## 3. Feature engineering

Dos representaciones complementarias concatenadas:

- **TF-IDF char_wb (2,5):** 194,794 features. Captura la dimension ortografica. `char_wb` respeta los limites de palabra, lo que permite detectar ofuscacion deliberada (`a$$hole`, `f*ck`) sin explosion de vocabulario.
- **nomic-embed-text-v1.5 (768d):** 768 features densos. Captura la dimension semantica via la Synthetic API con `task_type=classification`. Las amenazas indirectas y el odio encubierto producen embeddings distintos a los de texto innocuo.

Los embeddings se cachean en GCS (`cache/nomic_embeddings_full.npz`, 272 MB) con hash de los IDs para validar integridad. El cache evita ~67 minutos de llamadas API en cada reentrenamiento.

## 4. Entrenamiento

El Vertex AI Custom Training Job ejecuta `python entrypoint.py --mode train`:

1. Descarga `train.csv` de GCS.
2. Limpia texto (lowercase, URLs, IPs, caracteres especiales).
3. Computa TF-IDF char_wb sobre el texto limpio.
4. Descarga embeddings cacheados de GCS (o computa via API y sube cache).
5. Concatena TF-IDF (sparse) + embeddings (dense) con `scipy.sparse.hstack`.
6. Para cada etiqueta, entrena LinearSVC (C=0.1, class_weight=balanced) con CalibratedClassifierCV (sigmoid, cv=3).
7. Sube los 7 joblib + metadata.json a GCS.

Tipo de maquina: `n1-highmem-4` (26 GB RAM). `n1-standard-4` (15 GB) produce OOM.

## 5. Evaluacion

El componente `evaluate_model` (`pipeline/components/evaluate.py`) calcula:

- AUC-ROC por etiqueta.
- AUC macro (promedio simple).
- Decision de despliegue: `auc_macro >= deploy_threshold` (default 0.95).

Si la metrica no pasa, el pipeline guarda el reporte y se detiene antes del despliegue.

## 6. Despliegue condicional

Si la metrica pasa, los artefactos ya estan en GCS. Cloud Run los carga al arrancar. No se usa Vertex AI Model Registry ni Vertex AI Endpoints porque el modelo es pequeno (30 MB) y predice en milisegundos. Cloud Run escala a cero y es mas economico para trafico intermitente.

## 7. Serving

Cloud Run ejecuta `python entrypoint.py --mode serve`. La app FastAPI (`src/serving/predictor.py`):

1. Al arrancar, descarga los artefactos del modelo desde GCS si no existen localmente.
2. Carga TF-IDF + 6 LinearSVC calibrados en memoria.
3. Cada request a `/predict` limpia el texto, computa TF-IDF localmente, obtiene embeddings via Synthetic API, concatena y predice.
4. Retorna probabilidades por etiqueta y etiquetas binarias aplicando umbrales F2-optimal.

Endpoints:

| Ruta | Metodo | Descripcion |
|---|---|---|
| `/health` | GET | Estado del modelo |
| `/predict` | POST | Prediccion de toxicidad |
| `/model_info` | GET | Metadatos del modelo |

## 8. Reentrenamiento

Cloud Scheduler publica un mensaje semanal en Pub/Sub. La Cloud Function `trigger-retraining` recibe el evento y lanza un Vertex AI Custom Training Job con los parametros configurados (datos, cache de embeddings, URI de salida). El job sobreescribe los artefactos en GCS. La siguiente instancia de Cloud Run carga el modelo actualizado.

Disparo manual:

```bash
gcloud pubsub topics publish retrain-trigger \
  --message='{"reason":"manual-retraining"}'
```

## 9. Rollback

Los artefactos previos pueden archivarse en GCS con versiones (`model/v1/`, `model/v2/`). Para revertir, copiar la version anterior a `model/` y reiniciar Cloud Run (forzar cold start).

```bash
# Archivar modelo actual
gsutil -m cp gs://mlops-toxic-classifier-ml/model/* gs://mlops-toxic-classifier-ml/model_archive/v1/

# Restaurar version anterior
gsutil -m cp gs://mlops-toxic-classifier-ml/model_archive/v0/* gs://mlops-toxic-classifier-ml/model/

# Forzar cold start en Cloud Run
gcloud run services update toxic-comment-classifier --region=us-central1 --no-traffic
```
