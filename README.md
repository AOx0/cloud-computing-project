# Jigsaw Toxic Comment Classifier - MLOps en GCP

Plataforma MLOps en Google Cloud Platform para el Jigsaw Toxic Comment Classification Challenge. Incluye analisis estadistico riguroso, seleccion de modelo justificada, pipeline de entrenamiento en Vertex AI, y API de prediccion en Cloud Run.

## Resultado principal

Modelo productivo: **LinearSVC + CalibratedClassifierCV** con features TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d). AUC macro **0.9903**, F1 macro **0.6388** en 6 etiquetas de toxicidad.

## Arquitectura

```
 Cloud Build ──build+push──▶ Artifact Registry
                                  │
                                  ▼
 Vertex AI Custom Training Job ────┤  (n1-highmem-4, --mode train)
   lee datos de GCS               │
   lee embeddings cache de GCS    │
   entrena 6 LinearSVC            │
   escribe artefactos a GCS ──────┤
                                  │
 Cloud Run ───────────────────────┘  (--mode serve)
   descarga modelo de GCS al arrancar
   sirve /predict + /health
```

Principio de diseno: **GCS es el contrato**. La imagen Docker es generica (no contiene el modelo). El pipeline escribe artefactos a GCS. Cloud Run los lee al arrancar. Misma imagen para entrenar y servir.

## GCP Resources

| Recurso | Identificador |
|---|---|
| Proyecto | `mlops-toxic-classifier` (943214853579) |
| Billing | `017496-4917BA-727421` |
| Region | `us-central1` |
| GCS bucket | `gs://mlops-toxic-classifier-ml/` |
| Artifact Registry | `us-central1-docker.pkg.dev/.../mlops-containers/toxic-classifier` |
| Cloud Run | `https://toxic-comment-classifier-943214853579.us-central1.run.app` |
| Service Account | `mlops-vertex-pipeline@mlops-toxic-classifier.iam.gserviceaccount.com` |
| Secret | `synthetic-api-key` (Synthetic API para nomic-embed) |

## Quick start

```bash
# Clonar
git clone https://github.com/jyaru1110/cloud-computing-project.git
cd cloud-computing-project

# Dependencias (requiere uv)
uv sync

# EDA + analisis estadistico
uv run python src/analysis_toxic_comments.py

# Entrenamiento local
uv run python src/trainer/train.py

# Probar la API (requiere gcloud auth)
curl https://toxic-comment-classifier-943214853579.us-central1.run.app/health
curl -X POST https://toxic-comment-classifier-943214853579.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["You are a stupid idiot"]}'
```

## Estructura del repositorio

```
cloud-computing-project/
  Dockerfile                    # Imagen generica (train + serve)
  .dockerignore
  cloudbuild-analysis.yaml      # Cloud Build para EDA
  docs/
    mlops_platform.md           # Documentacion operativa completa (comandos GCP)
    analysis_report.md          # Reporte del analisis estadistico (20 hipotesis)
    architecture.md             # Diagrama de arquitectura actual
    monitoring_strategy.md      # Estrategia de monitoreo
    pitch.md                    # Sales pitch
  pipeline/
    pipeline.py                 # Definicion KFP del pipeline
    compile_pipeline.py         # Script para compilar
    compiled/                   # Pipeline compilado (json)
    components/                 # Componentes custom (validacion, evaluacion, etc.)
  raw/juegos/                   # Dataset original (train.csv 159k filas)
  reports/
    eda/                        # Reporte Typst + 54 figuras
    training/                   # Metricas, modelos, figuras de comparacion
  src/
    analysis_toxic_comments.py  # EDA principal (H1-H5)
    analysis_sentiment.py       # Analisis VADER (H6-H8)
    analysis_empath.py          # Analisis EMPATH (H9-H11)
    serving/
      predictor.py              # FastAPI con GCS-first model loading
      train.py                  # Entrypoint dual (train/serve) + embeddings cache
    trainer/
      compare_models.py         # Comparacion 6 modelos (H12-H14)
      nb_variants.py            # Variantes NB (H15-H16)
      charwb_ridge.py           # char_wb + Ridge (H17-H18)
      embeddings_experiment.py  # TF-IDF vs embeddings (H19-H20)
      validation.py             # Validacion estadistica (bootstrap+DeLong+McNemar)
      evaluation.py             # Evaluacion multi-label (bootstrap CI, ECE, F2)
      features.py               # FeaturePipeline
      model.py                  # ClassifierChainLGBM
      train.py                  # Entrenamiento local
  statistical_toolbelt/         # Libreria de analisis estadistico
```

## Hipotesis evaluadas

20 hipotesis formuladas antes del modelado, evaluadas con el flujo hipotesis -> EDA -> pruebas -> modelos -> conclusion.

| # | Hipotesis | Resultado |
|---|---|---|
| H1 | Ningun par de etiquetas es independiente | Confirmada |
| H2 | Comentarios toxicos son mas cortos y con mas enfasis | Parcialmente confirmada |
| H3 | Accuracy no es metrica valida por desbalance | Confirmada |
| H4 | Baselines simples lejos de rendimiento competitivo | Confirmada |
| H5 | Etiquetas raras tienen metricas inestables | Confirmada |
| H6 | Sentimiento negativo asocia con toxicidad | Confirmada |
| H7 | VADER mejora AUC del baseline | Confirmada |
| H8 | VADER correlaciona menos con threat/identity_hate | Confirmada |
| H9 | EMPATH supera a VADER en correlacion | Parcialmente confirmada |
| H10 | EMPATH complementa a VADER como features | Confirmada |
| H11 | Categorias EMPATH especificas por etiqueta | Parcialmente confirmada |
| H12 | MNB mejor en prevalentes que en raras | Refutada |
| H13 | LinearSVC supera LR en F1 | Confirmada |
| H14 | Classifier Chain supera independientes | Refutada |
| H15 | Undersampling mejora NB vs balanced | Confirmada |
| H16 | TF-IDF sin max_features mejora NB | Refutada (vanilla), confirmada (undersample) |
| H17 | char_wb supera a word n-gramas | Confirmada |
| H18 | Ridge con target continuo supera LinearSVC | Refutada |
| H19 | Embeddings capturan senal que char_wb no ve | Parcialmente confirmada |
| H20 | TF-IDF + embeddings > cada uno por separado | Confirmada |

Detalle completo en `docs/analysis_report.md`.

## Metricas del modelo final

LinearSVC + CalibratedClassifierCV (sigmoid, cv=3), features TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d).

| Etiqueta | AUC | F1 | Umbral F2-opt |
|---|---|---|---|
| toxic | 0.9847 | 0.769 | 0.15 |
| severe_toxic | 0.9907 | 0.506 | 0.10 |
| obscene | 0.9935 | 0.785 | 0.10 |
| threat | 0.9944 | 0.546 | 0.15 |
| insult | 0.9877 | 0.740 | 0.15 |
| identity_hate | 0.9905 | 0.487 | 0.10 |
| **Macro** | **0.9903** | **0.639** | |

## Documentacion

| Documento | Contenido |
|---|---|
| `docs/mlops_platform.md` | Comandos GCP, IAM, deploy, troubleshooting |
| `docs/analysis_report.md` | 20 hipotesis, metricas, validacion estadistica |
| `docs/architecture.md` | Diagrama de arquitectura actual |
| `docs/monitoring_strategy.md` | Alertas, drift, reentrenamiento |
| `docs/pitch.md` | Propuesta de valor |

## Herramientas

- **GCP:** Vertex AI, Cloud Build, Cloud Run, Cloud Storage, Artifact Registry, Secret Manager
- **ML:** scikit-learn (LinearSVC, CalibratedClassifierCV), nomic-embed-text-v1.5 (Synthetic API)
- **Analisis:** VADER, EMPATH, scipy, statsmodels, numpy
- **MLOps:** KFP/Vertex AI Pipelines, Cloud Scheduler + Pub/Sub + Cloud Functions
- **Reportes:** Typst, matplotlib, seaborn
