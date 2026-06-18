# Estrategia de Monitoreo

## Data drift

El drift en datos de entrada ocurre cuando la distribucion de los comentarios de produccion difiere de la del dataset de entrenamiento. Para un clasificador de texto, las senales relevantes son:

- Longitud media de comentarios (los comentarios de Wikipedia pueden ser distintos a los de redes sociales).
- Distribucion de TF-IDF features (nuevas formas de ofuscacion, neologismos).
- Prevalencia de categorias EMPATH (cambio en el tipo de contenido hostil).
- Distribucion de embeddings (cambio semantico en el vocabulario).

Vertex AI Model Monitoring puede configurarse con baseline del dataset de entrenamiento y comparar contra requests de produccion. Para texto, se recomienda monitorear las estadisticas agregadas (longitud, features mas activos) en vez de comparar feature por feature.

## Prediction drift

Prediction drift ocurre cuando la distribucion de probabilidades de salida cambia. Ejemplos concretos para este modelo:

- El modelo predice `toxic` en mas del 20% de los comentarios cuando antes predicia en 10%.
- La media de `prob(threat)` sube de 0.02 a 0.10 sin cambio aparente en el input.
- Una etiqueta especifica (e.g. identity_hate) comienza a activarse con frecuencia inusual.

Prediction drift no siempre indica que el modelo esta mal, pero senala que el entorno cambio y el modelo debe revisarse.

## Degradacion del modelo

La degradacion requiere etiquetas reales (ground truth) para medirse. En moderacion de contenido, las etiquetas llegan con demora (revision humana). La arquitectura debe almacenar predicciones y, cuando las etiquetas humanas esten disponibles, comparar las metricas actuales contra las del entrenamiento.

Para este modelo, las metricas a monitorear son:

| Metrica | Valor baseline (entrenamiento) | Umbral de alerta |
|---|---|---|
| AUC macro | 0.9903 | < 0.97 |
| F1 macro | 0.6388 | < 0.55 |
| AUC threat | 0.9944 | < 0.98 |
| AUC identity_hate | 0.9905 | < 0.97 |

Estas etiquetas (threat, identity_hate) son las mas vulnerables a degradacion porque su prevalencia es baja (0.3% y 0.88%) y la senal es mas semantica que ortografica.

## Alertas en Cloud Monitoring

| Alerta | Condicion | Severidad |
|---|---|---|
| Custom Training Job fallo | Job state = FAILED | Critical |
| Cloud Run 5xx | Error rate > 1% por 5 min | Warning |
| Cloud Run latencia | p95 > 2s por 5 min | Warning |
| API de embeddings fallo | Error rate > 5% en Synthetic API | Warning |
| Sin trafico | 0 requests en 1 hora | Info |
| Memoria Cloud Run | Uso > 90% por 5 min | Warning |

### Configuracion de alertas

```bash
# Alerta de pipeline failure
gcloud alpha monitoring policies create \
  --display-name="Vertex AI Training Failure" \
  --condition-display-name="Custom Job Failed" \
  --condition-filter='resource.type="aiplatform_custom_job" AND severity=ERROR' \
  --project=mlops-toxic-classifier
```

## Logs relevantes

```bash
# Logs de Cloud Run
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="toxic-comment-classifier"' \
  --project=mlops-toxic-classifier --limit=50

# Logs de Custom Training Job
gcloud logging read \
  'resource.type="aiplatform_custom_job"' \
  --project=mlops-toxic-classifier --limit=50

# Logs de Cloud Build
gcloud logging read \
  'resource.type="cloud_build"' \
  --project=mlops-toxic-classifier --limit=20
```

## Politica de reentrenamiento

El reentrenamiento se puede activar por:

| Trigger | Configuracion |
|---|---|
| Schedule semanal | Cloud Scheduler (lunes 2am, America/Mexico_City) |
| Drift detectado | Alerta de prediction drift (manual por ahora) |
| Degradacion | AUC macro < 0.97 en evaluacion con etiquetas humanas |
| Manual | `gcloud pubsub topics publish retrain-trigger --message='...'` |

La configuracion actual usa schedule semanal via Cloud Scheduler + Pub/Sub + Cloud Function.

## Estrategia de rollback

Mantener versiones anteriores de artefactos en GCS:

```
gs://mlops-toxic-classifier-ml/model/              ← version actual (sirve Cloud Run)
gs://mlops-toxic-classifier-ml/model_archive/v0/   ← version anterior
gs://mlops-toxic-classifier-ml/model_archive/v1/   ← version anterior
```

Procedimiento de rollback:

1. Copiar version anterior de `model_archive/` a `model/`.
2. Forzar cold start en Cloud Run (la instancia nueva carga el modelo de GCS).
3. Inspeccionar logs y metricas para confirmar estabilidad.
4. Bloquear futuros despliegues hasta identificar la causa raiz.

Para sistemas de alto riesgo, usar despliegue canary o shadow antes de enviar todo el trafico al modelo nuevo.
