# Sales Pitch

## Problema

Las plataformas de contenido (Wikipedia, Reddit, Twitter) procesan millones de comentarios diarios. La moderacion manual no escala. Los modelos de toxicidad existentes se entrenan en laboratorios con GPU y se despliegen como cajas negras. Cuando el modelo se degrada o el contenido evoluciona, no hay mecanismo automatico para detectarlo ni reentrenar.

## Solucion

Una plataforma MLOps completa en Google Cloud Platform que automatiza el ciclo de vida de un clasificador de toxicidad multi-etiqueta. No es un modelo en un notebook. Es un sistema que entrena, evalua, despliega, monitorea y reentrena de forma automatica.

## Arquitectura clave

**GCS como contrato.** La imagen Docker no contiene el modelo. El pipeline de entrenamiento escribe artefactos a Cloud Storage. Cloud Run los lee al arrancar. Misma imagen para entrenar y servir. Cambiar el modelo no requiere reconstruir la imagen.

**Desacoplamiento entrenamiento-serving.** Vertex AI Custom Training Job corre en una VM efimera, escribe artefactos, y desaparece. Cloud Run corre como servicio persistente, lee artefactos, y predice. No comparten estado excepto GCS.

**Cache de embeddings.** Los embeddings de nomic-embed (768d) para 159k comentarios se cachean en GCS. El primer entrenamiento toma ~67 minutos para computarlos. Los subsiguientes los descargan en ~30 segundos. Esto reduce el costo de reentrenamiento de 2 horas a 15 minutos.

**Gate de despliegue.** El pipeline solo despliega si AUC macro >= umbral configurable. Un modelo que degrada no llega a produccion.

## Modelo

LinearSVC + CalibratedClassifierCV con features TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d).

| Metrica | Valor |
|---|---|
| AUC macro | 0.9903 |
| F1 macro | 0.6388 |
| Latencia de prediccion | ~450 ms |
| Tamano del modelo | ~30 MB |

El modelo se selecciono con rigor estadistico: 20 hipotesis formuladas antes del modelado, evaluadas con bootstrap pareado, DeLong y McNemar con correccion Bonferroni. LinearSVC supera a LogisticRegression, NaiveBayes, Ridge y LightGBM en AUC macro. char_wb supera a word n-gramas. La combinacion TF-IDF + embeddings supera a cada uno individualmente en 6/6 etiquetas.

## Valor de negocio

| Beneficio | Como se logra |
|---|---|
| Automatizacion completa | Cloud Build + Vertex AI + Cloud Run sin intervencion manual |
| Deteccion de degradacion | Monitoreo de drift + alertas + metricas baseline |
| Reentrenamiento automatico | Cloud Scheduler semanal o trigger por drift |
| Costo eficiente | Cloud Run escala a cero, entrenamiento en VM efimera |
| Reproducibilidad | Artefactos versionados en GCS, misma imagen para train y serve |
| Transparencia | 20 hipotesis evaluadas, validacion estadistica con Bonferroni |

## Diferenciador

No es otro modelo de NLP en un endpoint. Es una plataforma donde cada decision de diseno esta justificada con evidencia estadistica, donde el entrenamiento y el serving estan desacoplados por diseno, y donde el modelo se actualiza solo cuando la metrica lo aprueba. El resultado no es solo un clasificador con AUC 0.99. Es un sistema que se mantiene relevante sin intervencion humana.
