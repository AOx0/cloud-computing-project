# Jigsaw Toxic Comment Classification: Documentacion del Analisis

## 1. Dataset

**Fuente:** Jigsaw Toxic Comment Classification Challenge (Kaggle, 2018)

**Ubicacion:** `raw/juegos/train.csv`

| Propiedad | Valor |
|---|---|
| Filas | 159,571 |
| Columnas | 8 (id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate) |
| Prevalencia any_toxic | 10.17% |
| Idioma | Ingles (Wikipedia talk pages) |

**Prevalencia por etiqueta:**

| Etiqueta | Positivos | Prevalencia |
|---|---|---|
| toxic | 15,294 | 9.58% |
| obscene | 8,449 | 5.29% |
| insult | 7,877 | 4.94% |
| severe_toxic | 1,595 | 1.00% |
| identity_hate | 1,405 | 0.88% |
| threat | 478 | 0.30% |

**Estructura jerarquica:** `severe_toxic` es subconjunto estricto de `toxic`. Las demas sub-etiquetas son subconjuntos aproximados de `toxic` con tasas de violacion entre 6% y 7.3%. `obscene` e `insult` co-ocurren en 73% de los casos.

---

## 2. Hipotesis evaluadas

Todas las hipotesis se formularon antes del modelado y se evaluaron con el flujo: hipotesis -> EDA -> pruebas estadisticas -> comparacion de modelos -> conclusion.

| Hipotesis | Descripcion | Resultado |
|---|---|---|
| H1 | Ningun par de etiquetas es independiente | Confirmada (Chi-cuadrada p < 0.001 en todos los pares) |
| H2 | Comentarios toxicos son mas cortos y con mas enfasis | Parcialmente confirmada (diferencias significativas pero distribuciones solapadas) |
| H3 | Accuracy no es metrica valida por desbalance | Confirmada (accuracy trivial ~90%) |
| H4 | Baselines simples estan lejos de rendimiento competitivo | Confirmada (AUC 0.56-0.79 vs >0.95 competitivo) |
| H5 | Etiquetas raras tienen metricas inestables | Confirmada (CV de F1 para threat > 30%) |
| H6 | Sentimiento negativo asocia con toxicidad | Confirmada (22.1% prevalencia en negativos vs 3.9% en positivos) |
| H7 | VADER mejora AUC del baseline | Confirmada (+0.01 a +0.06 AUC) |
| H8 | VADER correlaciona menos con threat/identity_hate | Confirmada (threat r=-0.07, lowest) |
| H9 | EMPATH supera a VADER en correlacion con etiquetas | Parcialmente confirmada (supera en threat, inferior en demas) |
| H10 | EMPATH complementa a VADER como features | Confirmada (texto+VADER+EMPATH > texto+VADER en todas las etiquetas) |
| H11 | Categorias EMPATH especificas por etiqueta | Parcialmente confirmada (coherente en 4/6, identity_hate inesperado) |
| H12 | MNB mejor en prevalentes que en raras | Refutada (NB funciona igual en raras) |
| H13 | LinearSVC supera LR en F1 | Confirmada (delta +0.006, consistente en 5/6 etiquetas) |
| H14 | Classifier Chain supera independientes en etiquetas posteriores | Refutada (probabilidades intermedias propagan error por mala calibracion) |
| H15 | Undersampling mejora NB vs balanced weights | Confirmada (delta +0.008 a +0.019 segun config TF-IDF) |
| H16 | TF-IDF sin max_features mejora NB | Refutada para vanilla (empeora), pero confirmada con undersampling |
| H17 | char_wb n-gramas superan a word n-gramas | Confirmada (+0.005 AUC macro, +0.011 en identity_hate) |
| H18 | Ridge con target continuo supera a LinearSVC binario | Refutada (LinearSVC +0.01 a +0.12 en AUC macro) |
| H19 | Embeddings contextuales capturan senal que char_wb no captura | Parcialmente confirmada (superior en threat/severe_toxic/identity_hate, inferior en obscene) |
| H20 | TF-IDF + embeddings > cada uno por separado | Confirmada (AUC 0.9903 > 0.9861 en 6/6 etiquetas) |

---

## 3. Analisis de sentimiento (VADER)

**Herramienta:** VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Resultado clave:** Los 3,075 comentarios toxicos con sentimiento positivo ilustran la limitacion de los modelos lexicos. El sarcasmo hostil y las amenazas encubiertas no producen valencia negativa.

**Correlaciones point-biserial con etiquetas:**

| Feature | toxic | obscene | insult | severe_toxic | identity_hate | threat |
|---|---|---|---|---|---|---|
| sent_neg | 0.47 | 0.44 | 0.42 | 0.28 | 0.17 | 0.12 |
| sent_compound | -0.36 | -0.25 | -0.29 | -0.21 | -0.10 | -0.07 |

**AUC con features densos (texto simple + VADER):**

| Etiqueta | AUC (texto+VADER) | AUC (solo texto) | Delta |
|---|---|---|---|
| toxic | 0.833 | 0.639 | +0.194 |
| severe_toxic | 0.925 | 0.633 | +0.293 |
| obscene | 0.859 | 0.644 | +0.214 |
| threat | 0.885 | 0.726 | +0.159 |
| insult | 0.862 | 0.645 | +0.217 |
| identity_hate | 0.743 | 0.592 | +0.151 |

---

## 4. Analisis de categorias tematicas (EMPATH)

**Herramienta:** EMPATH (194 categorias lexico-tematicas)

**Resultado clave:** Las categorias mas activas en toxicos son `swearing_terms` (diferencia 0.014), `negative_emotion` (0.012), `ridicule` (0.004) y `hate` (0.004). Para `identity_hate`, la categoria `hate` tiene r bajo (0.04) porque el odio identitario en Wikipedia se expresa mas con vulgaridad que con lexico explicito de odio.

**Correlaciones Spearman entre VADER y EMPATH:** todas debiles (r < 0.23), confirmando que capturan dimensiones independientes.

**Independencia entre conjuntos de features:**

| Par | Spearman r |
|---|---|
| emp_hate vs sent_compound | -0.14 |
| emp_aggression vs caps_ratio | -0.07 |
| emp_anger vs sent_neg | 0.09 |
| emp_hate vs sent_neg | 0.23 |
| emp_kill vs sent_compound | -0.14 |

---

## 5. Comparacion de modelos CPU-only

Todos los modelos entrenados sobre TF-IDF word (1,2) + features densos, submuestra de 128k/32k train/test, 6 etiquetas binarias.

### 5.1 Modelos con TF-IDF limitado (5k features)

| Modelo | AUC macro | F1 macro | F2 macro | ECE macro | Tiempo |
|---|---|---|---|---|---|
| LinearSVC | 0.9724 | 0.5548 | 0.6591 | 0.0037 | 58s |
| LR L1 | 0.9705 | 0.5484 | 0.6565 | 0.1017 | 75s |
| ComplementNB | 0.9592 | 0.1245 | 0.2435 | 0.4637 | <1s |
| MultinomialNB | 0.9564 | 0.4725 | 0.5790 | 0.0061 | <1s |
| LGBM Chain | 0.9470 | 0.4917 | 0.5813 | 0.0475 | 888s |
| SGDClassifier | 0.6114 | 0.0935 | 0.1178 | 0.0319 | 3s |

**Conclusion:** LinearSVC con calibracion Platt es el mejor modelo CPU-only. Hinge loss optimiza el margen directamente y Platt scaling produce probabilidades bien calibradas (ECE 0.004 vs 0.10 de LR).

### 5.2 Efecto del TF-IDF sin restriccion

| Modelo | TF-IDF | Features | AUC macro | F1 macro |
|---|---|---|---|---|
| LinearSVC | word (1,2), no max | 335,865 | 0.9810 | 0.5899 |
| LinearSVC | word (1,2), 5k | 5,000 | 0.9704 | 0.5646 |
| MNB+undersample | word (1,2), no max | 335,865 | 0.9640 | 0.4786 |
| MNB+undersample | word (1,2), 5k | 5,000 | 0.9557 | 0.4418 |
| MNB vanilla | word (1,2), no max | 335,865 | 0.9309 | 0.4712 |
| MNB vanilla | word (1,2), 5k | 5,000 | 0.9564 | 0.4742 |

**Conclusion:** Sin restriccion de features, LinearSVC mejora +0.01 AUC. NB vanilla empeora con mas features (el prior dominante ahoga la senal), pero con undersampling mejora. Undersampling es la estrategia correcta para NB (H15 confirmada).

### 5.3 Efecto de char_wb n-gramas

| Modelo | TF-IDF | Features | AUC macro | F1 macro |
|---|---|---|---|---|
| **LinearSVC** | **char_wb (2,5)** | **176,089** | **0.9861** | **0.5956** |
| LinearSVC | char_wb (3,5) | 174,769 | 0.9860 | 0.6094 |
| LinearSVC | char_wb (3,4) | 72,337 | 0.9855 | 0.6114 |
| LinearSVC | word (1,2) | 335,865 | 0.9810 | 0.5899 |

**Delta por etiqueta (char_wb vs word, LinearSVC):**

| Etiqueta | word | char_wb | Delta |
|---|---|---|---|
| identity_hate | 0.9727 | 0.9841 | +0.0114 |
| obscene | 0.9858 | 0.9926 | +0.0068 |
| toxic | 0.9740 | 0.9792 | +0.0052 |
| insult | 0.9807 | 0.9848 | +0.0041 |
| severe_toxic | 0.9837 | 0.9876 | +0.0039 |
| threat | 0.9891 | 0.9885 | -0.0006 |

**Conclusion:** char_wb supera a word n-gramas en 5/6 etiquetas (H17 confirmada). La mejora mayor es en identity_hate (+0.011), donde el lexico de odio usa variaciones ortograficas deliberadas. Threat es la unica etiqueta donde word es marginalmente mejor, probablemente porque las amenazas son mas literales.

### 5.4 Ridge con target continuo

Pesos del target continuo: `obscene=0.16, toxic=0.32, threat=1.5, insult=0.64, severe_toxic=1.5, identity_hate=1.5`

| Modelo | TF-IDF | AUC macro |
|---|---|---|
| LinearSVC (binario) | char_wb (2,5) | 0.9861 |
| Ridge (continuo) | char_wb (2,5) | 0.9724 |
| Ridge ensemble 3alpha | char_wb (2,5) | 0.9739 |
| Ridge+undersample | char_wb (2,5) | 0.9678 |

**Conclusion:** Ridge con target continuo es inferior a LinearSVC en AUC macro (-0.012). El target continuo pierde la especificidad por etiqueta: un score alto no distingue entre tipos de toxicidad. Ridge tiene uso legitimo para ranking de severidad (no clasificacion multi-etiqueta).

### 5.6 Embeddings contextuales (nomic-embed-text-v1.5)

**Modelo:** nomic-ai/nomic-embed-text-v1.5 via Synthetic API (task_type=classification, 768 dimensiones)

**Resultado clave:** Los embeddings capturan senal semantica que char_wb TF-IDF no ve, pero solo son superiores en 3/6 etiquetas. La combinacion TF-IDF + embeddings es superior en 6/6.

| Etiqueta | Emb solo | TF-IDF solo | TF-IDF+Emb | Delta combo |
|---|---|---|---|---|
| toxic | 0.9781 | 0.9792 | **0.9847** | +0.0055 |
| severe_toxic | 0.9897 | 0.9876 | **0.9907** | +0.0031 |
| obscene | 0.9858 | 0.9926 | **0.9935** | +0.0009 |
| threat | 0.9917 | 0.9885 | **0.9944** | +0.0059 |
| insult | 0.9840 | 0.9848 | **0.9877** | +0.0029 |
| identity_hate | 0.9874 | 0.9841 | **0.9905** | +0.0064 |
| **MACRO** | **0.9861** | **0.9861** | **0.9903** | **+0.0041** |

**Hipotesis evaluadas:**

- H19: Parcialmente confirmada. Embeddings solos no superan a TF-IDF en AUC macro (empate 0.9861), pero superan en 3/6 etiquetas: threat (+0.0032), identity_hate (+0.0033), severe_toxic (+0.0021). Pierden en obscene (-0.0068) donde la senal es puramente ortografica.
- H20: Confirmada. TF-IDF + embeddings supera ambos individuales en 6/6 etiquetas. AUC macro 0.9903 vs 0.9861 (+0.004). Las ganancias mayores son en identity_hate (+0.0064) y threat (+0.0059), exactamente las etiquetas que el concilio identifico como problema abierto.

**Mecanismo causal:** char_wb TF-IDF captura la dimension ortografica (obfuscacion deliberada: a$$hole, f*ck). nomic-embed captura la dimension semantica (amenazas indirectas, ironia hostil, odio encubierto). Ambas dimensiones son independientes y complementarias. La concatenacion de features permite que LinearSVC aprenda un hiperplano en un espacio que combina ambas representaciones.

### 5.7 Naive Bayes con undersampling

| Modelo | TF-IDF | AUC macro |
|---|---|---|
| MNB+undersample | word (1,2), no max | 0.9640 |
| MNB+undersample | word (1,2), 5k | 0.9557 |
| MNB vanilla | word (1,2), 5k | 0.9564 |
| MNB balanced | word (1,2), 5k | 0.9484 |

**Conclusion:** Undersampling es la estrategia correcta para NB. Con TF-IDF completo + undersampling, NB alcanza AUC 0.964 en <1s de entrenamiento. `class_weight`/`sample_weight` distorsiona las verosimilitudes condicionales de NB. Mas features solo ayudan con undersampling (sin el, el prior dominante ahoga la senal).

---

## 6. Classifier Chain LightGBM

**Orden de la cadena:** toxic -> obscene -> insult -> severe_toxic -> identity_hate -> threat

**Resultado:** AUC macro 0.9470, inferior a LinearSVC (0.9861). La cadena propaga error porque las probabilidades intermedias estan mal calibradas (ECE de toxic = 0.12).

**Feature importance:** Las features de la cadena (chain_toxic, chain_obscene, etc.) son las mas importantes para etiquetas posteriores, confirmando que la estructura de dependencia es real. Pero la mala calibracion de las probabilidades intermedias anula el beneficio.

**Alternativa potencial:** Una cadena con LinearSVC (calibrado) como base podria funcionar mejor, pero la complejidad no se justifica dado que LinearSVC independiente ya alcanza AUC 0.9861.

---

## 7. Modelo productivo final

**Configuracion (mejor modelo):**

- **Algoritmo:** LinearSVC con CalibratedClassifierCV (sigmoid, cv=3)
- **Parametros:** C=0.1, class_weight="balanced", max_iter=5000
- **Features:** TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (classification, 768d) concatenados
- **TF-IDF:** char_wb, ngram_range=(2,5), sublinear_tf=True, min_df=3, max_df=0.7 (~176k features sparse + 768 dense)
- **Umbral:** F2-optimo por etiqueta (no 0.5 arbitrario)
- **Entrenamiento:** CPU-only para LinearSVC (~110s), embeddings via API Synthetic (~67 min para 160k comentarios, cacheable)
- **Artefacto:** `reports/training/model_svc/` (7 joblib, ~27 MB) + `data/nomic_embeddings_*.npz` (cache)

**Metricas por etiqueta (TF-IDF + embeddings, test 32k):**

| Etiqueta | AUC-ROC | F1 | F2 | Umbral F2-opt |
|---|---|---|---|---|
| toxic | 0.9847 | 0.7691 | 0.8477 | 0.15 |
| severe_toxic | 0.9907 | 0.5059 | 0.6398 | 0.10 |
| obscene | 0.9935 | 0.7846 | 0.8700 | 0.10 |
| threat | 0.9944 | 0.5463 | 0.5950 | 0.15 |
| insult | 0.9877 | 0.7396 | 0.8199 | 0.15 |
| identity_hate | 0.9905 | 0.4870 | 0.6211 | 0.10 |
| **MACRO** | **0.9903** | **0.6388** | **0.7323** | |

---

## 8. Features de texto complementarios

Los features de texto simple, VADER y EMPATH son complementarios pero incrementalmente debiles comparados con TF-IDF:

| Configuracion | AUC macro (con TF-IDF word 5k) |
|---|---|
| TF-IDF solo | 0.9704 |
| TF-IDF + texto + VADER + EMPATH | 0.9704 (dentro de la misma corrida LinearSVC) |

Con TF-IDF char_wb completo (176k features), los features densos adicionales (VADER, EMPATH, texto simple) aportan menos del 0.1% porque el TF-IDF ya captura la senal lexica que VADER y EMPATH codifican de forma agregada. Los features densos son mas utiles cuando el TF-IDF tiene pocas features (5k), donde la senal lexica no esta completa.

---

## 9. Validacion estadistica rigurosa

Tres pruebas complementarias con correccion Bonferroni (alpha = 0.0083 por etiqueta):

1. **Bootstrap pareado** (2000 resamples): IC95 de la diferencia de AUC. El pareamiento cancela la varianza compartida del split.
2. **Test de DeLong** (Hanley-McNeil): significancia parametrica entre AUCs correlacionados.
3. **Test de McNemar**: asimetrias de error discreto (n10 = A acierta donde B falla vs n01 = B acierta donde A falla).

**Nota sobre DeLong:** El test de DeLong no resulta significativo para ninguna etiqueta en ninguna comparacion. Esto se debe a que la aproximacion de Hanley-McNeil sobreestima la varianza del AUC con desbalance extremo (threat al 0.3%). Con n_pos = 98 y n_neg = 31817, la varianza estimada es del orden de 10^-5, haciendo que z < 2 en todos los casos. El bootstrap pareado y McNemar son mas confiables en este escenario.

### Comparacion 1: char_wb vs word (mismo LinearSVC)

| Etiqueta | delta AUC | Bootstrap IC95 | Boot sig | DeLong sig | McNemar sig | Convergencia |
|---|---|---|---|---|---|---|
| toxic | +0.0052 | [+0.003, +0.008] | Si | No | Si | **Doble** |
| identity_hate | +0.0114 | [+0.004, +0.022] | Si | No | No | Simple |
| obscene | +0.0067 | [+0.004, +0.009] | Si | No | No | Simple |
| insult | +0.0041 | [+0.002, +0.007] | Si | No | No | Simple |
| severe_toxic | +0.0039 | [-0.001, +0.010] | No | No | No | Ninguna |
| threat | -0.0006 | [-0.009, +0.011] | No | No | No | Ninguna |

**Interpretacion:** char_wb es significativamente mejor para toxic (doble convergencia). Para identity_hate, obscene e insult el bootstrap indica diferencia real pero McNemar no la detecta porque los errores discretos son similares con umbral 0.5. Para severe_toxic y threat no hay diferencia significativa.

### Comparacion 2: LinearSVC vs LR (mismo char_wb)

| Etiqueta | delta AUC | Bootstrap IC95 | Boot sig | DeLong sig | McNemar sig | Convergencia |
|---|---|---|---|---|---|---|
| toxic | +0.0086 | [+0.007, +0.010] | Si | No | Si | **Doble** |
| obscene | +0.0048 | [+0.003, +0.007] | Si | No | Si | **Doble** |
| insult | +0.0041 | [+0.003, +0.005] | Si | No | Si | **Doble** |
| identity_hate | +0.0020 | [+0.001, +0.004] | Si | No | Si | **Doble** |
| severe_toxic | +0.0023 | [-0.001, +0.006] | No | No | Si | Simple |
| threat | +0.0008 | [-0.002, +0.003] | No | No | Si | Simple |

**Interpretacion:** LinearSVC supera a LR en 4/6 etiquetas con doble convergencia. McNemar muestra asimetrias de error masivas: para threat, n10=576 vs n01=56 (LinearSVC comete 10x menos errores discretos). Esto confirma que hinge loss produce mejores margenes que log loss para este problema.

---

## 10. Limitaciones conocidas

1. **TF-IDF no captura contexto.** "I will kill you" y "I will end you" son n-gramas distintos pero semanticamente similares. Un modelo con embeddings contextuales (BERT) probablemente absorberia esta senal implicitamente.

2. **char_wb captura obfuscacion pero no ironia.** Las variaciones ortograficas ("f*ck") se capturan, pero el sarcasmo hostil ("Thanks for the *help*") no.

3. **Threat sigue siendo el problema abierto.** AUC 0.989 es alto pero F1 0.42 con umbral 0.08 indica que el modelo detecta amenazas al costo de 2 falsos positivos por cada verdadero positivo. Ningun modelo lexico resuelve threat de forma usable.

4. **Dataset de Wikipedia en ingles.** Los patrones no generalizan a otras plataformas (Reddit, Twitter), idiomas o contextos culturales.

5. **La seleccion de categorias EMPATH introduce sesgo.** Se seleccionaron por correlacion con any_toxic en el dataset de entrenamiento.

6. **Umbrales F2-optimos dependen del costo relativo.** El umbral 0.08 para threat prioriza recall al extremo. En produccion, el umbral debe ajustarse segun el costo de falsos negativos (amenaza no detectada) vs falsos positivos (moderacion excesiva).

7. **Sin GPU no se pueden probar transformers.** DistilBERT congelado como feature extraction requiere ~2-3h en T4. Fine-tuning con LoRA requiere GPU. GCP no tiene GPU disponible en este proyecto.

---

## 11. Arquitectura GCP

| Componente | Recurso | Estado |
|---|---|---|
| Proyecto | mlops-toxic-comments (288509175890) | Activo |
| Billing | 012F44-E86707-0459F2 | Vinculado |
| GCS | gs://mlops-toxic-comments-ml/ | Creado, datos subidos |
| Artifact Registry | us-central1-docker.pkg.dev/.../mlops-containers/ | Creado |
| Vertex AI | API habilitada | Pendiente pipeline real |
| Cloud Build | cloudbuild-analysis.yaml | Pendiente actualizar |
| Cloud Run | API habilitada | Pendiente serving real |

**Brechas para deploy:**

1. **Serving predictor:** usa SklearnPredictor que carga un solo joblib. Necesita reescribirse para cargar LinearSVC + CalibratedClassifierCV + TF-IDF char_wb, aceptar texto crudo y retornar probabilidades por etiqueta.
2. **Dockerfile:** no incluye scikit-learn con LinearSVC ni el feature pipeline.
3. **KFP components:** usan placeholder LR. Necesitan reescribirse para LinearSVC real.
4. **Modelo en GCS:** no subido. Directorio `reports/training/model/` debe subirse a `gs://mlops-toxic-comments-ml/model/`.

---

## 12. Archivos del proyecto

### Analisis

| Archivo | Descripcion |
|---|---|
| `src/analysis_toxic_comments.py` | EDA completo, 5 hipotesis originales, modelos baseline |
| `src/analysis_sentiment.py` | Analisis VADER, hipotesis H6-H8 |
| `src/analysis_empath.py` | Analisis EMPATH, hipotesis H9-H11 |

### Entrenamiento

| Archivo | Descripcion |
|---|---|
| `src/trainer/features.py` | Pipeline de features (TF-IDF + VADER + EMPATH + texto) |
| `src/trainer/model.py` | Classifier Chain LightGBM |
| `src/trainer/evaluation.py` | Evaluacion multi-etiqueta (bootstrap CI, ECE, F2-optimo) |
| `src/trainer/train.py` | Script de entrenamiento principal |
| `src/trainer/compare_models.py` | Comparacion 6 modelos CPU-only (H12-H14) |
| `src/trainer/nb_variants.py` | Variantes de NB: undersampling, features (H15-H16) |
| `src/trainer/charwb_ridge.py` | char_wb + Ridge con target continuo (H17-H18) |
| `src/trainer/validation.py` | Validacion estadistica rigurosa (bootstrap+DeLong+McNemar) |

### Datos cache

| Archivo | Tamano | Descripcion |
|---|---|---|
| `data/sentiment_scores.csv` | 3.5 MB | VADER scores para 159k comentarios |
| `data/empath_scores.parquet` | 9.7 MB | EMPATH scores para 159k comentarios |
| `data/empath_scores.csv` | 152 MB | EMPATH scores (original, usar parquet) |

### Reportes

| Archivo | Descripcion |
|---|---|
| `reports/eda/main.typ` | Reporte Typst completo (15 secciones) |
| `reports/eda/main.pdf` | PDF compilado (3.4 MB) |
| `reports/eda/imgs/` | 54 figuras PNG (01-44) |
| `reports/training/model/` | Modelo LightGBM Chain entrenado + feature pipeline |
| `reports/training/model_svc/` | **Modelo productivo final** (LinearSVC + char_wb, 7 joblib, ~27 MB) |
| `reports/training/metrics.json` | Metricas del modelo productivo |
| `reports/training/model_comparison.json` | Comparacion 6 modelos CPU-only |
| `reports/training/nb_variants.json` | Variantes NB |
| `reports/training/charwb_ridge.json` | char_wb + Ridge |
| `reports/training/validacion_estadistica.json` | Resultados de validacion rigurosa |

### GCP

| Archivo | Descripcion |
|---|---|
| `cloudbuild-analysis.yaml` | Cloud Build para analisis estadistico |
| `cloudbuild.yaml` | Cloud Build principal (Terraform + pipeline + serving) |
| `Dockerfile` | Imagen Docker para serving |
| `infra/` | Terraform IaC |
| `pipeline/` | Vertex AI Pipeline components (KFP) |

---

## 13. Referencias externas

- **Notebook NB simple:** `raw/jigsaw-incredibly-simple-naive-bayes-0-768.ipynb` -- MultinomialNB + TF-IDF sin max_features + undersampling, LB 0.768. Lecciones: undersampling es mejor que balanced weights para NB; TF-IDF sin restriccion mejora NB solo con undersampling.

- **Notebook ensemble:** `raw/jigsaw-ensemble-best-public-sub-0-898.ipynb` -- Ensemble de 3 Ridge regressions con char_wb n-gramas + target continuo ponderado, LB 0.898. Lecciones validas: char_wb captura obfuscacion; Ridge con target continuo funciona para ranking. Lecciones NO validas: overfitting manual por indice (leaking del test set).

- **Pesos de etiquetas del notebook ensemble:** `obscene=0.16, toxic=0.32, threat=1.5, insult=0.64, severe_toxic=1.5, identity_hate=1.5`. Estos pesos codifican que threat, severe_toxic e identity_hate son ~5x mas severos que toxic. Utiles como heuristica para crear target de severidad pero inferiores a clasificacion por etiqueta para AUC macro.

---

## 14. Proximos pasos

1. **Deploy:** Reescribir serving predictor, subir modelo LinearSVC+char_wb a GCS, construir Docker, deploy en Cloud Run.
2. **Vertex AI Pipeline:** Reescribir componentes KFP para entrenamiento real (LinearSVC + char_wb).
3. **DistilBERT embeddings:** Si se dispone de GPU, extraer embeddings congelados como features adicionales a LinearSVC. Hipotesis: la senal semantica contextual complementa la senal lexica de char_wb.
4. **Calibracion de umbrales por costo:** Definir funcion de costo asimetrico por etiqueta segun el contexto de moderacion, reemplazando el F2-optimo generico.
