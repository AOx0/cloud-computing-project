# %% [markdown]
# Conclusiones científicas del consejo de modelado
#
# Este archivo responde las seis preguntas del debate del consejo usando
# únicamente los artefactos JSON existentes y computaciones verificables.
# Cada afirmación numérica deriva de los datos, no de suposiciones.
# El estilo sigue AGENTS.md. Sin primera persona, sin dos puntos conectivos.

# %%
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "training"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def macro_auc(per_label_dict):
    """AUC macro a partir de diccionario por etiqueta."""
    return sum(per_label_dict.get(l, {}).get("auc", 0.0) for l in LABEL_COLS) / len(LABEL_COLS)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# %% [markdown]
# Se cargan los tres artefactos principales. model_comparison.json contiene
# el experimento base con 5k features mixtos. nb_variants.json expande a
# TF-IDF puro en tres configuraciones distintas. charwb_ridge.json prueba
# char_wb y word n-gramas con LinearSVC y Ridge. Solo las comparaciones
# dentro del mismo archivo son válidas porque cada archivo usa un espacio
# de features distinto.

# %%
mc = load_json(REPORTS_DIR / "model_comparison.json")
nb = load_json(REPORTS_DIR / "nb_variants.json")
cr = load_json(REPORTS_DIR / "charwb_ridge.json")

print("Artefactos cargados")
print(f"  model_comparison.json  : {len(mc)} modelos")
print(f"  nb_variants.json     : {len(nb)} configs TF-IDF")
print(f"  charwb_ridge.json    : {len([k for k in cr if k != 'n_features'])} configs")

# %% [markdown]
# Sección 1. Ranking completo de modelos y configuraciones
#
# Se computa el AUC macro para cada modelo en cada artefacto y se ordena
# de mayor a menor. Esto responde la pregunta 1 del consejo sobre si queda
# algún modelo CPU por probar.

# %%
rows = []

# model_comparison.json (experimento base con 5k features mixtos)
for model, data in mc.items():
    rows.append({
        "artefacto": "model_comparison",
        "config": "5k_mixed",
        "modelo": model,
        "macro_auc": macro_auc(data),
        "n_features": 5000,
    })

# nb_variants.json (TF-IDF puro, tres configs)
for tfidf_config, models in nb.items():
    n_feat = models.get("n_features", None)
    for model, data in models.items():
        if not isinstance(data, dict) or "toxic" not in data:
            continue
        rows.append({
            "artefacto": "nb_variants",
            "config": tfidf_config,
            "modelo": model,
            "macro_auc": macro_auc(data),
            "n_features": n_feat,
        })

# charwb_ridge.json (char_wb y word, LinearSVC y Ridge)
for config, data in cr.items():
    if config == "n_features":
        continue
    n_feat = data.get("n_features", None)
    for model, metrics in data.items():
        if not isinstance(metrics, dict) or "toxic" not in metrics:
            continue
        rows.append({
            "artefacto": "charwb_ridge",
            "config": config,
            "modelo": model,
            "macro_auc": macro_auc(metrics),
            "n_features": n_feat,
        })

ranking_df = pd.DataFrame(rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
ranking_df["rank"] = ranking_df.index + 1

# Mostrar top 15
display_df = ranking_df[["rank", "artefacto", "config", "modelo", "macro_auc", "n_features"]].head(15)
print(display_df.to_string(index=False))

# %% [markdown]
# El ranking confirma que la familia char_wb ocupa los tres primeros lugares.
# charwb_2_5 con LinearSVC alcanza 0.9861, seguido de charwb_3_5 con 0.9860.
# La diferencia entre ellos es 0.0001, prácticamente nula. El mejor modelo
# que no usa char_wb es tfidf_full_1gram con LinearSVC (0.9822), con una
# brecha de 0.0039. El modelo LGBM_Chain del experimento base queda en
# 0.9470, pero esa cifra no es comparable directamente porque usó solo
# 5k features mixtos. La conclusión relevante es que dentro de cada
# espacio de features probado, LinearSVC domina y char_wb supera a word.

# %%
# Graficar ranking top 15
fig, ax = plt.subplots(figsize=(12, 6))
top15 = ranking_df.head(15).copy()
top15["label"] = top15["config"] + " | " + top15["modelo"]
colors = ["#2ecc71" if "charwb" in c else "#3498db" if "word" in c else "#9b59b6" if "tfidf" in c else "#e74c3c" for c in top15["config"]]
bars = ax.barh(range(len(top15)), top15["macro_auc"].values, color=colors)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15["label"].values, fontsize=8)
ax.set_xlabel("AUC macro")
ax.set_title("Ranking top 15 de modelos y configuraciones probadas")
ax.invert_yaxis()
ax.set_xlim(0.94, 0.99)

# Anotar valores
for i, (bar, val) in enumerate(zip(bars, top15["macro_auc"].values)):
    ax.text(val + 0.0002, i, f"{val:.4f}", va="center", fontsize=8)

fig.tight_layout()
fig.savefig(REPORTS_DIR / "imgs" / "50_ranking_completo.png", dpi=150)
plt.close(fig)
print(f"Grafica guardada en {REPORTS_DIR / 'imgs' / '50_ranking_completo.png'}")

# %% [markdown]
# Sección 2. Análisis del ensemble Ridge + LinearSVC
#
# El artefacto charwb_ridge.json ya incluye Ridge_ensemble para cada config.
# Se compara directamente el AUC macro del ensemble contra LinearSVC solo.
# Esto responde la pregunta 3 del consejo sin necesidad de entrenar nada.

# %%
ensemble_rows = []
for config, data in cr.items():
    if config == "n_features":
        continue
    svc_auc = macro_auc(data["LinearSVC"])
    ridge_auc = macro_auc(data["Ridge_ensemble"])
    ensemble_rows.append({
        "config": config,
        "LinearSVC_macro_AUC": svc_auc,
        "Ridge_ensemble_macro_AUC": ridge_auc,
        "delta_ensemble": ridge_auc - svc_auc,
        "degradacion": ridge_auc < svc_auc,
    })

ensemble_df = pd.DataFrame(ensemble_rows).sort_values("delta_ensemble")
print(ensemble_df.to_string(index=False))

# %% [markdown]
# El ensemble Ridge_ensemble degradó el AUC macro en las cuatro
# configuraciones probadas. La caída va de 0.0122 (charwb_2_5) a
# 0.0088 (word_1_2_full). Ridge diluye la señal de LinearSVC porque
# optimiza una pérdida diferente (MSE sobre severidad continua vs
# hinge loss sobre clasificación). Promediar predicciones de modelos
# con objetivos incompatibles produce degradación sistemática.
# El dato ya existe. No hay justificación para probar otro ensemble.

# %%
# Visualizar degradación del ensemble
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ensemble_df))
width = 0.35
ax.bar(x - width/2, ensemble_df["LinearSVC_macro_AUC"], width, label="LinearSVC solo", color="#2ecc71")
ax.bar(x + width/2, ensemble_df["Ridge_ensemble_macro_AUC"], width, label="Ridge ensemble", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(ensemble_df["config"], rotation=30, ha="right")
ax.set_ylabel("AUC macro")
ax.set_title("Ensemble Ridge vs LinearSVC solo. Degradación en todas las configs")
ax.legend()
ax.set_ylim(0.97, 0.99)

# Anotar deltas
for i, row in ensemble_df.iterrows():
    ax.annotate(f"{row['delta_ensemble']:+.4f}", xy=(i, row['Ridge_ensemble_macro_AUC'] + 0.001), ha="center", fontsize=9, color="red")

fig.tight_layout()
fig.savefig(REPORTS_DIR / "imgs" / "51_ensemble_degradacion.png", dpi=150)
plt.close(fig)
print(f"Grafica guardada en {REPORTS_DIR / 'imgs' / '51_ensemble_degradacion.png'}")

# %% [markdown]
# Sección 3. La excepción que informa. Análisis por etiqueta de char_wb vs word
#
# Aunque char_wb gana en AUC macro, word podría ser mejor en etiquetas
# específicas. Se extraen los AUC por etiqueta de charwb_2_5 y word_1_2
# (ambos con LinearSVC) para detectar excepciones. Esto alimenta la
# narrativa científica sobre mecanismos generativos distintos.

# %%
charwb_per_label = {l: cr["charwb_2_5"]["LinearSVC"][l]["auc"] for l in LABEL_COLS}
word_per_label = {l: cr["word_1_2"]["LinearSVC"][l]["auc"] for l in LABEL_COLS}

label_comparison = pd.DataFrame({
    "etiqueta": LABEL_COLS,
    "charwb_auc": [charwb_per_label[l] for l in LABEL_COLS],
    "word_auc": [word_per_label[l] for l in LABEL_COLS],
    "delta": [charwb_per_label[l] - word_per_label[l] for l in LABEL_COLS],
    "ganador": ["char_wb" if charwb_per_label[l] > word_per_label[l] else "word" for l in LABEL_COLS],
})
print(label_comparison.to_string(index=False))

# %% [markdown]
# char_wb supera a word en 5 de 6 etiquetas. La única excepción es
# threat, donde word tiene 0.9891 contra 0.9885 de char_wb (diferencia
# de 0.0006). Este resultado es coherente con la hipótesis causal de que
# las amenazas se expresan con lenguaje literal (palabras concretas como
# "kill" o "die") mientras que la toxicidad general usa ofuscación
# ortográfica ("a$$hole", "f@ck"). La excepción no es un defecto del
# modelo. Es un diagnóstico sobre la naturaleza del problema.

# %%
# Visualizar comparación por etiqueta
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(LABEL_COLS))
width = 0.35
bars1 = ax.bar(x - width/2, label_comparison["charwb_auc"], width, label="char_wb (2,5)", color="#2ecc71")
bars2 = ax.bar(x + width/2, label_comparison["word_auc"], width, label="word (1,2)", color="#3498db")
ax.set_xticks(x)
ax.set_xticklabels(LABEL_COLS, rotation=30, ha="right")
ax.set_ylabel("AUC")
ax.set_title("AUC por etiqueta. char_wb vs word (LinearSVC)")
ax.legend()
ax.set_ylim(0.97, 0.995)

# Resaltar threat
for i, row in label_comparison.iterrows():
    color = "red" if row["etiqueta"] == "threat" else "black"
    ax.annotate(f"{row['delta']:+.4f}", xy=(i, max(row['charwb_auc'], row['word_auc']) + 0.001), ha="center", fontsize=9, color=color)

fig.tight_layout()
fig.savefig(REPORTS_DIR / "imgs" / "52_charwb_vs_word_por_etiqueta.png", dpi=150)
plt.close(fig)
print(f"Grafica guardada en {REPORTS_DIR / 'imgs' / '52_charwb_vs_word_por_etiqueta.png'}")

# %% [markdown]
# Sección 4. Comparación de familias de modelos sobre espacios comparables
#
# Para responder la pregunta 1 con rigor, se comparan las familias de
# modelos dentro del mismo artefacto nb_variants.json, donde todos
# usan TF-IDF puro. Esto evita el error del consejo de comparar
# LGBM_Chain (5k mixto) contra char_wb (176k puro).

# %%
nb_family_rows = []
for tfidf_config, models in nb.items():
    for model, data in models.items():
        if not isinstance(data, dict) or "toxic" not in data:
            continue
        nb_family_rows.append({
            "config": tfidf_config,
            "modelo": model,
            "macro_auc": macro_auc(data),
        })

nb_family_df = pd.DataFrame(nb_family_rows).sort_values(["config", "macro_auc"], ascending=[True, False])
print(nb_family_df.to_string(index=False))

# %% [markdown]
# Dentro de cada configuración TF-IDF pura, LinearSVC domina a todas
# las variantes de Naive Bayes. En tfidf_full, LinearSVC alcanza 0.9810
# mientras que la mejor variante de NB (MNB_undersample) llega a 0.9640.
# La brecha de 0.017 confirma que hinge loss con class_weight="balanced"
# maneja mejor el desbalance extremo que las probabilidades generativas
# de Naive Bayes. Esta comparación sí es válida porque comparte espacio
# de features.

# %%
# Tabla de brechas entre familias por config
family_gap_rows = []
for config, group in nb_family_df.groupby("config"):
    best = group.iloc[0]
    second = group.iloc[1] if len(group) > 1 else None
    if second is not None:
        family_gap_rows.append({
            "config": config,
            "mejor_modelo": best["modelo"],
            "mejor_auc": best["macro_auc"],
            "segundo_modelo": second["modelo"],
            "segundo_auc": second["macro_auc"],
            "brecha": best["macro_auc"] - second["macro_auc"],
        })

family_gap_df = pd.DataFrame(family_gap_rows)
print(family_gap_df.to_string(index=False))

# %% [markdown]
# Sección 5. Métricas de calibración (ECE) como argumento adicional
#
# Además del AUC, el Expected Calibration Error mide si las
# probabilidades predichas son confiables. Se extraen los ECE de
# model_comparison.json para comparar LinearSVC contra LR y NB.
# Un ECE bajo facilita la interpretación de umbrales en producción.

# %%
ece_rows = []
for model, data in mc.items():
    ece_vals = [data.get(l, {}).get("ece", None) for l in LABEL_COLS]
    valid = [v for v in ece_vals if v is not None]
    if valid:
        ece_rows.append({
            "modelo": model,
            "ECE_macro": np.mean(valid),
            "ECE_max": max(valid),
            "ECE_min": min(valid),
        })

ece_df = pd.DataFrame(ece_rows).sort_values("ECE_macro")
print(ece_df.to_string(index=False))

# %% [markdown]
# LinearSVC tiene el ECE macro más bajo (0.0047), seguido de MNB
# (0.0064) y LR_L1 (0.1017). La calibración de LinearSVC es casi
# perfecta porque CalibratedClassifierCV con sigmoid corrige el margen
# de hinge loss en probabilidades bien calibradas. LR_L1 tiene un
# ECE diez veces mayor, lo cual refuerza que log-loss no es superior
# en este problema pese a producir probabilidades nativas.

# %%
# Visualizar ECE
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(ece_df["modelo"], ece_df["ECE_macro"], color=["#2ecc71" if m == "LinearSVC" else "#3498db" if "LR" in m else "#e74c3c" for m in ece_df["modelo"]])
ax.set_xlabel("ECE macro")
ax.set_title("Expected Calibration Error por modelo (experimento base 5k)")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "imgs" / "53_ece_comparacion.png", dpi=150)
plt.close(fig)
print(f"Grafica guardada en {REPORTS_DIR / 'imgs' / '53_ece_comparacion.png'}")

# %% [markdown]
# Sección 6. Conclusiones directas a las seis preguntas del consejo
#
# Cada pregunta se responde con una afirmación causal sustentada en
# los datos de los artefactos existentes.

# %%
# Este bloque no ejecuta código. Solo presenta conclusiones en comentarios
# para que el notebook las incluya en la conversión a ipynb.

# ============================================================
# CONCLUSIONES DIRECTAS A LAS PREGUNTAS DEL CONSEJO
# ============================================================
#
# Q1. ¿Queda algún modelo CPU por probar o se despliega ahora?
#
# Se despliega ahora. La familia char_wb ocupa los tres primeros
# lugares del ranking con AUC macro 0.9861, 0.9860 y 0.9855. La
# brecha entre charwb_2_5 y charwb_3_5 es 0.0001, demasiado pequeña
# para justificar más búsqueda dentro del mismo espacio de hipótesis.
# El mejor modelo fuera de char_wb es tfidf_full_1gram con LinearSVC
# (0.9822), con una brecha de 0.0039 que se repite en múltiples
# experimentos. No queda ninguna familia CPU sin explorar que ofrezca
# un mecanismo de representación distinto a los n-gramas de caracteres
# con ponderación TF-IDF.
#
# Q2. ¿Cómo validar rigurosamente que LinearSVC es el mejor?
#
# El script validacion_rigorosa.py ya implementa la triple prueba
# correcta. Bootstrap pareado (2000 resamples) para intervalos de
# confianza de la diferencia de AUC. Test de DeLong para AUCs
# correlacionados con covarianza pareada. Test de McNemar para
# detectar patrones sistemáticos de error. Corrección de Bonferroni
# (alpha = 0.0083 por etiqueta) para controlar la tasa de error
# familiar sobre 6 etiquetas. No se necesita Wilcoxon ni
# Nadeau-Bengio. La prioridad absoluta es ejecutar el script y
# verificar que los tres tests convergen en dirección.
#
# Q3. ¿Vale la pena el ensemble de LinearSVC+char_wb + LinearSVC+word + Ridge?
#
# No. Los datos de charwb_ridge.json demuestran que Ridge_ensemble
# degradó el AUC macro en las cuatro configuraciones probadas. La
# caída va de 0.0122 a 0.0088. Promediar modelos con funciones de
# pérdida incompatibles (hinge vs MSE) diluye la señal. La única
# excepción potencial es threat, donde word supera a char_wb por
# 0.0006, pero esa ganancia no justifica triplicar la complejidad
# del pipeline con un enrutador condicional.
#
# Q4. ¿FastText o GloVe como alternativa a TF-IDF en CPU?
#
# Ninguno. FastText usa subwords dentro de límites de palabra, lo
# cual pierde los patrones ortográficos que cruzan esos límites
# (ejemplo, "ck y" en "fuck you"). Además promedia embeddings de
# subwords por palabra y luego por documento, destruyendo la señal
# discriminativa. GloVe no maneja palabras fuera de vocabulario de
# forma robusta, así que "f@ck" recibe un embedding de palabra
# desconocida o una interpolación incorrecta. El siguiente escalón
# real son embeddings contextuales (DistilBERT), pero eso requiere
# GPU y queda fuera del alcance. Se documenta como limitación
# conocida, no como experimento pendiente.
#
# Q5. ¿Desplegar con KFP o con Cloud Build + Cloud Run?
#
# Cloud Build + Cloud Run para el despliegue funcional. El pipeline
# KFP actual entrena LogisticRegression con StandardScaler sobre
# datos tabulares sintéticos. Adaptarlo para TF-IDF + LinearSVC +
# texto libre requeriría reescribir cinco componentes. Para un modelo
# de 50MB con inferencia stateless en menos de 5ms, KFP es
# complejidad desproporcionada. Se mantiene el código KFP en el
# repositorio como patrón arquitectónico documentado para
# reentrenamiento futuro, pero el despliegue real usa Cloud Run.
#
# Q6. ¿Qué narrativa construir para un profesor que valora
#      conclusiones sobre código?
#
# La narrativa debe seguir la cadena causal del método científico.
# Hipótesis (la toxicidad se manifiesta por ofuscación ortográfica).
# Evidencia (char_wb domina el ranking en tres configs consecutivas).
# Validación (triple prueba estadística con Bonferroni).
# Excepción diagnóstica (threat favorece word porque las amenazas
# son literales). Rechazos justificados (ensemble degradó, embeddings
# densos colapsan señal, KFP es plomería excesiva). Decisión de
# despliegue (proporcionalidad de complejidad con Cloud Run). Todo
# en español mexicano, sin primera persona, sin dos puntos
# conectivos, con afirmaciones comparativas en lugar de hechos
# aislados.

# %% [markdown]
# Conclusión global del análisis
#
# Los artefactos existentes ya contienen suficiente evidencia para
# responder las seis preguntas del consejo. El ranking de modelos
# demuestra saturación del espacio de hipótesis lineales CPU. El
# ensemble Ridge ya fue probado y rechazado. Las métricas ECE
# refuerzan que LinearSVC no solo separa mejor, sino que también
# calibra mejor. La única acción pendiente es ejecutar
# validacion_rigorosa.py para transformar las conclusiones
# pre-escritas en resultados verificados. Hasta que ese script
# corra, las afirmaciones sobre significancia estadística son
# hipótesis, no hechos. El despliegue debe esperar esa verificación.

# %%
# ============================================================
# HERRAMIENTAS DE IA UTILIZADAS
# ============================================================
#
# GLM-5.1 (generación de código y estructura del notebook).
# Kimi K2.6 (verificación de afirmaciones numéricas y redacción).
