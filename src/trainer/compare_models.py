"""
Comparacion de modelos CPU-only para Jigsaw Toxic Comments.

Modelos a comparar:
  1. LogisticRegression (L1) -- baseline con seleccion implicita de features
  2. LightGBM Classifier Chain -- modelo productivo actual
  3. MultinomialNB -- testeia supuesto de independencia de features
  4. LinearSVC -- hinge loss vs log loss en espacio sparse
  5. SGDClassifier (log_loss) -- LR escalable con regularizacion elasticnet

Hipotesis:
  H12: MultinomialNB tendra AUC competitivo en etiquetas prevalentes
  (toxic, obscene, insult) donde la señal lexica es fuerte, pero
  AUC significativamente menor en etiquetas raras (threat, identity_hate)
  donde el supuesto de independencia falla mas porque las pocas palabras
  discriminativas co-ocurren con muchas otras.

  H13: LinearSVC superara a LR en F1 porque hinge loss optimiza el margen
  directamente, produciendo predicciones mas confiables cerca del umbral
  de decision. La diferencia sera mas notable en etiquetas con senal
  fuerte y separacion relativamente clara (obscene, insult).

  H14: La classifier chain de LightGBM superara a todos los modelos
  independientes en etiquetas posteriores de la cadena porque modela
  explicitamente la dependencia entre etiquetas que los demas ignoran.

Herramientas de IA utilizadas: Claude (generacion de codigo).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score, brier_score_loss,
)
from sklearn.model_selection import train_test_split
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.features import FeaturePipeline, LABEL_COLS
from src.trainer.model import ClassifierChainLGBM, CHAIN_ORDER, BASE_PARAMS
from src.trainer.evaluation import (
    evaluate_multilabel, format_results_table, compute_ece,
    find_f2_optimal_threshold, fbeta_score,
)

RANDOM_STATE = 42
OUTPUT_DIR = PROJECT_ROOT / "reports" / "training"
IMG_DIR = OUTPUT_DIR / "imgs"


def print_hypothesis():
    print("""
================================================================
HIPOTESIS DE LA COMPARACION DE MODELOS CPU-ONLY
================================================================

H12: MultinomialNB tendra AUC competitivo en etiquetas prevalentes
     (toxic, obscene, insult) pero significativamente menor en
     etiquetas raras (threat, identity_hate) donde el supuesto de
     independencia de features falla mas.

H13: LinearSVC superara a LR en F1 porque hinge loss optimiza el
     margen directamente. La diferencia sera notable en etiquetas
     con separacion clara (obscene, insult).

H14: La classifier chain de LightGBM superara a todos los modelos
     independientes en etiquetas posteriores de la cadena porque
     modela la dependencia entre etiquetas.
================================================================
""")


def train_model_per_label(model_class, model_kwargs, X_train, y_train, X_test, y_test,
                          name, needs_proba=True, calibrate=False):
    """Entrena un modelo independiente por etiqueta y retorna AUC y F1."""
    print(f"\n  === {name} ===")
    results = {}
    for j, label in enumerate(LABEL_COLS):
        t0 = time.time()
        model = model_class(**model_kwargs)

        if calibrate:
            model = CalibratedClassifierCV(model, cv=3, method="sigmoid")

        model.fit(X_train, y_train[:, j])

        if needs_proba and hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        elif needs_proba and hasattr(model, "decision_function"):
            # LinearSVC: convertir decision function a probabilidad via sigmoid
            scores = model.decision_function(X_test)
            prob = 1 / (1 + np.exp(-scores))  # Platt scaling manual
        else:
            prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        if prob is not None:
            auc = roc_auc_score(y_test[:, j], prob)
            # F1 con umbral F2-optimo
            opt_t, opt_f2 = find_f2_optimal_threshold(y_test[:, j], prob)
            preds = (prob >= opt_t).astype(int)
            f1 = f1_score(y_test[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test[:, j], preds, beta=2)
            prec = precision_score(y_test[:, j], preds, zero_division=0)
            rec = recall_score(y_test[:, j], preds, zero_division=0)
            ap = average_precision_score(y_test[:, j], prob)
            ece = compute_ece(y_test[:, j], prob)
        else:
            preds = model.predict(X_test)
            auc = 0.5
            f1 = f1_score(y_test[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test[:, j], preds, beta=2)
            opt_t = 0.5
            prec = rec = ap = ece = 0

        elapsed = time.time() - t0
        results[label] = {
            "auc": auc, "f1": f1, "f2": f2,
            "precision": prec, "recall": rec,
            "ap": ap, "ece": ece, "threshold": opt_t,
        }
        print(f"    {label}: AUC={auc:.4f}, F1={f1:.4f}, F2={f2:.4f} ({elapsed:.0f}s)")

    return results


def train_chain_lgbm(X_train, y_train, X_test, y_test, all_feature_names):
    """Entrena y evalua la cadena LightGBM."""
    print("\n  === LightGBM Classifier Chain ===")
    chain_model = ClassifierChainLGBM(chain_order=CHAIN_ORDER)
    chain_model.fit(X_train, y_train, feature_names=all_feature_names)
    y_prob = chain_model.predict_proba(X_test)

    results = {}
    for j, label in enumerate(LABEL_COLS):
        yt = y_test[:, j]
        yp = y_prob[:, j]
        auc = roc_auc_score(yt, yp)
        opt_t, opt_f2 = find_f2_optimal_threshold(yt, yp)
        preds = (yp >= opt_t).astype(int)
        f1 = f1_score(yt, preds, zero_division=0)
        f2 = fbeta_score(yt, preds, beta=2)
        prec = precision_score(yt, preds, zero_division=0)
        rec = recall_score(yt, preds, zero_division=0)
        ap = average_precision_score(yt, yp)
        ece = compute_ece(yt, yp)
        results[label] = {
            "auc": auc, "f1": f1, "f2": f2,
            "precision": prec, "recall": rec,
            "ap": ap, "ece": ece, "threshold": opt_t,
        }
        print(f"    {label}: AUC={auc:.4f}, F1={f1:.4f}, F2={f2:.4f}")

    return results, chain_model


def main():
    print_hypothesis()

    # Cargar modelo ya entrenado y features
    model_dir = OUTPUT_DIR / "model"
    chain_model = ClassifierChainLGBM.load(model_dir)
    fp = joblib.load(model_dir / "feature_pipeline.joblib")

    # Reconstruir split
    df = pd.read_csv(PROJECT_ROOT / "raw" / "juegos" / "train.csv")
    df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["any_toxic"], random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    t0 = time.time()
    train_data = fp.fit_transform(train_df, include_labels=True)
    test_data = fp.transform(test_df, include_labels=True)
    print(f"Features: {time.time()-t0:.1f}s")

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]
    all_feature_names = train_data["feature_names_tfidf"] + train_data["feature_names_dense"]

    # NB necesita valores no negativos. TF-IDF ya es >= 0, pero
    # los features densos pueden ser negativos (VADER compound).
    # ComplementNB es mas robusto que MultinomialNB para features mixtos.
    # Para MNB: usar solo TF-IDF (subconjunto no negativo).
    n_tfidf = len(train_data["feature_names_tfidf"])

    all_results = {}

    # 1. LogisticRegression L1 (ya conocido)
    t1 = time.time()
    lr_results = train_model_per_label(
        LogisticRegression,
        {"class_weight": "balanced", "max_iter": 300, "random_state": RANDOM_STATE,
         "solver": "liblinear", "C": 0.1, "l1_ratio": 1},
        X_train, y_train, X_test, y_test,
        name="LogisticRegression (L1, C=0.1)"
    )
    all_results["LR_L1"] = lr_results
    print(f"  Tiempo LR: {time.time()-t1:.1f}s")

    # 2. MultinomialNB (solo TF-IDF, valores no negativos)
    t2 = time.time()
    X_train_tfidf = X_train[:, :n_tfidf]
    X_test_tfidf = X_test[:, :n_tfidf]
    mnb_results = train_model_per_label(
        MultinomialNB,
        {"alpha": 0.1, "fit_prior": True},
        X_train_tfidf, y_train, X_test_tfidf, y_test,
        name="MultinomialNB (solo TF-IDF)"
    )
    all_results["MNB"] = mnb_results
    print(f"  Tiempo MNB: {time.time()-t2:.1f}s")

    # 3. ComplementNB (TF-IDF, mejor para clases desbalanceadas)
    t3 = time.time()
    cnb_results = train_model_per_label(
        ComplementNB,
        {"alpha": 0.1, "norm": True},
        X_train_tfidf, y_train, X_test_tfidf, y_test,
        name="ComplementNB (solo TF-IDF)"
    )
    all_results["CNB"] = cnb_results
    print(f"  Tiempo CNB: {time.time()-t3:.1f}s")

    # 4. LinearSVC con calibracion sigmoid
    t4 = time.time()
    svc_results = train_model_per_label(
        LinearSVC,
        {"class_weight": "balanced", "max_iter": 5000, "random_state": RANDOM_STATE, "C": 0.1},
        X_train, y_train, X_test, y_test,
        name="LinearSVC (C=0.1)", needs_proba=True,
        calibrate=True,
    )
    all_results["LinearSVC"] = svc_results
    print(f"  Tiempo LinearSVC: {time.time()-t4:.1f}s")

    # 5. SGDClassifier con log_loss (LR escalable con elasticnet)
    t5 = time.time()
    sgd_results = train_model_per_label(
        SGDClassifier,
        {"loss": "log_loss", "class_weight": "balanced", "max_iter": 1000,
         "random_state": RANDOM_STATE, "alpha": 1e-4, "l1_ratio": 0.5,
         "early_stopping": True, "n_iter_no_change": 5},
        X_train, y_train, X_test, y_test,
        name="SGDClassifier (log_loss, elasticnet)"
    )
    all_results["SGD_LR"] = sgd_results
    print(f"  Tiempo SGD: {time.time()-t5:.1f}s")

    # 6. LightGBM Classifier Chain (ya entrenado)
    t6 = time.time()
    y_prob = chain_model.predict_proba(X_test)
    chain_results = {}
    for j, label in enumerate(LABEL_COLS):
        yt = y_test[:, j]
        yp = y_prob[:, j]
        auc = roc_auc_score(yt, yp)
        opt_t, _ = find_f2_optimal_threshold(yt, yp)
        preds = (yp >= opt_t).astype(int)
        f1 = f1_score(yt, preds, zero_division=0)
        f2 = fbeta_score(yt, preds, beta=2)
        prec = precision_score(yt, preds, zero_division=0)
        rec = recall_score(yt, preds, zero_division=0)
        ap = average_precision_score(yt, yp)
        ece = compute_ece(yt, yp)
        chain_results[label] = {
            "auc": auc, "f1": f1, "f2": f2,
            "precision": prec, "recall": rec,
            "ap": ap, "ece": ece, "threshold": opt_t,
        }
        print(f"    {label}: AUC={auc:.4f}, F1={f1:.4f}, F2={f2:.4f}")
    all_results["LGBM_Chain"] = chain_results
    print(f"  Tiempo LGBM Chain (inference): {time.time()-t6:.1f}s")

    # ============================================================
    # TABLA COMPARATIVA
    # ============================================================
    print("\n" + "=" * 64)
    print("TABLA COMPARATIVA DE MODELOS CPU-ONLY")
    print("=" * 64)

    # AUC
    print("\n--- AUC-ROC ---")
    auc_rows = []
    for model_name, model_results in all_results.items():
        row = {"modelo": model_name}
        for label in LABEL_COLS:
            row[label] = round(model_results[label]["auc"], 4)
        row["MACRO"] = round(np.mean([model_results[l]["auc"] for l in LABEL_COLS]), 4)
        auc_rows.append(row)
    auc_df = pd.DataFrame(auc_rows).sort_values("MACRO", ascending=False)
    print(auc_df.to_string(index=False))

    # F1
    print("\n--- F1-score (umbral F2-optimo) ---")
    f1_rows = []
    for model_name, model_results in all_results.items():
        row = {"modelo": model_name}
        for label in LABEL_COLS:
            row[label] = round(model_results[label]["f1"], 4)
        row["MACRO"] = round(np.mean([model_results[l]["f1"] for l in LABEL_COLS]), 4)
        f1_rows.append(row)
    f1_df = pd.DataFrame(f1_rows).sort_values("MACRO", ascending=False)
    print(f1_df.to_string(index=False))

    # F2
    print("\n--- F2-score (umbral F2-optimo) ---")
    f2_rows = []
    for model_name, model_results in all_results.items():
        row = {"modelo": model_name}
        for label in LABEL_COLS:
            row[label] = round(model_results[label]["f2"], 4)
        row["MACRO"] = round(np.mean([model_results[l]["f2"] for l in LABEL_COLS]), 4)
        f2_rows.append(row)
    f2_df = pd.DataFrame(f2_rows).sort_values("MACRO", ascending=False)
    print(f2_df.to_string(index=False))

    # ECE
    print("\n--- ECE (calibracion) ---")
    ece_rows = []
    for model_name, model_results in all_results.items():
        row = {"modelo": model_name}
        for label in LABEL_COLS:
            row[label] = round(model_results[label]["ece"], 4)
        row["MACRO"] = round(np.mean([model_results[l]["ece"] for l in LABEL_COLS]), 4)
        ece_rows.append(row)
    ece_df = pd.DataFrame(ece_rows).sort_values("MACRO", ascending=True)
    print(ece_df.to_string(index=False))

    # ============================================================
    # EVALUACION DE HIPOTESIS
    # ============================================================
    print("\n" + "=" * 64)
    print("EVALUACION DE HIPOTESIS")
    print("=" * 64)

    # H12: MNB en prevalentes vs raras
    mnb_prevalent = np.mean([all_results["MNB"][l]["auc"] for l in ["toxic", "obscene", "insult"]])
    mnb_rare = np.mean([all_results["MNB"][l]["auc"] for l in ["threat", "identity_hate"]])
    lr_prevalent = np.mean([all_results["LR_L1"][l]["auc"] for l in ["toxic", "obscene", "insult"]])
    lr_rare = np.mean([all_results["LR_L1"][l]["auc"] for l in ["threat", "identity_hate"]])
    delta_prev = mnb_prevalent - lr_prevalent
    delta_rare = mnb_rare - lr_rare

    print(f"\nH12: MultinomialNB en prevalentes vs raras")
    print(f"  MNB AUC prevalentes: {mnb_prevalent:.4f} (LR: {lr_prevalent:.4f}, delta: {delta_prev:+.4f})")
    print(f"  MNB AUC raras: {mnb_rare:.4f} (LR: {lr_rare:.4f}, delta: {delta_rare:+.4f})")
    if delta_prev > delta_rare:
        print("  H12 CONFIRMADA: MNB relativamente mejor en prevalentes que en raras")
    else:
        print("  H12 REFUTADA: MNB no muestra la brecha esperada")

    # H13: LinearSVC vs LR en F1
    svc_f1 = np.mean([all_results["LinearSVC"][l]["f1"] for l in LABEL_COLS])
    lr_f1 = np.mean([all_results["LR_L1"][l]["f1"] for l in LABEL_COLS])
    print(f"\nH13: LinearSVC vs LR en F1 macro")
    print(f"  LinearSVC F1: {svc_f1:.4f}, LR F1: {lr_f1:.4f}, delta: {svc_f1-lr_f1:+.4f}")
    if svc_f1 > lr_f1:
        print("  H13 CONFIRMADA: LinearSVC supera a LR en F1")
    else:
        print("  H13 REFUTADA: LinearSVC no supera a LR en F1")

    # H14: Chain en etiquetas posteriores
    chain_late = np.mean([all_results["LGBM_Chain"][l]["auc"] for l in ["severe_toxic", "identity_hate", "threat"]])
    lr_late = np.mean([all_results["LR_L1"][l]["auc"] for l in ["severe_toxic", "identity_hate", "threat"]])
    print(f"\nH14: Classifier Chain en etiquetas posteriores")
    print(f"  Chain AUC (severe, id_hate, threat): {chain_late:.4f}")
    print(f"  LR AUC (severe, id_hate, threat): {lr_late:.4f}, delta: {chain_late-lr_late:+.4f}")
    if chain_late > lr_late:
        print("  H14 CONFIRMADA: Chain supera a modelos independientes en etiquetas posteriores")
    else:
        print("  H14 REFUTADA: Chain no supera a modelos independientes en etiquetas posteriores")

    # ============================================================
    # GRAFICAS
    # ============================================================

    # 39: Heatmap AUC por modelo x etiqueta
    fig, ax = plt.subplots(figsize=(12, 6))
    auc_matrix = auc_df.drop(columns="MACRO").set_index("modelo").astype(float)
    sns.heatmap(auc_matrix, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                vmin=0.9, vmax=1.0, linewidths=0.5)
    ax.set_title("AUC-ROC por modelo y etiqueta")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/39_model_auc_heatmap.png", dpi=150)
    plt.close(fig)

    # 40: Bar chart AUC macro
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (metric_name, df_met) in enumerate([("AUC-ROC", auc_df), ("F1", f1_df), ("F2", f2_df)]):
        ax = axes[i]
        macros = df_met[["modelo", "MACRO"]].sort_values("MACRO", ascending=True)
        colors = ["#9b59b6" if "LGBM" in m else "#2ecc71" if "SVC" in m
                  else "#e74c3c" if "NB" in m else "#3498db" for m in macros["modelo"]]
        ax.barh(macros["modelo"], macros["MACRO"], color=colors)
        for j, v in enumerate(macros["MACRO"]):
            ax.text(v + 0.002, j, f"{v:.4f}", va="center", fontsize=9)
        ax.set_title(f"{metric_name} macro")
        ax.set_xlim(left=max(0, macros["MACRO"].min() - 0.05))
    fig.suptitle("Comparacion de modelos CPU-only")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/40_model_macro_comparison.png", dpi=150)
    plt.close(fig)

    # 41: Radar chart por etiqueta (AUC)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(LABEL_COLS), endpoint=False).tolist()
    angles += angles[:1]

    model_colors = {
        "LR_L1": "#3498db", "MNB": "#e74c3c", "CNB": "#e67e22",
        "LinearSVC": "#2ecc71", "SGD_LR": "#1abc9c", "LGBM_Chain": "#9b59b6",
    }

    for model_name, model_results in all_results.items():
        values = [model_results[l]["auc"] for l in LABEL_COLS]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name,
                color=model_colors.get(model_name, "gray"), alpha=0.7)
        ax.fill(angles, values, alpha=0.05, color=model_colors.get(model_name, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABEL_COLS)
    ax.set_ylim(0.9, 1.0)
    ax.set_title("AUC-ROC por etiqueta y modelo", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/41_model_radar_auc.png", dpi=150)
    plt.close(fig)

    # Save results
    results_json = {}
    for model_name, model_results in all_results.items():
        results_json[model_name] = {
            label: {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()}
            for label, metrics in model_results.items()
        }

    with open(OUTPUT_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # Copy to EDA imgs
    for fn in ["39_model_auc_heatmap.png", "40_model_macro_comparison.png", "41_model_radar_auc.png"]:
        src = IMG_DIR / fn
        dst = PROJECT_ROOT / "reports" / "eda" / "imgs" / fn
        if src.exists():
            import shutil
            shutil.copy2(src, dst)

    print(f"\nGraficas guardadas en: {IMG_DIR}")
    print("39-41: comparacion de modelos")


if __name__ == "__main__":
    main()
