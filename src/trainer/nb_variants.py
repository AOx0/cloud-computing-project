"""
Test de lecciones del notebook 'incredibly-simple-naive-bayes'.

El notebook alcanza LB=0.768 con solo MultinomialNB + TF-IDF sin
restriccion de features + undersampling. Queremos saber:

  H15: Undersampling mejora el rendimiento de NB comparado con
  class_weight="balanced" porque NB estima P(feature|clase)
  y el undersampling no distorsiona las verosimilitudes pero
  produce mejor calibracion del prior.

  H16: TF-IDF sin max_features mejora el AUC de NB porque NB
  es inmune a la maldicion de la dimensionalidad (cada feature
  se trata independientemente) y features adicionales solo
  agregan senal.

Herramientas de IA utilizadas: Claude (generacion de codigo).
"""

from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score,
)
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.features import LABEL_COLS
from src.trainer.evaluation import compute_ece, find_f2_optimal_threshold, fbeta_score

RANDOM_STATE = 42
IMG_DIR = PROJECT_ROOT / "reports" / "training" / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
    return text


def eval_per_label(y_true, y_prob):
    """Calcula metricas por etiqueta."""
    results = {}
    for j, label in enumerate(LABEL_COLS):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        try:
            auc = roc_auc_score(yt, yp)
        except Exception:
            auc = 0.5
        opt_t, _ = find_f2_optimal_threshold(yt, yp)
        preds = (yp >= opt_t).astype(int)
        f1 = f1_score(yt, preds, zero_division=0)
        f2 = fbeta_score(yt, preds, beta=2)
        prec = precision_score(yt, preds, zero_division=0)
        rec = recall_score(yt, preds, zero_division=0)
        try:
            ap = average_precision_score(yt, yp)
        except Exception:
            ap = 0
        ece = compute_ece(yt, yp)
        results[label] = {
            "auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4),
            "prec": round(prec, 4), "rec": round(rec, 4),
            "ap": round(ap, 4), "ece": round(ece, 4), "threshold": round(opt_t, 2),
        }
    return results


def main():
    print("=" * 64)
    print("TEST DE LECCIONES DEL NOTEBOOK NAIVE BAYES")
    print("=" * 64)

    # Cargar datos
    df = pd.read_csv(PROJECT_ROOT / "raw" / "juegos" / "train.csv")
    df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)
    df["clean_text"] = df["comment_text"].fillna("").apply(clean_text)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["any_toxic"], random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    y_train = train_df[LABEL_COLS].values
    y_test = test_df[LABEL_COLS].values

    # ============================================================
    # CONFIGURACION 1: TF-IDF con max_features=5000 (actual)
    # CONFIGURACION 2: TF-IDF sin max_features (notebook style)
    # ============================================================

    configs = {
        "tfidf_5k": {
            "max_features": 5000, "ngram_range": (1, 2),
            "sublinear_tf": True, "min_df": 3, "max_df": 0.95,
            "token_pattern": r"(?u)\b\w+\b",
        },
        "tfidf_full": {
            "max_features": None, "ngram_range": (1, 2),
            "sublinear_tf": True, "min_df": 3, "max_df": 0.95,
            "token_pattern": r"(?u)\b\w+\b",
        },
        "tfidf_full_1gram": {
            "max_features": None, "ngram_range": (1, 1),
            "sublinear_tf": True, "min_df": 1, "max_df": 0.95,
            "token_pattern": r"(?u)\b\w+\b",
        },
    }

    all_results = {}

    for cfg_name, cfg_kwargs in configs.items():
        print(f"\n{'='*50}")
        print(f"TF-IDF config: {cfg_name}")
        print(f"{'='*50}")

        t0 = time.time()
        tfidf = TfidfVectorizer(**cfg_kwargs)
        X_train_tfidf = tfidf.fit_transform(train_df["clean_text"])
        X_test_tfidf = tfidf.transform(test_df["clean_text"])
        feat_time = time.time() - t0
        n_features = X_train_tfidf.shape[1]
        print(f"  Features: {n_features}, tiempo: {feat_time:.1f}s")

        # --- Modelo A: MNB con class_weight (alpha=1 default) ---
        t1 = time.time()
        mnb_bal = MultinomialNB(alpha=0.1, fit_prior=True)
        mnb_bal_results = {}
        for j, label in enumerate(LABEL_COLS):
            # NB no tiene class_weight. Usamos sample_weight para simularlo
            # class_weight balanced: w_i = N / (2 * n_class_i) para clase i
            n_pos = y_train[:, j].sum()
            n_neg = len(y_train) - n_pos
            w_pos = len(y_train) / (2 * n_pos)
            w_neg = len(y_train) / (2 * n_neg)
            sample_weight = np.where(y_train[:, j] == 1, w_pos, w_neg)
            mnb_bal.fit(X_train_tfidf, y_train[:, j], sample_weight=sample_weight)
            prob = mnb_bal.predict_proba(X_test_tfidf)[:, 1]
            auc = roc_auc_score(y_test[:, j], prob)
            opt_t, _ = find_f2_optimal_threshold(y_test[:, j], prob)
            preds = (prob >= opt_t).astype(int)
            f1 = f1_score(y_test[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test[:, j], preds, beta=2)
            mnb_bal_results[label] = {"auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4)}
        t_mnb = time.time() - t1
        print(f"  MNB+balanced: AUC macro={np.mean([v['auc'] for v in mnb_bal_results.values()]):.4f}, "
              f"F1 macro={np.mean([v['f1'] for v in mnb_bal_results.values()]):.4f} ({t_mnb:.1f}s)")

        # --- Modelo B: MNB con undersampling (notebook style) ---
        t2 = time.time()
        mnb_us_results = {}
        for j, label in enumerate(LABEL_COLS):
            n_pos = y_train[:, j].sum()
            pos_idx = np.where(y_train[:, j] == 1)[0]
            neg_idx = np.where(y_train[:, j] == 0)[0]
            # Undersample negativos al tamano de positivos
            rng = np.random.RandomState(RANDOM_STATE)
            neg_sample = rng.choice(neg_idx, size=min(n_pos, len(neg_idx)), replace=False)
            us_idx = np.concatenate([pos_idx, neg_sample])
            # Re-fit TF-IDF en undersampled data (notebook lo hace asi)
            X_us = X_train_tfidf[us_idx]
            y_us = y_train[us_idx, j]
            mnb_us = MultinomialNB(alpha=0.1, fit_prior=True)
            mnb_us.fit(X_us, y_us)
            prob = mnb_us.predict_proba(X_test_tfidf)[:, 1]
            auc = roc_auc_score(y_test[:, j], prob)
            opt_t, _ = find_f2_optimal_threshold(y_test[:, j], prob)
            preds = (prob >= opt_t).astype(int)
            f1 = f1_score(y_test[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test[:, j], preds, beta=2)
            mnb_us_results[label] = {"auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4)}
        t_us = time.time() - t2
        print(f"  MNB+undersample: AUC macro={np.mean([v['auc'] for v in mnb_us_results.values()]):.4f}, "
              f"F1 macro={np.mean([v['f1'] for v in mnb_us_results.values()]):.4f} ({t_us:.1f}s)")

        # --- Modelo C: MNB vanilla (sin balanceo) ---
        t3 = time.time()
        mnb_vanilla_results = {}
        for j, label in enumerate(LABEL_COLS):
            mnb_v = MultinomialNB(alpha=0.1, fit_prior=True)
            mnb_v.fit(X_train_tfidf, y_train[:, j])
            prob = mnb_v.predict_proba(X_test_tfidf)[:, 1]
            auc = roc_auc_score(y_test[:, j], prob)
            opt_t, _ = find_f2_optimal_threshold(y_test[:, j], prob)
            preds = (prob >= opt_t).astype(int)
            f1 = f1_score(y_test[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test[:, j], preds, beta=2)
            mnb_vanilla_results[label] = {"auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4)}
        t_van = time.time() - t3
        print(f"  MNB vanilla: AUC macro={np.mean([v['auc'] for v in mnb_vanilla_results.values()]):.4f}, "
              f"F1 macro={np.mean([v['f1'] for v in mnb_vanilla_results.values()]):.4f} ({t_van:.1f}s)")

        # --- Modelo D: ComplementNB vanilla ---
        t4 = time.time()
        cnb_results = {}
        for j, label in enumerate(LABEL_COLS):
            cnb = ComplementNB(alpha=0.1, norm=True)
            cnb.fit(X_train_tfidf, y_train[:, j])
            prob = cnb.predict_proba(X_test_tfidf)[:, 1]
            auc = roc_auc_score(y_test[:, j], prob)
            opt_t, _ = find_f2_optimal_threshold(y_test[:, j], prob)
            preds = (prob >= opt_t).astype(int)
            f1 = f1_score(y_test[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test[:, j], preds, beta=2)
            cnb_results[label] = {"auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4)}
        t_cnb = time.time() - t4
        print(f"  ComplementNB: AUC macro={np.mean([v['auc'] for v in cnb_results.values()]):.4f}, "
              f"F1 macro={np.mean([v['f1'] for v in cnb_results.values()]):.4f} ({t_cnb:.1f}s)")

        # --- Modelo E: LinearSVC (referencia) ---
        t5 = time.time()
        svc_results = {}
        for j, label in enumerate(LABEL_COLS):
            svc = LinearSVC(class_weight="balanced", max_iter=5000, C=0.1, random_state=RANDOM_STATE)
            svc_cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
            svc_cal.fit(X_train_tfidf, y_train[:, j])
            prob = svc_cal.predict_proba(X_test_tfidf)[:, 1]
            auc = roc_auc_score(y_test[:, j], prob)
            opt_t, _ = find_f2_optimal_threshold(y_test[:, j], prob)
            preds = (prob >= opt_t).astype(int)
            f1 = f1_score(y_test[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test[:, j], preds, beta=2)
            svc_results[label] = {"auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4)}
        t_svc = time.time() - t5
        print(f"  LinearSVC: AUC macro={np.mean([v['auc'] for v in svc_results.values()]):.4f}, "
              f"F1 macro={np.mean([v['f1'] for v in svc_results.values()]):.4f} ({t_svc:.1f}s)")

        all_results[cfg_name] = {
            "n_features": n_features,
            "MNB_balanced": mnb_bal_results,
            "MNB_undersample": mnb_us_results,
            "MNB_vanilla": mnb_vanilla_results,
            "ComplementNB": cnb_results,
            "LinearSVC": svc_results,
        }

    # ============================================================
    # TABLA COMPARATIVA
    # ============================================================
    print("\n" + "=" * 64)
    print("TABLA COMPARATIVA")
    print("=" * 64)

    rows = []
    for cfg_name, cfg_results in all_results.items():
        n_feat = cfg_results["n_features"]
        for model_name in ["MNB_vanilla", "MNB_balanced", "MNB_undersample", "ComplementNB", "LinearSVC"]:
            mr = cfg_results[model_name]
            auc_macro = np.mean([mr[l]["auc"] for l in LABEL_COLS])
            f1_macro = np.mean([mr[l]["f1"] for l in LABEL_COLS])
            f2_macro = np.mean([mr[l]["f2"] for l in LABEL_COLS])
            rows.append({
                "tfidf": cfg_name, "n_features": n_feat, "modelo": model_name,
                "AUC_macro": round(auc_macro, 4),
                "F1_macro": round(f1_macro, 4),
                "F2_macro": round(f2_macro, 4),
            })

    comp_df = pd.DataFrame(rows).sort_values("AUC_macro", ascending=False)
    print(comp_df.to_string(index=False))

    # ============================================================
    # DETALLE POR ETIQUETA: mejor NB vs mejor SVC
    # ============================================================
    print("\n--- Detalle AUC por etiqueta ---")
    detail_rows = []
    # Mejor NB config + modelo
    best_nb = comp_df[comp_df["modelo"].str.contains("MNB|Complement")].iloc[0]
    best_nb_cfg = best_nb["tfidf"]
    best_nb_model = best_nb["modelo"]
    # Mejor SVC
    best_svc = comp_df[comp_df["modelo"] == "LinearSVC"].sort_values("AUC_macro", ascending=False).iloc[0]
    best_svc_cfg = best_svc["tfidf"]

    for label in LABEL_COLS:
        nb_auc = all_results[best_nb_cfg][best_nb_model][label]["auc"]
        svc_auc = all_results[best_svc_cfg]["LinearSVC"][label]["auc"]
        nb_f1 = all_results[best_nb_cfg][best_nb_model][label]["f1"]
        svc_f1 = all_results[best_svc_cfg]["LinearSVC"][label]["f1"]
        detail_rows.append({
            "etiqueta": label,
            f"NB_{best_nb_cfg}_{best_nb_model}_AUC": nb_auc,
            f"SVC_{best_svc_cfg}_AUC": svc_auc,
            "delta_AUC": round(nb_auc - svc_auc, 4),
            f"NB_F1": nb_f1,
            f"SVC_F1": svc_f1,
            "delta_F1": round(nb_f1 - svc_f1, 4),
        })

    detail_df = pd.DataFrame(detail_rows)
    print(detail_df.to_string(index=False))

    # ============================================================
    # EVALUACION DE HIPOTESIS
    # ============================================================
    print("\n" + "=" * 64)
    print("EVALUACION DE HIPOTESIS")
    print("=" * 64)

    # H15: Undersampling mejora NB
    for cfg_name in configs.keys():
        bal_auc = np.mean([all_results[cfg_name]["MNB_balanced"][l]["auc"] for l in LABEL_COLS])
        us_auc = np.mean([all_results[cfg_name]["MNB_undersample"][l]["auc"] for l in LABEL_COLS])
        van_auc = np.mean([all_results[cfg_name]["MNB_vanilla"][l]["auc"] for l in LABEL_COLS])
        print(f"\nH15 ({cfg_name}): MNB vanilla={van_auc:.4f}, balanced={bal_auc:.4f}, undersample={us_auc:.4f}")
        if us_auc > bal_auc:
            print(f"  Undersampling > balanced (delta={us_auc-bal_auc:+.4f})")
        else:
            print(f"  Balanced >= undersampling (delta={us_auc-bal_auc:+.4f})")

    # H16: TF-IDF sin max_features mejora NB
    nb_5k = np.mean([all_results["tfidf_5k"]["MNB_vanilla"][l]["auc"] for l in LABEL_COLS])
    nb_full = np.mean([all_results["tfidf_full"]["MNB_vanilla"][l]["auc"] for l in LABEL_COLS])
    print(f"\nH16: MNB vanilla AUC con 5k features={nb_5k:.4f} vs full features={nb_full:.4f}, delta={nb_full-nb_5k:+.4f}")
    if nb_full > nb_5k:
        print("  H16 CONFIRMADA: mas features mejoran NB")
    else:
        print("  H16 REFUTADA: mas features no mejoran NB")

    # ============================================================
    # GRAFICA
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AUC macro por config y modelo
    ax = axes[0]
    for i, (_, row) in enumerate(comp_df.iterrows()):
        label = f"{row['modelo']}\n({row['tfidf']}, {row['n_features']:.0f})"
    # Simplified bar chart
    models_plot = comp_df["modelo"].unique()
    tfidf_configs = comp_df["tfidf"].unique()
    x = np.arange(len(models_plot))
    width = 0.2
    colors = {"tfidf_5k": "#3498db", "tfidf_full": "#e74c3c", "tfidf_full_1gram": "#2ecc71"}
    for j, cfg in enumerate(tfidf_configs):
        vals = []
        for m in models_plot:
            sub = comp_df[(comp_df["modelo"] == m) & (comp_df["tfidf"] == cfg)]
            vals.append(sub["AUC_macro"].values[0] if len(sub) > 0 else 0)
        ax.bar(x + j * width, vals, width, label=f"{cfg}", color=colors.get(cfg, "gray"), alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models_plot, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC-ROC macro")
    ax.set_title("NB variants: AUC por modelo y config TF-IDF")
    ax.legend(fontsize=8)

    # F1
    ax = axes[1]
    for j, cfg in enumerate(tfidf_configs):
        vals = []
        for m in models_plot:
            sub = comp_df[(comp_df["modelo"] == m) & (comp_df["tfidf"] == cfg)]
            vals.append(sub["F1_macro"].values[0] if len(sub) > 0 else 0)
        ax.bar(x + j * width, vals, width, label=f"{cfg}", color=colors.get(cfg, "gray"), alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models_plot, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1 macro")
    ax.set_title("NB variants: F1 por modelo y config TF-IDF")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/42_nb_variants.png", dpi=150)
    plt.close(fig)

    # Copy
    import shutil
    shutil.copy2(f"{IMG_DIR}/42_nb_variants.png", PROJECT_ROOT / "reports" / "eda" / "imgs" / "42_nb_variants.png")

    # Save JSON
    with open(PROJECT_ROOT / "reports" / "training" / "nb_variants.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nGrafica 42 guardada")


if __name__ == "__main__":
    main()
