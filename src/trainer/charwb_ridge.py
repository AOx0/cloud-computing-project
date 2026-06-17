"""
Test de ideas del notebook ensemble LB=0.898.

Dos ideas validas a testear:

  H17: char_wb n-gramas superan a word n-gramas en AUC porque
  capturan variaciones ortograficas deliberadas (f*ck, fck, etc.)
  que los word n-gramas pierden. La mejora sera mayor en
  obscene e insult donde el lexico soez tiene mas variaciones.

  H18: Ridge regression con target continuo (etiquetas ponderadas)
  supera a LinearSVC binario por etiqueta en AUC porque el target
  continuo codifica severidad relativa (threat=1.5 vs toxic=0.32),
  lo que da mas senal al modelo que 6 clasificaciones binarias
  independientes que ignoran la severidad relativa entre etiquetas.

Herramientas de IA utilizadas: Claude (generacion de codigo).
"""

from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    average_precision_score,
)
from scipy.stats import rankdata
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.features import LABEL_COLS
from src.trainer.evaluation import compute_ece, find_f2_optimal_threshold, fbeta_score

RANDOM_STATE = 42
IMG_DIR = PROJECT_ROOT / "reports" / "training" / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Pesos del notebook para crear score continuo
CAT_WEIGHTS = {
    "obscene": 0.16, "toxic": 0.32, "threat": 1.5,
    "insult": 0.64, "severe_toxic": 1.5, "identity_hate": 1.5,
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def eval_per_label_binary(y_true, y_prob):
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


def eval_ridge_per_label(y_true, y_scores):
    """Evalua Ridge como regresor: convierte scores continuos a AUC por etiqueta."""
    results = {}
    for j, label in enumerate(LABEL_COLS):
        yt = y_true[:, j]
        ys = y_scores
        try:
            auc = roc_auc_score(yt, ys)
        except Exception:
            auc = 0.5
        # Umbral F2-optimo sobre el score continuo
        opt_t, _ = find_f2_optimal_threshold(yt, ys)
        preds = (ys >= opt_t).astype(int)
        f1 = f1_score(yt, preds, zero_division=0)
        f2 = fbeta_score(yt, preds, beta=2)
        prec = precision_score(yt, preds, zero_division=0)
        rec = recall_score(yt, preds, zero_division=0)
        try:
            ap = average_precision_score(yt, ys)
        except Exception:
            ap = 0
        results[label] = {
            "auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4),
            "prec": round(prec, 4), "rec": round(rec, 4),
            "ap": round(ap, 4), "threshold": round(opt_t, 4),
        }
    return results


def main():
    print("=" * 64)
    print("TEST: char_wb + Ridge con target continuo")
    print("=" * 64)

    # Cargar datos
    df = pd.read_csv(PROJECT_ROOT / "raw" / "juegos" / "train.csv")
    df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)

    # Target continuo: suma ponderada de etiquetas
    for cat, weight in CAT_WEIGHTS.items():
        df[f"w_{cat}"] = df[cat] * weight
    weight_cols = [f"w_{c}" for c in LABEL_COLS]
    df["severity_score"] = df[weight_cols].sum(axis=1)

    # Text cleaning como el notebook
    df["clean_text"] = df["comment_text"].fillna("").apply(clean_text)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["any_toxic"], random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    y_train_labels = train_df[LABEL_COLS].values
    y_test_labels = test_df[LABEL_COLS].values
    y_train_severity = train_df["severity_score"].values
    y_test_severity = test_df["severity_score"].values

    # ============================================================
    # CONFIGURACIONES TF-IDF
    # ============================================================
    tfidf_configs = {
        "word_1_2": {
            "analyzer": "word", "ngram_range": (1, 2),
            "sublinear_tf": True, "min_df": 3, "max_df": 0.5,
            "token_pattern": r"(?u)\b\w+\b",
        },
        "word_1_2_full": {
            "analyzer": "word", "ngram_range": (1, 2),
            "sublinear_tf": True, "min_df": 1, "max_df": 0.5,
        },
        "charwb_2_5": {
            "analyzer": "char_wb", "ngram_range": (2, 5),
            "sublinear_tf": True, "min_df": 3, "max_df": 0.7,
        },
        "charwb_3_5": {
            "analyzer": "char_wb", "ngram_range": (3, 5),
            "sublinear_tf": True, "min_df": 3, "max_df": 0.5,
        },
        "charwb_3_4": {
            "analyzer": "char_wb", "ngram_range": (3, 4),
            "sublinear_tf": True, "min_df": 3, "max_df": 0.7,
        },
    }

    all_results = {}

    for cfg_name, cfg_kwargs in tfidf_configs.items():
        print(f"\n{'='*50}")
        print(f"TF-IDF: {cfg_name}")
        print(f"{'='*50}")

        t0 = time.time()
        tfidf = TfidfVectorizer(**cfg_kwargs)
        X_train = tfidf.fit_transform(train_df["clean_text"])
        X_test = tfidf.transform(test_df["clean_text"])
        n_features = X_train.shape[1]
        feat_time = time.time() - t0
        print(f"  Features: {n_features}, tiempo: {feat_time:.1f}s")

        # --- Modelo A: LinearSVC binario por etiqueta (referencia) ---
        t1 = time.time()
        svc_results = {}
        for j, label in enumerate(LABEL_COLS):
            svc = LinearSVC(class_weight="balanced", max_iter=5000, C=0.1, random_state=RANDOM_STATE)
            svc_cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
            svc_cal.fit(X_train, y_train_labels[:, j])
            prob = svc_cal.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test_labels[:, j], prob)
            opt_t, _ = find_f2_optimal_threshold(y_test_labels[:, j], prob)
            preds = (prob >= opt_t).astype(int)
            f1 = f1_score(y_test_labels[:, j], preds, zero_division=0)
            f2 = fbeta_score(y_test_labels[:, j], preds, beta=2)
            svc_results[label] = {"auc": round(auc, 4), "f1": round(f1, 4), "f2": round(f2, 4)}
        t_svc = time.time() - t1
        svc_auc = np.mean([v["auc"] for v in svc_results.values()])
        svc_f1 = np.mean([v["f1"] for v in svc_results.values()])
        print(f"  LinearSVC: AUC={svc_auc:.4f}, F1={svc_f1:.4f} ({t_svc:.1f}s)")

        # --- Modelo B: Ridge con target continuo (notebook style) ---
        t2 = time.time()
        ridge = Ridge(alpha=0.5, random_state=RANDOM_STATE)
        ridge.fit(X_train, y_train_severity)
        scores_test = ridge.predict(X_test)
        # Evaluar AUC por etiqueta: el score continuo deberia separar toxicos de limpios
        ridge_results = eval_ridge_per_label(y_test_labels, scores_test)
        ridge_auc = np.mean([v["auc"] for v in ridge_results.values()])
        ridge_f1 = np.mean([v["f1"] for v in ridge_results.values()])
        t_ridge = time.time() - t2
        print(f"  Ridge(severity): AUC={ridge_auc:.4f}, F1={ridge_f1:.4f} ({t_ridge:.1f}s)")

        # --- Modelo C: Ridge con undersampling (notebook style completo) ---
        t3 = time.time()
        n_pos = (y_train_severity > 0).sum()
        pos_idx = np.where(y_train_severity > 0)[0]
        neg_idx = np.where(y_train_severity == 0)[0]
        rng = np.random.RandomState(RANDOM_STATE)
        neg_sample = rng.choice(neg_idx, size=min(n_pos * 2, len(neg_idx)), replace=False)
        us_idx = np.concatenate([pos_idx, neg_sample])
        ridge_us = Ridge(alpha=0.5, random_state=RANDOM_STATE)
        ridge_us.fit(X_train[us_idx], y_train_severity[us_idx])
        scores_us_test = ridge_us.predict(X_test)
        ridge_us_results = eval_ridge_per_label(y_test_labels, scores_us_test)
        ridge_us_auc = np.mean([v["auc"] for v in ridge_us_results.values()])
        ridge_us_f1 = np.mean([v["f1"] for v in ridge_us_results.values()])
        t_us = time.time() - t3
        print(f"  Ridge(severity+US): AUC={ridge_us_auc:.4f}, F1={ridge_us_f1:.4f} ({t_us:.1f}s)")

        # --- Modelo D: Ensemble de 3 Ridge con diferentes alpha ---
        t4 = time.time()
        ridge_05 = Ridge(alpha=0.5, random_state=RANDOM_STATE).fit(X_train, y_train_severity)
        ridge_10 = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_train, y_train_severity)
        ridge_20 = Ridge(alpha=2.0, random_state=RANDOM_STATE).fit(X_train, y_train_severity)
        scores_ens = (ridge_05.predict(X_test) + ridge_10.predict(X_test) + ridge_20.predict(X_test)) / 3.0
        ens_results = eval_ridge_per_label(y_test_labels, scores_ens)
        ens_auc = np.mean([v["auc"] for v in ens_results.values()])
        ens_f1 = np.mean([v["f1"] for v in ens_results.values()])
        t_ens = time.time() - t4
        print(f"  Ridge(ensemble 3alpha): AUC={ens_auc:.4f}, F1={ens_f1:.4f} ({t_ens:.1f}s)")

        all_results[cfg_name] = {
            "n_features": n_features,
            "LinearSVC": svc_results,
            "Ridge_severity": ridge_results,
            "Ridge_severity_US": ridge_us_results,
            "Ridge_ensemble": ens_results,
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
        for model_name in ["LinearSVC", "Ridge_severity", "Ridge_severity_US", "Ridge_ensemble"]:
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

    # Detalle por etiqueta del mejor modelo
    best_row = comp_df.iloc[0]
    best_cfg = best_row["tfidf"]
    best_model = best_row["modelo"]
    print(f"\nMejor configuracion: {best_cfg} + {best_model} (AUC={best_row['AUC_macro']})")
    print("Detalle AUC por etiqueta:")
    mr = all_results[best_cfg][best_model]
    for label in LABEL_COLS:
        print(f"  {label}: AUC={mr[label]['auc']}, F1={mr[label]['f1']}, F2={mr[label]['f2']}")

    # ============================================================
    # EVALUACION DE HIPOTESIS
    # ============================================================
    print("\n" + "=" * 64)
    print("EVALUACION DE HIPOTESIS")
    print("=" * 64)

    # H17: char_wb > word n-gramas
    word_cfgs = [c for c in all_results.keys() if c.startswith("word")]
    char_cfgs = [c for c in all_results.keys() if c.startswith("char")]
    for model_name in ["LinearSVC", "Ridge_severity"]:
        word_aucs = [np.mean([all_results[c][model_name][l]["auc"] for l in LABEL_COLS]) for c in word_cfgs]
        char_aucs = [np.mean([all_results[c][model_name][l]["auc"] for l in LABEL_COLS]) for c in char_cfgs]
        best_word = max(word_aucs)
        best_char = max(char_aucs)
        print(f"\nH17 ({model_name}): mejor word AUC={best_word:.4f}, mejor char_wb AUC={best_char:.4f}, delta={best_char-best_word:+.4f}")
        if best_char > best_word:
            print("  H17 CONFIRMADA: char_wb supera a word n-gramas")
        else:
            print("  H17 REFUTADA: word n-gramas supera a char_wb")

        # Detalle por etiqueta
        best_word_cfg = word_cfgs[word_aucs.index(best_word)]
        best_char_cfg = char_cfgs[char_aucs.index(best_char)]
        for label in LABEL_COLS:
            w_auc = all_results[best_word_cfg][model_name][label]["auc"]
            c_auc = all_results[best_char_cfg][model_name][label]["auc"]
            print(f"  {label}: word={w_auc}, char_wb={c_auc}, delta={c_auc-w_auc:+.4f}")

    # H18: Ridge con target continuo > LinearSVC binario
    for cfg_name in all_results.keys():
        svc_auc = np.mean([all_results[cfg_name]["LinearSVC"][l]["auc"] for l in LABEL_COLS])
        ridge_auc = np.mean([all_results[cfg_name]["Ridge_severity"][l]["auc"] for l in LABEL_COLS])
        ridge_us_auc = np.mean([all_results[cfg_name]["Ridge_severity_US"][l]["auc"] for l in LABEL_COLS])
        ens_auc = np.mean([all_results[cfg_name]["Ridge_ensemble"][l]["auc"] for l in LABEL_COLS])
        print(f"\nH18 ({cfg_name}): SVC={svc_auc:.4f}, Ridge={ridge_auc:.4f}, Ridge+US={ridge_us_auc:.4f}, Ridge ens={ens_auc:.4f}")
        best_ridge = max(ridge_auc, ridge_us_auc, ens_auc)
        if best_ridge > svc_auc:
            print(f"  Ridge ({best_ridge:.4f}) supera SVC ({svc_auc:.4f}), delta={best_ridge-svc_auc:+.4f}")
        else:
            print(f"  SVC ({svc_auc:.4f}) supera Ridge ({best_ridge:.4f}), delta={svc_auc-best_ridge:+.4f}")

    # ============================================================
    # GRAFICAS
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # AUC macro por config y modelo
    ax = axes[0]
    models_plot = ["LinearSVC", "Ridge_severity", "Ridge_severity_US", "Ridge_ensemble"]
    tfidf_order = list(tfidf_configs.keys())
    x = np.arange(len(tfidf_order))
    width = 0.2
    model_colors = {
        "LinearSVC": "#2ecc71",
        "Ridge_severity": "#3498db",
        "Ridge_severity_US": "#e67e22",
        "Ridge_ensemble": "#9b59b6",
    }

    for i, model_name in enumerate(models_plot):
        vals = []
        for cfg in tfidf_order:
            mr = all_results[cfg][model_name]
            vals.append(np.mean([mr[l]["auc"] for l in LABEL_COLS]))
        ax.bar(x + i * width, vals, width, label=model_name,
               color=model_colors[model_name], alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tfidf_order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("AUC-ROC macro")
    ax.set_title("AUC macro por TF-IDF y modelo")
    ax.legend(fontsize=8)
    ax.axhline(0.98, color="gray", linestyle=":", alpha=0.3)

    # F1 macro
    ax = axes[1]
    for i, model_name in enumerate(models_plot):
        vals = []
        for cfg in tfidf_order:
            mr = all_results[cfg][model_name]
            vals.append(np.mean([mr[l]["f1"] for l in LABEL_COLS]))
        ax.bar(x + i * width, vals, width, label=model_name,
               color=model_colors[model_name], alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tfidf_order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("F1 macro")
    ax.set_title("F1 macro por TF-IDF y modelo")
    ax.legend(fontsize=8)

    fig.suptitle("char_wb vs word + Ridge severity vs LinearSVC binario")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/43_charwb_ridge.png", dpi=150)
    plt.close(fig)

    # Detalle por etiqueta: mejor config vs anterior mejor (LinearSVC word)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    best_overall = comp_df.iloc[0]
    best_cfg_name = best_overall["tfidf"]
    best_model_name = best_overall["modelo"]
    # Comparar contra LinearSVC word_1_2_full (mejor anterior)
    baseline_cfg = "word_1_2_full"
    baseline_model = "LinearSVC"

    for j, label in enumerate(LABEL_COLS):
        ax = axes[j]
        new_auc = all_results[best_cfg_name][best_model_name][label]["auc"]
        old_auc = all_results[baseline_cfg][baseline_model][label]["auc"]
        new_f1 = all_results[best_cfg_name][best_model_name][label]["f1"]
        old_f1 = all_results[baseline_cfg][baseline_model][label]["f1"]
        x_pos = [0, 1]
        ax.bar([0], [old_auc], color="#3498db", label=f"SVC word AUC={old_auc:.4f}")
        ax.bar([1], [new_auc], color="#9b59b6", label=f"{best_model_name} {best_cfg_name} AUC={new_auc:.4f}")
        ax.set_title(label)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SVC\nword", f"{best_model_name}\n{best_cfg_name}"], fontsize=8)
        ax.set_ylim(0.9, 1.0)
        ax.legend(fontsize=7)

    fig.suptitle(f"Mejor modelo nuevo vs baseline anterior (por etiqueta)")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/44_best_vs_baseline.png", dpi=150)
    plt.close(fig)

    # Copy
    import shutil
    for fn in ["43_charwb_ridge.png", "44_best_vs_baseline.png"]:
        shutil.copy2(f"{IMG_DIR}/{fn}", PROJECT_ROOT / "reports" / "eda" / "imgs" / fn)

    # Save JSON
    with open(PROJECT_ROOT / "reports" / "training" / "charwb_ridge.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nGraficas 43-44 guardadas")


if __name__ == "__main__":
    main()
