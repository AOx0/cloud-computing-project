"""
Validacion estadistica rigurosa del modelo LinearSVC + char_wb.

Tres pruebas complementarias:
  1. Bootstrap pareado (2000 resamples): IC95 de la diferencia de AUC
  2. Test de DeLong: significancia estadistica entre AUCs correlacionados
  3. Test de McNemar: asimetrias de error discreto entre modelos

Correccion Bonferroni: alpha = 0.05 / 6 = 0.0083 por etiqueta.

Comparaciones:
  A: LinearSVC + char_wb (2,5)  [mejor modelo]
  B: LinearSVC + word (1,2)      [segundo mejor]
  C: LogisticRegression + char_wb [descomposicion: char_wb vs modelo]

La convergencia de las tres pruebas elimina dudas sobre supuestos
distribucionales y confirma si la ventaja es real o artefacto del split.

Herramientas de IA utilizadas: Claude (generacion de codigo).
"""

from __future__ import annotations

import json, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.features import LABEL_COLS

RANDOM_STATE = 42
IMG_DIR = PROJECT_ROOT / "reports" / "training" / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)
N_BOOTSTRAP = 2000
ALPHA = 0.05
N_LABELS = len(LABEL_COLS)
ALPHA_BONF = ALPHA / N_LABELS  # 0.0083


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


# ============================================================
# 1. BOOTSTRAP PAREADO
# ============================================================
def bootstrap_paired_auc(y_true, y_prob_a, y_prob_b, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA_BONF):
    """
    Bootstrap pareado de la diferencia de AUC entre dos modelos.
    El pareamiento cancela la varianza compartida del split.
    """
    n = len(y_true)
    rng = np.random.RandomState(42)

    boot_delta = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            auc_a = roc_auc_score(y_true[idx], y_prob_a[idx])
            auc_b = roc_auc_score(y_true[idx], y_prob_b[idx])
            boot_delta.append(auc_a - auc_b)
        except Exception:
            continue

    boot_delta = np.array(boot_delta)
    delta_obs = roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b)

    ci_lower = np.percentile(boot_delta, 100 * alpha / 2)
    ci_upper = np.percentile(boot_delta, 100 * (1 - alpha / 2))
    p_value = np.mean(boot_delta <= 0) * 2  # two-sided

    return {
        "delta_AUC": round(delta_obs, 6),
        "IC95": [round(ci_lower, 6), round(ci_upper, 6)],
        "p_value": round(p_value, 6),
        "significativo": "Si" if ci_lower > 0 else "No",
        "n_bootstrap": len(boot_delta),
    }


# ============================================================
# 2. TEST DE DELONG (aproximacion por covarianza de Hanley-McNeil)
# ============================================================
def delong_roc_test(y_true, y_prob_a, y_prob_b):
    """
    Aproximacion del test de DeLong para comparar dos AUCs correlacionados.
    Usa la estructura de covarianza de Hanley & McNeil (1982).
    """
    n1 = y_true.sum()
    n0 = len(y_true) - n1

    if n1 == 0 or n0 == 0:
        return {"p_value": 1.0, "z_stat": 0, "significativo": "No"}

    # Positivos y negativos
    pos_a = y_prob_a[y_true == 1]
    neg_a = y_prob_a[y_true == 0]
    pos_b = y_prob_b[y_true == 1]
    neg_b = y_prob_b[y_true == 0]

    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)

    # Varianza de AUC (Hanley-McNeil)
    Q1_a = auc_a / (1 - auc_a) if auc_a < 1 else 1.0
    Q2_a = (2 * auc_a**2) / (1 + auc_a) if auc_a < 1 else 1.0
    var_a = (auc_a * (1 - auc_a) * (1 + (n0 - 1) * Q1_a + (n1 - 1) * Q2_a)) / (n0 * n1) if n0 * n1 > 0 else 0

    Q1_b = auc_b / (1 - auc_b) if auc_b < 1 else 1.0
    Q2_b = (2 * auc_b**2) / (1 + auc_b) if auc_b < 1 else 1.0
    var_b = (auc_b * (1 - auc_b) * (1 + (n0 - 1) * Q1_b + (n1 - 1) * Q2_b)) / (n0 * n1) if n0 * n1 > 0 else 0

    # Covarianza (aproximacion: r entre scores -> covarianza de AUCs)
    # Metodo simplificado: cov = r * sqrt(var_a * var_b)
    # r se estima como correlacion de rangos entre predicciones
    from scipy.stats import spearmanr
    r, _ = spearmanr(y_prob_a, y_prob_b)
    cov = r * np.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else 0

    # Z-test
    var_diff = var_a + var_b - 2 * cov
    z = (auc_a - auc_b) / np.sqrt(var_diff) if var_diff > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))

    return {
        "z_stat": round(z, 4),
        "p_value": round(p, 6),
        "significativo": "Si" if p < ALPHA_BONF else "No",
        "AUC_A": round(auc_a, 4),
        "AUC_B": round(auc_b, 4),
        "var_A": round(var_a, 8),
        "var_B": round(var_b, 8),
        "cov_AB": round(cov, 8),
    }


# ============================================================
# 3. TEST DE MCNEMAR
# ============================================================
def mcnemar_test(y_true, y_pred_a, y_pred_b, alpha=ALPHA_BONF):
    """
    Test de McNemar para asimetrias de error discreto.
    n10 = A acierta donde B falla
    n01 = B acierta donde A falla
    """
    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)

    n10 = int(((correct_a == 1) & (correct_b == 0)).sum())
    n01 = int(((correct_a == 0) & (correct_b == 1)).sum())

    if n10 + n01 == 0:
        return {"n10": 0, "n01": 0, "chi2": 0, "p_value": 1.0, "significativo": "No"}

    chi2 = (abs(n10 - n01) - 0.5)**2 / (n10 + n01)
    p = 1 - chi2_dist_cdf(chi2)

    return {
        "n10": n10, "n01": n01,
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "significativo": "Si" if p < alpha else "No",
        "A_mejor": "Si" if n10 > n01 else "No",
    }


def chi2_dist_cdf(x):
    """CDF de chi-cuadrada con 1 df."""
    from scipy.stats import chi2 as chi2_dist
    return chi2_dist.cdf(x, df=1)


def main():
    print("=" * 64)
    print("VALIDACION ESTADISTICA RIGUROSA")
    print("Bootstrap pareado + DeLong + McNemar + Bonferroni")
    print(f"Alpha por etiqueta: {ALPHA_BONF:.4f}")
    print("=" * 64)

    # Cargar datos
    df = pd.read_csv(PROJECT_ROOT / "raw" / "juegos" / "train.csv")
    df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)
    df["clean_text"] = df["comment_text"].fillna("").apply(clean_text)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["any_toxic"], random_state=RANDOM_STATE
    )
    test_df = test_df.reset_index(drop=True)
    print(f"Test set: {len(test_df)}")
    y_test = test_df[LABEL_COLS].values

    # ============================================================
    # Entrenar 3 modelos
    # ============================================================
    print("\nEntrenando modelos...")

    # Modelo A: LinearSVC + char_wb (2,5)
    print("  A: LinearSVC + char_wb (2,5)...")
    t0 = time.time()
    tfidf_a = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3, max_df=0.7)
    X_train_a = tfidf_a.fit_transform(train_df["clean_text"])
    X_test_a = tfidf_a.transform(test_df["clean_text"])

    probs_a = np.zeros((len(test_df), N_LABELS))
    preds_a = np.zeros((len(test_df), N_LABELS))
    for j, label in enumerate(LABEL_COLS):
        svc = LinearSVC(class_weight="balanced", max_iter=5000, C=0.1, random_state=RANDOM_STATE)
        cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
        cal.fit(X_train_a, train_df[label].values)
        probs_a[:, j] = cal.predict_proba(X_test_a)[:, 1]
        preds_a[:, j] = cal.predict(X_test_a)
    print(f"  A listo: {time.time()-t0:.1f}s")

    # Modelo B: LinearSVC + word (1,2)
    print("  B: LinearSVC + word (1,2)...")
    t1 = time.time()
    tfidf_b = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_df=0.5,
                              token_pattern=r"(?u)\b\w+\b")
    X_train_b = tfidf_b.fit_transform(train_df["clean_text"])
    X_test_b = tfidf_b.transform(test_df["clean_text"])

    probs_b = np.zeros((len(test_df), N_LABELS))
    preds_b = np.zeros((len(test_df), N_LABELS))
    for j, label in enumerate(LABEL_COLS):
        svc = LinearSVC(class_weight="balanced", max_iter=5000, C=0.1, random_state=RANDOM_STATE)
        cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
        cal.fit(X_train_b, train_df[label].values)
        probs_b[:, j] = cal.predict_proba(X_test_b)[:, 1]
        preds_b[:, j] = cal.predict(X_test_b)
    print(f"  B listo: {time.time()-t1:.1f}s")

    # Modelo C: LR + char_wb (2,5)
    print("  C: LogisticRegression + char_wb (2,5)...")
    t2 = time.time()
    probs_c = np.zeros((len(test_df), N_LABELS))
    preds_c = np.zeros((len(test_df), N_LABELS))
    for j, label in enumerate(LABEL_COLS):
        lr = LogisticRegression(class_weight="balanced", max_iter=300, solver="liblinear",
                                C=0.1, random_state=RANDOM_STATE)
        lr.fit(X_train_a, train_df[label].values)
        probs_c[:, j] = lr.predict_proba(X_test_a)[:, 1]
        preds_c[:, j] = lr.predict(X_test_a)
    print(f"  C listo: {time.time()-t2:.1f}s")

    # ============================================================
    # Pruebas por etiqueta
    # ============================================================
    comparisons = {
        "char_wb vs word (mismo algoritmo)": ("A", "B", probs_a, probs_b, preds_a, preds_b),
        "LinearSVC vs LR (mismo TF-IDF)": ("A", "C", probs_a, probs_c, preds_a, preds_c),
    }

    all_results = {}

    for comp_name, (name_a, name_b, pa, pb, pred_a, pred_b) in comparisons.items():
        print(f"\n{'='*60}")
        print(f"Comparacion: {comp_name}")
        print(f"{'='*60}")

        comp_results = {}

        for j, label in enumerate(LABEL_COLS):
            yt = y_test[:, j]
            ya = pa[:, j]
            yb = pb[:, j]
            pred_a_label = pred_a[:, j]
            pred_b_label = pred_b[:, j]

            print(f"\n  {label}:")

            # 1. Bootstrap pareado
            t0 = time.time()
            boot = bootstrap_paired_auc(yt, ya, yb)
            print(f"    Bootstrap: delta={boot['delta_AUC']:+.6f}, IC95=[{boot['IC95'][0]:+.6f}, {boot['IC95'][1]:+.6f}], p={boot['p_value']:.4f}, sig={boot['significativo']} ({time.time()-t0:.1f}s)")

            # 2. DeLong
            t1 = time.time()
            delong = delong_roc_test(yt, ya, yb)
            print(f"    DeLong: z={delong['z_stat']:.4f}, p={delong['p_value']:.6f}, sig={delong['significativo']} ({time.time()-t1:.1f}s)")

            # 3. McNemar
            t2 = time.time()
            mcnemar = mcnemar_test(yt, pred_a_label, pred_b_label)
            print(f"    McNemar: n10={mcnemar['n10']}, n01={mcnemar['n01']}, chi2={mcnemar['chi2']:.2f}, p={mcnemar['p_value']:.4f}, sig={mcnemar['significativo']}, A_mejor={mcnemar['A_mejor']} ({time.time()-t2:.1f}s)")

            # Convergencia
            sig_count = sum([
                boot["significativo"] == "Si",
                delong["significativo"] == "Si",
                mcnemar["significativo"] == "Si",
            ])
            convergencia = "Triple" if sig_count == 3 else "Doble" if sig_count == 2 else "Simple" if sig_count == 1 else "Ninguna"

            comp_results[label] = {
                "bootstrap": boot,
                "delong": delong,
                "mcnemar": mcnemar,
                "convergencia": convergencia,
                "AUC_A": round(roc_auc_score(yt, ya), 4),
                "AUC_B": round(roc_auc_score(yt, yb), 4),
            }
            print(f"    Convergencia: {convergencia} ({sig_count}/3 pruebas significativas)")

        all_results[comp_name] = comp_results

    # ============================================================
    # TABLA DE DECISION
    # ============================================================
    print("\n" + "=" * 64)
    print("TABLA DE DECISION DE VALIDACION")
    print("=" * 64)

    rows = []
    for comp_name, comp_results in all_results.items():
        for label in LABEL_COLS:
            r = comp_results[label]
            rows.append({
                "comparacion": comp_name,
                "etiqueta": label,
                "AUC_A": r["AUC_A"],
                "AUC_B": r["AUC_B"],
                "delta_AUC": f"{r['bootstrap']['delta_AUC']:+.6f}",
                "bootstrap_sig": r["bootstrap"]["significativo"],
                "delong_sig": r["delong"]["significativo"],
                "mcnemar_sig": r["mcnemar"]["significativo"],
                "convergencia": r["convergencia"],
                "McNemar_A_mejor": r["mcnemar"]["A_mejor"],
            })

    decision_df = pd.DataFrame(rows)
    print(decision_df.to_string(index=False))

    # Resumen
    print("\n=== RESUMEN ===")
    for comp_name, comp_results in all_results.items():
        sig_labels = [l for l in LABEL_COLS if comp_results[l]["convergencia"] in ("Triple", "Doble")]
        no_sig = [l for l in LABEL_COLS if comp_results[l]["convergencia"] in ("Simple", "Ninguna")]
        print(f"\n{comp_name}:")
        if sig_labels:
            print(f"  Significativo (doble o triple): {', '.join(sig_labels)}")
        if no_sig:
            print(f"  No significativo: {', '.join(no_sig)}")

    # ============================================================
    # GRAFICAS
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, label in enumerate(LABEL_COLS):
        ax = axes[i]

        # Barras de delta AUC con IC95 del bootstrap
        comparisons_list = list(all_results.keys())
        short_names = ["char_wb vs word", "SVC vs LR"]
        colors = ["#9b59b6", "#2ecc71"]

        for k, comp_name in enumerate(comparisons_list):
            r = all_results[comp_name][label]
            delta = r["bootstrap"]["delta_AUC"]
            ci = r["bootstrap"]["IC95"]
            color = colors[k]

            ax.bar(k, delta, color=color, alpha=0.7, width=0.6)
            ax.plot([k, k], ci, color="black", linewidth=1.5)
            ax.plot(k, ci[0], "_", color="black", markersize=8)
            ax.plot(k, ci[1], "_", color="black", markersize=8)

            sig_marker = "*" if r["convergencia"] in ("Triple", "Doble") else ""
            ax.text(k, delta + 0.002 if delta > 0 else delta - 0.004,
                    f"{delta:+.4f}{sig_marker}", ha="center", fontsize=8)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(len(comparisons_list)))
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_title(label)
        ax.set_ylabel("Delta AUC-ROC")

    fig.suptitle("Validacion estadistica: delta AUC con IC95 (Bootstrap pareado)\n* = doble o triple convergencia (p < 0.0083, Bonferroni)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{IMG_DIR}/45_validation_deltas.png", dpi=150)
    plt.close(fig)

    # Copy
    import shutil
    shutil.copy2(f"{IMG_DIR}/45_validation_deltas.png",
                 PROJECT_ROOT / "reports" / "eda" / "imgs" / "45_validation_deltas.png")

    # Save JSON
    with open(PROJECT_ROOT / "reports" / "training" / "validacion_estadistica.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResultados guardados en validacion_estadistica.json")
    print("Grafica 45 guardada")


if __name__ == "__main__":
    main()
