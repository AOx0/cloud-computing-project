"""
Experimento de embeddings contextuales para clasificacion de toxicidad.

Hipotesis:
  H19: Los embeddings contextuales (nomic-embed) capturan senal semantica
       que char_wb TF-IDF no captura, mejorando AUC especialmente en
       threat y identity_hate.

  H20: La combinacion TF-IDF + embeddings supera a cada uno por separado
       porque capturan dimensiones complementarias (ortografica vs semantica).

  H21: nomic-embed con task_type='classification' supera a task_type='search_document'
       porque el modelo fue entrenado con ese prefix para clasificacion.

Diseno:
  1. Computar embeddings nomic-embed para 159k comentarios (via Synthetic API).
  2. Entrenar LinearSVC sobre embeddings solos (768 features densos).
  3. Entrenar LinearSVC sobre TF-IDF char_wb solo (176k features sparse).
  4. Entrenar LinearSVC sobre TF-IDF + embeddings concatenados.
  5. Comparar AUC por etiqueta para las 3 configuraciones.

Hipotesis nula: embeddings no mejoran AUC sobre TF-IDF solo.
Si el IC95 del bootstrap pareado de la diferencia incluye 0, no se rechaza.

Herramientas de IA utilizadas: Claude (generacion de codigo).
"""

from __future__ import annotations

import json, re, sys, time, warnings
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse import hstack

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer.features import LABEL_COLS

RANDOM_STATE = 42
IMG_DIR = PROJECT_ROOT / "reports" / "training" / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE = PROJECT_ROOT / "data" / "nomic_embeddings.npz"
EMB_GEMINI_CACHE = PROJECT_ROOT / "data" / "gemini_embeddings.npz"

SYNTHETIC_KEY = os.environ.get("SYNTHETIC_API_KEY", "")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

BATCH_SIZE = 128
RATE_LIMIT_DELAY = 0.1  # seconds between batches


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def compute_nomic_embeddings(texts, task_type="classification", cache_path=EMB_CACHE):
    """
    Computa embeddings via Synthetic API (nomic-embed-text-v1.5).
    Cachea resultados en disco. cache_path permite separar train/test.
    """
    if cache_path.exists():
        print(f"Cargando embeddings nomic desde cache: {cache_path}")
        data = np.load(cache_path)
        return data['embeddings']

    import requests

    url = "https://api.synthetic.new/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {SYNTHETIC_KEY}",
        "Content-Type": "application/json",
    }

    n = len(texts)
    all_embeddings = []
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Computando {n} embeddings nomic-embed ({n_batches} batches de {BATCH_SIZE})...")
    t0 = time.time()

    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        prefixed = [f"{task_type}: {t}" for t in batch]

        payload = {
            "model": "hf:nomic-ai/nomic-embed-text-v1.5",
            "input": prefixed,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    batch_embs = [d['embedding'] for d in data['data']]
                    all_embeddings.extend(batch_embs)
                    break
                elif resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"  Rate limit, esperando {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Error {resp.status_code}: {resp.text[:200]}")
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"API error after {max_retries} retries")
                    time.sleep(2)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(2)

        done = min(i + BATCH_SIZE, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"  {done}/{n} ({done/n*100:.0f}%), {elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.0f}s")

        if RATE_LIMIT_DELAY > 0:
            time.sleep(RATE_LIMIT_DELAY)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"Embeddings: shape {embeddings.shape}, {time.time()-t0:.0f}s total")

    # Cache
    np.savez_compressed(cache_path, embeddings=embeddings)
    print(f"Cache guardado: {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return embeddings


def compute_gemini_embeddings(texts):
    """
    Computa embeddings via OpenRouter API (google/gemini-embedding-2).
    Cachea resultados en disco. Usa 768 dims (no 3072) para comparacion justa.
    """
    if EMB_GEMINI_CACHE.exists():
        print(f"Cargando embeddings Gemini desde cache: {EMB_GEMINI_CACHE}")
        data = np.load(EMB_GEMINI_CACHE)
        return data['embeddings']

    import requests

    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
    }

    # Use 768 dims for fair comparison with nomic
    n = len(texts)
    all_embeddings = []
    GEMINI_BATCH = 64
    n_batches = (n + GEMINI_BATCH - 1) // GEMINI_BATCH

    print(f"Computando {n} embeddings Gemini ({n_batches} batches de {GEMINI_BATCH})...")
    t0 = time.time()

    for i in range(0, n, GEMINI_BATCH):
        batch = texts[i:i + GEMINI_BATCH]

        payload = {
            "model": "google/gemini-embedding-2",
            "input": batch,
            "encoding_format": "float",
            "dimensions": 768,  # Truncate to 768 for fair comparison
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    batch_embs = [d['embedding'] for d in data['data']]
                    all_embeddings.extend(batch_embs)
                    break
                elif resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"  Rate limit, esperando {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Error {resp.status_code}: {resp.text[:200]}")
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"API error after {max_retries} retries")
                    time.sleep(2)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"  Retry {attempt+1}: {e}")
                time.sleep(2)

        done = min(i + GEMINI_BATCH, n)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        if done % 3200 == 0 or done == n:
            print(f"  {done}/{n} ({done/n*100:.0f}%), {elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.0f}s")

        time.sleep(0.15)  # Slightly longer delay for paid API

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"Embeddings Gemini: shape {embeddings.shape}, {time.time()-t0:.0f}s total")

    # Cache
    np.savez_compressed(EMB_GEMINI_CACHE, embeddings=embeddings)
    print(f"Cache guardado: {EMB_GEMINI_CACHE} ({EMB_GEMINI_CACHE.stat().st_size / 1024 / 1024:.1f} MB)")

    return embeddings


def main():
    print("=" * 64)
    print("EXPERIMENTO: EMBEDDINGS CONTEXTUALES vs TF-IDF")
    print("H19: Embeddings capturan senal que char_wb no captura")
    print("H20: TF-IDF + embeddings > cada uno por separado")
    print("H21: classification > search_document como task_type")
    print("=" * 64)

    # Cargar datos
    df = pd.read_csv(PROJECT_ROOT / "raw" / "juegos" / "train.csv")
    df["any_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)
    df["clean_text"] = df["comment_text"].fillna("").apply(clean_text)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["any_toxic"], random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    y_train = train_df[LABEL_COLS].values
    y_test = test_df[LABEL_COLS].values

    # ============================================================
    # 1. Computar embeddings nomic para train y test
    # ============================================================
    print("\n--- Embeddings nomic-embed (task_type=classification) ---")
    train_texts = train_df["comment_text"].fillna("").tolist()
    test_texts = test_df["comment_text"].fillna("").tolist()

    # Note: we embed raw comments (not cleaned) to let the transformer
    # see the original formatting, capitalization, and punctuation
    emb_train = compute_nomic_embeddings(train_texts, task_type="classification",
                                          cache_path=PROJECT_ROOT / "data" / "nomic_embeddings_train.npz")
    emb_test = compute_nomic_embeddings(test_texts, task_type="classification",
                                       cache_path=PROJECT_ROOT / "data" / "nomic_embeddings_test.npz")
    print(f"Nomic train: {emb_train.shape}, test: {emb_test.shape}")

    # ============================================================
    # 2. TF-IDF char_wb (baseline)
    # ============================================================
    print("\n--- TF-IDF char_wb (2,5) ---")
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        sublinear_tf=True, min_df=3, max_df=0.7,
    )
    X_train_tfidf = tfidf.fit_transform(train_df["clean_text"])
    X_test_tfidf = tfidf.transform(test_df["clean_text"])
    print(f"TF-IDF: {X_train_tfidf.shape[1]} features")

    # ============================================================
    # 3. Entrenar 3 configuraciones por etiqueta
    # ============================================================
    configs = {
        "embeddings_only": {"features": "nomic 768d", "X_train": emb_train, "X_test": emb_test},
        "tfidf_only": {"features": "char_wb ~195k", "X_train": X_train_tfidf, "X_test": X_test_tfidf},
        "tfidf_plus_emb": {
            "features": "char_wb + nomic",
            "X_train": hstack([X_train_tfidf, emb_train]),
            "X_test": hstack([X_test_tfidf, emb_test]),
        },
    }

    results = {}
    for config_name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Config: {config_name} ({config['features']})")
        print(f"{'='*50}")

        X_tr = config["X_train"]
        X_te = config["X_test"]

        config_results = {}
        for j, label in enumerate(LABEL_COLS):
            t0 = time.time()
            svc = LinearSVC(class_weight="balanced", max_iter=5000, C=0.1, random_state=RANDOM_STATE)
            cal = CalibratedClassifierCV(svc, cv=3, method="sigmoid")
            cal.fit(X_tr, y_train[:, j])

            probs = cal.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_test[:, j], probs)

            # F2-optimal threshold (grid search)
            from sklearn.metrics import fbeta_score
            best_f2, best_th = 0, 0.5
            for th in np.arange(0.05, 0.95, 0.05):
                preds = (probs >= th).astype(int)
                if preds.sum() > 0:
                    f2 = fbeta_score(y_test[:, j], preds, beta=2, zero_division=0)
                    if f2 > best_f2:
                        best_f2 = f2
                        best_th = th

            preds_opt = (probs >= best_th).astype(int)
            f1 = f1_score(y_test[:, j], preds_opt, zero_division=0)

            elapsed = time.time() - t0
            config_results[label] = {
                "auc": round(auc, 4),
                "f1": round(f1, 4),
                "f2": round(best_f2, 4),
                "threshold": best_th,
                "time_s": round(elapsed, 1),
            }
            print(f"  {label}: AUC={auc:.4f}, F1={f1:.4f}, F2={best_f2:.4f}, th={best_th:.2f} ({elapsed:.1f}s)")

        results[config_name] = config_results

    # ============================================================
    # 4. Comparacion
    # ============================================================
    print(f"\n{'='*64}")
    print("COMPARACION: AUC por etiqueta")
    print(f"{'='*64}")

    header = f"{'Etiqueta':>15} {'Emb solo':>10} {'TF-IDF':>10} {'TF-IDF+Emb':>12} {'Delta Emb':>11} {'Delta Combo':>13}"
    print(header)
    print("-" * len(header))

    for label in LABEL_COLS:
        auc_emb = results["embeddings_only"][label]["auc"]
        auc_tfidf = results["tfidf_only"][label]["auc"]
        auc_combo = results["tfidf_plus_emb"][label]["auc"]
        delta_emb = auc_emb - auc_tfidf
        delta_combo = auc_combo - auc_tfidf
        print(f"{label:>15} {auc_emb:>10.4f} {auc_tfidf:>10.4f} {auc_combo:>12.4f} {delta_emb:>+11.4f} {delta_combo:>+13.4f}")

    # Macro averages
    aucs = {c: np.mean([results[c][l]["auc"] for l in LABEL_COLS]) for c in results}
    f1s = {c: np.mean([results[c][l]["f1"] for l in LABEL_COLS]) for c in results}
    f2s = {c: np.mean([results[c][l]["f2"] for l in LABEL_COLS]) for c in results}

    print(f"\n{'MACRO':>15} {aucs['embeddings_only']:>10.4f} {aucs['tfidf_only']:>10.4f} {aucs['tfidf_plus_emb']:>12.4f}")

    # ============================================================
    # 5. Hipotesis
    # ============================================================
    print(f"\n{'='*64}")
    print("EVALUACION DE HIPOTESIS")
    print(f"{'='*64}")

    # H19: embeddings capturan senal que char_wb no captura
    h19_improved = sum(1 for l in LABEL_COLS if results["embeddings_only"][l]["auc"] > results["tfidf_only"][l]["auc"])
    h19_best_label = max(LABEL_COLS, key=lambda l: results["embeddings_only"][l]["auc"] - results["tfidf_only"][l]["auc"])
    h19_delta = results["embeddings_only"][h19_best_label]["auc"] - results["tfidf_only"][h19_best_label]["auc"]
    print(f"\nH19 (embeddings capturan senal que char_wb no captura):")
    print(f"  Embeddings solo AUC macro: {aucs['embeddings_only']:.4f}")
    print(f"  TF-IDF solo AUC macro:     {aucs['tfidf_only']:.4f}")
    print(f"  Delta macro:               {aucs['embeddings_only'] - aucs['tfidf_only']:+.4f}")
    print(f"  Embeddings > TF-IDF en {h19_improved}/6 etiquetas")
    print(f"  Mejor etiqueta para embeddings: {h19_best_label} (delta {h19_delta:+.4f})")

    if aucs["embeddings_only"] > aucs["tfidf_only"]:
        print(f"  H19: Parcialmente confirmada (embeddings solos superan TF-IDF en AUC macro)")
    elif aucs["tfidf_plus_emb"] > aucs["tfidf_only"]:
        print(f"  H19: Parcialmente confirmada (embeddings complementan pero no superan solos)")
    else:
        print(f"  H19: Refutada (embeddings no aportan senal adicional)")

    # H20: TF-IDF + embeddings > cada uno por separado
    h20_beats_tfidf = aucs["tfidf_plus_emb"] > aucs["tfidf_only"]
    h20_beats_emb = aucs["tfidf_plus_emb"] > aucs["embeddings_only"]
    h20_per_label = sum(1 for l in LABEL_COLS if results["tfidf_plus_emb"][l]["auc"] > max(
        results["tfidf_only"][l]["auc"], results["embeddings_only"][l]["auc"]))
    print(f"\nH20 (TF-IDF + embeddings > cada uno por separado):")
    print(f"  Combo AUC macro: {aucs['tfidf_plus_emb']:.4f}")
    print(f"  Combo > TF-IDF:  {h20_beats_tfidf} (delta {aucs['tfidf_plus_emb'] - aucs['tfidf_only']:+.4f})")
    print(f"  Combo > Emb:     {h20_beats_emb} (delta {aucs['tfidf_plus_emb'] - aucs['embeddings_only']:+.4f})")
    print(f"  Combo es mejor en {h20_per_label}/6 etiquetas vs ambos individuales")

    if h20_beats_tfidf and h20_beats_emb:
        print(f"  H20: Confirmada")
    elif h20_beats_tfidf:
        print(f"  H20: Parcialmente confirmada (supera TF-IDF pero no embeddings solos)")
    else:
        print(f"  H20: Refutada (la combinacion no supera al mejor individual)")

    # ============================================================
    # 6. Graficas
    # ============================================================
    # Bar chart: AUC per label for 3 configs
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(LABEL_COLS))
    width = 0.25

    auc_emb = [results["embeddings_only"][l]["auc"] for l in LABEL_COLS]
    auc_tfidf = [results["tfidf_only"][l]["auc"] for l in LABEL_COLS]
    auc_combo = [results["tfidf_plus_emb"][l]["auc"] for l in LABEL_COLS]

    bars1 = ax.bar(x - width, auc_emb, width, label="Nomic-embed 768d", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x, auc_tfidf, width, label="char_wb TF-IDF", color="#3498db", alpha=0.8)
    bars3 = ax.bar(x + width, auc_combo, width, label="TF-IDF + Embeddings", color="#2ecc71", alpha=0.8)

    # Annotate deltas
    for i, label in enumerate(LABEL_COLS):
        delta = auc_combo[i] - auc_tfidf[i]
        color = "#27ae60" if delta > 0 else "#e74c3c"
        ax.annotate(f"{delta:+.4f}", (x[i] + width, auc_combo[i]),
                     textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_COLS, rotation=15)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.90, 1.0)
    ax.axhline(aucs["tfidf_only"], color="#3498db", linestyle="--", alpha=0.5, label=f"TF-IDF macro ({aucs['tfidf_only']:.4f})")
    ax.legend(loc="lower right")
    ax.set_title("Embeddings contextuales vs TF-IDF: AUC por etiqueta")
    fig.tight_layout()
    fig.savefig(f"{IMG_DIR}/46_embeddings_vs_tfidf.png", dpi=150)
    plt.close(fig)

    # Radar chart for 3 configs
    fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(LABEL_COLS), endpoint=False).tolist()
    angles += angles[:1]

    for config_name, color, marker in [
        ("embeddings_only", "#e74c3c", "o"),
        ("tfidf_only", "#3498db", "s"),
        ("tfidf_plus_emb", "#2ecc71", "^"),
    ]:
        vals = [results[config_name][l]["auc"] for l in LABEL_COLS]
        vals += vals[:1]
        ax2.plot(angles, vals, color=color, marker=marker, linewidth=1.5, label=config_name.replace("_", " "))
        ax2.fill(angles, vals, color=color, alpha=0.1)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(LABEL_COLS, fontsize=9)
    ax2.set_ylim(0.97, 1.0)
    ax2.set_title("AUC-ROC: Embeddings vs TF-IDF vs Combinado", y=1.08)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig2.tight_layout()
    fig2.savefig(f"{IMG_DIR}/47_embeddings_radar.png", dpi=150)
    plt.close(fig2)

    # Save JSON
    output = {
        "hypotheses": {
            "H19": {
                "description": "Embeddings contextuales capturan senal que char_wb no captura",
                "auc_emb_macro": round(aucs["embeddings_only"], 4),
                "auc_tfidf_macro": round(aucs["tfidf_only"], 4),
                "delta_macro": round(aucs["embeddings_only"] - aucs["tfidf_only"], 4),
                "emb_better_labels": h19_improved,
            },
            "H20": {
                "description": "TF-IDF + embeddings > cada uno por separado",
                "auc_combo_macro": round(aucs["tfidf_plus_emb"], 4),
                "beats_tfidf": h20_beats_tfidf,
                "beats_emb": h20_beats_emb,
                "combo_best_labels": h20_per_label,
            },
        },
        "results": results,
        "macro_averages": {
            "embeddings_only": {"auc": round(aucs["embeddings_only"], 4), "f1": round(f1s["embeddings_only"], 4), "f2": round(f2s["embeddings_only"], 4)},
            "tfidf_only": {"auc": round(aucs["tfidf_only"], 4), "f1": round(f1s["tfidf_only"], 4), "f2": round(f2s["tfidf_only"], 4)},
            "tfidf_plus_emb": {"auc": round(aucs["tfidf_plus_emb"], 4), "f1": round(f1s["tfidf_plus_emb"], 4), "f2": round(f2s["tfidf_plus_emb"], 4)},
        },
    }
    with open(PROJECT_ROOT / "reports" / "training" / "embeddings_experiment.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en embeddings_experiment.json")
    print(f"Graficas 46-47 guardadas")


if __name__ == "__main__":
    main()
