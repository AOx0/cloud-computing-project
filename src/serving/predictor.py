"""
Toxic Comment Classification API.

Serving endpoint that accepts raw text, computes TF-IDF char_wb features
locally and nomic-embed embeddings via Synthetic API, concatenates them,
and runs 6 calibrated LinearSVC models to produce per-label probabilities.

Deployed on Cloud Run.
"""

from __future__ import annotations

import os, re, time, json
from typing import Optional

import numpy as np
import joblib
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# Configuration
# ============================================================

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

SYNTHETIC_API_KEY = os.environ.get("SYNTHETIC_API_KEY", "")
SYNTHETIC_API_URL = "https://api.synthetic.new/openai/v1/embeddings"
EMBEDDING_MODEL = "hf:nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
EMBEDDING_TASK_TYPE = "classification"

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")

# F2-optimal thresholds from training
THRESHOLDS = {
    "toxic": 0.15,
    "severe_toxic": 0.10,
    "obscene": 0.10,
    "threat": 0.15,
    "insult": 0.15,
    "identity_hate": 0.10,
}

# ============================================================
# Text preprocessing
# ============================================================

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ip:\d+\.\d+\.\d+\.\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\d]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

# ============================================================
# Embedding client
# ============================================================

def get_embeddings(texts: list[str], api_key: str) -> np.ndarray:
    """Compute nomic-embed embeddings via Synthetic API."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prefixed = [f"{EMBEDDING_TASK_TYPE}: {t}" for t in texts]
    payload = {
        "model": EMBEDDING_MODEL,
        "input": prefixed,
    }

    for attempt in range(3):
        try:
            resp = requests.post(SYNTHETIC_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                embeddings = [d["embedding"] for d in data["data"]]
                return np.array(embeddings, dtype=np.float32)
            elif resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Embedding API error: {resp.status_code} {resp.text[:200]}"
                )
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise HTTPException(status_code=504, detail="Embedding API timeout")
            time.sleep(1)

    raise HTTPException(status_code=502, detail="Embedding API failed after 3 retries")


# ============================================================
# Model loading
# ============================================================

class ToxicityPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tfidf = None
        self.models = {}
        self.thresholds = THRESHOLDS
        self._load_models()

    def _load_models(self):
        t0 = time.time()

        # Load TF-IDF vectorizer
        tfidf_path = os.path.join(self.model_dir, "tfidf_charwb_2_5.joblib")
        self.tfidf = joblib.load(tfidf_path)

        # Load per-label LinearSVC models
        for label in LABEL_COLS:
            model_path = os.path.join(self.model_dir, f"svc_{label}.joblib")
            self.models[label] = joblib.load(model_path)

        elapsed = time.time() - t0
        n_features = self.tfidf.get_feature_names_out().shape[0] if hasattr(self.tfidf, 'get_feature_names_out') else 0
        print(f"Loaded {len(self.models)} models + TF-IDF ({n_features} features) in {elapsed:.1f}s")

    def predict(self, texts: list[str], api_key: str) -> list[dict]:
        """
        Predict toxicity probabilities for a batch of texts.

        Returns list of dicts with keys: probabilities, labels, threshold
        """
        t_total = time.time()

        # Step 1: Clean text for TF-IDF
        clean_texts = [clean_text(t) for t in texts]

        # Step 2: TF-IDF features (local, fast)
        t0 = time.time()
        X_tfidf = self.tfidf.transform(clean_texts)
        tfidf_time = time.time() - t0

        # Step 3: Embedding features (API call)
        t0 = time.time()
        X_emb = get_embeddings(texts, api_key)
        emb_time = time.time() - t0

        # Step 4: Concatenate
        X = hstack([X_tfidf, X_emb]).tocsr()

        # Step 5: Predict per label
        t0 = time.time()
        results = []
        for i, text in enumerate(texts):
            row = X[i]
            probs = {}
            labels = {}
            for label in LABEL_COLS:
                prob = self.models[label].predict_proba(row)[0, 1]
                probs[label] = round(float(prob), 4)
                labels[label] = prob >= self.thresholds[label]
            results.append({
                "text": text[:200] + "..." if len(text) > 200 else text,
                "probabilities": probs,
                "labels": labels,
                "thresholds": self.thresholds,
            })
        predict_time = time.time() - t0

        total_time = time.time() - t_total
        print(f"Predicted {len(texts)} texts: tfidf={tfidf_time:.2f}s, emb={emb_time:.2f}s, predict={predict_time:.2f}s, total={total_time:.2f}s")

        return results


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Toxic Comment Classification API",
    description="Multi-label toxicity classifier using LinearSVC + char_wb TF-IDF + nomic-embed",
    version="1.0.0",
)

predictor: Optional[ToxicityPredictor] = None


@app.on_event("startup")
def startup():
    global predictor
    predictor = ToxicityPredictor(MODEL_DIR)


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128, description="List of comments to classify")
    api_key: Optional[str] = Field(None, description="Synthetic API key (or set SYNTHETIC_API_KEY env var)")


class PredictionResult(BaseModel):
    text: str
    probabilities: dict
    labels: dict
    thresholds: dict


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    model_info: dict
    latency_ms: int


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
        "n_labels": len(LABEL_COLS) if predictor else 0,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    api_key = request.api_key or SYNTHETIC_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key provided. Set SYNTHETIC_API_KEY env var or pass api_key in request."
        )

    t0 = time.time()
    predictions = predictor.predict(request.texts, api_key)
    latency_ms = int((time.time() - t0) * 1000)

    return PredictResponse(
        predictions=predictions,
        model_info={
            "algorithm": "LinearSVC + CalibratedClassifierCV (sigmoid)",
            "features": "TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (768d)",
            "n_labels": len(LABEL_COLS),
            "labels": LABEL_COLS,
            "auc_macro": 0.9903,
        },
        latency_ms=latency_ms,
    )


@app.get("/model_info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {
        "algorithm": "LinearSVC + CalibratedClassifierCV (sigmoid, cv=3)",
        "features": "TF-IDF char_wb (2,5) + nomic-embed-text-v1.5 (classification, 768d)",
        "tfidf_features": predictor.tfidf.get_feature_names_out().shape[0] if hasattr(predictor.tfidf, 'get_feature_names_out') else "N/A",
        "embedding_features": EMBEDDING_DIM,
        "labels": LABEL_COLS,
        "thresholds": THRESHOLDS,
        "metrics": {
            "auc_macro": 0.9903,
            "f1_macro": 0.6388,
        },
    }
