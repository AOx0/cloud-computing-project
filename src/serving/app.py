from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serving.predictor import SklearnPredictor


class PredictRequest(BaseModel):
    instances: list


app = FastAPI(title="MLOps GCP Placeholder Predictor")
predictor = SklearnPredictor()


@app.on_event("startup")
def startup() -> None:
    try:
        predictor.load()
    except Exception:
        # Vertex AI may start the container before the model artifact is ready.
        # The first prediction request retries the load.
        pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, list]:
    try:
        predictions = predictor.predict(request.instances)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"predictions": predictions}

