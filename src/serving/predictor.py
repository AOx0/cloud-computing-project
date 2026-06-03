from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from google.cloud import storage


class SklearnPredictor:
    """Small custom predictor for the demo sklearn artifact.

    Replace this class when the real model requires TensorFlow, PyTorch, NLP
    tokenization, image preprocessing, or a different prediction protocol.
    """

    def __init__(self) -> None:
        self.model = None

    def load(self, model_uri: str | None = None) -> None:
        model_uri = model_uri or os.getenv("AIP_STORAGE_URI") or os.getenv("MODEL_DIR", "/model")
        model_path = self._resolve_model_path(model_uri)
        self.model = joblib.load(model_path)

    def predict(self, instances: list[dict[str, Any]] | list[list[float]]) -> list[Any]:
        if self.model is None:
            self.load()
        dataframe = pd.DataFrame(instances)
        predictions = self.model.predict(dataframe)
        return predictions.tolist()

    def _resolve_model_path(self, model_uri: str) -> Path:
        if model_uri.startswith("gs://"):
            return self._download_from_gcs(model_uri)

        model_dir = Path(model_uri)
        if model_dir.is_file():
            return model_dir
        return model_dir / "model.joblib"

    def _download_from_gcs(self, model_uri: str) -> Path:
        bucket_name, prefix = model_uri.replace("gs://", "", 1).split("/", 1)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        local_dir = Path(tempfile.mkdtemp(prefix="model-artifact-"))

        for blob in client.list_blobs(bucket, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            relative_path = Path(blob.name).relative_to(prefix)
            destination = local_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(destination)

        return local_dir / "model.joblib"

