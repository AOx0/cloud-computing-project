from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_placeholder_model(random_state: int = 42) -> Pipeline:
    """Return the demo sklearn model used only to validate the architecture."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=random_state)),
        ]
    )

