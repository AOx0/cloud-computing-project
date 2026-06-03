from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.4.0",
    ],
)
def train_model_component(
    train_data: Input[Dataset],
    label_column: str,
    model_artifact: Output[Model],
    model_type: str = "sklearn_logistic_regression",
    random_state: int = 42,
) -> NamedTuple("Outputs", [("model_artifact_uri", str)]):
    """Train a simple placeholder model.

    The architecture is intentionally independent from this model. Replace this
    component with TensorFlow, PyTorch, XGBoost, CV, NLP, regression, or another
    training routine when the final use case is selected.
    """

    import json
    import os
    from typing import NamedTuple

    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_df = pd.read_csv(os.path.join(train_data.path, "train.csv"))
    feature_columns = [column for column in train_df.columns if column != label_column]

    x_train = train_df[feature_columns]
    y_train = train_df[label_column]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=random_state)),
        ]
    )
    model.fit(x_train, y_train)

    os.makedirs(model_artifact.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_artifact.path, "model.joblib"))

    metadata = {
        "framework": "scikit-learn",
        "model_type": model_type,
        "label_column": label_column,
        "feature_columns": feature_columns,
        "replacement_note": "Replace this placeholder training code with the real model implementation.",
    }
    with open(os.path.join(model_artifact.path, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    model_artifact.metadata["framework"] = "scikit-learn"
    model_artifact.metadata["model_type"] = model_type
    model_artifact.metadata["feature_columns"] = ",".join(feature_columns)

    outputs = NamedTuple("Outputs", [("model_artifact_uri", str)])
    return outputs(model_artifact.uri)
