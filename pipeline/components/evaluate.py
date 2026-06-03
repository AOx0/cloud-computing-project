from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Model, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.4.0",
    ],
)
def evaluate_model_component(
    model_artifact: Input[Model],
    test_data: Input[Dataset],
    label_column: str,
    metric_name: str,
    deploy_threshold: float,
    evaluation_report: Output[Artifact],
) -> NamedTuple("Outputs", [("metric_value", float), ("deploy_decision", bool)]):
    """Evaluate the model and return a generic deployment decision."""

    import json
    import os
    from typing import NamedTuple

    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score

    model = joblib.load(os.path.join(model_artifact.path, "model.joblib"))
    test_df = pd.read_csv(os.path.join(test_data.path, "test.csv"))
    feature_columns = [column for column in test_df.columns if column != label_column]

    x_test = test_df[feature_columns]
    y_test = test_df[label_column]
    predictions = model.predict(x_test)

    metric_values = {"accuracy": float(accuracy_score(y_test, predictions))}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]
        metric_values["roc_auc"] = float(roc_auc_score(y_test, probabilities))

    if metric_name not in metric_values:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Available metrics: {sorted(metric_values)}"
        )

    metric_value = metric_values[metric_name]
    deploy_decision = metric_value >= deploy_threshold

    report = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "deploy_threshold": deploy_threshold,
        "deploy_decision": deploy_decision,
        "all_metrics": metric_values,
        "business_note": "In a real project, the threshold must be defined by the business risk and model objective.",
    }

    with open(evaluation_report.path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    outputs = NamedTuple("Outputs", [("metric_value", float), ("deploy_decision", bool)])
    return outputs(metric_value, deploy_decision)

