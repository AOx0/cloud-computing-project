from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-storage>=2.14.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
    ],
)
def validate_data_component(
    project_id: str,
    gcs_data_uri: str,
    validation_report: Output[Artifact],
) -> NamedTuple("Outputs", [("is_valid", bool)]):
    """Validate Jigsaw toxic comment data from GCS.

    Checks: row count, label columns present, no empty texts, class balance.
    """

    import json
    from typing import NamedTuple
    from io import BytesIO

    from google.cloud import storage
    import pandas as pd

    # Parse GCS URI
    parts = gcs_data_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else "train.csv"

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    # Download CSV
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data))

    # Required columns
    required = ["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    checks = {
        "row_count_positive": len(df) > 0,
        "required_columns_present": all(c in df.columns for c in required),
        "no_empty_comments": df["comment_text"].fillna("").str.strip().str.len().gt(0).sum() > len(df) * 0.99,
        "label_columns_binary": all(df[c].isin([0, 1]).all() for c in label_cols if c in df.columns),
        "min_toxic_prevalence": (df[label_cols].sum(axis=1) > 0).mean() > 0.05 if all(c in df.columns for c in label_cols) else False,
        "row_count": int(len(df)),
        "toxic_prevalence": float((df[label_cols].sum(axis=1) > 0).mean()) if all(c in df.columns for c in label_cols) else 0,
        "columns": list(df.columns),
    }

    is_valid = (
        checks["row_count_positive"]
        and checks["required_columns_present"]
        and checks["no_empty_comments"]
        and checks["label_columns_binary"]
        and checks["min_toxic_prevalence"]
    )

    with open(validation_report.path, "w", encoding="utf-8") as f:
        json.dump({"is_valid": is_valid, "checks": checks}, f, indent=2)

    outputs = NamedTuple("Outputs", [("is_valid", bool)])
    if not is_valid:
        raise ValueError(f"Data validation failed: {json.dumps(checks, indent=2)}")
    return outputs(is_valid)
