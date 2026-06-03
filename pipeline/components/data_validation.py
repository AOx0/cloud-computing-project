from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
    ],
)
def validate_data_component(
    project_id: str,
    bq_source_table: str,
    label_column: str,
    validation_report: Output[Artifact],
) -> NamedTuple("Outputs", [("is_valid", bool)]):
    """Validate source data before training.

    Replace this with domain-specific validation later: TensorFlow Data
    Validation, Great Expectations, custom rules, image checks, NLP text quality
    checks, or business constraints.
    """

    import json
    from typing import NamedTuple

    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    dataframe = client.query(f"SELECT * FROM `{bq_source_table}`").to_dataframe()

    missing_values = dataframe.isna().sum().to_dict()
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    checks = {
        "row_count_positive": len(dataframe) > 0,
        "label_column_present": label_column in dataframe.columns,
        "has_numeric_features": len([col for col in numeric_columns if col != label_column]) > 0,
        "missing_values": missing_values,
        "columns": list(dataframe.columns),
        "row_count": int(len(dataframe)),
    }

    is_valid = (
        checks["row_count_positive"]
        and checks["label_column_present"]
        and checks["has_numeric_features"]
    )

    with open(validation_report.path, "w", encoding="utf-8") as file:
        json.dump({"is_valid": is_valid, "checks": checks}, file, indent=2)

    outputs = NamedTuple("Outputs", [("is_valid", bool)])

    if not is_valid:
        raise ValueError(f"Data validation failed: {checks}")

    return outputs(is_valid)

