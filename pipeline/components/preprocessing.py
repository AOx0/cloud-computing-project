from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "scikit-learn>=1.4.0",
    ],
)
def preprocess_data_component(
    project_id: str,
    bq_source_table: str,
    label_column: str,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Apply generic tabular preprocessing and create train/test artifacts.

    This is a placeholder for real feature engineering. A future use case can
    replace it with image transforms, tokenization, embeddings, feature stores,
    time-series windows, or any model-specific preprocessing code.
    """

    import os

    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split

    client = bigquery.Client(project=project_id)
    dataframe = client.query(f"SELECT * FROM `{bq_source_table}`").to_dataframe()

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    dataframe[numeric_columns] = dataframe[numeric_columns].fillna(dataframe[numeric_columns].mean())

    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe[label_column] if label_column in dataframe.columns else None,
    )

    os.makedirs(train_data.path, exist_ok=True)
    os.makedirs(test_data.path, exist_ok=True)
    train_df.to_csv(os.path.join(train_data.path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(test_data.path, "test.csv"), index=False)

