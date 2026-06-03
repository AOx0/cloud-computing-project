"""Generate synthetic tabular classification data for the demo pipeline.

This file is intentionally simple. In a real project, replace this generator
with ingestion from the business data source: BigQuery, Cloud Storage, APIs,
streaming events, images, documents, or another governed dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from sklearn.datasets import make_classification


def generate_dataframe(rows: int, random_state: int) -> pd.DataFrame:
    features, labels = make_classification(
        n_samples=rows,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=random_state,
    )
    df = pd.DataFrame(
        features,
        columns=["feature_1", "feature_2", "feature_3", "feature_4"],
    )
    df["label"] = labels.astype(int)
    return df


def upload_to_bigquery(df: pd.DataFrame, project_id: str, table_id: str) -> None:
    client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    load_job.result()
    print(f"Loaded {len(df)} rows to {table_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_classification.csv")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--bq-table", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_dataframe(rows=args.rows, random_state=args.random_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")

    if args.project_id and args.bq_table:
      upload_to_bigquery(df, args.project_id, args.bq_table)


if __name__ == "__main__":
    main()

