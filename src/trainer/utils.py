from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_label(dataframe: pd.DataFrame, label_column: str):
    feature_columns = [column for column in dataframe.columns if column != label_column]
    return dataframe[feature_columns], dataframe[label_column], feature_columns

