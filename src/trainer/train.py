from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.trainer.model import build_placeholder_model
from src.trainer.utils import load_csv, split_features_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--model-dir", default="model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = load_csv(args.train_csv)
    x_train, y_train, _ = split_features_label(dataframe, args.label_column)
    model = build_placeholder_model()
    model.fit(x_train, y_train)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    print(f"Saved placeholder model to {model_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
