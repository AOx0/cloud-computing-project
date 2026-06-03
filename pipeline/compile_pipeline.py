from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kfp import compiler

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from pipeline.pipeline import mlops_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="pipeline/compiled/mlops_pipeline.json",
        help="Output path for the compiled Vertex AI Pipeline template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path=str(output_path),
    )
    print(f"Compiled pipeline to {output_path}")


if __name__ == "__main__":
    main()
