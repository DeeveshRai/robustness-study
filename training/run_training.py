"""CLI for one tabular walk-forward configuration; no configuration selection."""

import argparse
import json
from pathlib import Path

import pandas as pd

from training.data_splitter import WindowConfig
from training.walk_forward import WalkForwardRunner
from training.models import MODEL_TYPES, create_model
from training.evaluation import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ready", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["expanding", "fixed"], default="expanding")
    parser.add_argument("--train-years", type=int)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--model", choices=list(MODEL_TYPES), default="ridge")
    parameters_group = parser.add_mutually_exclusive_group()
    parameters_group.add_argument("--parameters", type=json.loads, default={})
    parameters_group.add_argument("--parameters-file", type=Path)
    parser.add_argument("--alpha", type=float, help="Ridge alpha; shorthand for --parameters")
    args = parser.parse_args()
    parameters = dict(json.loads(args.parameters_file.read_text()) if args.parameters_file else args.parameters)
    if args.alpha is not None:
        if args.model != "ridge":
            parser.error("--alpha shorthand applies to Ridge; use --parameters for other models")
        parameters["alpha"] = args.alpha
    windows = WindowConfig(args.mode, args.train_years, args.months, args.months)
    result = WalkForwardRunner().run(
        pd.read_csv(args.model_ready), windows,
        model_factory=lambda: create_model(args.model, parameters),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.output, index=False)
    evaluate_predictions(result.predictions).to_csv(
        args.output.with_name(args.output.stem + "_metrics.csv"), index=False
    )
    print(f"Saved {len(result.predictions)} predictions from {len(result.windows)} fitted windows to {args.output}")


if __name__ == "__main__":
    main()
