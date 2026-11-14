"""CLI for one Ridge walk-forward configuration; no configuration selection."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

from training.data_splitter import WindowConfig
from training.walk_forward import WalkForwardRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_ready", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["expanding", "fixed"], default="expanding")
    parser.add_argument("--train-years", type=int)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    windows = WindowConfig(args.mode, args.train_years, args.months, args.months)
    result = WalkForwardRunner().run(
        pd.read_csv(args.model_ready), windows,
        model_factory=lambda: Ridge(alpha=args.alpha),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.output, index=False)
    print(f"Saved {len(result.predictions)} predictions from {len(result.windows)} fitted windows to {args.output}")


if __name__ == "__main__":
    main()
