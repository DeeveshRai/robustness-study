"""Select on 2018--2021, or evaluate an already frozen selection on 2022--2025."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from training.experiments import Candidate, FrozenSelection, select_configuration, evaluate_frozen_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Compare configurations using validation only")
    select.add_argument("model_ready", type=Path)
    select.add_argument("candidates", type=Path)
    select.add_argument("output_dir", type=Path)
    select.add_argument("--metric", choices=["mae", "rmse", "directional_accuracy"], default="mae")
    test = commands.add_parser("test", help="Run frozen settings on the final test period")
    test.add_argument("model_ready", type=Path)
    test.add_argument("selected", type=Path)
    test.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.model_ready)
    if args.command == "select":
        candidates = [Candidate.from_dict(item) for item in json.loads(args.candidates.read_text())]
        result = select_configuration(data, candidates, args.metric)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "selected.json").write_text(json.dumps(result.selected.to_dict(), indent=2))
        result.leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
        reports, predictions = [], []
        for name in result.predictions:
            predictions.append(result.predictions[name].assign(candidate=name))
            reports.append(result.metrics[name].assign(candidate=name))
        pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "validation_predictions.csv", index=False)
        pd.concat(reports, ignore_index=True).to_csv(args.output_dir / "validation_metrics.csv", index=False)
        (args.output_dir / "candidates.json").write_text(json.dumps([asdict(c) for c in candidates], indent=2))
        print(f"Selected {result.selected.to_dict()['candidate']['name']} using validation {args.metric}")
    else:
        selected = FrozenSelection.from_dict(json.loads(args.selected.read_text()))
        result, metrics = evaluate_frozen_configuration(data, selected)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
        metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)
        (args.output_dir / "selected.json").write_text(json.dumps(selected.to_dict(), indent=2))
        pd.DataFrame([{
            "window_id": window.window_id, "fit_before": window.fit_before,
            "train_start": window.train_start, "train_end": window.train_end,
            "last_training_target_exit": window.last_training_target_exit,
            "training_samples": window.training_samples,
        } for window in result.windows]).to_csv(args.output_dir / "test_windows.csv", index=False)
        print(f"Saved {len(result.predictions)} final test predictions")


if __name__ == "__main__":
    main()
