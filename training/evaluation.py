"""Predictive metrics only; no execution or trading-cost assumptions."""

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or len(actual) == 0:
        raise ValueError("Metrics require equally sized non-empty vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted returns")
    error = predicted - actual
    return {
        "n_samples": len(actual),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "directional_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
    }


def evaluate_predictions(predictions):
    """Overall metrics are pooled across dates, not averages of fold metrics."""
    rows = []
    groups = [("overall", "all", predictions)]
    groups += [("window", str(window), group) for window, group in predictions.groupby("window_id", sort=True)]
    for scope, window, group in groups:
        for name, column in (("model", "predicted_return"), ("zero_return", "baseline_prediction")):
            rows.append({"scope": scope, "window_id": window, "predictor": name,
                         **regression_metrics(group.actual_return, group[column])})
    return pd.DataFrame(rows)
