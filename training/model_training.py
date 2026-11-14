from dataclasses import dataclass
from typing import Any

import numpy as np

from training.dataset_gen import PreparedDataset

@dataclass(frozen=True)
class TrainingConfig:
    dataset_dimensions: str = "tabular"
    sequence_length: int | None = None

    def __post_init__(self):
        if self.dataset_dimensions not in ("tabular", "sequence"):
            raise ValueError("dataset_dimensions must be 'tabular' or 'sequence'")
        if self.dataset_dimensions == "sequence":
            if type(self.sequence_length) is not int or self.sequence_length <= 0:
                raise ValueError("Sequence models require a positive integer sequence_length")
        elif self.sequence_length is not None:
            raise ValueError("Tabular models do not use sequence_length")


class ModelTrainer:
    """Fit a supplied model exposing fit(X, y) and predict(X)."""

    def __init__(self, model: Any, config: TrainingConfig):
        self.model = model
        self.config = config
        self.feature_names = None
        self.asset = None

    def _validate(self, dataset: PreparedDataset, require_target=False):
        expected = 2 if self.config.dataset_dimensions == "tabular" else 3
        if dataset.X.ndim != expected or len(dataset.X) == 0:
            raise ValueError(f"Expected non-empty {expected}D model inputs")
        if expected == 3 and dataset.X.shape[1] != self.config.sequence_length:
            raise ValueError("Sequence length does not match training configuration")
        if not np.isfinite(dataset.X).all():
            raise ValueError("Model inputs must be finite")
        if require_target and (dataset.y is None or not np.isfinite(dataset.y).all()):
            raise ValueError("Fitting requires finite targets")
        if dataset.metadata.asset.nunique() != 1:
            raise ValueError("Train and predict one asset at a time")

    def fit(self, dataset: PreparedDataset):
        self._validate(dataset, require_target=True)
        self.model.fit(dataset.X, dataset.y)
        self.feature_names = dataset.feature_names
        self.asset = dataset.metadata.asset.iloc[0]
        return self

    def predict(self, dataset: PreparedDataset):
        if self.feature_names is None:
            raise ValueError("Fit the model before predicting")
        self._validate(dataset)
        if dataset.feature_names != self.feature_names or dataset.metadata.asset.iloc[0] != self.asset:
            raise ValueError("Prediction feature order and asset must match training")
        predictions = np.asarray(self.model.predict(dataset.X), dtype=float)
        if predictions.shape != (len(dataset.X),) or not np.isfinite(predictions).all():
            raise ValueError("Model must return one finite return per observation")
        return predictions

