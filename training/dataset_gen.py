"""Aligned numeric arrays and metadata for tabular and sequence models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreparedDataset:
    X: np.ndarray
    y: np.ndarray | None
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]

    def __post_init__(self):
        if self.X.ndim not in (2, 3) or self.X.shape[-1] != len(self.feature_names):
            raise ValueError("Expected samples x features or samples x time x features")
        if len(self.metadata) != len(self.X):
            raise ValueError("Metadata must align with each sample")
        if self.y is not None and (self.y.ndim != 1 or len(self.y) != len(self.X)):
            raise ValueError("Targets must be a vector aligned with each sample")


class DatasetGenerator:
    def build_tabular(self, data):
        if data.X.ndim != 2 or not np.isfinite(data.X).all():
            raise ValueError("Tabular data requires finite 2D inputs")
        return data

    def build_sequences(self, data, sequence_length, history=None):
        """Build sequences ending at each selected observation.

        Supply the full feature history transformed by the SAME fitted
        preprocessor, including rows excluded from training for unavailable
        labels. No history labels are read. With explicit history, every
        endpoint needs sufficient finite context; otherwise raise an error.
        Without history, the initial length-1 endpoints are omitted.
        """
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        explicit_history = history is not None
        history = data if history is None else history
        if data.X.ndim != 2 or history.X.ndim != 2:
            raise ValueError("Sequence construction requires 2D source inputs")
        if data.feature_names != history.feature_names:
            raise ValueError("History and endpoints must use identical feature order")
        for source in (data, history):
            if source.metadata.asset.nunique() != 1:
                raise ValueError("Build sequences separately for each asset")
            if source.metadata.date.duplicated().any() or not source.metadata.date.is_monotonic_increasing:
                raise ValueError("Sequence dates must be ordered and unique")
        if data.metadata.asset.iloc[0] != history.metadata.asset.iloc[0]:
            raise ValueError("History and endpoints must belong to the same asset")
        positions = pd.Index(history.metadata.date).get_indexer(data.metadata.date)
        if (positions < 0).any():
            raise ValueError("Every endpoint must exist in the feature history")
        if not np.allclose(history.X[positions], data.X, equal_nan=True):
            raise ValueError("History and endpoints must contain the same transformed features")
        sequences, selected = [], []
        for endpoint, position in enumerate(positions):
            start = position - sequence_length + 1
            if start < 0:
                if explicit_history:
                    raise ValueError("Insufficient history for a requested prediction date")
                continue
            window = history.X[start:position + 1]
            if not np.isfinite(window).all():
                raise ValueError("Sequence context contains undefined features")
            sequences.append(window)
            selected.append(endpoint)
        if not sequences:
            raise ValueError("No complete sequences available")
        y = data.y[selected].copy() if data.y is not None else None
        return PreparedDataset(
            np.stack(sequences), y,
            data.metadata.iloc[selected].reset_index(drop=True).copy(),
            data.feature_names,
        )
