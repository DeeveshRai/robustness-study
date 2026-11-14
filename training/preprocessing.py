"""One fitted scaler per historical training window."""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from sklearn.preprocessing import StandardScaler

from data_pipeline.contracts import prepare_daily_observations
from features.schema import FEATURE_COLUMNS
from training.dataset_gen import PreparedDataset


class Preprocessor:
    def __init__(self):
        self.columns = FEATURE_COLUMNS.copy()
        self.scaler = None
        self.asset = None

    def _read(self, data, require_target):
        rows = prepare_daily_observations(data)
        for column in self.columns:
            if not is_numeric_dtype(rows[column]) or is_bool_dtype(rows[column]):
                raise ValueError(f"Expected numeric feature: {column}")
        values = rows[self.columns].to_numpy(dtype=float, na_value=np.nan)
        metadata = rows[["asset", "date"]].copy()
        for column in ("target_entry_date", "target_exit_date"):
            if column in rows:
                metadata[column] = pd.to_datetime(rows[column], utc=True, errors="raise")
        y = None
        if require_target:
            if not is_numeric_dtype(rows["target_return"]) or is_bool_dtype(rows["target_return"]):
                raise ValueError("Expected numeric target_return")
            y = rows["target_return"].to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(values).all() or not np.isfinite(y).all():
                raise ValueError("Supervised rows must have finite features and targets")
            if not {"target_entry_date", "target_exit_date"}.issubset(metadata):
                raise ValueError("Supervised rows require target dates")
            if not ((metadata.date < metadata.target_entry_date)
                    & (metadata.target_entry_date < metadata.target_exit_date)).all():
                raise ValueError("Expected feature date < target entry < target exit")
        return values, y, metadata

    def fit(self, train):
        values, _, metadata = self._read(train, require_target=True)
        scaler = StandardScaler().fit(values)
        self.scaler = scaler
        self.asset = metadata.asset.iloc[0]
        return self

    def transform(self, data, require_target=True):
        """Transform supervised endpoints, or feature-only history/context.

        Context may include undefined indicators. They stay NaN, and any
        requested sequence that touches them is rejected by the builder.
        """
        if self.scaler is None:
            raise ValueError("Fit on the training window before transforming data")
        values, y, metadata = self._read(data, require_target)
        if metadata.asset.iloc[0] != self.asset:
            raise ValueError("Transform data must match the fitted asset")
        finite_rows = np.isfinite(values).all(axis=1)
        scaled = np.full(values.shape, np.nan)
        if finite_rows.any():
            scaled[finite_rows] = self.scaler.transform(values[finite_rows])
        return PreparedDataset(scaled, y, metadata, tuple(self.columns))

    def standardise_input(self, train, validation):
        """Return aligned containers; the fitted scaler remains on this instance."""
        self.fit(train)
        return self.transform(train), self.transform(validation)
