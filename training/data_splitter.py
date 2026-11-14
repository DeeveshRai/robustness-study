"""Walk-forward selection using daily feature and target availability dates."""

from dataclasses import dataclass

import pandas as pd

from data_pipeline.contracts import prepare_daily_observations
from features.schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class WindowConfig:
    mode: str  # "expanding" or "fixed"
    train_years: int | None = None
    validation_months: int | None = None
    step_months: int | None = None

    def __post_init__(self):
        if self.mode not in ("expanding", "fixed"):
            raise ValueError("mode must be 'expanding' or 'fixed'")
        for name in ("validation_months", "step_months"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.step_months != self.validation_months:
            raise ValueError("step_months must equal validation_months for contiguous, non-overlapping predictions")
        if self.mode == "fixed":
            if type(self.train_years) is not int or self.train_years <= 0:
                raise ValueError("Fixed windows require positive integer train_years")
        elif self.train_years is not None:
            raise ValueError("Expanding windows do not use train_years")


class DataSplitter:
    TRAIN_START = pd.Timestamp("2000-01-01", tz="UTC")
    VALIDATION_START = pd.Timestamp("2018-01-01", tz="UTC")
    TEST_START = pd.Timestamp("2022-01-01", tz="UTC")
    TEST_END = pd.Timestamp("2026-01-01", tz="UTC")

    def _prepare(self, df):
        """Read model-ready rows, including CSV-loaded date strings."""
        result = prepare_daily_observations(df)
        for column in ("target_entry_date", "target_exit_date"):
            result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
        if result[["target_entry_date", "target_exit_date"]].isna().any().any():
            raise ValueError("Use model-ready rows with complete target dates")
        if not ((result["date"] < result["target_entry_date"])
                & (result["target_entry_date"] < result["target_exit_date"])).all():
            raise ValueError("Expected feature date < target entry < target exit")
        return result

    def split(self, df, config: WindowConfig):
        """Yield (train, validation) for tuning during 2018--2021.

        Fit after the last bar before validation_start. Exit dates on or after
        validation_start are unavailable at that fit. Validation outcomes may
        cross internal windows, but must finish before the final test period.
        """
        data = self._prepare(df)
        validation_start = self.VALIDATION_START
        while validation_start < self.TEST_START:
            validation_end = min(
                validation_start + pd.DateOffset(months=config.validation_months),
                self.TEST_START,
            )
            train_start = self.TRAIN_START
            if config.mode == "fixed":
                train_start = max(
                    train_start,
                    validation_start - pd.DateOffset(years=config.train_years),
                )
            train = data.loc[
                (data["date"] >= train_start)
                & (data["date"] < validation_start)
                & (data["target_exit_date"] < validation_start)
            ].copy()
            validation = data.loc[
                (data["date"] >= validation_start)
                & (data["date"] < validation_end)
                & (data["target_exit_date"] < self.TEST_START)
            ].copy()
            if train.empty or validation.empty:
                raise ValueError(f"Empty training or validation window at {validation_start.date()}; check data coverage")
            yield train, validation
            validation_start += pd.DateOffset(months=config.step_months)

    def get_test_set(self, df):
        """Return scored test rows whose complete outcome lies in 2022--2025.

        This selects evaluation rows; it does not fit models or choose configs.
        """
        data = self._prepare(df)
        return data.loc[
            (data["date"] >= self.TEST_START)
            & (data["date"] < self.TEST_END)
            & (data["target_exit_date"] < self.TEST_END)
        ].copy()


class Preprocessor:
    def __init__(self):
        self.columns = FEATURE_COLUMNS.copy()

    def standardise_input(self, train, validation):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        X_train = scaler.fit_transform(train[self.columns])
        X_val = scaler.transform(validation[self.columns])

        y_train = train["target_return"].to_numpy()
        y_val = validation["target_return"].to_numpy()

        return X_train, X_val, y_train, y_val



