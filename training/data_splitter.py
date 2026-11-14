'''
- **Training:** 3 January 2000 2000-01-03 – 31 December 2017 2017-12-31
- **Validation / tuning:** 1 January 2018 – 31 December 2021
- **Final historical test:** 1 January 2022 – 31 December 2025
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd
from dataclasses import dataclass

@dataclass
class WindowConfig:
    mode: str  # "expanding" or "fixed"
    train_years: int | None = None
    validation_months: int | None = None
    step_months: int | None = None

class DataSplitter:
    def __init__(self):
        pass


    def split(self, df, config: WindowConfig):
        initial_train_start = pd.Timestamp("2000-01-01")
        validation_start = pd.Timestamp("2018-01-01")
        test_boundary = pd.Timestamp("2022-01-01")

        while True:
            validation_end = validation_start + pd.DateOffset(
                months=config.validation_months
            )

            if validation_end > test_boundary:
                break

            train_end = validation_start

            if config.mode == "fixed":
                train_start = train_end - pd.DateOffset(
                    years=config.train_years
                )

            elif config.mode == "expanding":
                train_start = initial_train_start

            else:
                raise ValueError(f"Unknown mode: {config.mode}")

            train = df[
                (df["date"] >= train_start) &
                (df["date"] < train_end)
            ]

            validation = df[
                (df["date"] >= validation_start) &
                (df["date"] < validation_end)
            ]

            yield train, validation

            validation_start += pd.DateOffset(
                months=config.step_months
            )

    def get_test_set(self, df):
        test_start = pd.Timestamp("2022-01-01")
        test_end = pd.Timestamp("2026-01-01")

        test = df[
                    (df["date"] >= test_start) &
                    (df["date"] < test_end)
                ]

        return test

class Preprocessor:
    def __init__(self):
        self.columns =[
                                'returns_1d', 
                                'returns_5d',
                                'returns_20d',
                                'returns_60d',
                                'returns_lag1',
                                'returns_lag2',
                                'sma_20_vs_prices',
                                'sma_50_vs_prices',
                                'sma_200_vs_prices',
                                'macd_norm',
                                'rsi_14',
                                'volatility_5d',
                                'volatility_20d',
                                'atr14_norm',
                                'bollinger_width20',
                                'relative_volume20',
                                'volume_change_1d',
                                'high_low_range',
                                'open_close_return',
                                'overnight_gap',
                                ]

    def standardise_input(self, train, validation):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(train[self.columns])
        X_val = scaler.transform(validation[self.columns])

        y_train = train["target_return"].to_numpy()
        y_val = validation["target_return"].to_numpy()

        return X_train, X_val, y_train, y_val



