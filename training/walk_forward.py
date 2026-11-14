"""Run one configuration through the 2018--2021 tuning windows."""

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from training.data_splitter import DataSplitter, WindowConfig
from training.dataset_gen import DatasetGenerator
from training.model_training import ModelTrainer, TrainingConfig
from training.preprocessing import Preprocessor


@dataclass
class FittedWindow:
    window_id: int
    fit_before: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    last_training_target_exit: pd.Timestamp
    training_samples: int
    preprocessor: Preprocessor
    trainer: ModelTrainer


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    windows: list[FittedWindow]
    window_config: WindowConfig
    training_config: TrainingConfig


class WalkForwardRunner:
    def run(self, model_ready, window_config: WindowConfig,
            training_config: TrainingConfig | None = None,
            model_factory: Callable[[], Any] = Ridge, full_features=None):
        """Fit a fresh scaler and model per window and return dated predictions.

        The factory must return a new, unfitted model on every call. Sequence
        models require the complete feature history for prediction context.
        This runner does not select configurations or access the final test set.
        """
        config = training_config or TrainingConfig()
        if config.dataset_dimensions == "sequence" and full_features is None:
            raise ValueError("Sequence validation requires full_features history")
        splitter, builder = DataSplitter(), DatasetGenerator()
        records, artifacts = [], []
        for window_id, (train, validation) in enumerate(splitter.split(model_ready, window_config)):
            fit_before = splitter.VALIDATION_START + pd.DateOffset(
                months=window_id * window_config.step_months
            )
            preprocessor = Preprocessor()
            training_data, prediction_data = preprocessor.standardise_input(train, validation)
            if config.dataset_dimensions == "sequence":
                # Training starts within its selected window; context for
                # validation includes observed bars excluded by label purging.
                history = full_features.copy()
                dates = pd.to_datetime(history["date"], utc=True, errors="raise")
                history = history.loc[dates <= prediction_data.metadata.date.max()]
                context = preprocessor.transform(history, require_target=False)
                # Preserve all observed bars inside the training window too;
                # label filtering must not silently compress sequence time.
                training_history = history.loc[dates.loc[history.index] >= training_data.metadata.date.min()]
                training_context = preprocessor.transform(training_history, require_target=False)
                positions = pd.Index(training_context.metadata.date).get_indexer(training_data.metadata.date)
                if (positions < 0).any():
                    raise ValueError("Training endpoints are missing from full_features")
                endpoints = train.iloc[np.flatnonzero(positions >= config.sequence_length - 1)]
                if endpoints.empty:
                    raise ValueError("Training window is too short for sequence_length")
                training_data = builder.build_sequences(
                    preprocessor.transform(endpoints), config.sequence_length, training_context
                )
                prediction_data = builder.build_sequences(prediction_data, config.sequence_length, context)
            else:
                training_data = builder.build_tabular(training_data)
                prediction_data = builder.build_tabular(prediction_data)

            model = model_factory()
            if any(model is fitted.trainer.model for fitted in artifacts):
                raise ValueError("model_factory must return a fresh model for each window")
            trainer = ModelTrainer(model, config).fit(training_data)
            output = prediction_data.metadata.copy()
            output["predicted_return"] = trainer.predict(prediction_data)
            output["actual_return"] = prediction_data.y
            output["baseline_prediction"] = np.zeros(len(output))
            output["window_id"] = window_id
            output["fit_before"] = fit_before
            records.append(output)
            artifacts.append(FittedWindow(
                window_id, fit_before, training_data.metadata.date.min(),
                training_data.metadata.date.max(),
                training_data.metadata.target_exit_date.max(),
                len(training_data.X), preprocessor, trainer,
            ))
        predictions = pd.concat(records, ignore_index=True)
        if predictions.duplicated(["asset", "date"]).any():
            raise ValueError("Walk-forward predictions must be unique per asset/date")
        return WalkForwardResult(predictions, artifacts, window_config, config)
