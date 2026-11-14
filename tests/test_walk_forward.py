import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from features.schema import FEATURE_COLUMNS
from training.data_splitter import WindowConfig, DataSplitter
from training.model_training import TrainingConfig, ModelTrainer
from training.preprocessing import Preprocessor
from training.walk_forward import WalkForwardRunner


class SequenceMeanModel:
    """Small test adapter that accepts 3D arrays, not a research model."""
    def fit(self, X, y):
        self.value = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2015-01-01", "2022-01-10", tz="UTC")
        n = len(dates)
        self.full = pd.DataFrame({"date": dates, "asset": "SPY"})
        for i, column in enumerate(FEATURE_COLUMNS):
            self.full[column] = np.sin(np.arange(n) / (i + 3))
        self.full["target_return"] = self.full[FEATURE_COLUMNS[0]] * 0.01
        self.full["target_entry_date"] = self.full.date.shift(-1)
        self.full["target_exit_date"] = self.full.date.shift(-2)
        self.ready = self.full.iloc[:-2].copy()
        self.windows = WindowConfig("fixed", 2, 12, 12)

    def test_ridge_outputs_and_fitted_windows_are_aligned(self):
        result = WalkForwardRunner().run(self.ready, self.windows)
        self.assertEqual(len(result.windows), 4)
        expected = self.ready[(self.ready.date >= "2018-01-01") & (self.ready.target_exit_date < "2022-01-01")]
        self.assertEqual(result.predictions.date.tolist(), expected.date.tolist())
        np.testing.assert_allclose(result.predictions.actual_return, expected.target_return)
        self.assertTrue(np.isfinite(result.predictions.predicted_return).all())
        self.assertTrue(result.predictions.baseline_prediction.eq(0).all())
        self.assertFalse(result.predictions.duplicated(["asset", "date"]).any())
        for artifact, (train, validation) in zip(result.windows, DataSplitter().split(self.ready, self.windows)):
            self.assertLess(artifact.last_training_target_exit, artifact.fit_before)
            np.testing.assert_allclose(artifact.preprocessor.scaler.mean_, train[FEATURE_COLUMNS].mean())
            transformed = artifact.preprocessor.transform(validation)
            predictions = result.predictions[result.predictions.window_id.eq(artifact.window_id)]
            np.testing.assert_allclose(artifact.trainer.predict(transformed), predictions.predicted_return)
        self.assertEqual(len({id(w.trainer.model) for w in result.windows}), 4)
        self.assertEqual(len({id(w.preprocessor.scaler) for w in result.windows}), 4)

    def test_future_changes_cannot_affect_first_window_fit_or_predictions(self):
        original = WalkForwardRunner().run(self.ready, self.windows)
        changed = self.ready.copy()
        changed.loc[changed.date >= "2019-01-01", FEATURE_COLUMNS] += 100
        changed.loc[changed.date >= "2019-01-01", "target_return"] += 1
        modified = WalkForwardRunner().run(changed, self.windows)
        pd.testing.assert_frame_equal(original.predictions.query("window_id == 0"), modified.predictions.query("window_id == 0"))

    def test_sequence_adapter_preserves_prediction_dates_and_context(self):
        config = TrainingConfig("sequence", 5)
        result = WalkForwardRunner().run(self.ready, self.windows, config, SequenceMeanModel, self.full)
        tabular = WalkForwardRunner().run(self.ready, self.windows)
        self.assertEqual(result.predictions.date.tolist(), tabular.predictions.date.tolist())
        self.assertTrue(np.isfinite(result.predictions.predicted_return).all())
        with self.assertRaises(ValueError):
            WalkForwardRunner().run(self.ready, self.windows, config)

    def test_invalid_config_and_reused_model_fail(self):
        for args in (("bad", None), ("tabular", 5), ("sequence", None), ("sequence", 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TrainingConfig(*args)
        reused = Ridge()
        with self.assertRaises(ValueError):
            WalkForwardRunner().run(self.ready, self.windows, model_factory=lambda: reused)
        data = Preprocessor().fit(self.ready.iloc[:10]).transform(self.ready.iloc[:10])
        trainer = ModelTrainer(Ridge(), TrainingConfig())
        with self.assertRaises(ValueError):
            trainer.predict(data)
        trainer.fit(data)
        trainer.model.predict = lambda X: np.full((len(X), 1), 0.0)
        with self.assertRaises(ValueError):
            trainer.predict(data)


if __name__ == "__main__":
    unittest.main()
