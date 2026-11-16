import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from features.schema import FEATURE_COLUMNS
from training.data_splitter import WindowConfig
from training.evaluation import regression_metrics, evaluate_predictions
from training.experiments import Candidate, FrozenSelection, select_configuration, evaluate_frozen_configuration
from training.models import MODEL_TYPES, create_model
from training.run_experiment import main
from training.run_training import main as single_model_main


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2015-01-01", "2025-12-31", tz="UTC")
        self.data = pd.DataFrame({"date": dates, "asset": "SPY"})
        for i, column in enumerate(FEATURE_COLUMNS):
            self.data[column] = np.sin(np.arange(len(dates)) / (i + 3))
        self.data["target_return"] = self.data[FEATURE_COLUMNS[0]] * 0.01
        self.data["target_entry_date"] = self.data.date.shift(-1)
        self.data["target_exit_date"] = self.data.date.shift(-2)
        self.data = self.data.iloc[:-2].copy()
        self.window = WindowConfig("fixed", 2, 12, 12)
        self.candidates = [Candidate("ridge", "ridge", {"alpha": 1.0}, self.window),
                           Candidate("linear", "linear", {}, self.window)]

    def test_metrics_and_zero_baseline_convention(self):
        metrics = regression_metrics([1, -1, 0], [0.5, -0.5, 0])
        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1 / 6))
        self.assertEqual(metrics["directional_accuracy"], 1)
        self.assertEqual(regression_metrics([1, -1, 0], [0, 0, 0])["directional_accuracy"], 1 / 3)
        with self.assertRaises(ValueError):
            regression_metrics([], [])

    def test_test_data_cannot_influence_selection(self):
        first = select_configuration(self.data, self.candidates)
        changed = self.data.copy()
        changed.loc[changed.date >= "2022-01-01", FEATURE_COLUMNS + ["target_return"]] += 1000
        second = select_configuration(changed, self.candidates)
        self.assertEqual(first.selected, second.selected)
        pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
        self.assertEqual(first.selected.to_dict()["candidate"]["name"], "linear")
        # Parameter edits after selection cannot alter the frozen snapshot.
        self.candidates[1].parameters["fit_intercept"] = False
        self.assertEqual(first.selected.to_dict()["candidate"]["parameters"], {})

    def test_ties_are_deterministic_and_metric_can_be_chosen(self):
        a = Candidate("a", "ridge", {"alpha": 1.0}, self.window)
        z = Candidate("z", "ridge", {"alpha": 1.0}, self.window)
        result = select_configuration(self.data, [z, a], "rmse")
        self.assertEqual(result.selected.to_dict()["candidate"]["name"], "a")
        with self.assertRaises(ValueError):
            select_configuration(self.data, [a, a])

    def test_frozen_test_retrains_causally_and_needs_no_2026_prices(self):
        selected = select_configuration(self.data, self.candidates).selected
        run, report = evaluate_frozen_configuration(self.data, selected)
        self.assertEqual(len(run.windows), 4)
        self.assertTrue(run.predictions.period.eq("test").all())
        self.assertTrue((run.predictions.date >= "2022-01-01").all())
        self.assertTrue((run.predictions.target_exit_date < "2026-01-01").all())
        self.assertEqual(run.predictions.target_exit_date.max(), pd.Timestamp("2025-12-31", tz="UTC"))
        self.assertTrue((run.predictions.date < "2026-01-01").all())
        for window in run.windows:
            self.assertLess(window.last_training_target_exit, window.fit_before)
        self.assertGreater(run.windows[1].last_training_target_exit, pd.Timestamp("2022-01-01", tz="UTC"))
        self.assertEqual(len(report), 10)  # model + baseline, overall + 4 windows
        self.assertEqual(report.iloc[0].n_samples, len(run.predictions))
        # Altering later test features/outcomes leaves the first test window unchanged.
        changed = self.data.copy()
        changed.loc[changed.date >= "2023-01-01", FEATURE_COLUMNS + ["target_return"]] += 100
        again, _ = evaluate_frozen_configuration(changed, selected)
        pd.testing.assert_frame_equal(run.predictions.query("window_id == 0"), again.predictions.query("window_id == 0"))

    def test_builtin_models_fit_and_predict(self):
        X = self.data[FEATURE_COLUMNS].iloc[:40].to_numpy()
        y = self.data.target_return.iloc[:40].to_numpy()
        for name in MODEL_TYPES:
            params = {"n_estimators": 3} if name == "random_forest" else {}
            model = create_model(name, params)
            model.fit(X, y)
            self.assertEqual(model.predict(X[:3]).shape, (3,))

    def test_select_and_test_cli_write_separate_reports(self):
        root = Path.cwd()
        with tempfile.TemporaryDirectory(dir=root) as temp:
            directory = Path(temp)
            data_path = directory / "ready.csv"
            candidate_path = directory / "candidates.json"
            self.data.to_csv(data_path, index=False)
            candidate_path.write_text(json.dumps([{
                "name": "ridge", "model": "ridge", "parameters": {"alpha": 1},
                "window": {"mode": "fixed", "train_years": 2, "validation_months": 12, "step_months": 12},
            }]))
            validation = directory / "validation"
            with patch("sys.argv", ["run_experiment", "select", str(data_path), str(candidate_path), str(validation)]):
                main()
            frozen_bytes = (validation / "selected.json").read_bytes()
            test = directory / "test"
            with patch("sys.argv", ["run_experiment", "test", str(data_path), str(validation / "selected.json"), str(test)]):
                main()
            self.assertEqual((test / "selected.json").read_bytes(), frozen_bytes)
            self.assertTrue((test / "test_windows.csv").exists())
            self.assertTrue((test / "test_metrics.csv").exists())
            self.assertFalse((validation / "test_predictions.csv").exists())
            # Parameter files avoid PowerShell's inline JSON quoting issues.
            parameters_path = directory / "parameters.json"
            parameters_path.write_text('{"alpha": 2.0}')
            single_output = directory / "single" / "ridge.csv"
            with patch("sys.argv", ["run_training", str(data_path), str(single_output),
                                    "--mode", "fixed", "--train-years", "2",
                                    "--parameters-file", str(parameters_path)]):
                single_model_main()
            self.assertTrue(single_output.exists())
            self.assertTrue(single_output.with_name("ridge_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
