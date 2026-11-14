import unittest

import pandas as pd

from training.data_splitter import DataSplitter, WindowConfig


class DataSplitterTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("1999-01-01", "2026-01-07", tz="UTC")
        self.data = pd.DataFrame({"date": dates, "asset": "SPY"})
        self.data["target_entry_date"] = self.data.date.shift(-1)
        self.data["target_exit_date"] = self.data.date.shift(-2)
        self.data = self.data.iloc[:-2].copy()
        self.splitter = DataSplitter()

    def test_training_labels_are_known_before_each_cutoff(self):
        windows = list(self.splitter.split(self.data, WindowConfig("expanding", validation_months=12, step_months=12)))
        self.assertEqual(len(windows), 4)
        for year, (train, validation) in zip(range(2018, 2022), windows):
            cutoff = pd.Timestamp(f"{year}-01-01", tz="UTC")
            self.assertTrue((train.target_exit_date < cutoff).all())
            self.assertTrue((train.date >= self.splitter.TRAIN_START).all())
            self.assertTrue((validation.date >= cutoff).all())
            self.assertTrue((validation.target_exit_date < self.splitter.TEST_START).all())
            crossing = self.data[(self.data.date < cutoff) & (self.data.target_exit_date >= cutoff)]
            self.assertEqual(len(crossing), 2)
            self.assertTrue(set(crossing.date).isdisjoint(set(train.date)))

    def test_fixed_window_and_csv_dates(self):
        strings = self.data.astype({"date": str, "target_entry_date": str, "target_exit_date": str})
        train, validation = next(self.splitter.split(strings, WindowConfig("fixed", 3, 12, 12)))
        self.assertEqual(str(train.date.dt.tz), "UTC")
        self.assertEqual(train.date.min(), pd.Timestamp("2015-01-01", tz="UTC"))
        self.assertTrue((train.target_exit_date < validation.date.min()).all())

    def test_predictions_are_unique_and_final_partial_window_is_retained(self):
        windows = list(self.splitter.split(self.data, WindowConfig("expanding", validation_months=7, step_months=7)))
        self.assertEqual(len(windows), 7)
        predictions = pd.concat([validation for _, validation in windows])
        self.assertFalse(predictions.date.duplicated().any())
        expected = self.data[(self.data.date >= self.splitter.VALIDATION_START)
                             & (self.data.target_exit_date < self.splitter.TEST_START)]
        self.assertEqual(predictions.date.tolist(), expected.date.tolist())

    def test_label_on_test_boundary_is_excluded_from_tuning(self):
        # An exit exactly on the boundary must also fail the strict cutoff.
        last_tuning_row = self.data.index[self.data.date.eq(pd.Timestamp("2021-12-30", tz="UTC"))][0]
        self.data.loc[last_tuning_row, "target_exit_date"] = self.splitter.TEST_START
        boundary_row = self.data.loc[[last_tuning_row]]
        self.assertFalse(boundary_row.empty)
        validation = pd.concat([v for _, v in self.splitter.split(self.data, WindowConfig("expanding", validation_months=12, step_months=12))])
        self.assertTrue(set(boundary_row.date).isdisjoint(set(validation.date)))
        test = self.splitter.get_test_set(self.data)
        self.assertTrue((test.date >= self.splitter.TEST_START).all())
        self.assertTrue((test.target_exit_date < self.splitter.TEST_END).all())

    def test_invalid_configs_are_rejected(self):
        cases = [dict(mode="unknown"), dict(validation_months=0), dict(step_months=0),
                 dict(step_months=6), dict(step_months=24), dict(train_years=3),
                 dict(mode="fixed"), dict(mode="fixed", train_years=-1),
                 dict(validation_months=True), dict(step_months=1.5)]
        for changes in cases:
            values = dict(mode="expanding", validation_months=12, step_months=12)
            values.update(changes)
            with self.subTest(values=values), self.assertRaises(ValueError):
                WindowConfig(**values)

    def test_empty_windows_and_missing_labels_fail_explicitly(self):
        config = WindowConfig("expanding", validation_months=12, step_months=12)
        with self.assertRaises(ValueError):
            next(self.splitter.split(self.data[self.data.date.dt.year < 2018], config))
        self.data.loc[0, "target_exit_date"] = pd.NaT
        with self.assertRaises(ValueError):
            next(self.splitter.split(self.data, config))


if __name__ == "__main__":
    unittest.main()
