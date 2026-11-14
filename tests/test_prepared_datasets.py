import unittest

import numpy as np
import pandas as pd

from features.schema import FEATURE_COLUMNS
from training.preprocessing import Preprocessor
from training.dataset_gen import DatasetGenerator, PreparedDataset


class PreparedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({
            "date": pd.bdate_range("2017-12-18", periods=20, tz="UTC"),
            "asset": "SPY", "target_return": np.arange(20) / 100,
        })
        self.rows["target_entry_date"] = self.rows.date + pd.Timedelta(days=3)
        self.rows["target_exit_date"] = self.rows.date + pd.Timedelta(days=4)
        for i, column in enumerate(FEATURE_COLUMNS):
            self.rows[column] = np.arange(20, dtype=float) + i
        self.preprocessor = Preprocessor()
        self.builder = DatasetGenerator()

    def test_scaler_only_fits_training_and_retains_metadata(self):
        train, validation = self.preprocessor.standardise_input(self.rows.iloc[:8], self.rows.iloc[10:])
        scaler = self.preprocessor.scaler
        expected_mean = self.rows.iloc[:8][FEATURE_COLUMNS].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected_mean)
        np.testing.assert_allclose(train.X.mean(axis=0), 0, atol=1e-12)
        changed = self.rows.iloc[10:].copy()
        changed[FEATURE_COLUMNS] += 1000
        transformed = self.preprocessor.transform(changed)
        self.assertIs(self.preprocessor.scaler, scaler)
        np.testing.assert_allclose(scaler.mean_, expected_mean)
        self.assertGreater(transformed.X[0, 0], validation.X[0, 0])
        self.assertEqual(validation.metadata.date.tolist(), self.rows.date.iloc[10:].tolist())
        np.testing.assert_array_equal(validation.y, self.rows.target_return.iloc[10:])
        self.assertEqual(self.builder.build_tabular(validation).X.shape, (10, 20))

    def test_context_recovers_first_prediction_and_ignores_history_labels(self):
        self.preprocessor.fit(self.rows.iloc[:8])
        endpoints = self.preprocessor.transform(self.rows.iloc[10:])
        # Rows 8 and 9 were excluded from fitting but remain valid feature context.
        raw_history = self.rows.drop(columns=["target_return", "target_entry_date", "target_exit_date"])
        history = self.preprocessor.transform(raw_history, require_target=False)
        sequences = self.builder.build_sequences(endpoints, 4, history)
        self.assertEqual(sequences.X.shape, (10, 4, 20))
        np.testing.assert_array_equal(sequences.X[0], history.X[7:11])
        np.testing.assert_array_equal(sequences.y, endpoints.y)
        pd.testing.assert_frame_equal(sequences.metadata, endpoints.metadata)
        # Later feature rows cannot affect an earlier sequence.
        history.X[11:] += 1000
        first_only = self.preprocessor.transform(self.rows.iloc[10:11])
        again = self.builder.build_sequences(first_only, 4, history)
        np.testing.assert_array_equal(again.X[0], sequences.X[0])

    def test_training_sequences_align_to_final_observation(self):
        self.preprocessor.fit(self.rows.iloc[:8])
        train = self.preprocessor.transform(self.rows.iloc[:8])
        sequence = self.builder.build_sequences(train, 3)
        self.assertEqual(sequence.X.shape, (6, 3, 20))
        np.testing.assert_array_equal(sequence.y, train.y[2:])
        self.assertEqual(sequence.metadata.date.tolist(), train.metadata.date.iloc[2:].tolist())

    def test_bad_shapes_and_invalid_sequence_context_fail(self):
        with self.assertRaises(ValueError):
            self.preprocessor.transform(self.rows)
        self.preprocessor.fit(self.rows.iloc[:8])
        data = self.preprocessor.transform(self.rows)
        for length in (0, -1, True, 2.5, 21):
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.builder.build_sequences(data, length)
        with self.assertRaises(ValueError):
            self.builder.build_sequences(data, 3, history=data)
        with self.assertRaises(ValueError):
            PreparedDataset(data.X, data.y[:-1], data.metadata, data.feature_names)
        context = self.rows.copy()
        context.loc[8, FEATURE_COLUMNS[0]] = np.nan
        context = self.preprocessor.transform(context, require_target=False)
        endpoints = self.preprocessor.transform(self.rows.iloc[10:])
        with self.assertRaises(ValueError):
            self.builder.build_sequences(endpoints, 4, context)
        foreign = self.rows.assign(asset="AAPL")
        with self.assertRaises(ValueError):
            self.preprocessor.transform(foreign)


if __name__ == "__main__":
    unittest.main()
