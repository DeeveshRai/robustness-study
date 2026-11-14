import unittest

import numpy as np
import pandas as pd
import pandas_ta as ta

from data_pipeline.contracts import prepare_daily_observations
from features.featureEngineer import FeatureEngineer
from features.schema import FEATURE_COLUMNS
from training.row_eligibility import select_model_rows
from training.target_gen import TargetGenerator


class FeatureEligibilityTests(unittest.TestCase):
    def history(self, length=260):
        price = 100 + np.arange(length) * 0.1 + np.sin(np.arange(length))
        return prepare_daily_observations(pd.DataFrame({
            "date": pd.bdate_range("1999-01-01", periods=length),
            "adjClose": price, "adjHigh": price + 2, "adjLow": price - 2,
            "adjOpen": price - 0.5, "adjVolume": np.full(length, 1000),
        }), "SPY")

    def generate(self, data):
        return TargetGenerator(data, FeatureEngineer(data).create_feature_dataframe()).calculate_next_open_to_open_return()

    def test_atr_is_dimensionless_and_price_scale_invariant(self):
        data = self.history()
        features = self.generate(data)
        expected = ta.atr(data.adjHigh, data.adjLow, data.adjClose, length=14) / data.adjClose
        np.testing.assert_allclose(features.atr14_norm, expected, equal_nan=True)
        scaled = data.copy()
        for column in ("adjClose", "adjHigh", "adjLow", "adjOpen"):
            scaled[column] *= 10
        np.testing.assert_allclose(features.atr14_norm, self.generate(scaled).atr14_norm, equal_nan=True)

    def test_warmup_and_tail_are_excluded_without_shifting_targets(self):
        full = self.generate(self.history())
        eligible, excluded = select_model_rows(full)
        self.assertEqual(eligible.index.tolist(), list(range(199, 258)))
        self.assertEqual(len(excluded), 201)
        pd.testing.assert_frame_equal(eligible, full.loc[eligible.index])
        self.assertTrue(excluded.loc[:198, "exclusion_reason"].str.contains("warmup").all())
        self.assertTrue(excluded.loc[258:, "exclusion_reason"].str.contains("incomplete_target").all())
        self.assertTrue(np.isfinite(eligible[FEATURE_COLUMNS + ["target_return"]]).all().all())
        # Features use earlier history, even if the model's date window starts later.
        self.assertEqual(len(eligible[eligible.date >= full.date.iloc[220]]), 38)

    def test_infinite_values_are_excluded_and_schema_errors_rejected(self):
        full = self.generate(self.history())
        full.loc[210, "relative_volume20"] = np.inf
        eligible, excluded = select_model_rows(full)
        self.assertNotIn(210, eligible.index)
        self.assertEqual(excluded.loc[210, "exclusion_reason"], "invalid_features")
        with self.assertRaises(ValueError):
            select_model_rows(full.drop(columns=[FEATURE_COLUMNS[0]]))
        full[FEATURE_COLUMNS[0]] = "bad value"
        with self.assertRaises(ValueError):
            select_model_rows(full)

    def test_short_history_produces_no_eligible_rows(self):
        eligible, excluded = select_model_rows(self.generate(self.history(5)))
        self.assertTrue(eligible.empty)
        self.assertEqual(len(excluded), 5)

    def test_later_prices_cannot_change_earlier_features(self):
        data = self.history()
        before = self.generate(data)
        data.loc[240:, ["adjOpen", "adjClose", "adjHigh", "adjLow"]] *= 2
        after = self.generate(data)
        pd.testing.assert_frame_equal(before.loc[:239, FEATURE_COLUMNS], after.loc[:239, FEATURE_COLUMNS])


if __name__ == "__main__":
    unittest.main()
