import unittest

import pandas as pd

from data_pipeline.contracts import prepare_daily_observations, require_daily_frequency
from training.target_gen import TargetGenerator


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.prices = pd.DataFrame({
            "date": ["2020-01-03", "2020-01-06", "2020-01-07"],
            "adjOpen": [100.0, 110.0, 121.0],
        }, index=[5, 8, 12])

    def test_utc_copy_and_asset(self):
        result = prepare_daily_observations(self.prices, " spy ")
        self.assertEqual(str(result.date.dt.tz), "UTC")
        self.assertEqual(result.asset.tolist(), ["SPY"] * 3)
        self.assertEqual(result.index.tolist(), [0, 1, 2])
        self.assertNotIn("asset", self.prices)

    def test_invalid_histories_are_rejected(self):
        for data in (self.prices.iloc[::-1],
                     pd.concat([self.prices, self.prices.iloc[:1]]),
                     self.prices.assign(date=[None, "2020-01-06", "2020-01-07"])):
            with self.subTest(data=data), self.assertRaises(ValueError):
                prepare_daily_observations(data, "SPY")
        data = self.prices.assign(asset=["SPY", "AAPL", "SPY"])
        with self.assertRaises(ValueError):
            prepare_daily_observations(data)

    def test_only_daily_frequency_is_supported(self):
        self.assertEqual(require_daily_frequency("daily"), "daily")
        with self.assertRaises(ValueError):
            require_daily_frequency("weekly")

    def test_target_dates_follow_observations_across_weekend(self):
        prices = prepare_daily_observations(self.prices, "SPY")
        result = TargetGenerator(prices, prices[["asset", "date"]]).calculate_next_open_to_open_return()
        self.assertEqual(result.target_entry_date.iloc[0], pd.Timestamp("2020-01-06", tz="UTC"))
        self.assertEqual(result.target_exit_date.iloc[0], pd.Timestamp("2020-01-07", tz="UTC"))
        self.assertAlmostEqual(result.target_return.iloc[0], 0.1)
        self.assertTrue(result.target_return.iloc[-2:].isna().all())

    def test_misaligned_features_are_rejected(self):
        prices = prepare_daily_observations(self.prices, "SPY")
        with self.assertRaises(ValueError):
            TargetGenerator(prices, prices[["asset", "date"]].iloc[1:])


if __name__ == "__main__":
    unittest.main()
