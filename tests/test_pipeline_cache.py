import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_pipeline.caching import Caching
from main import run_pipeline


class PipelineCacheTests(unittest.TestCase):
    def test_fresh_and_cached_runs_match_with_global_cleaning(self):
        dates = pd.bdate_range("2015-01-01", periods=260, tz="UTC")
        price = 100 + np.arange(260) * 0.1 + np.sin(np.arange(260))
        data = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%dT00:00:00Z"),
            "close": price, "high": price + 2, "low": price - 2,
            "open": price - 0.5, "volume": 1000,
            "adjClose": price, "adjHigh": price + 2, "adjLow": price - 2,
            "adjOpen": price - 0.5, "adjVolume": 1000,
            "divCash": 0, "splitFactor": 1,
        })
        # Reverse all chunks and repeat a date across chunk boundaries.
        raw = pd.concat([data.iloc[::-1], data.iloc[[-1]]], ignore_index=True)
        # Deliberately reorder columns: values must still keep their meanings.
        raw = raw[raw.columns[::-1]]
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory(dir=original_directory) as directory:
            try:
                os.chdir(directory)

                def fetch_to_cache():
                    Caching("SPY", "2015-01-01", "2015-12-31").cache(raw.to_dict("records"))
                    # Also exercise normalisation with a differently ordered CSV.
                    raw.to_csv("data/raw/SPY_2015-01-01_2015-12-31.csv", index=False)

                with patch("main.Fetcher.fetch", side_effect=fetch_to_cache) as fetch:
                    first = run_pipeline("SPY", "2015-01-01", "2015-12-31", "daily")
                    snapshots = {
                        folder: Path(f"data/{folder}/SPY_2015-01-01_2015-12-31.csv").read_bytes()
                        for folder in ("raw", "normalised", "cleaned", "features", "model_ready", "excluded")
                    }
                    second = run_pipeline("SPY", "2015-01-01", "2015-12-31", "daily")
                    fetch.assert_called_once_with()

                pd.testing.assert_frame_equal(first, second)
                self.assertEqual(len(second), 260)
                self.assertTrue(second.date.is_monotonic_increasing)
                self.assertFalse(second.date.duplicated().any())
                self.assertAlmostEqual(second.target_return.iloc[0],
                                       (price[2] - price[1]) / (price[1] - 0.5))
                for folder, contents in snapshots.items():
                    self.assertEqual(
                        Path(f"data/{folder}/SPY_2015-01-01_2015-12-31.csv").read_bytes(),
                        contents,
                    )
                saved = pd.read_csv("data/features/SPY_2015-01-01_2015-12-31.csv")
                self.assertEqual(len(saved), 260)
                self.assertEqual(len(pd.read_csv("data/model_ready/SPY_2015-01-01_2015-12-31.csv")), 59)
                self.assertEqual(len(pd.read_csv("data/excluded/SPY_2015-01-01_2015-12-31.csv")), 201)
            finally:
                os.chdir(original_directory)

    def test_replacing_raw_cache_retains_header(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory(dir=original_directory) as directory:
            try:
                os.chdir(directory)
                cache = Caching("SPY", "start", "end")
                cache.cache([{"date": "2020-01-01", "open": 100}])
                cache.cache([{"date": "2020-01-02", "open": 101}])
                saved = pd.read_csv("data/raw/SPY_start_end.csv")
                self.assertEqual(saved.date.tolist(), ["2020-01-02"])
                self.assertEqual(saved.open.tolist(), [101])
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
