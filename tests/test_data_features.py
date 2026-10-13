"""Data quality and causal feature checks, independent of network access."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecasting.data import download_prices, load_prices, save_prices, synthetic_prices, validate_prices
from forecasting.features import build_features


class DataFeatureTests(unittest.TestCase):
    def test_synthetic_is_reproducible_and_explicit(self):
        data = synthetic_prices(120, seed=19)
        pd.testing.assert_frame_equal(data, synthetic_prices(120, seed=19))
        self.assertEqual(data.attrs["data_source"], "synthetic")
        self.assertTrue((data.index.dayofweek < 5).all())
        self.assertFalse(data.equals(synthetic_prices(120, seed=20)))

    def test_csv_round_trip_and_missing_stock_values_rejected(self):
        data = synthetic_prices(80)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "prices.csv"
            save_prices(data, path)
            loaded = load_prices(path)
        pd.testing.assert_frame_equal(data, loaded, check_freq=False)
        bad_data = data.copy()
        bad_data.iloc[20, bad_data.columns.get_loc("close")] = np.nan
        with self.assertRaisesRegex(ValueError, "do not fill unknown bars"):
            validate_prices(bad_data)

    def test_prefix_invariance_and_future_price_perturbation(self):
        data = synthetic_prices(160)
        features = build_features(data)
        self.assertGreaterEqual(features.shape[1], 15)
        pd.testing.assert_index_equal(features.index, data.index)
        pd.testing.assert_frame_equal(features.iloc[:100], build_features(data.iloc[:100]))
        perturbed = data.copy()
        perturbed.iloc[100:, :4] *= 3.0
        perturbed.iloc[100:, 4] *= 4.0
        pd.testing.assert_frame_equal(features.iloc[:100], build_features(perturbed).iloc[:100])
        self.assertTrue(features.iloc[0].isna().any())
        self.assertGreater(len(features.dropna()), 0)

    def test_vix_is_lagged_and_unknown_values_not_backfilled(self):
        data = synthetic_prices(100)
        data.loc[data.index[70], "vix"] = np.nan
        features = build_features(data)
        self.assertTrue(pd.isna(features.loc[data.index[71], "vix_lag_1"]))
        self.assertEqual(features.loc[data.index[70], "vix_lag_1"], data.loc[data.index[69], "vix"])
        changed = data.copy()
        changed.loc[data.index[80], "vix"] = 100.0
        changed_features = build_features(changed)
        pd.testing.assert_frame_equal(features.iloc[:81], changed_features.iloc[:81])
        self.assertEqual(changed_features.loc[data.index[81], "vix_lag_1"], 100.0)

    def test_download_adjusts_ohlc_and_keeps_unmatched_vix_missing(self):
        source = synthetic_prices(20)
        stock = source.drop(columns="vix").rename(columns=str.title)
        vix = pd.DataFrame({"Close": source["vix"]}).drop(source.index[4])
        calls = []

        def download(ticker, **kwargs):
            calls.append((ticker, kwargs))
            return stock if ticker == "ABC" else vix

        with patch.dict("sys.modules", {"yfinance": SimpleNamespace(download=download)}):
            actual = download_prices("abc", "2015-01-01", "2015-03-01")
        self.assertTrue(pd.isna(actual.loc[source.index[4], "vix"]))
        self.assertEqual(actual.attrs["data_source"], "yahoo_finance")
        for _, kwargs in calls:
            self.assertTrue(kwargs["auto_adjust"])
            self.assertEqual(kwargs["end"], "2015-03-01")

    def test_download_failure_does_not_substitute_synthetic_prices(self):
        fake_yfinance = SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame())
        with patch.dict("sys.modules", {"yfinance": fake_yfinance}):
            with self.assertRaisesRegex(RuntimeError, "Could not download"):
                download_prices("ABC", "2020-01-01", "2021-01-01")


if __name__ == "__main__":
    unittest.main()
