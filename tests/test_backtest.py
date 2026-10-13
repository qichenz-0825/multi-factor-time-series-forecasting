"""Execution timing, trading costs, and initial-wealth risk metric checks."""

import unittest

import numpy as np
import pandas as pd

from forecasting.backtest import backtest


def prices_at_opens(opens):
    opens = np.asarray(opens, dtype=float)
    return pd.DataFrame({"open": opens, "high": opens * 1.01, "low": opens * 0.99,
                         "close": opens, "volume": 1_000_000.0, "vix": 20.0},
                        index=pd.bdate_range("2020-01-02", periods=len(opens), name="date"))


def forecasts(data, row_numbers, values, horizon=5):
    origins = data.index[row_numbers]
    return pd.DataFrame({"origin": origins,
                         "target_date": origins + pd.offsets.BDay(horizon),
                         "horizon": horizon, "predicted_log_return": values,
                         "actual_log_return": 0.0, "model": "test"})


class BacktestTests(unittest.TestCase):
    def test_close_signal_executes_next_open_and_holds_until_new_signal(self):
        data = prices_at_opens([100, 200, 200, 220, 198, 207.9])
        predictions = forecasts(data, [0, 2], [0.1, -0.1])
        daily, metrics = backtest(data, predictions, cost_bps=0)
        self.assertEqual(daily.index[0], data.index[1])
        np.testing.assert_array_equal(daily["position"], [1.0, 1.0, 0.0, 0.0])
        # The 100% origin-close-to-execution-open gap cannot be earned.
        np.testing.assert_allclose(daily["gross_return"], [0.0, 0.1, 0.0, 0.0], atol=1e-12)
        self.assertAlmostEqual(metrics["total_return"], 0.1)
        self.assertAlmostEqual(metrics["benchmark_total_return"], 207.9 / 200 - 1)

    def test_entry_and_terminal_exit_costs_both_count(self):
        data = prices_at_opens([100, 100, 100, 100, 100])
        daily, metrics = backtest(data, forecasts(data, [0], [0.1]), cost_bps=10)
        np.testing.assert_allclose(daily["turnover"], [1.0, 0.0, 1.0])
        np.testing.assert_allclose(daily["net_return"], [-0.001, 0, -0.001])
        self.assertAlmostEqual(metrics["turnover"], 2)
        self.assertAlmostEqual(metrics["total_return"], 0.999**2 - 1)
        self.assertAlmostEqual(metrics["max_drawdown"], 0.999**2 - 1)

    def test_initial_loss_drawdown_uses_starting_wealth(self):
        data = prices_at_opens([100, 100, 90, 90])
        daily, metrics = backtest(data, forecasts(data, [0], [0.1]), cost_bps=0)
        self.assertAlmostEqual(daily["drawdown"].iloc[0], -0.1)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["benchmark_max_drawdown"], -0.1)

    def test_realized_target_is_never_used_to_set_positions(self):
        data = prices_at_opens([100, 101, 99, 100, 102])
        predictions = forecasts(data, [0, 1, 2], [0.03, -0.02, 0.0])
        expected, expected_metrics = backtest(data, predictions)
        predictions["actual_log_return"] = [2000, -3000, np.nan]
        actual, actual_metrics = backtest(data, predictions)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual_metrics, expected_metrics)

    def test_sharpe_uses_sample_std_and_threshold_keeps_cash(self):
        data = prices_at_opens([100, 100, 101, 103, 102])
        predictions = forecasts(data, [0], [0.02])
        daily, metrics = backtest(data, predictions, cost_bps=0)
        expected = np.sqrt(252) * daily["net_return"].mean() / daily["net_return"].std(ddof=1)
        self.assertAlmostEqual(metrics["sharpe"], expected)
        cash, cash_metrics = backtest(data, predictions, threshold=0.03)
        self.assertTrue((cash["position"] == 0).all())
        self.assertEqual(cash_metrics["turnover"], 0)
        self.assertEqual(cash_metrics["sharpe"], 0)

    def test_multiple_models_rejected(self):
        data = prices_at_opens([100, 100, 101, 102, 103])
        predictions = forecasts(data, [0, 1], [0.1, 0.1])
        predictions.loc[1, "model"] = "another"
        with self.assertRaisesRegex(ValueError, "one model at a time"):
            backtest(data, predictions)


if __name__ == "__main__":
    unittest.main()
