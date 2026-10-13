"""Causal technical and volatility predictors available after each session close."""

import numpy as np
import pandas as pd

from .data import validate_prices


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return same-index, past-and-present predictors without imputing warm-up rows.

    All stock inputs through session t are available only after t's close. VIX
    uses t-1 throughout to avoid relying on synchronized close/publication times.
    Rolling statistics and EMA never access future rows. Missing values, including
    unavailable VIX history, remain NaN; no backward fill is used.
    """
    prices = validate_prices(data)
    close = prices["close"]
    volume = prices["volume"]
    log_close = np.log(close)
    returns = log_close.diff()
    result = pd.DataFrame(index=prices.index)
    for lag in (1, 2, 3, 5, 10, 20):
        result[f"log_return_{lag}"] = log_close.diff(lag)
    for window in (5, 10, 20, 50):
        result[f"close_to_sma_{window}"] = close / close.rolling(window).mean() - 1.0
    for window in (5, 10, 20):
        result[f"realized_vol_{window}"] = returns.rolling(window).std(ddof=1) * np.sqrt(252)
    result["intraday_log_return"] = np.log(close / prices["open"])
    result["overnight_log_return"] = np.log(prices["open"] / close.shift(1))
    result["high_low_range"] = (prices["high"] - prices["low"]) / close
    price_range = (prices["high"] - prices["low"]).replace(0.0, np.nan)
    result["close_location"] = (close - prices["low"]) / price_range
    result["volume_log_change"] = np.log1p(volume).diff()
    result["volume_to_mean_20"] = volume / volume.rolling(20).mean().replace(0.0, np.nan) - 1.0

    change = close.diff()
    average_gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100.0 - 100.0 / (1.0 + average_gain / average_loss.replace(0.0, np.nan))
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    result["rsi_14"] = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_normalized"] = macd / close
    result["macd_signal_normalized"] = macd_signal / close
    result["macd_histogram_normalized"] = (macd - macd_signal) / close

    true_range = pd.concat([prices["high"] - prices["low"],
                            (prices["high"] - close.shift(1)).abs(),
                            (prices["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
    result["atr_14_normalized"] = true_range.rolling(14).mean() / close
    lagged_vix = prices["vix"].shift(1)
    result["vix_lag_1"] = lagged_vix
    result["vix_log_change_1"] = np.log(lagged_vix).diff()
    result["vix_log_change_5"] = np.log(lagged_vix).diff(5)
    result["vix_to_mean_20"] = lagged_vix / lagged_vix.rolling(20).mean() - 1.0
    return result.replace([np.inf, -np.inf], np.nan)
