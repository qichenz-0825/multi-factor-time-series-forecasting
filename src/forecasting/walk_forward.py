"""Chronological retraining with explicit feature and label availability bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import ForecastFit, fit_arima_predict, fit_bilstm_predict


@dataclass(frozen=True)
class TrainingSamples:
    x: np.ndarray
    y: np.ndarray
    end_positions: np.ndarray
    window_start: int
    lookback: int
    horizon: int

    @property
    def input_start(self) -> int:
        return int(self.end_positions.min()) - self.lookback + 1

    @property
    def input_cutoff(self) -> int:
        return int(self.end_positions.max())

    @property
    def label_cutoff(self) -> int:
        return int(self.end_positions.max()) + self.horizon


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_inputs(data: pd.DataFrame, features: pd.DataFrame) -> None:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have a DatetimeIndex.")
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("Dates must be unique and sorted in ascending order.")
    if data.index.hasnans:
        raise ValueError("Dates must not contain missing values.")
    if not data.index.equals(features.index):
        raise ValueError("features must have exactly the same date index as data.")
    if not len(features.columns) or not features.columns.is_unique:
        raise ValueError("features must contain uniquely named feature columns.")
    if "close" not in data:
        raise ValueError("data must contain close prices.")
    close = data["close"].to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Close prices must be positive and finite.")
    # Feature warm-up NaNs are allowed and excluded at the window level.
    features.to_numpy(dtype=np.float64)


def make_training_samples(
    data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    origin_pos: int,
    train_window: int,
    lookback: int,
    horizon: int,
) -> TrainingSamples:
    """Build direct-H samples wholly contained in the current rolling window.

    A sample ending at position t has inputs [t-L+1, t] and targets
    log(close[t+h] / close[t]), h=1..H. Its last label must be known at the
    after-close forecast origin: t+H <= origin_pos. No filling/backfilling is
    performed; windows with any nonfinite feature are omitted.
    """
    _validate_inputs(data, features)
    for name, value in (("train_window", train_window), ("lookback", lookback), ("horizon", horizon)):
        _positive_integer(value, name)
    if not isinstance(origin_pos, int) or isinstance(origin_pos, bool):
        raise ValueError("origin_pos must be an integer row position.")
    if origin_pos < train_window - 1 or origin_pos >= len(data):
        raise ValueError("origin_pos must have a complete training window in data.")
    window_start = origin_pos - train_window + 1
    values = features.to_numpy(dtype=np.float64)
    log_close = np.log(data["close"].to_numpy(dtype=np.float64))
    xs, ys, positions = [], [], []
    for end in range(window_start + lookback - 1, origin_pos - horizon + 1):
        x = values[end - lookback + 1 : end + 1]
        if not np.isfinite(x).all():
            continue
        xs.append(x)
        ys.append(log_close[end + 1 : end + horizon + 1] - log_close[end])
        positions.append(end)
    if not positions:
        raise ValueError(
            "No eligible training samples: increase train_window, reduce lookback/horizon, "
            "or provide more finite feature history."
        )
    return TrainingSamples(
        x=np.stack(xs),
        y=np.stack(ys),
        end_positions=np.asarray(positions, dtype=np.int64),
        window_start=window_start,
        lookback=lookback,
        horizon=horizon,
    )


def _validate_settings(model: str, train_window: int, lookback: int, horizon: int) -> None:
    for name, value in (
        ("train_window", train_window), ("lookback", lookback), ("horizon", horizon)
    ):
        _positive_integer(value, name)
    if model not in {"arima", "bilstm", "zero"}:
        raise ValueError("model must be 'arima', 'bilstm', or 'zero'.")
    if train_window < lookback + horizon:
        raise ValueError("train_window must be at least lookback + horizon.")


def _fit_at_origin(
    data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    origin_pos: int,
    model: str,
    train_window: int,
    lookback: int,
    horizon: int,
    epochs: int,
    hidden_size: int,
    seed: int,
) -> tuple[ForecastFit, dict]:
    """Shared training path for backtests and live forecasts."""
    origin = data.index[origin_pos]
    samples = make_training_samples(
        data, features, origin_pos=origin_pos, train_window=train_window,
        lookback=lookback, horizon=horizon,
    )
    prediction_x = features.iloc[origin_pos - lookback + 1 : origin_pos + 1].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction_x).all():
        raise ValueError(f"Nonfinite prediction features at origin {origin.date()}.")
    input_start, input_cutoff = samples.input_start, samples.input_cutoff
    label_cutoff, n_samples = samples.label_cutoff, len(samples.x)
    n_features = len(features.columns)
    if model == "bilstm":
        fit = fit_bilstm_predict(
            samples.x, samples.y, prediction_x,
            epochs=epochs, hidden_size=hidden_size, seed=seed,
        )
    elif model == "arima":
        log_close = np.log(data["close"].to_numpy(dtype=np.float64))
        daily_returns = np.diff(log_close[samples.window_start : origin_pos + 1])
        fit = fit_arima_predict(daily_returns, horizon)
        input_start, input_cutoff = samples.window_start, origin_pos
        label_cutoff, n_samples, n_features = origin_pos, len(daily_returns), 1
    else:
        fit = ForecastFit(np.zeros(horizon), {"status": "benchmark_no_fit"})
        n_features = 0
    audit = {
        "origin": origin,
        "model": model,
        "train_window_start": data.index[samples.window_start],
        "train_window_end": origin,
        "training_input_start": data.index[input_start],
        "training_input_cutoff": data.index[input_cutoff],
        "training_label_cutoff": data.index[label_cutoff],
        "prediction_input_start": data.index[origin_pos - lookback + 1],
        "prediction_input_cutoff": origin,
        "n_train_samples": n_samples,
        "n_supervised_samples": len(samples.x),
        "n_features": n_features,
        "train_window": train_window,
        "lookback": lookback,
        "horizon": horizon,
        **fit.diagnostics,
    }
    return fit, audit


def run_walk_forward(
    data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    model: str = "arima",
    train_window: int = 504,
    lookback: int = 30,
    horizon: int = 5,
    stride: int = 5,
    test_start: str = "2024-04-23",
    epochs: int = 10,
    hidden_size: int = 32,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit at each after-close origin; predict cumulative future log returns.

    Only origins with a full rolling train window and all H realized future
    targets are scored. ``stride`` counts trading observations, not calendar
    days. Feature windows are checked for finiteness without imputation. The
    caller is responsible for supplying causally computed features.

    ARIMA receives the daily close returns inside the same rolling date window;
    BiLSTM receives only samples whose entire H-dimensional label is observed.
    Evaluation data is never used for scaling, selection, or early stopping.
    """
    _validate_inputs(data, features)
    _validate_settings(model, train_window, lookback, horizon)
    _positive_integer(stride, "stride")
    start_date = pd.Timestamp(test_start)
    if data.index.tz is not None and start_date.tzinfo is None:
        start_date = start_date.tz_localize(data.index.tz)
    first_origin = max(int(data.index.searchsorted(start_date)), train_window - 1)
    origins = range(first_origin, len(data) - horizon, stride)
    if not len(origins):
        raise ValueError("No evaluable origins: check test_start, train_window, horizon, and data length.")
    log_close = np.log(data["close"].to_numpy(dtype=np.float64))
    prediction_records, audit_records = [], []
    for origin_pos in origins:
        origin = data.index[origin_pos]
        fit, audit = _fit_at_origin(
            data, features, origin_pos=origin_pos, model=model,
            train_window=train_window, lookback=lookback, horizon=horizon,
            epochs=epochs, hidden_size=hidden_size, seed=seed,
        )
        for h in range(1, horizon + 1):
            prediction_records.append({
                "origin": origin,
                "target_date": data.index[origin_pos + h],
                "horizon": h,
                "predicted_log_return": float(fit.prediction[h - 1]),
                "actual_log_return": float(log_close[origin_pos + h] - log_close[origin_pos]),
                "model": model,
            })
        audit_records.append(audit)
    return pd.DataFrame(prediction_records), pd.DataFrame(audit_records)


def forecast_latest(
    data: pd.DataFrame,
    features: pd.DataFrame,
    *,
    model: str = "arima",
    train_window: int = 504,
    lookback: int = 30,
    horizon: int = 5,
    epochs: int = 10,
    hidden_size: int = 32,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit at the final observed close and forecast H future trading periods.

    Future calendar dates are intentionally absent: a trading calendar is not
    assumed. Unlike a backtest, this requires no realized future observations.
    The caller must supply only completed daily bars for a daily-close forecast.
    """
    _validate_inputs(data, features)
    _validate_settings(model, train_window, lookback, horizon)
    if len(data) < train_window:
        raise ValueError("Latest forecast requires a complete training window.")
    fit, audit = _fit_at_origin(
        data, features, origin_pos=len(data) - 1, model=model,
        train_window=train_window, lookback=lookback, horizon=horizon,
        epochs=epochs, hidden_size=hidden_size, seed=seed,
    )
    predictions = pd.DataFrame({
        "origin": data.index[-1],
        "horizon": np.arange(1, horizon + 1),
        "predicted_log_return": fit.prediction,
        "model": model,
    })
    return predictions, pd.DataFrame([audit])
