"""A single-symbol long/cash backtest with next-session-open execution."""

import numpy as np
import pandas as pd

from .data import validate_prices


def _performance(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    """Daily arithmetic returns; 252 sessions/year, sample std, risk-free rate 0."""
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annualized_return = float(equity.iloc[-1] ** (252.0 / len(returns)) - 1.0)
    # Initial wealth is 1: an immediate first-period loss must be a drawdown.
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1.0
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "sharpe": float(np.sqrt(252.0) * returns.mean() / volatility) if volatility > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
    }


def backtest(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    signal_horizon: int = 5,
    cost_bps: float = 10.0,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Execute positive forecast signals at the following trading session open.

    For one model and the selected horizon, a forecast made after origin close
    sets a target weight of 1 when predicted_log_return > threshold, otherwise
    0. The target is executed at the next observed session open and held until
    another forecast arrives. Each row dated t earns t-open to (t+1)-open return;
    it cannot earn the close-to-next-open gap that precedes its execution.

    The evaluation begins at the first executable signal and ends at the last
    available price open. Entry/rebalance costs are cost_bps * absolute change
    in weight / 10,000. The final position is liquidated at the final open, with
    its exit cost assigned to the last interval row. Costs are subtracted from
    that interval's arithmetic return. Thus reported turnover includes entry and
    liquidation. The same-period benchmark buys and holds without trading costs.

    Forecast target_date and actual_log_return are metadata, never trade inputs.
    Returns and drawdowns start with wealth 1; Sharpe uses sample standard
    deviation, zero risk-free return and 252 trading sessions per year. An exactly
    constant return series receives Sharpe 0 rather than an undefined infinity.
    """
    prices = validate_prices(data)
    if isinstance(signal_horizon, bool) or not isinstance(signal_horizon, (int, np.integer)) or signal_horizon < 1:
        raise ValueError("signal_horizon must be a positive integer.")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10_000:
        raise ValueError("cost_bps must be finite and in [0, 10000).")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative.")
    required = {"origin", "target_date", "horizon", "predicted_log_return", "actual_log_return", "model"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns: {', '.join(sorted(required - set(predictions.columns)))}")
    signals = predictions.loc[predictions["horizon"] == signal_horizon].copy()
    if signals.empty:
        raise ValueError(f"No forecasts for signal_horizon={signal_horizon}.")
    if signals["model"].nunique(dropna=False) != 1:
        raise ValueError("Backtest one model at a time; filter the prediction table first.")
    origins = pd.DatetimeIndex(pd.to_datetime(signals["origin"], errors="raise"))
    if origins.tz is not None:
        origins = origins.tz_localize(None)
    origins = origins.normalize()
    if origins.has_duplicates or origins.hasnans:
        raise ValueError("Forecast origin dates must be unique and non-missing.")
    if not origins.isin(prices.index).all():
        raise ValueError("Every forecast origin must be an observed price date.")
    predicted = pd.to_numeric(signals["predicted_log_return"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError("Forecast signals must be finite.")
    close_signal = pd.Series(np.nan, index=prices.index, dtype=float)
    close_signal.loc[origins] = (predicted > threshold).astype(float)
    position = close_signal.shift(1).ffill().fillna(0.0)
    first_execution = int(prices.index.get_indexer([origins.min()])[0]) + 1
    if first_execution >= len(prices) - 1:
        raise ValueError("At least two observed opens after the first forecast origin are required.")
    interval_index = prices.index[first_execution:-1]
    position = position.loc[interval_index]
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    turnover.iloc[-1] += abs(position.iloc[-1])
    benchmark_return = (prices["open"].shift(-1) / prices["open"] - 1.0).loc[interval_index]
    gross_return = position * benchmark_return
    net_return = gross_return - turnover * cost_bps / 10_000.0
    if (net_return <= -1.0).any():
        raise ValueError("Trading costs/returns exhaust capital; reduce cost_bps or inspect inputs.")
    equity = (1.0 + net_return).cumprod()
    benchmark_equity = (1.0 + benchmark_return).cumprod()
    daily = pd.DataFrame({
        "date": interval_index,
        "position": position,
        "turnover": turnover,
        "gross_return": gross_return,
        "net_return": net_return,
        "equity": equity,
        "drawdown": equity / equity.cummax().clip(lower=1.0) - 1.0,
        "benchmark_return": benchmark_return,
        "benchmark_equity": benchmark_equity,
    }, index=interval_index)
    metrics = _performance(net_return, equity)
    metrics["turnover"] = float(turnover.sum())
    metrics.update({f"benchmark_{key}": value for key, value in
                    _performance(benchmark_return, benchmark_equity).items()})
    return daily, metrics
