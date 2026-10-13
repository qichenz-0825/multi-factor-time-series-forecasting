"""Daily adjusted OHLCV and VIX inputs, with an explicit offline demo source."""

from pathlib import Path

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "vix")


def validate_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy; reject invalid stock bars rather than filling them.

    The index must contain unique, increasing trading dates. OHLCV must be known
    on every row; missing VIX observations are retained as NaN for downstream
    complete-case filtering. No stock dates or unknown stock values are imputed.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Prices must be a non-empty pandas DataFrame.")
    result = data.copy()
    result.columns = [str(column).strip().lower() for column in result.columns]
    if result.columns.duplicated().any():
        raise ValueError("Price column names must be unique.")
    missing = set(PRICE_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(sorted(missing))}")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("Prices require a DatetimeIndex of trading dates.")
    if result.index.hasnans:
        raise ValueError("Trading dates cannot be missing.")
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result.index = result.index.normalize().rename("date")
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("Trading dates must be unique and sorted in ascending order.")
    result = result.loc[:, list(PRICE_COLUMNS)]
    for column in PRICE_COLUMNS:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {column!r} must contain numeric values.") from exc
    stock = result.loc[:, ["open", "high", "low", "close", "volume"]]
    if not np.isfinite(stock.to_numpy()).all():
        raise ValueError("Stock OHLCV contains missing or infinite values; do not fill unknown bars.")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive.")
    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    tolerance = result["close"] * 1e-8
    if ((result["high"] + tolerance < result[["open", "low", "close"]].max(axis=1)).any()
            or (result["low"] - tolerance > result[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("Inconsistent OHLC bar: low <= open/close <= high is required.")
    vix = result["vix"].dropna()
    if not np.isfinite(vix).all() or (vix <= 0).any():
        raise ValueError("Known VIX observations must be finite and strictly positive.")
    return result


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read a CSV with a date column and the six normalized input columns."""
    data = pd.read_csv(path)
    data.columns = [str(column).strip().lower() for column in data.columns]
    if "date" not in data.columns:
        raise ValueError("Price CSV must include a 'date' column.")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    return validate_prices(data.set_index("date"))


def save_prices(data: pd.DataFrame, path: str | Path) -> None:
    """Write a validated date-indexed CSV, creating its parent directory."""
    destination = Path(path)
    validated = validate_prices(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(destination, index_label="date", date_format="%Y-%m-%d")


def _single_ticker(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no daily observations for {ticker}.")
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        # Current yfinance uses (Price, Ticker); older versions return flat columns.
        if ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, level=-1, axis=1)
        elif len(frame.columns.get_level_values(-1).unique()) == 1:
            frame.columns = frame.columns.get_level_values(0)
        else:
            raise RuntimeError(f"Unexpected multi-ticker download for {ticker}.")
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize().rename("date")
    return frame.sort_index()


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily Yahoo auto-adjusted OHLC, raw volume, and VIX close.

    ``start`` is inclusive and ``end`` is exclusive, matching yfinance. Corporate
    action adjustment is applied consistently to all OHLC fields. VIX is joined
    by observed trading date without filling; build_features applies a further
    one-session lag. Network or source failures raise, never create demo prices.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("A non-empty ticker is required.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("start must be earlier than the exclusive end date.")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Real-data download requires yfinance; install the project dependencies.") from exc
    ticker = ticker.strip().upper()
    try:
        options = dict(start=start, end=end, interval="1d", auto_adjust=True,
                       progress=False, threads=False)
        stock = _single_ticker(yf.download(ticker, **options), ticker)
        vix = _single_ticker(yf.download("^VIX", **options), "^VIX")
        stock["vix"] = vix["close"].reindex(stock.index)
        if stock["vix"].notna().sum() == 0:
            raise ValueError("VIX has no observations matching the requested stock dates.")
        result = validate_prices(stock)
    except Exception as exc:
        raise RuntimeError(f"Could not download valid adjusted prices for {ticker}: {exc}") from exc
    result.attrs.update(data_source="yahoo_finance", ticker=ticker, adjusted=True)
    return result


def synthetic_prices(periods: int = 900, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible, fictional weekday prices for an offline smoke test.

    These observations are not market history and cannot support investment or
    resume performance claims. Weekdays are used, without a US holiday calendar.
    """
    if isinstance(periods, bool) or not isinstance(periods, (int, np.integer)) or periods < 2:
        raise ValueError("periods must be an integer of at least 2.")
    rng = np.random.default_rng(seed)
    shock = rng.normal(size=periods)
    vix = np.empty(periods)
    vix[0] = 20.0
    for index in range(1, periods):
        vix[index] = np.clip(20.0 + 0.965 * (vix[index - 1] - 20.0)
                             - 0.7 * shock[index] + rng.normal(0, 0.7), 9.0, 55.0)
    daily_scale = np.r_[vix[0], vix[:-1]] / (100.0 * np.sqrt(252.0))
    intraday = 0.00025 + daily_scale * 0.85 * shock
    overnight = rng.normal(0.00005, 0.003, periods)
    close = 100.0 * np.exp(np.cumsum(overnight + intraday))
    previous_close = np.r_[100.0, close[:-1]]
    opening = previous_close * np.exp(overnight)
    spread = np.abs(rng.normal(0.004, 0.002, periods))
    result = pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) * np.exp(spread),
        "low": np.minimum(opening, close) * np.exp(-spread),
        "close": close,
        "volume": np.round(rng.lognormal(np.log(3_000_000), 0.3, periods)
                           * (1.0 + 0.2 * np.abs(shock))),
        "vix": vix,
    }, index=pd.bdate_range("2015-01-02", periods=periods, name="date"))
    result = validate_prices(result)
    result.attrs.update(data_source="synthetic", seed=seed, adjusted=False)
    return result
