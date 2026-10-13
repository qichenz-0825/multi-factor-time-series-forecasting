# Methodology

This document describes the portfolio implementation. It is an extension of the coursework, not an exact reproduction of its price-level forecasts. Default settings are declared in `configs/research.json`.

## Observation and forecast timing

An origin `t` is an observed stock trading session. All stock features through `t` become available **after its close**. The model then forecasts cumulative log returns

`y(t, h) = log(close[t + h] / close[t])`, for `h = 1, ..., H`.

Here, `h` counts observed trading rows, not calendar days. Historical evaluation records actual observed target dates. Latest forecasts report horizons from their observed origin without guessing future trading dates. A model is not given future OHLCV, VIX, target prices, or realized returns as predictors.

## Inputs and features

The online data path uses Yahoo Finance through `yfinance`: automatically adjusted daily stock OHLC, reported volume, and the `^VIX` daily close. Stock dates are the alignment index. VIX is joined by observed date without imputation and then lagged one stock session before feature construction.

| Feature family | Definitions | Count |
| --- | --- | ---: |
| Trailing log returns | Close log differences over 1, 2, 3, 5, 10, 20 observations | 6 |
| Moving-average distance | Close / trailing simple mean − 1 over 5, 10, 20, 50 observations | 4 |
| Realized volatility | Sample standard deviation of daily log returns over 5, 10, 20 observations × √252 | 3 |
| Session price structure | Intraday and overnight log returns, high/low range divided by close, close location in the daily range | 4 |
| Volume | Change in `log(1 + volume)` and volume / 20-observation mean − 1 | 2 |
| Momentum | RSI(14), MACD(12,26), signal(9), and histogram; MACD quantities divided by close | 4 |
| True range | 14-observation simple mean of true range divided by close | 1 |
| Lagged VIX | Level, 1- and 5-observation log changes, and distance from 20-observation mean | 4 |
| **Total** | | **28** |

RSI uses exponentially weighted gains and losses with `alpha = 1/14`; MACD uses exponential moving averages. Trailing computations may draw on earlier observed data before a particular training-window boundary. This uses past information and does not extend a model's fitting window into the future. Warm-up values and undefined ratios remain missing. Incomplete training sequences are omitted; an incomplete forecast sequence causes an error instead of fabricated inputs.

## Rolling training and label availability

For each forecast origin, the rolling date window ends at the origin and spans `train_window` observed rows, including that origin. Its start is `origin − train_window + 1` in row positions.

A BiLSTM training sequence ending at `s` contains features from `s − lookback + 1` through `s`. Its final label occurs at `s + H`. A sample is eligible only if its whole input sequence lies inside the rolling window and `s + H <= origin`. Thus no partly unrealized multi-horizon label enters training. The inference sequence ends at the origin and may include observations more recent than the last training input sequence.

Feature means and standard deviations are estimated from the flattened eligible training sequences. Overlapping observations therefore receive the multiplicity implied by those sequences. Target standardization uses eligible training labels. Both transformations are refitted for each origin and applied unchanged to the inference input and predicted outputs. Zero-variance dimensions use scale one.

Historical evaluation includes only origins with a complete training window and all `H` observed future targets. `stride` controls the number of observed rows between refits. With defaults of `H = 5` and `stride = 5`, each origin produces five cumulative return forecasts. Changing stride can introduce overlapping target intervals; correlated forecast errors should not be treated as independent observations.

Audit records retain origin, training window bounds, training input and label cutoffs, prediction input bounds, sample counts, feature count, and fit diagnostics. These make the temporal boundary inspectable rather than relying solely on a claim of no leakage.

## Models

### ARIMA

ARIMA fits daily close log returns within the rolling date window. It selects the lowest BIC among converged, finite-BIC candidates in the fixed grid:

`(0,0,0), (1,0,0), (0,0,1), (1,0,1), (2,0,0), (0,0,2)`.

Because the input is already a return series, the candidate differencing order is zero. A constant term is included. Daily forecasts are cumulatively summed to obtain the same multi-horizon targets as the neural model. Selection uses training data only. Failed or nonconverged candidates and warnings are recorded; if every candidate fails, the run raises an error. ARIMA is a univariate comparator and does not consume the 28-factor feature matrix.

### Bidirectional LSTM

The PyTorch model is a single bidirectional LSTM layer followed by a linear output layer with `H` outputs. It concatenates the final hidden state from each direction. Both directions process only the already observed lookback sequence. Bidirectionality within that historical sequence does not provide access to observations after the forecast origin.

Every origin trains a new model from scratch. Defaults use 32 hidden units per direction, 10 full-batch CPU epochs, Adam at learning rate 0.001, mean squared error on standardized targets, and gradient norm clipping at 1.0. Seed 42 and deterministic PyTorch operations support repeatability within the same platform and dependency versions. There is no test-set early stopping or test-set hyperparameter search. These modest settings are a baseline, not an extensively tuned architecture.

### Zero-return forecast

The zero model predicts zero cumulative return at every horizon. Under the default strict positive-signal rule, its trading strategy remains in cash. Its purpose is to show whether a learned forecast improves on a simple no-change reference.

## Trading simulation

Each ticker/model combination is evaluated separately. A forecast made after the close sets desired weight `1` if its selected-horizon predicted return exceeds `threshold`, otherwise `0`. The default uses horizon 5 and threshold 0. The desired weight becomes executable at the next observed open, then persists until a newer forecast takes effect. The strategy does not short or use leverage.

For an interval beginning at open `u` and ending at the following observed open:

`gross_return[u] = position[u] × (open[u + 1] / open[u] − 1)`

`net_return[u] = gross_return[u] − (cost_bps / 10,000) × turnover[u]`.

Turnover is the absolute change in target weight, including the initial entry. The final position is liquidated at the last available open; its cost is charged to the last interval. A 0→1→0 round trip therefore costs 20 basis points at the default 10-basis-point one-way rate. Costs are subtracted from interval arithmetic returns. Equity is the product of `1 + net_return`, starting at one. A run that exhausts capital raises an error.

The evaluation interval begins at the first executable signal and ends at the last available price open, even when the final signal has been held beyond its forecast horizon. Changing the available price tail can therefore change trading results. The same-period buy-and-hold comparison has no trading costs.

The forecast target uses close-to-close returns, while execution uses next-open fills and open-to-open holding returns. The overnight gap before the first executable open cannot be earned by that forecast. This target/execution mismatch is deliberate and should be considered when interpreting trading outcomes.

## Metrics and interpretation

Forecast errors compare predicted and realized cumulative log returns at the same horizon. The report provides root mean squared error (RMSE), mean absolute error (MAE), and directional accuracy per ticker, model, and horizon. Directional accuracy compares signs, including an exact zero as its own sign; this means a zero-return forecast does not receive a conventional up/down classification score. Trading statistics use daily arithmetic returns:

- Total return: terminal equity minus one.
- Annualized return: `terminal_equity ** (252 / number_of_intervals) − 1`.
- Sharpe: `sqrt(252) × mean(net_return) / sample_std(net_return)`, with zero risk-free rate. Exactly constant series receive Sharpe zero by convention.
- Maximum drawdown: the minimum of equity divided by its running peak, including initial wealth one, minus one. A loss is reported as a negative fraction.
- Turnover: sum of absolute changes in weight, including entry and final liquidation.

A favorable backtest is not established by architecture or implementation alone. The original coursework's price-error comparison is not directly comparable with these return targets and execution assumptions. The resume's Sharpe and drawdown figures have not been verified. No statistical claim of model superiority should be inferred from a synthetic demo or a single two-stock historical run.

## Remaining limitations

Adjusted vendor history is not a point-in-time data archive: future corporate actions or corrections may revise earlier bars. The feature implementation is causal on the supplied snapshot, but that does not prove complete real-world point-in-time availability. The fixed choice of NVDA and XOM also introduces selection and survivorship limitations and cannot establish index-wide generalization.

The simulator assumes fills at the reported adjusted opening price with a fixed turnover charge. It omits market impact, variable spread/slippage, partial fills, order rejection, taxes, and cash interest. It does not model a multi-asset allocation, liquidity or capacity constraints, a historical constituent universe, or live execution. Further research should specify a point-in-time universe and data archive, separate development from final evaluation periods, and assess sensitivity to costs, regimes, seeds, and target/execution alignment before making performance claims.
