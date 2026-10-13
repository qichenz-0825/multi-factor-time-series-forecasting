# Resume alignment and verification

This repository implements a research system inspired by the “Multi-Factor Time Series Forecasting System” project description. It makes the relationship between the original coursework, the extension, and performance claims explicit.

## Capability mapping

| Resume description | Repository implementation | Scope / evidence boundary |
| --- | --- | --- |
| Multi-step log-return forecasting | Cumulative return targets for horizons 1 through `H`; default `H = 5` | The coursework originally forecast price levels; this is an extension |
| ARIMA and bidirectional LSTM | BIC-selected ARIMA on daily returns and a direct multi-output PyTorch BiLSTM | The coursework used a conventional LSTM; the bidirectional model is implemented here |
| Rolling-window retraining | A fresh fit at each historical forecast origin | Default 504-row window and five-row refit stride are declared in configuration |
| OHLCV, RSI, MACD, and VIX; 15+ features | 28 trailing features with lagged VIX | ARIMA remains univariate; the feature matrix feeds the neural model |
| Avoiding look-ahead bias | Training-only standardization, fully observed labels, after-close origins, next-open execution, and audit records | These controls apply to supplied data; retrospective adjusted vendor history is not a point-in-time archive |
| Automated daily pipeline | CLI refresh/latest-forecast command and a scheduled GitHub Actions workflow | Workflow execution depends on GitHub scheduling and vendor availability; this is not a production uptime claim |
| Simulated trading evaluation | Per-ticker long/cash backtest with configurable turnover costs and liquidation | There is no live broker integration or portfolio-wide risk allocation |
| S&P 500 equities | Default case study: NVDA and XOM | This is not a point-in-time S&P 500 constituent backtest; do not describe it as index-wide coverage |
| Docker | Containerized project entry point | Provides a repeatable execution setup; it is not evidence of hosted production deployment |
| Sharpe 1.45; maximum drawdown 12.4% | Not established by the supplied original project materials | **Unverified; do not present these figures as reproduced results** |

## What the original materials support

The supplied ISE 537 report describes an ARIMA-versus-LSTM price forecasting study using NVDA and XOM daily observations from January 2, 2015 through April 25, 2025. Its stated training period ends April 22, 2024 and its testing period starts April 23, 2024. It reports rolling five-step price forecasts and price-level forecast errors.

Those materials do not establish the current 28-feature pipeline, PyTorch bidirectional model, next-open trading simulation, Docker setup, or the resume's Sharpe and drawdown claims. These repository capabilities are a subsequent implementation. A repository publication date or an automation schedule must not be used to imply that the entire extension was operating during the earlier coursework period.

## Suggested resume wording

The following wording describes the implemented scope without claiming unverified market performance:

> Built a configurable Python/PyTorch research pipeline for multi-horizon equity log-return forecasting, comparing BIC-selected ARIMA and a bidirectional LSTM with rolling-window retraining on NVDA and XOM.
>
> Engineered 28 OHLCV, RSI, MACD, volatility, and lagged VIX features with chronological label filtering, training-only standardization, and per-origin audit records.
>
> Implemented a next-open long/cash backtest with transaction costs, reproducible CLI and Docker workflows, and scheduled latest-forecast generation; evaluated forecast error and risk-adjusted trading metrics against simple benchmarks.

Use the project's actual development dates on a resume. Only add numerical performance after a real-data run has been retained with its configuration, data provenance, evaluation interval, costs, metrics, and code revision. Identify the ticker/model or precisely defined portfolio associated with each number. A number from an offline synthetic demo should never be used as a historical trading achievement.

## 中文核对说明

原始课程材料能够支持的是：2015–2025 年 NVDA 和 XOM 的 ARIMA/LSTM 股价预测研究。当前仓库是在这一基础上增加的工程实现，包括 28 个特征、PyTorch 双向 LSTM、多步收益率目标、滚动训练、含交易成本的次日开盘回测和自动化流程。

简历中的 Sharpe 1.45 与最大回撤 12.4% 尚未由原始材料或本仓库的可追溯实验复现。建议先采用上面的功能性描述；获得真实数据实验结果后，再按具体模型、股票、样本区间及交易成本更新数字。模拟数据结果不应写成真实市场业绩，也不应将两只股票的案例描述为覆盖全部标普 500 成分股。
