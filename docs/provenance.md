# Provenance and data use

## Project lineage

The project extends an ISE 537 final-project study supplied by the repository owner. The source report compares ARIMA and conventional LSTM price forecasts for NVDA and XOM using daily data from January 2, 2015 through April 25, 2025. It describes a held-out period beginning April 23, 2024 and rolling five-step price forecasts.

The public implementation changes the prediction target to cumulative log returns and adds a causal factor pipeline, a PyTorch bidirectional LSTM, explicit label cutoffs, execution-aware trading evaluation, and automation. It is a research extension, not a byte-for-byte recreation of the coursework experiments. The original report's conclusions and numerical price errors are not treated as validated results of this implementation.

Private contact details, the resume PDF, course lecture materials, and the original local project bundle are not needed to run this repository and are not part of its public deliverables. This document describes the lineage without redistributing those materials or attributing third-party course material to the repository author.

## Online market data

The online download path uses the `yfinance` package to request Yahoo Finance daily observations for configured stock tickers and `^VIX`. The stock OHLC fields use the vendor's automatic adjustment option; volume is the reported vendor volume. Start is inclusive and end is exclusive. VIX is aligned to stock dates without backfilling; the feature pipeline applies an additional one-stock-session lag.

Access to a data endpoint is not a redistribution license. Users are responsible for reviewing the applicable provider terms and using the data within the permissions available to them. Any code license in this repository applies to the code, not to Yahoo Finance, VIX, index, exchange, or other third-party market data. Raw vendor downloads should remain local and should not be committed to the public repository or included in publicly shared artifacts without appropriate permission.

A failed vendor request raises an error. The real-data path never silently replaces missing observations with synthetic data. Vendor corrections, historical adjustments, availability changes, and package versions can produce different snapshots on different download dates. Automatically adjusted bars are retrospective values and do not establish what a trader could have downloaded at each historical forecast origin.

The original report lists 2,593 observations per ticker. The snapshot downloaded for this implementation contains 2,594 observations per ticker over the requested period. That difference is another reason not to describe the extension as an exact reproduction of the original experiment. The run metadata records the actual input coverage and checksum used for a specific execution.

## Local input format

The CLI accepts a data directory containing a ticker-named CSV for each configured symbol, for example `NVDA.csv` and `XOM.csv`:

```text
date,open,high,low,close,volume,vix
```

Dates must be observed trading dates in strictly increasing order without duplicates. OHLC prices must be positive and consistent within each bar; volume must be nonnegative. Stock OHLCV cannot be missing or infinite. Known VIX values must be positive and finite; missing VIX values remain missing and affect feature/sample eligibility. Do not mix adjusted close prices with unadjusted open, high, or low prices.

Preserve the original downloaded snapshot locally when comparing experiments. A reproducible result should retain the exact code revision, configuration, dependency environment, source/retrieval details, input date coverage and checksums where available, predictions, training audit, and trading outputs. A download refresh deliberately creates a new snapshot and may change results.

Online downloads save a local `<TICKER>.metadata.json` sidecar with provider, retrieval time, requested interval, coverage, and checksum information. Experiment `run_metadata.json` records a checksum of the normalized input used in that run, along with dependency versions and resolved configuration. Third-party CSVs without a download sidecar are labeled `user_supplied_csv`; users should preserve their source and adjustment documentation separately.

## Synthetic demonstration data

The offline demo uses seeded, fictional OHLCV and volatility series. Its dates are generated from weekdays, without a US exchange-holiday calendar. It exists to exercise data validation, feature construction, fitting, chronological evaluation, and artifact generation without a network connection.

Synthetic charts, forecast errors, Sharpe ratios, or drawdowns are software demonstration outputs. They are not historical stock observations, investment evidence, or support for resume performance claims. Demo artifacts should retain their synthetic labeling when shared.

## Scope and selection

The default universe is a fixed pair of stocks chosen in the original coursework, representing two different industries. This is not a reconstruction of historical S&P 500 membership. Selection and survivorship limitations therefore remain even when the pipeline's chronological boundaries are correct. Extending the ticker list alone would not address those limitations; an index-wide study needs a dated constituent universe and appropriate handling of delistings, corporate actions, missing data, and historical availability.
