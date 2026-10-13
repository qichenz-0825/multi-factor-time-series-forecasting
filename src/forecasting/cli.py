"""Commands for offline checks, historical experiments, and daily forecast updates."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .backtest import backtest
from .data import download_prices, load_prices, save_prices, synthetic_prices
from .features import build_features
from .walk_forward import forecast_latest, run_walk_forward

DEFAULTS = dict(tickers=['NVDA', 'XOM'], start='2015-01-02', end='2025-04-26',
                test_start='2024-04-23', models=['zero', 'arima', 'bilstm'],
                train_window=504, lookback=30, horizon=5, stride=5,
                epochs=10, hidden_size=32, seed=42, signal_horizon=5,
                cost_bps=10.0, threshold=0.0)
MODEL_KEYS = ('train_window', 'lookback', 'horizon', 'epochs', 'hidden_size', 'seed')


def load_config(path=None):
    config = DEFAULTS.copy()
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        config.update(supplied)
    if not config['tickers'] or not all(isinstance(t, str) and re.fullmatch(r'[A-Z0-9.^-]+', t)
                                      for t in config['tickers']):
        raise ValueError('tickers must be a nonempty list of uppercase symbols.')
    if len(set(config['tickers'])) != len(config['tickers']):
        raise ValueError('tickers must be unique.')
    if not config['models'] or not set(config['models']) <= {'zero', 'arima', 'bilstm'}:
        raise ValueError('models must contain zero, arima, or bilstm.')
    if len(set(config['models'])) != len(config['models']):
        raise ValueError('models must be unique.')
    for key in ('train_window', 'lookback', 'horizon', 'stride', 'epochs', 'hidden_size', 'signal_horizon'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if not 1 <= config['signal_horizon'] <= config['horizon']:
        raise ValueError('signal_horizon must be within the forecast horizon.')
    if config['stride'] > config['horizon']:
        raise ValueError('stride must not exceed horizon between scheduled forecast origins.')
    if config['train_window'] <= config['lookback'] + config['horizon']:
        raise ValueError('train_window must exceed lookback + horizon.')
    if pd.Timestamp(config['start']) >= pd.Timestamp(config['end']):
        raise ValueError('start must precede exclusive end.')
    return config


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + '\n')


def _provenance(data, source):
    canonical = data.to_csv(index_label='date', date_format='%Y-%m-%d').encode()
    return {'source': source, 'rows': len(data), 'first_date': str(data.index[0].date()),
            'last_date': str(data.index[-1].date()), 'canonical_csv_sha256': hashlib.sha256(canonical).hexdigest(),
            'adjustment': {'yahoo_finance': 'auto_adjusted_OHLC_raw_volume',
                           'synthetic': 'fictional'}.get(source, 'user_supplied_unverified')}


def _prices(ticker, config, data_dir, refresh):
    path = Path(data_dir) / f'{ticker}.csv'
    metadata_path = path.with_suffix('.metadata.json')
    requested_start = pd.Timestamp(config['start'])
    requested_end = pd.Timestamp(config['end'])
    test_start = pd.Timestamp(config['test_start'])
    if refresh or not path.exists():
        data = download_prices(ticker, config['start'], config['end'])
        save_prices(data, path)
        write_json(metadata_path, {
            'provider': 'Yahoo Finance via yfinance', 'ticker': ticker,
            'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
            'requested_start': config['start'], 'requested_end_exclusive': config['end'],
            'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            **_provenance(data, 'yahoo_finance')})
    else:
        data = load_prices(path)
    source = 'user_supplied_csv'
    if metadata_path.exists():
        try:
            snapshot = json.loads(metadata_path.read_text())
            if (snapshot['ticker'] != ticker
                    or snapshot['provider'] != 'Yahoo Finance via yfinance'
                    or snapshot.get('source') != 'yahoo_finance'
                    or snapshot.get('adjustment') != 'auto_adjusted_OHLC_raw_volume'):
                raise ValueError('provider, ticker, source, or adjustment does not match')
            cached_start = pd.Timestamp(snapshot['requested_start'])
            cached_end = pd.Timestamp(snapshot['requested_end_exclusive'])
            if pd.isna(cached_start) or pd.isna(cached_end):
                raise ValueError('snapshot metadata has missing requested dates')
            if cached_start > requested_start or cached_end < requested_end:
                raise ValueError('cached download request does not cover the requested dates')
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            csv_hash = snapshot.get('csv_sha256')
            canonical_hash = snapshot.get('canonical_csv_sha256')
            if not csv_hash and not canonical_hash:
                raise ValueError('snapshot metadata has no recorded data hash')
            if csv_hash and csv_hash != file_hash:
                raise ValueError('CSV content no longer matches its recorded hash')
            if canonical_hash:
                normalized_hash = _provenance(data, 'yahoo_finance')['canonical_csv_sha256']
                # save_prices writes this same canonical layout; raw-byte equality
                # also avoids false failures from CSV floating-point round trips.
                if canonical_hash not in (file_hash, normalized_hash):
                    raise ValueError('CSV content no longer matches its canonical hash')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{ticker}: invalid cache provenance ({exc}); use --refresh.') from exc
        source = 'yahoo_finance'
    data = data.loc[(data.index >= requested_start) & (data.index < requested_end)]
    if data.empty:
        raise ValueError(f'{ticker}: cached data does not overlap the requested dates; use --refresh.')
    if data.index[0] > test_start:
        raise ValueError(f'{ticker}: cache begins after test_start; use --refresh.')
    if data.index[-1] < test_start:
        raise ValueError(f'{ticker}: cache has no requested test-period observations; use --refresh.')
    # This conservative guard allows weekends/holiday closures without assuming
    # an exchange calendar, while rejecting materially stale or truncated tails.
    if requested_end - data.index[-1] > pd.Timedelta(days=7):
        raise ValueError(f'{ticker}: cache ends over 7 calendar days before requested end; use --refresh.')
    return data, source


def _forecast_scores(predictions):
    rows = []
    for (ticker, model, horizon), frame in predictions.groupby(['ticker', 'model', 'horizon']):
        error = frame.predicted_log_return - frame.actual_log_return
        rows.append(dict(ticker=ticker, model=model, horizon=int(horizon), n=len(frame),
                         rmse_log_return=float(np.sqrt(np.mean(error ** 2))),
                         mae_log_return=float(np.mean(np.abs(error))),
                         directional_accuracy=float(np.mean(np.sign(frame.predicted_log_return)
                                                            == np.sign(frame.actual_log_return)))))
    return rows


def _plot(curves, destination, synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tickers = list(dict.fromkeys(t for t, _ in curves))
    fig, axes = plt.subplots(len(tickers), 2, figsize=(12, 3.6 * len(tickers)), squeeze=False)
    for row, ticker in enumerate(tickers):
        first = True
        for (symbol, model), frame in curves.items():
            if symbol != ticker:
                continue
            axes[row, 0].plot(frame.date, frame.equity, label=model.upper(), linewidth=1.6)
            axes[row, 1].plot(frame.date, 100 * frame.drawdown, label=model.upper(), linewidth=1.4)
            if first:
                axes[row, 0].plot(frame.date, frame.benchmark_equity, label='BUY & HOLD',
                                  linestyle='--', color='#7c8797', linewidth=1.3)
                first = False
        axes[row, 0].set_title(f'{ticker} | net wealth (initial = 1)')
        axes[row, 1].set_title(f'{ticker} | strategy drawdown (%)')
        for axis in axes[row]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
            axis.tick_params(axis='x', rotation=25)
    fig.suptitle('SYNTHETIC SMOKE TEST - not market performance' if synthetic else
                 'NVDA / XOM | rolling out-of-sample experiment', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_experiment(config, output, data_dir='data/raw', refresh=False, demo=False, latest=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Remove a prior success marker before any work; failed reruns must not look successful.
    status = output / 'run_metadata.json'
    write_json(status, {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat()})
    predictions, audits, metrics, curves, provenance = [], [], [], {}, {}
    try:
        for index, ticker in enumerate(config['tickers']):
            if demo:
                data, source = synthetic_prices(periods=420, seed=config['seed'] + index), 'synthetic'
            else:
                data, source = _prices(ticker, config, data_dir, refresh)
            features = build_features(data)
            provenance[ticker] = _provenance(data, source)
            feature_summary = {'feature_count': features.shape[1], 'features': features.columns.tolist()}
            for model in config['models']:
                print(f'{ticker}: fitting {model} ({len(data)} sessions)', flush=True)
                options = {key: config[key] for key in MODEL_KEYS}
                if latest:
                    pred, audit = forecast_latest(data, features, model=model, **options)
                else:
                    pred, audit = run_walk_forward(data, features, model=model, **options,
                                                  stride=config['stride'], test_start=config['test_start'])
                if pred.empty:
                    raise ValueError(f'{ticker}/{model}: no forecasts; check available history and test_start.')
                pred['ticker'], audit['ticker'] = ticker, ticker
                predictions.append(pred)
                audits.append(audit)
                if not latest:
                    daily, summary = backtest(data, pred, signal_horizon=config['signal_horizon'],
                                              cost_bps=config['cost_bps'], threshold=config['threshold'])
                    daily.to_csv(output / f'{ticker}_{model}_daily.csv', index=False)
                    metrics.append({'ticker': ticker, 'model': model, **summary})
                    curves[ticker, model] = daily
        combined = pd.concat(predictions, ignore_index=True)
        combined.to_csv(output / 'predictions.csv', index=False)
        pd.concat(audits, ignore_index=True).to_csv(output / 'training_audit.csv', index=False)
        write_json(output / 'features.json', feature_summary)
        write_json(output / 'config.json', config)
        if not latest:
            scores = _forecast_scores(combined)
            pd.DataFrame(scores).to_csv(output / 'forecast_metrics.csv', index=False)
            write_json(output / 'metrics.json', metrics)
            _plot(curves, output / 'performance.png', demo)
            lines = ['# Synthetic smoke test' if demo else '# Historical research run', '',
                     'Synthetic observations; these numbers are not market evidence.' if demo else
                     'Separate single-stock simulations; these are not combined portfolio returns.', '',
                     '| Ticker | Model | Sharpe | Max drawdown | Total return |',
                     '|---|---|---:|---:|---:|']
            for item in metrics:
                sharpe = f"{item['sharpe']:.3f}" if item['sharpe'] is not None else 'N/A'
                lines.append(f"| {item['ticker']} | {item['model']} | {sharpe} | "
                             f"{item['max_drawdown']:.2%} | {item['total_return']:.2%} |")
            lines += ['', '![Equity and drawdown](performance.png)', '',
                      'See `config.json`, `run_metadata.json`, `training_audit.csv`, and daily ledgers for reproducibility.',
                      '10 bps one-way costs by default; signals execute next open. Sharpe uses 252 sessions and zero risk-free return.']
            (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
        versions = {name: metadata.version(name) for name in
                    ('numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'torch', 'yfinance', 'matplotlib')}
        write_json(status, {'status': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                            'mode': 'synthetic_demo' if demo else ('latest' if latest else 'historical'),
                            'python': platform.python_version(), 'dependencies': versions,
                            'datasets': provenance, 'config': config,
                            'resume_performance_claim_verified': False})
        print(f'Completed. Results: {output.resolve()}', flush=True)
    except Exception as exc:
        write_json(status, {'status': 'failed', 'error': str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'demo', 'update'):
        item = sub.add_parser(command)
        item.add_argument('--config', help='JSON config; defaults to the NVDA/XOM research settings')
        item.add_argument('--output', default=f'reports/{command}')
        item.add_argument('--data-dir', default='data/raw')
        item.add_argument('--refresh', action='store_true', help='Replace cached input snapshots with new downloads')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == 'demo':
            config.update(tickers=['SYNTHETIC'], train_window=180, lookback=20, epochs=2,
                          hidden_size=8, test_start='2016-06-01', stride=5,
                          horizon=5, signal_horizon=5)
        if args.command == 'update':
            now = datetime.now(ZoneInfo('America/New_York'))
            # Include today's bar only after a conservative 18:00 NY publication buffer.
            end_date = now.date() + timedelta(days=1) if now.hour >= 18 else now.date()
            config['end'] = end_date.isoformat()
        run_experiment(config, args.output, args.data_dir,
                       args.refresh or args.command == 'update', args.command == 'demo',
                       args.command == 'update')
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    if return_code:
        raise SystemExit(return_code)
