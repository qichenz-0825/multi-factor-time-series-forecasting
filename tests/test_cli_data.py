"""Cache reuse must preserve provenance and the requested evaluation coverage."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from forecasting.cli import _prices, _provenance
from forecasting.data import save_prices, synthetic_prices


class CliCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'ABC.csv'
        self.data = synthetic_prices(120)
        self.config = {
            'start': str(self.data.index[0].date()),
            'end': str((self.data.index[-1] + pd.Timedelta(days=1)).date()),
            'test_start': str(self.data.index[80].date()),
        }
        save_prices(self.data, self.path)

    def metadata(self, **updates):
        snapshot = {
            'provider': 'Yahoo Finance via yfinance', 'ticker': 'ABC',
            'source': 'yahoo_finance', 'adjustment': 'auto_adjusted_OHLC_raw_volume',
            'requested_start': self.config['start'],
            'requested_end_exclusive': self.config['end'],
            'csv_sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }
        snapshot.update(updates)
        self.path.with_suffix('.metadata.json').write_text(json.dumps(snapshot))

    def test_valid_legacy_byte_hash_reuses_provider_snapshot(self):
        self.metadata()
        with patch('forecasting.cli.download_prices') as download:
            actual, source = _prices('ABC', self.config, self.directory.name, False)
        download.assert_not_called()
        self.assertEqual(source, 'yahoo_finance')
        self.assertEqual(len(actual), len(self.data))

    def test_canonical_hash_accepts_valid_csv_round_trip(self):
        canonical = _provenance(self.data, 'yahoo_finance')['canonical_csv_sha256']
        self.metadata(csv_sha256=None, canonical_csv_sha256=canonical)
        _, source = _prices('ABC', self.config, self.directory.name, False)
        self.assertEqual(source, 'yahoo_finance')

    def test_changed_values_in_provider_snapshot_are_rejected(self):
        self.metadata()
        changed = self.data.copy()
        changed['volume'] *= 2
        save_prices(changed, self.path)
        with self.assertRaisesRegex(ValueError, 'no longer matches'):
            _prices('ABC', self.config, self.directory.name, False)

    def test_metadata_requires_identity_and_hash(self):
        self.metadata(ticker='OTHER')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            _prices('ABC', self.config, self.directory.name, False)
        self.metadata(csv_sha256=None)
        with self.assertRaisesRegex(ValueError, 'no recorded data hash'):
            _prices('ABC', self.config, self.directory.name, False)

    def test_provider_metadata_must_cover_requested_range(self):
        self.metadata(requested_end_exclusive=str(self.data.index[-10].date()))
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            _prices('ABC', self.config, self.directory.name, False)

    def test_truncated_manual_csv_does_not_shorten_test_silently(self):
        save_prices(self.data.iloc[:-20], self.path)
        with self.assertRaisesRegex(ValueError, 'over 7 calendar days'):
            _prices('ABC', self.config, self.directory.name, False)

    def test_manual_csv_without_test_period_is_rejected(self):
        save_prices(self.data.iloc[:60], self.path)
        with self.assertRaisesRegex(ValueError, 'no requested test-period'):
            _prices('ABC', self.config, self.directory.name, False)

    def test_valid_manual_csv_is_not_claimed_as_provider_data(self):
        _, source = _prices('ABC', self.config, self.directory.name, False)
        self.assertEqual(source, 'user_supplied_csv')

    def test_refresh_replaces_stale_cache_and_writes_both_hashes(self):
        self.metadata(ticker='WRONG')
        with patch('forecasting.cli.download_prices', return_value=self.data) as download:
            _, source = _prices('ABC', self.config, self.directory.name, True)
        download.assert_called_once_with('ABC', self.config['start'], self.config['end'])
        self.assertEqual(source, 'yahoo_finance')
        snapshot = json.loads(self.path.with_suffix('.metadata.json').read_text())
        self.assertTrue(snapshot['csv_sha256'])
        self.assertTrue(snapshot['canonical_csv_sha256'])


if __name__ == '__main__':
    unittest.main()
