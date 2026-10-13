"""Availability-boundary tests plus small, real model fits."""

import numpy as np
import pandas as pd
import pytest

from forecasting.models import Standardizer, fit_arima_predict, fit_bilstm_predict
from forecasting.walk_forward import forecast_latest, make_training_samples, run_walk_forward


def example_data(n=80):
    index = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0003, 0.01, n)
    data = pd.DataFrame({"close": 100 * np.exp(np.cumsum(returns))}, index=index)
    features = pd.DataFrame({"return": returns, "trend": np.arange(n) / n}, index=index)
    return data, features


def test_every_training_input_and_label_stays_inside_window():
    data, features = example_data()
    samples = make_training_samples(
        data, features, origin_pos=24, train_window=20, lookback=3, horizon=4
    )
    assert samples.window_start == 5
    np.testing.assert_array_equal(samples.end_positions, np.arange(7, 21))
    assert samples.input_start == 5
    assert samples.input_cutoff == 20
    assert samples.label_cutoff == 24
    assert samples.x.shape == (14, 3, 2)
    assert samples.y.shape == (14, 4)
    np.testing.assert_allclose(
        samples.y[-1], np.log(data.close.iloc[21:25].values / data.close.iloc[20])
    )
    # Entire rolling window advances, including the earliest feature window.
    later = make_training_samples(
        data, features, origin_pos=30, train_window=20, lookback=3, horizon=4
    )
    assert later.window_start == later.input_start == 11
    assert later.label_cutoff == 30


def test_future_mutations_cannot_change_training_samples():
    data, features = example_data()
    kwargs = dict(origin_pos=40, train_window=30, lookback=5, horizon=5)
    before = make_training_samples(data, features, **kwargs)
    altered_data, altered_features = data.copy(), features.copy()
    altered_data.iloc[41:, 0] *= 50
    altered_features.iloc[41:] = 1e8
    after = make_training_samples(altered_data, altered_features, **kwargs)
    np.testing.assert_array_equal(before.x, after.x)
    np.testing.assert_array_equal(before.y, after.y)


def test_nonfinite_training_windows_are_removed_without_backfill():
    data, features = example_data()
    features.iloc[8, 0] = np.nan
    samples = make_training_samples(
        data, features, origin_pos=24, train_window=20, lookback=3, horizon=4
    )
    assert not {8, 9, 10}.intersection(samples.end_positions)
    assert 7 in samples.end_positions
    assert 11 in samples.end_positions
    assert np.isfinite(samples.x).all()


def test_zero_backtest_uses_actual_future_targets_and_records_cutoffs():
    data, features = example_data(40)
    predictions, audit = run_walk_forward(
        data, features, model="zero", train_window=20, lookback=4,
        horizon=3, stride=4, test_start=str(data.index[29].date()),
    )
    assert len(audit) == 2
    assert len(predictions) == 6
    assert (predictions.predicted_log_return == 0).all()
    np.testing.assert_array_equal(audit.origin.values, data.index[[29, 33]].values)
    assert (audit.training_label_cutoff <= audit.origin).all()
    assert (audit.training_input_cutoff < audit.origin).all()
    assert (audit.prediction_input_cutoff == audit.origin).all()
    first = predictions.iloc[0]
    assert first.target_date == data.index[30]
    assert first.actual_log_return == pytest.approx(np.log(data.close.iloc[30] / data.close.iloc[29]))


def test_latest_forecast_needs_no_future_bars():
    data, features = example_data(25)
    predictions, audit = forecast_latest(
        data, features, model="zero", train_window=20, lookback=3, horizon=4
    )
    assert list(predictions.columns) == ["origin", "horizon", "predicted_log_return", "model"]
    assert len(predictions) == 4
    assert (predictions.origin == data.index[-1]).all()
    assert audit.training_label_cutoff.iloc[0] == data.index[-1]
    assert audit.training_input_cutoff.iloc[0] == data.index[-5]


def test_invalid_inputs_fail_explicitly():
    data, features = example_data(40)
    with pytest.raises(ValueError, match="exactly the same date index"):
        forecast_latest(data, features.iloc[1:], train_window=20, lookback=3)
    with pytest.raises(ValueError, match="positive integer"):
        run_walk_forward(data, features, stride=0)
    with pytest.raises(ValueError, match="No evaluable origins"):
        run_walk_forward(data, features, train_window=20, lookback=3, test_start="2030-01-01")
    features.iloc[-1, 0] = np.inf
    with pytest.raises(ValueError, match="Nonfinite prediction features"):
        forecast_latest(data, features, model="zero", train_window=20, lookback=3)


def test_standardizer_is_fitted_on_supplied_training_data_only():
    train = np.array([[1.0, 7.0], [3.0, 7.0]])
    scaler = Standardizer.fit(train)
    np.testing.assert_array_equal(scaler.mean, [2.0, 7.0])
    np.testing.assert_array_equal(scaler.scale, [1.0, 1.0])
    held_out = np.array([[1000.0, 7.0]])
    np.testing.assert_array_equal(scaler.transform(held_out), [[998.0, 0.0]])
    np.testing.assert_array_equal(scaler.inverse_transform(scaler.transform(train)), train)


def test_real_bilstm_fit_shape_seed_and_training_only_scaling():
    pytest.importorskip("torch")
    rng = np.random.default_rng(23)
    x = rng.normal(size=(12, 4, 2))
    y = np.column_stack([x[:, -1, 0] * 0.01, x[:, -1, 1] * 0.02])
    kwargs = dict(epochs=3, hidden_size=4, seed=19)
    first = fit_bilstm_predict(x, y, x[-1], **kwargs)
    repeated = fit_bilstm_predict(x, y, x[-1], **kwargs)
    held_out_changed = fit_bilstm_predict(x, y, x[-1] + 50, **kwargs)
    assert first.prediction.shape == (2,)
    assert np.isfinite(first.prediction).all()
    np.testing.assert_array_equal(first.prediction, repeated.prediction)
    # Changing prediction inputs cannot alter training scaling or training loss.
    assert first.diagnostics["train_loss"] == held_out_changed.diagnostics["train_loss"]
    assert first.diagnostics["train_loss"] < first.diagnostics["initial_train_loss"]


def test_real_arima_fit_has_finite_cumulative_forecast_and_diagnostics():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(7)
    daily_returns = rng.normal(0.002, 0.01, size=100)
    result = fit_arima_predict(daily_returns, 3, orders=((0, 0, 0), (1, 0, 0)))
    assert result.prediction.shape == (3,)
    assert np.isfinite(result.prediction).all()
    assert result.diagnostics["converged"] is True
    assert np.isfinite(result.diagnostics["bic"])
    assert result.diagnostics["candidate_count"] == 2
    assert result.diagnostics["successful_candidates"] >= 1


@pytest.mark.parametrize("model, dependency", [("bilstm", "torch"), ("arima", "statsmodels")])
def test_real_backtest_matches_latest_on_truncated_history(model, dependency):
    pytest.importorskip(dependency)
    data, features = example_data(48)
    kwargs = dict(model=model, train_window=30, lookback=4, horizon=3, epochs=2, hidden_size=4)
    backtest, backtest_audit = run_walk_forward(
        data, features, test_start=str(data.index[40].date()), stride=100, **kwargs
    )
    latest, latest_audit = forecast_latest(data.iloc[:41], features.iloc[:41], **kwargs)
    np.testing.assert_array_equal(backtest.predicted_log_return, latest.predicted_log_return)
    pd.testing.assert_frame_equal(backtest_audit, latest_audit)
    assert (latest_audit.training_label_cutoff <= latest_audit.origin).all()


def test_arima_all_candidate_failures_raise_instead_of_returning_baseline(monkeypatch):
    arima_module = pytest.importorskip("statsmodels.tsa.arima.model")

    class BrokenARIMA:
        def __init__(self, *args, **kwargs):
            raise ValueError("intentional fit failure")

    monkeypatch.setattr(arima_module, "ARIMA", BrokenARIMA)
    with pytest.raises(RuntimeError, match="All ARIMA candidates failed"):
        fit_arima_predict(np.linspace(-0.01, 0.01, 20), 2)
