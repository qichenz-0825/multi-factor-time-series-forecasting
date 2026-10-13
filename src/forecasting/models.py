"""Small, auditable forecasting models with training-only preprocessing.

Heavy model libraries are imported only when the respective model is requested.
The neural model is bidirectional *within observed historical inputs*: it never
receives future observations, targets, or covariates as an input sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import warnings
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastFit:
    prediction: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class Standardizer:
    """Scikit-Learn scaler, fitted exclusively on supplied training data.

    The wrapper preserves sequence shapes while StandardScaler operates on
    two-dimensional feature/target matrices.
    """

    _scaler: Any

    @property
    def mean(self) -> np.ndarray:
        return self._scaler.mean_

    @property
    def scale(self) -> np.ndarray:
        return self._scaler.scale_

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardizer:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Standardizer needs a nonempty, finite 2D training array.")
        from sklearn.preprocessing import StandardScaler

        return cls(StandardScaler().fit(values))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return self._scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return self._scaler.inverse_transform(values.reshape(-1, values.shape[-1])).reshape(values.shape)


def fit_bilstm_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    prediction_x: np.ndarray,
    *,
    epochs: int = 10,
    hidden_size: int = 32,
    seed: int = 42,
) -> ForecastFit:
    """Fit a direct multi-horizon BiLSTM from scratch and forecast one origin.

    Shapes are (samples, lookback, features), (samples, horizon), and
    (lookback, features). Both scalers use only the supplied training arrays.
    Full-batch CPU optimization and deterministic algorithms make a fixed seed
    reproducible within the same PyTorch/platform version.
    """
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    prediction_x = np.asarray(prediction_x, dtype=np.float64)
    if train_x.ndim != 3 or train_y.ndim != 2 or prediction_x.ndim != 2:
        raise ValueError("Expected X=(N,L,F), y=(N,H), prediction X=(L,F).")
    if (
        len(train_x) < 2
        or len(train_x) != len(train_y)
        or prediction_x.shape != train_x.shape[1:]
        or min(train_x.shape[1:]) < 1
        or train_y.shape[1] < 1
    ):
        raise ValueError("Incompatible training shapes or fewer than two samples.")
    if not all(np.isfinite(a).all() for a in (train_x, train_y, prediction_x)):
        raise ValueError("BiLSTM inputs and training labels must all be finite.")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer.")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("BiLSTM requires PyTorch; install the project requirements.") from exc

    x_scaler = Standardizer.fit(train_x.reshape(-1, train_x.shape[-1]))
    y_scaler = Standardizer.fit(train_y)
    x = torch.from_numpy(x_scaler.transform(train_x).astype(np.float32))
    y = torch.from_numpy(y_scaler.transform(train_y).astype(np.float32))
    x_next = torch.from_numpy(
        x_scaler.transform(prediction_x)[None, ...].astype(np.float32)
    )

    class DirectBiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=train_x.shape[-1],
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.output = nn.Linear(2 * hidden_size, train_y.shape[-1])

        def forward(self, values):
            _, (hidden, _) = self.lstm(values)
            # Both final states summarize the entire observed lookback window.
            return self.output(torch.cat((hidden[-2], hidden[-1]), dim=1))

    # Restore process-global settings so a library call does not change its host.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_threads = torch.get_num_threads()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        network = DirectBiLSTM().cpu()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        network.train()
        initial_loss = None
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("BiLSTM training produced a nonfinite loss.")
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
        network.eval()
        with torch.no_grad():
            final_loss = float(criterion(network(x), y))
            standardized_prediction = network(x_next).numpy()[0]
        prediction = y_scaler.inverse_transform(standardized_prediction)
        if not np.isfinite(prediction).all() or not np.isfinite(final_loss):
            raise RuntimeError("BiLSTM produced nonfinite predictions or final loss.")
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        torch.use_deterministic_algorithms(previous_determinism, warn_only=previous_warn_only)
        torch.set_num_threads(previous_threads)

    return ForecastFit(
        prediction=np.asarray(prediction, dtype=np.float64),
        diagnostics={
            "epochs": epochs,
            "hidden_size": hidden_size,
            "seed": seed,
            "initial_train_loss": initial_loss,
            "train_loss": final_loss,
            "loss_scale": "standardized_targets",
            "optimizer": "Adam(lr=0.001)",
            "device": "cpu",
            "status": "fixed_epochs_completed",
        },
    )


def fit_arima_predict(
    log_returns: np.ndarray,
    horizon: int,
    *,
    orders: tuple[tuple[int, int, int], ...] = (
        (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2)
    ),
) -> ForecastFit:
    """Select a converged ARIMA by training-only BIC and sum daily forecasts.

    The fixed candidate grid is applied to daily close log returns, hence d=0.
    Failed/nonconverged candidates are recorded, not silently substituted. If
    every candidate fails, this raises instead of claiming a successful fit.
    """
    log_returns = np.asarray(log_returns, dtype=np.float64)
    if log_returns.ndim != 1 or len(log_returns) < 10 or not np.isfinite(log_returns).all():
        raise ValueError("ARIMA needs at least 10 finite daily log returns.")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    if not orders:
        raise ValueError("At least one ARIMA candidate order is required.")
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("ARIMA requires statsmodels; install the project requirements.") from exc

    best_result = None
    best_order = None
    best_bic = np.inf
    failures: list[str] = []
    recorded_warnings: list[str] = []
    successful_candidates = 0
    for order in orders:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = ARIMA(
                    log_returns,
                    order=order,
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit()
                converged = bool(result.mle_retvals.get("converged", False))
                bic = float(result.bic)
                if not converged or not np.isfinite(bic):
                    failures.append(f"{order}: converged={converged}, BIC={bic}")
                    continue
                successful_candidates += 1
                if bic < best_bic:
                    best_result, best_order, best_bic = result, order, bic
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                failures.append(f"{order}: {type(exc).__name__}: {exc}")
            finally:
                recorded_warnings.extend(f"{order}: {item.message}" for item in caught)
    if best_result is None:
        raise RuntimeError("All ARIMA candidates failed: " + "; ".join(failures))
    prediction = np.cumsum(np.asarray(best_result.forecast(steps=horizon), dtype=np.float64))
    if prediction.shape != (horizon,) or not np.isfinite(prediction).all():
        raise RuntimeError("Selected ARIMA produced invalid forecasts.")
    return ForecastFit(
        prediction=prediction,
        diagnostics={
            "order": str(best_order),
            "bic": best_bic,
            "converged": True,
            "candidate_count": len(orders),
            "successful_candidates": successful_candidates,
            "candidate_failures": "; ".join(failures),
            "fit_warnings": "; ".join(dict.fromkeys(recorded_warnings)),
            "status": "converged",
        },
    )
