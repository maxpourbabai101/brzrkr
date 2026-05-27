"""Volatility modelling: GARCH(1,1) + IV‑surface interpolation.

Two responsibilities:

* :class:`Garch11` — maximum‑likelihood fit of a standard GARCH(1,1) on
  daily log returns, exposing 1‑step‑ahead forecast plus an N‑day
  variance forecast.
* :class:`IVSurface` — bivariate cubic‑spline interpolation over
  (log‑moneyness, time‑to‑expiry) from an options chain DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------
@dataclass
class GarchParams:
    omega: float
    alpha: float
    beta: float
    log_likelihood: float


class Garch11:
    """Plain GARCH(1,1): ``sigma_t^2 = omega + alpha*r_{t-1}^2 + beta*sigma_{t-1}^2``."""

    def __init__(self) -> None:
        self.params: GarchParams | None = None
        self._last_variance: float | None = None

    @staticmethod
    def _negloglik(theta: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
            return 1e10
        var = np.empty_like(returns)
        var[0] = np.var(returns)
        for t in range(1, len(returns)):
            var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
        # Penalize non‑positive variance.
        if np.any(var <= 0):
            return 1e10
        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + returns ** 2 / var)
        return -ll

    def fit(self, returns: Sequence[float]) -> GarchParams:
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        if r.size < 50:
            raise ValueError("Need at least 50 return observations to fit GARCH(1,1).")

        x0 = np.array([np.var(r) * 0.05, 0.05, 0.9])
        bounds = [(1e-10, None), (1e-6, 1.0), (1e-6, 1.0)]
        res = minimize(self._negloglik, x0, args=(r,), method="L-BFGS-B", bounds=bounds)
        if not res.success:
            logger.warning("GARCH fit did not fully converge: %s", res.message)

        omega, alpha, beta = res.x
        self.params = GarchParams(omega, alpha, beta, -res.fun)

        # Cache the final conditional variance for forecasting.
        var = np.var(r)
        for t in range(1, len(r)):
            var = omega + alpha * r[t - 1] ** 2 + beta * var
        self._last_variance = float(var)
        return self.params

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Return forecasted conditional variances for the next `horizon` steps."""
        if self.params is None or self._last_variance is None:
            raise RuntimeError("Garch11 must be fit before forecasting.")
        omega, alpha, beta = self.params.omega, self.params.alpha, self.params.beta
        unconditional = omega / max(1 - alpha - beta, 1e-9)
        out = np.empty(horizon)
        var = self._last_variance
        for h in range(horizon):
            var = omega + (alpha + beta) * var
            out[h] = var
        # Long‑horizon variance reverts toward unconditional level.
        return np.minimum(out, unconditional * 5.0)


# ---------------------------------------------------------------------------
# IV surface
# ---------------------------------------------------------------------------
class IVSurface:
    """Bivariate spline interpolator over (log‑moneyness, T)."""

    def __init__(self, smoothing: float = 0.0) -> None:
        self.smoothing = smoothing
        self._spline: SmoothBivariateSpline | None = None

    def fit(
        self,
        options: pd.DataFrame,
        spot_price: float,
        *,
        moneyness_col: str = "strike",
        expiry_col: str = "expiry",
        iv_col: str = "iv",
    ) -> "IVSurface":
        df = options.dropna(subset=[moneyness_col, expiry_col, iv_col]).copy()
        if df.empty:
            raise ValueError("No valid rows in options DataFrame to fit IV surface.")
        df["log_moneyness"] = np.log(df[moneyness_col].astype(float) / float(spot_price))
        # Time to expiry in years (ACT/365).
        now = pd.Timestamp.utcnow()
        df["T"] = (pd.to_datetime(df[expiry_col], utc=True) - now).dt.total_seconds() / (365 * 86400)
        df = df[df["T"] > 0]

        if df.shape[0] < 16:
            raise ValueError("Need at least 16 valid quotes to fit a 2D spline.")

        self._spline = SmoothBivariateSpline(
            df["log_moneyness"].values,
            df["T"].values,
            df[iv_col].astype(float).values,
            s=self.smoothing,
        )
        return self

    def __call__(self, log_moneyness: float, t: float) -> float:
        if self._spline is None:
            raise RuntimeError("IVSurface must be fit before evaluation.")
        return float(self._spline.ev(log_moneyness, t))

    def grid(
        self,
        log_moneyness: Sequence[float],
        t: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate on a regular grid; returns (X, Y, Z)."""
        if self._spline is None:
            raise RuntimeError("IVSurface must be fit before evaluation.")
        X, Y = np.meshgrid(log_moneyness, t)
        Z = self._spline.ev(X.ravel(), Y.ravel()).reshape(X.shape)
        return X, Y, Z
