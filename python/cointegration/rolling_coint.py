"""
Rolling cointegration with pair classification.

Classifies pairs into regimes over a rolling window:
  - COINTEGRATED: p_EG < threshold and Johansen rank >= 1
  - WEAKLY_COINTEGRATED: only one test passes
  - BROKEN: neither test passes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from python.cointegration.engle_granger import engle_granger_test, EGResult
from python.cointegration.johansen import johansen_test, JohansenResult


class PairStatus(str, Enum):
    COINTEGRATED = "COINTEGRATED"
    WEAKLY_COINTEGRATED = "WEAKLY_COINTEGRATED"
    BROKEN = "BROKEN"


@dataclass
class RollingCointResult:
    """Rolling cointegration result for one pair at one date."""
    date: pd.Timestamp
    sym_y: str
    sym_x: str
    eg_pvalue: float
    johansen_rank: int
    status: PairStatus
    beta_ols: float
    half_life: float          # estimated mean-reversion half-life in bars
    hurst_exp: float          # Hurst exponent of residuals
    spread_vol: float         # rolling spread volatility


@dataclass
class PairTimeSeries:
    """Full rolling time series for one pair."""
    sym_y: str
    sym_x: str
    history: List[RollingCointResult] = field(default_factory=list)

    def latest_status(self) -> PairStatus:
        if not self.history:
            return PairStatus.BROKEN
        return self.history[-1].status

    def status_series(self) -> pd.Series:
        dates = [r.date for r in self.history]
        vals = [r.status.value for r in self.history]
        return pd.Series(vals, index=dates)


def estimate_half_life(residuals: np.ndarray) -> float:
    """
    Estimate mean-reversion half-life via OLS on AR(1):
      Δε_t = λ * ε_{t-1} + noise
    half-life = -ln(2) / ln(1 + λ)
    """
    y = np.diff(residuals)
    x = residuals[:-1]
    if len(x) < 5:
        return np.inf
    X = np.column_stack([np.ones(len(x)), x])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    lam = coeffs[1]
    if lam >= 0:
        return np.inf  # no mean reversion
    return float(-np.log(2) / np.log(1 + lam))


def hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.
    H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending.
    """
    n = len(ts)
    max_lag = min(max_lag, n // 2)
    lags = range(2, max_lag)
    rs_vals = []

    for lag in lags:
        sub = ts[:lag]
        mean_sub = np.mean(sub)
        deviation = np.cumsum(sub - mean_sub)
        R = deviation.max() - deviation.min()
        S = np.std(sub, ddof=1)
        if S > 1e-10:
            rs_vals.append(R / S)
        else:
            rs_vals.append(np.nan)

    rs_arr = np.array(rs_vals)
    valid = ~np.isnan(rs_arr) & (rs_arr > 0)
    if valid.sum() < 3:
        return 0.5

    log_lags = np.log(list(lags))[valid]
    log_rs = np.log(rs_arr[valid])
    slope = np.polyfit(log_lags, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


class RollingCointegrationEngine:
    """
    Runs rolling Engle-Granger + Johansen tests over a sliding window.

    Parameters
    ----------
    window       : number of bars in each test window
    step         : step size (bars) between tests
    eg_threshold : p-value threshold for EG test
    """

    def __init__(
        self,
        window: int = 252,
        step: int = 21,
        eg_threshold: float = 0.05,
    ) -> None:
        self.window = window
        self.step = step
        self.eg_threshold = eg_threshold

    def classify(
        self,
        eg_p: float,
        johansen_rank: int,
    ) -> PairStatus:
        eg_ok = eg_p <= self.eg_threshold
        joh_ok = johansen_rank >= 1
        if eg_ok and joh_ok:
            return PairStatus.COINTEGRATED
        if eg_ok or joh_ok:
            return PairStatus.WEAKLY_COINTEGRATED
        return PairStatus.BROKEN

    def run_pair(
        self,
        prices_y: pd.Series,
        prices_x: pd.Series,
    ) -> PairTimeSeries:
        """Roll the window over the full price history for one pair."""
        sym_y = str(prices_y.name or "Y")
        sym_x = str(prices_x.name or "X")

        # Align
        df = pd.DataFrame({"y": prices_y, "x": prices_x}).dropna()
        log_y = np.log(df["y"].values)
        log_x = np.log(df["x"].values)
        dates = df.index

        result_ts = PairTimeSeries(sym_y=sym_y, sym_x=sym_x)
        n = len(df)

        for end in range(self.window, n + 1, self.step):
            start = end - self.window
            y_w = log_y[start:end]
            x_w = log_x[start:end]

            # EG test
            try:
                eg: EGResult = engle_granger_test(y_w, x_w)
                eg_p = eg.p_value
                beta_ols = eg.beta
                residuals = eg.residuals
            except Exception:  # noqa: BLE001
                continue

            # Johansen test
            try:
                data_2d = np.column_stack([y_w, x_w])
                joh: JohansenResult = johansen_test(data_2d, lags=1)
                joh_rank = joh.r_trace
            except Exception:  # noqa: BLE001
                joh_rank = 0

            status = self.classify(eg_p, joh_rank)
            hl = estimate_half_life(residuals)
            hurst = hurst_exponent(residuals)
            spread_vol = float(np.std(residuals))

            result_ts.history.append(
                RollingCointResult(
                    date=dates[end - 1],
                    sym_y=sym_y,
                    sym_x=sym_x,
                    eg_pvalue=eg_p,
                    johansen_rank=joh_rank,
                    status=status,
                    beta_ols=beta_ols,
                    half_life=hl,
                    hurst_exp=hurst,
                    spread_vol=spread_vol,
                )
            )

        return result_ts

    def run_all_pairs(
        self,
        price_data: Dict[str, pd.Series],
        pairs: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], PairTimeSeries]:
        """Run rolling cointegration for a list of (sym_y, sym_x) pairs."""
        results: Dict[Tuple[str, str], PairTimeSeries] = {}
        for sym_y, sym_x in pairs:
            if sym_y not in price_data or sym_x not in price_data:
                continue
            ts = self.run_pair(price_data[sym_y], price_data[sym_x])
            results[(sym_y, sym_x)] = ts
        return results

    def currently_active_pairs(
        self,
        results: Dict[Tuple[str, str], PairTimeSeries],
        require_status: PairStatus = PairStatus.COINTEGRATED,
    ) -> List[Tuple[str, str]]:
        """Return pairs whose most recent status matches *require_status*."""
        return [
            pair
            for pair, ts in results.items()
            if ts.latest_status() == require_status
        ]
