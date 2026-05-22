"""
Statistical validation tests for strategy results.

Includes:
  - Paired t-test / Wilcoxon on fold returns
  - Ljung-Box autocorrelation test on residuals
  - ARCH LM test for volatility clustering
  - Kupiec POF test for VaR backtesting
  - White's Reality Check (re-exported from metrics)
  - DSR (Deflated Sharpe Ratio, re-exported from metrics)
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import List

from python.backtest.metrics import (
    ttest_significance,
    deflated_sharpe_ratio,
    whites_reality_check,
)


__all__ = [
    "ttest_significance",
    "deflated_sharpe_ratio",
    "whites_reality_check",
    "ljung_box_test",
    "arch_lm_test",
    "kupiec_pof_test",
    "walk_forward_t_test",
    "bootstrap_confidence_interval",
]


def ljung_box_test(
    residuals: np.ndarray,
    lags: int = 10,
) -> tuple[float, float]:
    """
    Ljung-Box portmanteau test for autocorrelation in *residuals*.

    H0: no autocorrelation up to *lags* lags.

    Returns (Q_stat, p_value).
    """
    r = np.asarray(residuals, dtype=np.float64)
    n = len(r)
    if n < lags + 2:
        return 0.0, 1.0

    # Autocorrelations
    mu = r.mean()
    r_demeaned = r - mu
    var = float(np.sum(r_demeaned ** 2))
    if var < 1e-12:
        return 0.0, 1.0

    acf = np.array([
        float(np.sum(r_demeaned[k:] * r_demeaned[:n - k])) / var
        for k in range(1, lags + 1)
    ])

    Q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    p_value = float(stats.chi2.sf(Q, df=lags))
    return float(Q), p_value


def arch_lm_test(
    returns: np.ndarray,
    lags: int = 5,
) -> tuple[float, float]:
    """
    ARCH LM test for volatility clustering.

    Regresses squared returns on lagged squared returns.
    H0: no ARCH effects.

    Returns (LM_stat, p_value).
    """
    r = np.asarray(returns, dtype=np.float64)
    r_sq = r ** 2
    n = len(r_sq)
    if n < lags + 5:
        return 0.0, 1.0

    y = r_sq[lags:]
    X_cols = [np.ones(n - lags)]
    for lag in range(1, lags + 1):
        X_cols.append(r_sq[lags - lag : n - lag])
    X = np.column_stack(X_cols)

    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ coeffs
    resid = y - y_hat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_sq_val = 1.0 - ss_res / max(ss_tot, 1e-12)

    lm_stat = (n - lags) * r_sq_val
    p_value = float(stats.chi2.sf(lm_stat, df=lags))
    return float(lm_stat), p_value


def kupiec_pof_test(
    returns: np.ndarray,
    var_estimates: np.ndarray,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """
    Kupiec (1995) Proportion of Failures (POF) test for VaR.

    Tests whether the observed exception rate equals the expected rate.

    Parameters
    ----------
    returns       : array of realised daily returns
    var_estimates : array of VaR estimates (positive values represent losses)
    confidence    : VaR confidence level (default 0.99 → expect 1% exceptions)

    Returns
    -------
    (LR_stat, p_value)
    """
    r = np.asarray(returns, dtype=np.float64)
    var_arr = np.asarray(var_estimates, dtype=np.float64)
    n = len(r)

    # Exceptions: actual loss exceeds VaR
    exceptions = r < -var_arr
    x = int(exceptions.sum())
    alpha = 1.0 - confidence  # expected exception rate

    if x == 0:
        x = 0
        lr = -2.0 * (n * np.log(1 - alpha) + 0 * np.log(alpha + 1e-12))
    elif x == n:
        lr = -2.0 * (0 + n * np.log(alpha + 1e-12) - n * np.log(x / n))
    else:
        p_hat = x / n
        lr = -2.0 * (
            x * np.log(alpha / p_hat) + (n - x) * np.log((1 - alpha) / (1 - p_hat))
        )

    p_value = float(stats.chi2.sf(abs(lr), df=1))
    return float(lr), p_value


def walk_forward_t_test(
    fold_sharpes: List[float],
) -> tuple[float, float]:
    """
    t-test on walk-forward fold Sharpe ratios.

    H0: mean fold Sharpe = 0 (strategy has no edge).

    Returns (t_stat, p_value).
    """
    arr = np.asarray(fold_sharpes, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return 0.0, 1.0
    t_stat = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(n)))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    return t_stat, p_value


def bootstrap_confidence_interval(
    metric_values: np.ndarray,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for any scalar metric.

    Parameters
    ----------
    metric_values : array of per-fold metric values
    n_bootstrap   : number of bootstrap replications
    ci_level      : confidence level (default 0.95)

    Returns
    -------
    (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(metric_values)
    if n < 2:
        v = float(metric_values[0]) if n == 1 else 0.0
        return v, v

    boot_means = np.array([
        np.mean(rng.choice(metric_values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper
