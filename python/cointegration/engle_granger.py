"""
Engle-Granger 2-step cointegration test with MacKinnon (2010) p-values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EGResult:
    beta: float           # OLS hedge ratio
    alpha: float          # OLS intercept
    adf_stat: float       # ADF test statistic on residuals
    p_value: float        # MacKinnon p-value
    n_lags: int           # AIC-selected lag count
    residuals: np.ndarray


def ols_regression(
    y: np.ndarray, x: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """OLS: y = alpha + beta*x + e.  Returns (alpha, beta, residuals)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    residuals = y - alpha - beta * x
    return alpha, beta, residuals


def adf_aic_lags(residuals: np.ndarray, max_lags: int = 12) -> int:
    """Select ADF lag count by AIC (Akaike Information Criterion)."""
    best_aic = np.inf
    best_k = 0

    for k in range(0, max_lags + 1):
        dy = np.diff(residuals)
        y_dep = dy[k:]
        cols = [residuals[k : len(residuals) - 1]]  # ε_{t-1}
        for i in range(1, k + 1):
            cols.append(dy[k - i : len(dy) - i])    # Δε_{t-i}
        X = np.column_stack(cols)

        n_obs = len(y_dep)
        if n_obs < k + 5:
            continue

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y_dep, rcond=None)
            resid = y_dep - X @ coeffs
            sigma2 = np.var(resid)
            if sigma2 <= 0:
                continue
            aic = n_obs * np.log(sigma2) + 2 * (k + 1)
            if aic < best_aic:
                best_aic = aic
                best_k = k
        except np.linalg.LinAlgError:
            continue

    return best_k


def _adf_statistic(
    residuals: np.ndarray, k: int
) -> tuple[float, float]:
    """Compute ADF t-statistic and SE for a given lag order *k*."""
    dy = np.diff(residuals)
    y_dep = dy[k:]
    cols = [residuals[k : len(residuals) - 1]]
    for i in range(1, k + 1):
        cols.append(dy[k - i : len(dy) - i])
    X = np.column_stack(cols)

    n_obs = len(y_dep)
    coeffs, _, _, _ = np.linalg.lstsq(X, y_dep, rcond=None)
    resid = y_dep - X @ coeffs
    sigma2 = np.sum(resid ** 2) / max(n_obs - len(coeffs), 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_gamma = np.sqrt(max(sigma2 * XtX_inv[0, 0], 1e-16))
    adf_stat = coeffs[0] / se_gamma
    return float(adf_stat), float(se_gamma)


def adf_test(
    residuals: np.ndarray, max_lags: int = 12
) -> tuple[float, float, int]:
    """ADF test on residuals.  Returns (stat, p_value, n_lags)."""
    k = adf_aic_lags(residuals, max_lags)
    adf_stat, _ = _adf_statistic(residuals, k)
    n_obs = len(residuals) - 1 - k
    p_value = mackinnon_pvalue(adf_stat, max(n_obs, 1))
    return adf_stat, p_value, k


def mackinnon_pvalue(tau: float, n: int) -> float:
    """
    MacKinnon (1991 / 2010) response-surface p-value for ADF test.

    Response surface:  c_p(T) = beta_inf + beta_1/T + beta_2/T^2
    Coefficients from MacKinnon (1991) Table B.6 (no constant, 1 variable).
    """
    # [beta_inf, beta_1, beta_2] for p = 1%, 5%, 10%
    cv_params = {
        0.01: [-3.4335, -5.999, -29.25],
        0.05: [-2.8621, -2.738, -8.36],
        0.10: [-2.5671, -1.438, -4.48],
    }

    def cv(p: float) -> float:
        b = cv_params[p]
        return b[0] + b[1] / n + b[2] / n ** 2

    cv_1pct = cv(0.01)
    cv_5pct = cv(0.05)
    cv_10pct = cv(0.10)

    if tau < cv_1pct:
        slope = (0.05 - 0.01) / (cv_5pct - cv_1pct)
        return max(0.001, 0.01 + slope * (tau - cv_1pct))
    elif tau < cv_5pct:
        t = (tau - cv_1pct) / (cv_5pct - cv_1pct)
        return 0.01 + t * 0.04
    elif tau < cv_10pct:
        t = (tau - cv_5pct) / (cv_10pct - cv_5pct)
        return 0.05 + t * 0.05
    else:
        slope = 0.05 / max(abs(cv_10pct - cv_5pct), 1e-8)
        return min(0.99, 0.10 + slope * abs(tau - cv_10pct))


def engle_granger_test(
    y: np.ndarray,
    x: np.ndarray,
    p_threshold: float = 0.01,
) -> EGResult:
    """Full Engle-Granger 2-step cointegration test."""
    alpha, beta, residuals = ols_regression(y, x)
    adf_stat, p_value, n_lags = adf_test(residuals)
    return EGResult(
        beta=beta,
        alpha=alpha,
        adf_stat=adf_stat,
        p_value=p_value,
        n_lags=n_lags,
        residuals=residuals,
    )
