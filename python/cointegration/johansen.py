"""
Johansen trace and maximum-eigenvalue cointegration tests.

Implements the full VECM framework from scratch using numpy only.
Critical values are embedded from Osterwald-Lenum (1992) / MacKinnon-Haug-Michelis (1999).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class JohansenResult:
    """Results of the Johansen cointegration test."""
    n_vars: int                    # number of variables
    r_trace: int                   # rank from trace test (5% level)
    r_max_eigen: int               # rank from max-eigenvalue test
    trace_stats: np.ndarray        # trace statistics for r=0,1,...,n-1
    max_eigen_stats: np.ndarray    # max-eigenvalue statistics
    trace_cv_90: np.ndarray        # 90% critical values (trace)
    trace_cv_95: np.ndarray        # 95% critical values (trace)
    trace_cv_99: np.ndarray        # 99% critical values (trace)
    max_eigen_cv_95: np.ndarray    # 95% critical values (max-eigen)
    eigenvalues: np.ndarray        # ordered eigenvalues (largest first)
    eigenvectors: np.ndarray       # cointegrating vectors (columns)
    beta: np.ndarray               # normalised cointegrating matrix (r x n)
    alpha: np.ndarray              # adjustment speed matrix (n x r)
    p_values_trace: np.ndarray     # approximate p-values for trace test


# ---------------------------------------------------------------------------
# Critical values — Osterwald-Lenum (1992), no deterministic trend (case I)
# Table rows: r=0,1,2,...,n-1   for n variables
# ---------------------------------------------------------------------------

# Trace test critical values indexed by (p - r) where p = n_vars, r = null rank.
# Row i corresponds to (p-r) = i+1.
# Columns: [90%, 95%, 99%]
# Source: Osterwald-Lenum (1992), Table 1, Case I (no deterministic component).
# (p-r=1) to (p-r=10)
_TRACE_CV = np.array([
    [ 6.50,  8.18, 11.65],   # p-r=1
    [12.91, 14.90, 19.19],   # p-r=2  ← n=2, r=0: 95% CV = 14.90 ≈ 15.41 w/ constant
    [18.90, 21.07, 26.79],   # p-r=3
    [24.78, 27.14, 32.99],   # p-r=4
    [30.84, 33.32, 39.89],   # p-r=5
    [36.84, 39.43, 46.53],   # p-r=6
    [42.94, 45.62, 53.12],   # p-r=7
    [48.88, 51.91, 59.70],   # p-r=8
    [54.90, 58.09, 66.23],   # p-r=9
    [60.85, 64.26, 72.72],   # p-r=10
], dtype=np.float64)

# Max-eigenvalue critical values indexed by (p - r).
# Source: Osterwald-Lenum (1992), Table 2, Case I.
_MAX_EIGEN_CV = np.array([
    [ 6.50,  8.18, 11.65],   # p-r=1
    [12.07, 14.07, 18.63],   # p-r=2
    [17.85, 20.16, 25.75],   # p-r=3
    [23.80, 26.23, 32.19],   # p-r=4
    [29.68, 32.46, 38.78],   # p-r=5
    [35.65, 38.77, 45.39],   # p-r=6
    [41.58, 44.91, 51.91],   # p-r=7
    [47.35, 51.07, 58.36],   # p-r=8
    [53.25, 57.12, 64.74],   # p-r=9
    [59.06, 63.17, 71.12],   # p-r=10
], dtype=np.float64)


def _lag_matrix(X: np.ndarray, lags: int) -> np.ndarray:
    """Stack lagged versions of X.  Returns (T-lags, n*lags) matrix."""
    T, n = X.shape
    cols = []
    for lag in range(1, lags + 1):
        cols.append(X[lags - lag : T - lag])
    return np.concatenate(cols, axis=1)


def _reduced_rank_regression(
    dX: np.ndarray, X_lag1: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Johansen's reduced-rank regression (S matrices approach).

    Parameters
    ----------
    dX    : (T, n)  - first differences
    X_lag1: (T, n)  - levels lagged one period
    Z     : (T, k)  - short-run regressors (lagged differences + constant)

    Returns eigenvalues λ, eigenvectors V (cols), and S11^{-1/2}.
    """
    T, n = dX.shape

    # Partial out Z from dX and X_lag1 using OLS
    def _resid(Y: np.ndarray) -> np.ndarray:
        if Z.shape[1] == 0:
            return Y
        coef = np.linalg.lstsq(Z, Y, rcond=None)[0]
        return Y - Z @ coef

    R0 = _resid(dX)       # residuals of dX on Z
    R1 = _resid(X_lag1)   # residuals of X_lag1 on Z

    # Moment matrices
    S00 = R0.T @ R0 / T
    S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T
    S10 = S01.T

    # Solve generalised eigenvalue problem
    # |λ S11 - S10 S00^{-1} S01| = 0
    S00_inv = np.linalg.inv(S00 + 1e-10 * np.eye(n))
    S11_inv = np.linalg.inv(S11 + 1e-10 * np.eye(n))

    M = S11_inv @ S10 @ S00_inv @ S01

    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Keep real parts (should be real for positive semi-definite M)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort descending
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Normalise: beta' S11 beta = I
    _S11_half = np.linalg.cholesky(S11 + 1e-10 * np.eye(n))
    for j in range(n):
        v = eigenvectors[:, j]
        scale = v @ S11 @ v
        if scale > 1e-12:
            eigenvectors[:, j] = v / np.sqrt(scale)

    return eigenvalues, eigenvectors, S11


def johansen_test(
    data: np.ndarray,
    lags: int = 1,
    det_order: int = 0,
) -> JohansenResult:
    """
    Johansen cointegration test.

    Parameters
    ----------
    data      : (T, n) array of price levels
    lags      : number of VAR lags (k-1 in VECM parameterisation)
    det_order : 0 = no constant, 1 = restricted constant, 2 = unrestricted constant

    Returns JohansenResult with trace/max-eigenvalue statistics and critical values.
    """
    T, n = data.shape
    assert n >= 2, "Need at least 2 variables"

    dX = np.diff(data, axis=0)          # (T-1, n)
    X_lag1 = data[:-1]                   # (T-1, n) — levels lagged 1

    T_eff = T - 1
    k = lags

    # Build short-run regressor matrix Z
    z_cols = []
    if k > 1:
        for lag in range(1, k):
            start = k - lag - 1
            end = T_eff - lag
            if start < end:
                z_cols.append(dX[start:end])

    if det_order >= 1:
        z_cols.append(np.ones((T_eff - (k - 1), 1)))

    # Trim all arrays to same length
    trim = k - 1
    dX_t = dX[trim:]
    X_lag1_t = X_lag1[trim:]

    if z_cols:
        Z = np.concatenate(z_cols, axis=1)
        min_len = min(len(dX_t), len(X_lag1_t), len(Z))
        dX_t = dX_t[:min_len]
        X_lag1_t = X_lag1_t[:min_len]
        Z = Z[:min_len]
    else:
        Z = np.zeros((len(dX_t), 0))

    T_eff = len(dX_t)

    eigenvalues, eigenvectors, S11 = _reduced_rank_regression(dX_t, X_lag1_t, Z)

    # Clip eigenvalues to (0, 1) for log computation
    eigenvalues = np.clip(eigenvalues, 1e-10, 1.0 - 1e-10)

    # Trace test statistics
    trace_stats = np.zeros(n)
    for r in range(n):
        trace_stats[r] = -T_eff * np.sum(np.log(1.0 - eigenvalues[r:]))

    # Max-eigenvalue statistics
    max_eigen_stats = -T_eff * np.log(1.0 - eigenvalues)

    # --- Critical values ---
    # CV is indexed by (p - r) = (n - r), i.e., table row index = (n - r - 1).
    max_n = _TRACE_CV.shape[0]

    trace_cv_90 = np.zeros(n)
    trace_cv_95 = np.zeros(n)
    trace_cv_99 = np.zeros(n)
    max_eigen_cv_95 = np.zeros(n)

    for r in range(n):
        p_minus_r = n - r                      # 1-based index into table
        idx = min(p_minus_r - 1, max_n - 1)   # 0-based row index
        trace_cv_90[r]      = _TRACE_CV[idx, 0]
        trace_cv_95[r]      = _TRACE_CV[idx, 1]
        trace_cv_99[r]      = _TRACE_CV[idx, 2]
        max_eigen_cv_95[r]  = _MAX_EIGEN_CV[idx, 1]

    # Determine rank via trace test at 5%
    r_trace = 0
    for r in range(n):
        if trace_stats[r] > trace_cv_95[r]:
            r_trace += 1
        else:
            break

    r_max_eigen = 0
    for r in range(n):
        if max_eigen_stats[r] > max_eigen_cv_95[r]:
            r_max_eigen += 1
        else:
            break

    # Approximate p-values via chi-squared with n-r degrees of freedom
    from scipy import stats as scipy_stats
    p_values_trace = np.zeros(n)
    for r in range(n):
        df = (n - r) ** 2  # approximate DOF for trace statistic
        p_values_trace[r] = float(scipy_stats.chi2.sf(trace_stats[r], df=df))

    # Normalised beta (cointegrating vectors as rows)
    beta_raw = eigenvectors[:, :max(r_trace, 1)]
    # Normalise so first element of each vector = 1
    beta = beta_raw.copy()
    for j in range(beta.shape[1]):
        if abs(beta[0, j]) > 1e-10:
            beta[:, j] /= beta[0, j]

    # Adjustment speeds: alpha = S01 * beta / (beta' S11 beta)
    _S00_inv_half = np.linalg.inv(S11 + 1e-10 * np.eye(n))
    S01 = dX_t.T @ X_lag1_t / T_eff
    alpha = S01 @ beta

    return JohansenResult(
        n_vars=n,
        r_trace=r_trace,
        r_max_eigen=r_max_eigen,
        trace_stats=trace_stats,
        max_eigen_stats=max_eigen_stats,
        trace_cv_90=trace_cv_90,
        trace_cv_95=trace_cv_95,
        trace_cv_99=trace_cv_99,
        max_eigen_cv_95=max_eigen_cv_95,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        beta=beta.T,   # (r, n)
        alpha=alpha,
        p_values_trace=p_values_trace,
    )


def select_pairs_johansen(
    price_matrix: np.ndarray,
    symbols: List[str],
    min_rank: int = 1,
    lags: int = 1,
) -> List[tuple]:
    """
    Test all pairs in *symbols* and return pairs with Johansen rank >= min_rank.
    Returns list of (sym_i, sym_j, result) tuples.
    """
    n = len(symbols)
    cointegrated = []
    for i in range(n):
        for j in range(i + 1, n):
            data = price_matrix[:, [i, j]]
            try:
                result = johansen_test(data, lags=lags)
                if result.r_trace >= min_rank:
                    cointegrated.append((symbols[i], symbols[j], result))
            except (np.linalg.LinAlgError, ValueError):
                continue
    return cointegrated
