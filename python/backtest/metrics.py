"""
Full performance metrics computation.

Includes:
  - Sharpe, Sortino, Calmar ratios
  - Maximum drawdown, drawdown duration
  - Win rate, profit factor
  - t-test significance
  - Bailey & Lopez de Prado Deflated Sharpe Ratio (DSR)
  - White's Reality Check p-value (block bootstrap)
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from typing import List


def compute_metrics(daily_returns: np.ndarray, trades: list) -> dict:
    """
    Compute comprehensive performance metrics from daily return series.

    Parameters
    ----------
    daily_returns : array of daily P&L fractions
    trades        : list of TradeRecord objects (may be empty)

    Returns
    -------
    dict with keys: sharpe, sortino, calmar, max_drawdown, n_trades,
                    win_rate, annualised_return, annualised_vol, turnover
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    n = len(r)

    if n < 2 or r.std() < 1e-10:
        return {
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "max_drawdown": 0.0, "n_trades": 0, "win_rate": 0.0,
            "annualised_return": 0.0, "annualised_vol": 0.0, "turnover": 0.0,
        }

    ann_factor = 252
    ann_return = float(r.mean() * ann_factor)
    ann_vol = float(r.std() * np.sqrt(ann_factor))

    sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0.0

    downside = r[r < 0]
    sortino_denom = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 1 else 1e-10
    sortino = ann_return / sortino_denom

    # Maximum drawdown
    equity = np.cumprod(1.0 + np.clip(r, -0.99, None))
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.maximum(peak, 1e-10)
    max_dd = float(drawdown.min())

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    n_trades = len(trades)
    win_rate = (
        sum(1 for t in trades if getattr(t, "pnl", 0.0) > 0) / n_trades
        if n_trades > 0
        else 0.0
    )

    years = n / ann_factor
    turnover = n_trades / max(years, 1e-3)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "annualised_return": ann_return,
        "annualised_vol": ann_vol,
        "turnover": turnover,
    }


def max_drawdown_duration(daily_returns: np.ndarray) -> int:
    """Return length (in bars) of the longest drawdown period."""
    r = np.asarray(daily_returns, dtype=np.float64)
    equity = np.cumprod(1.0 + np.clip(r, -0.99, None))
    peak = np.maximum.accumulate(equity)
    in_drawdown = equity < peak

    max_dur = 0
    current_dur = 0
    for flag in in_drawdown:
        if flag:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


def profit_factor(daily_returns: np.ndarray) -> float:
    """Gross profit / gross loss.  Returns inf if no losses."""
    r = np.asarray(daily_returns, dtype=np.float64)
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    return gross_profit / max(gross_loss, 1e-10)


def ttest_significance(daily_returns: np.ndarray) -> tuple[float, float]:
    """
    One-sample t-test on daily returns (H0: mean = 0).

    Returns (t_stat, p_value).
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    n = len(r)
    if n < 2:
        return 0.0, 1.0
    mu = r.mean()
    se = r.std(ddof=1) / np.sqrt(n)
    t_stat = mu / max(se, 1e-12)
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    return float(t_stat), p_value


def deflated_sharpe_ratio(
    sharpe_obs: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> float:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Adjusts for multiple testing over *n_trials* strategies evaluated
    on *n_obs* observations.  Returns P(true SR > 0 | observed SR).

    Parameters
    ----------
    sharpe_obs   : observed annualised Sharpe ratio
    n_trials     : number of strategy variations tested
    n_obs        : number of daily return observations
    skew         : skewness of daily return distribution
    excess_kurt  : excess kurtosis of daily returns
    """
    from scipy.special import ndtri

    gamma_euler = 0.5772156649

    if n_trials <= 1:
        sr_benchmark = 0.0
    else:
        z1 = ndtri(1.0 - 1.0 / n_trials)
        z2 = ndtri(max(1.0 - 1.0 / (n_trials * np.e), 1e-9))
        sr_benchmark = (1 - gamma_euler) * z1 + gamma_euler * z2

    # Non-normality adjustment (IID Sharpe, per obs not annualised)
    sr_per_obs = sharpe_obs / np.sqrt(252)
    sr_adj = sr_per_obs * (
        1.0
        - skew * sr_per_obs / 6.0
        + (excess_kurt - 1.0) * sr_per_obs ** 2 / 24.0
    )

    if n_obs > 1:
        dsr = float(stats.norm.cdf((sr_adj - sr_benchmark) * np.sqrt(n_obs - 1)))
    else:
        dsr = 0.5

    return float(np.clip(dsr, 0.0, 1.0))


def whites_reality_check(
    benchmark_returns: np.ndarray,
    strategy_returns_list: List[np.ndarray],
    n_bootstrap: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> float:
    """
    White (2000) Reality Check p-value via stationary block bootstrap.

    Tests H0: no strategy beats the benchmark (zero mean excess return).

    Parameters
    ----------
    benchmark_returns     : array of benchmark daily returns
    strategy_returns_list : list of arrays (one per strategy)
    n_bootstrap           : number of bootstrap replications
    block_size            : expected block length for circular bootstrap
    seed                  : random seed for reproducibility

    Returns
    -------
    p_value : probability of observing max SR >= observed by chance
    """
    rng = np.random.default_rng(seed)
    n = len(benchmark_returns)

    # Excess returns over benchmark
    excess_list = [
        np.asarray(sr, dtype=np.float64) - benchmark_returns
        for sr in strategy_returns_list
    ]

    # Observed test statistic: max mean excess return (scaled)
    obs_stat = max(np.mean(exc) for exc in excess_list)

    # Stationary block bootstrap
    def bootstrap_stat() -> float:
        boot_means = []
        for exc in excess_list:
            boot = _circular_block_bootstrap(exc, block_size, n, rng)
            boot_means.append(np.mean(boot))
        return max(boot_means)

    boot_stats = np.array([bootstrap_stat() for _ in range(n_bootstrap)])
    p_value = float(np.mean(boot_stats >= obs_stat))
    return p_value


def _circular_block_bootstrap(
    data: np.ndarray,
    block_size: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular block bootstrap resample of *data* to length *n*."""
    t = len(data)
    starts = rng.integers(0, t, size=n)
    result = []
    for start in starts:
        end = min(start + block_size, start + n - len(result))
        block = [data[(start + i) % t] for i in range(min(block_size, n - len(result)))]
        result.extend(block)
        if len(result) >= n:
            break
    return np.array(result[:n], dtype=np.float64)
