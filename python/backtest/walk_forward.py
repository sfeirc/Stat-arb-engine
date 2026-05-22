"""
Expanding-window walk-forward backtester.

Architecture
------------
  - Initial training window : 252 bars
  - Refit frequency         : 63 bars
  - Per fold: refit EG cointegration, Kalman Q/R (EM), HMM regime detector
  - Signal: rolling z-score on Kalman spread
  - Sizing: Kelly-inspired with regime weighting + vol scaling
  - Costs: CostModel (crypto taker fees + linear impact)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from python.cointegration.engle_granger import engle_granger_test
from python.kalman.hedge_ratio import KalmanHedgeRatio
from python.regime.hmm_regime import HMMRegimeDetector
from python.backtest.metrics import compute_metrics
from python.strategy.cost_model import CostModel
from python.strategy.signal_generator import ZScoreSignalGenerator, SignalParams
from python.strategy.position_sizer import PositionSizer, SizingParams


@dataclass
class TradeRecord:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    pair: str
    side: int                  # +1 = long spread, -1 = short spread
    z_entry: float
    z_exit: Optional[float]
    pnl: float = 0.0
    cost: float = 0.0
    holding_bars: int = 0
    exit_reason: str = ""      # "mean_revert", "stop_loss", "time", "regime"


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    daily_returns: np.ndarray
    trades: List[TradeRecord]
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    annualised_return: float
    annualised_vol: float
    turnover: float


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_hmm_features(spreads: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Build 5-dimensional feature matrix for the HMM:
      [spread_vol_21, spread_autocorr_lag1, market_vol_proxy, abs_z_63, innov_var_21]
    """
    n = len(spreads)
    if n < max(window, 63) + 1:
        return np.zeros((0, 5))

    s = pd.Series(spreads)

    vol_21 = s.rolling(window).std().values
    autocorr = s.rolling(window).apply(
        lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1])
        if len(x) > 2 and np.std(x) > 1e-10
        else 0.0,
        raw=True,
    ).values

    roll_mu_63 = s.rolling(63).mean().values
    roll_std_63 = s.rolling(63).std().values
    z_63 = (spreads - roll_mu_63) / np.maximum(roll_std_63, 1e-8)

    diffs = np.diff(spreads, prepend=spreads[0])
    innov_var = pd.Series(diffs).rolling(window).var().values
    market_vol_proxy = vol_21.copy()

    features = np.column_stack([
        vol_21,
        autocorr,
        market_vol_proxy,
        np.abs(z_63),
        innov_var,
    ])

    valid = ~np.any(np.isnan(features) | np.isinf(features), axis=1)
    return features[valid]


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

class WalkForwardBacktester:
    """
    Expanding-window walk-forward validation.

    Parameters
    ----------
    initial_train_days : size of the first training window (bars)
    refit_days         : refit frequency and test window size (bars)
    z_entry            : z-score entry threshold
    z_exit             : z-score exit threshold (mean-reversion)
    z_stop             : z-score stop-loss threshold
    max_hold_bars      : maximum bars to hold a position
    kelly_cap          : maximum Kelly fraction of NAV
    target_daily_vol   : target portfolio daily volatility for vol-scaling
    """

    def __init__(
        self,
        initial_train_days: int = 252,
        refit_days: int = 63,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 4.0,
        max_hold_bars: int = 21,
        kelly_cap: float = 0.25,
        target_daily_vol: float = 0.01,
        adv_usdt: float = 5e8,  # default ADV assumption for cost model
    ) -> None:
        self.initial_train_days = initial_train_days
        self.refit_days = refit_days
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.max_hold_bars = max_hold_bars
        self.kelly_cap = kelly_cap
        self.target_daily_vol = target_daily_vol
        self.adv_usdt = adv_usdt

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        price_data: Dict[str, pd.Series],
        pairs: List[Tuple[str, str]],
    ) -> List[FoldResult]:
        """
        Run the full walk-forward backtest.

        Parameters
        ----------
        price_data : {symbol: pd.Series of daily close prices}
        pairs      : list of (symbol_y, symbol_x) tuples to test

        Returns
        -------
        List of FoldResult, one per test window.
        """
        df = pd.DataFrame(price_data).dropna()
        dates = df.index
        n_days = len(dates)

        if n_days < self.initial_train_days + 5:
            raise ValueError(
                f"Need at least {self.initial_train_days + 5} bars; got {n_days}"
            )

        fold_results: List[FoldResult] = []
        fold_id = 0
        train_end_idx = self.initial_train_days

        while train_end_idx < n_days:
            test_start_idx = train_end_idx
            test_end_idx = min(train_end_idx + self.refit_days, n_days)
            test_dates = dates[test_start_idx:test_end_idx]

            if len(test_dates) < 5:
                break

            active_pairs = self._refit_models(df, pairs, train_end_idx)

            daily_pnl, all_trades = self._simulate_test_period(
                df, active_pairs, test_start_idx, test_end_idx, test_dates
            )

            # Vol-scale to target
            daily_pnl = self._vol_scale(daily_pnl)

            metrics = compute_metrics(daily_pnl, all_trades)

            fold_results.append(
                FoldResult(
                    fold_id=fold_id,
                    train_start=dates[0],
                    train_end=dates[train_end_idx - 1],
                    test_start=test_dates[0],
                    test_end=test_dates[-1],
                    daily_returns=daily_pnl,
                    trades=all_trades,
                    **metrics,
                )
            )

            train_end_idx += self.refit_days
            fold_id += 1

        return fold_results

    # ------------------------------------------------------------------
    # Refit models on training window
    # ------------------------------------------------------------------

    def _refit_models(
        self,
        df: pd.DataFrame,
        pairs: List[Tuple[str, str]],
        train_end_idx: int,
    ) -> list:
        """Refit EG, Kalman, HMM for all pairs on training data."""
        active_pairs = []
        cost_model = CostModel()

        for sym_y, sym_x in pairs:
            if sym_y not in df.columns or sym_x not in df.columns:
                continue

            y_train = np.log(df[sym_y].iloc[:train_end_idx].values)
            x_train = np.log(df[sym_x].iloc[:train_end_idx].values)

            # --- Engle-Granger test ---
            try:
                eg = engle_granger_test(y_train, x_train)
            except Exception:  # noqa: BLE001
                continue
            if eg.p_value > 0.05:
                continue

            # --- Kalman filter ---
            kf = KalmanHedgeRatio(sigma_beta=1e-4, sigma_alpha=1e-4)
            kf_result = kf.fit_series(y_train, x_train)

            # EM noise estimation
            Q_new, R_new = kf.em_estimate_noise(y_train, x_train, window=min(252, train_end_idx))
            kf.Q = np.diag(Q_new)
            kf.R = R_new
            # Keep the Kalman state and spread history from training so the
            # test period inherits the learned state (no cold-start reset).
            # Preserve the last 63 spreads to seed the z-score normaliser.
            spreads_train = kf_result["spreads"]
            seed_spreads = list(spreads_train[-63:])

            # --- HMM ---
            features = build_hmm_features(spreads_train)
            if len(features) < 50:
                continue

            hmm = HMMRegimeDetector(n_states=2, n_features=5)
            try:
                hmm.fit(features)
            except Exception:  # noqa: BLE001
                continue

            active_pairs.append({
                "sym_y": sym_y,
                "sym_x": sym_x,
                "kf": kf,
                "hmm": hmm,
                "eg_beta": eg.beta,
                "seed_spreads": seed_spreads,
            })

        return active_pairs

    # ------------------------------------------------------------------
    # Test period simulation
    # ------------------------------------------------------------------

    def _simulate_test_period(
        self,
        df: pd.DataFrame,
        active_pairs: list,
        test_start: int,
        test_end: int,
        test_dates: pd.DatetimeIndex,
    ) -> Tuple[np.ndarray, List[TradeRecord]]:
        """Simulate trading over the test window for all active pairs."""
        n = test_end - test_start
        daily_pnl = np.zeros(n)
        all_trades: List[TradeRecord] = []
        cost_model = CostModel()
        n_active = max(len(active_pairs), 1)

        for pair_info in active_pairs:
            sym_y = pair_info["sym_y"]
            sym_x = pair_info["sym_x"]
            kf: KalmanHedgeRatio = pair_info["kf"]
            hmm: HMMRegimeDetector = pair_info["hmm"]
            seed_spreads: list = pair_info.get("seed_spreads", [])

            y_test = np.log(df[sym_y].iloc[test_start:test_end].values)
            x_test = np.log(df[sym_x].iloc[test_start:test_end].values)

            pair_pnl, trades = self._simulate_pair(
                y_test, x_test, test_dates,
                f"{sym_y}/{sym_x}", kf, hmm, cost_model,
                seed_spreads=seed_spreads,
            )
            daily_pnl += pair_pnl / n_active
            all_trades.extend(trades)

        return daily_pnl, all_trades

    def _simulate_pair(
        self,
        y: np.ndarray,
        x: np.ndarray,
        dates: pd.DatetimeIndex,
        pair_label: str,
        kf: KalmanHedgeRatio,
        hmm: HMMRegimeDetector,
        cost_model: CostModel,
        seed_spreads: Optional[list] = None,
    ) -> Tuple[np.ndarray, List[TradeRecord]]:
        """Simulate one pair over the test period bar by bar."""
        n = len(y)
        daily_pnl = np.zeros(n)
        trades: List[TradeRecord] = []
        sizer = PositionSizer(SizingParams(kelly_cap=self.kelly_cap))

        position = 0
        entry_bar = -1
        entry_z = 0.0
        entry_date: Optional[pd.Timestamp] = None

        # Seed the spread buffer with the last 63 training spreads so
        # z-scores are immediately meaningful on the first test bar.
        spread_buf: list = list(seed_spreads or [])

        # z_scores_test: indexed by test-period bar t (length n at end)
        z_scores_test: list = []

        for t in range(n):
            # Update Kalman filter
            result = kf.update(float(y[t]), float(x[t]))
            spread = result["spread"]
            spread_buf.append(spread)

            # Compute rolling z-score (window = min(total bars seen, 63))
            z_window = min(len(spread_buf), 63)
            if z_window < 10:
                z_scores_test.append(0.0)
                continue

            recent = spread_buf[-z_window:]
            mu = float(np.mean(recent))
            sig = float(np.std(recent)) + 1e-8
            z = (spread - mu) / sig
            z_scores_test.append(z)
            # alias: use spreads for HMM feature building
            spreads = spread_buf

            # Regime probability
            n_test_z = len(z_scores_test)  # number of test-period z-scores so far

            if len(spread_buf) >= 21 + len(seed_spreads or []):
                # Build HMM features from the full spread buffer (seed + test)
                feats = build_hmm_features(np.array(spread_buf))
                if len(feats) >= 2:
                    regime_p = float(hmm.predict_regime(feats)[-1])
                else:
                    regime_p = 0.5
            else:
                regime_p = 0.5

            spread_vol = float(np.std(spread_buf[-21:])) + 1e-8
            size = sizer.compute_size(spread_vol, z, regime_p)

            # --- Exit logic ---
            if position != 0:
                hold_bars = t - entry_bar
                exit_reason = ""

                if abs(z) < self.z_exit:
                    exit_reason = "mean_revert"
                elif abs(z) > self.z_stop:
                    exit_reason = "stop_loss"
                elif hold_bars >= self.max_hold_bars:
                    exit_reason = "time"
                elif regime_p < 0.3:
                    exit_reason = "regime"

                if exit_reason:
                    # Long spread (position=+1): profit when z increases toward 0
                    # (entry_z < 0, z_exit closer to 0): pnl = +1 * (z_exit - entry_z) > 0
                    # Short spread (position=-1): profit when z decreases toward 0
                    # (entry_z > 0, z_exit closer to 0): pnl = -1 * (z_exit - entry_z) > 0
                    pnl = float(position * (z - entry_z) * spread_vol * size)
                    cost = cost_model.crypto_round_trip_cost(size, self.adv_usdt)
                    net = pnl - cost
                    daily_pnl[t] += net

                    trades.append(TradeRecord(
                        entry_date=entry_date,
                        exit_date=dates[t],
                        pair=pair_label,
                        side=position,
                        z_entry=entry_z,
                        z_exit=z,
                        pnl=net,
                        cost=cost,
                        holding_bars=hold_bars,
                        exit_reason=exit_reason,
                    ))

                    position = 0
                    entry_bar = -1
                else:
                    # Mark-to-market: long spread profits from z increasing toward 0
                    # daily_pnl[t] = position * Δz * spread_vol * size
                    if n_test_z >= 2:
                        dz = z_scores_test[-1] - z_scores_test[-2]
                        daily_pnl[t] += position * dz * spread_vol * size

            # --- Entry logic ---
            if position == 0 and regime_p >= 0.4:
                if z < -self.z_entry:
                    position = 1
                    entry_bar = t
                    entry_z = z
                    entry_date = dates[t]
                elif z > self.z_entry:
                    position = -1
                    entry_bar = t
                    entry_z = z
                    entry_date = dates[t]

        return daily_pnl, trades

    # ------------------------------------------------------------------
    # Vol scaling
    # ------------------------------------------------------------------

    def _vol_scale(self, pnl: np.ndarray) -> np.ndarray:
        """Scale daily P&L series to target daily volatility."""
        sig = float(np.std(pnl))
        if sig < 1e-10:
            return pnl
        scale = self.target_daily_vol / sig
        return pnl * min(scale, 5.0)  # hard cap at 5x leverage


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _generate_synthetic_prices(
    symbols: List[str],
    n_days: int = 500,
    seed: int = 0,
) -> Dict[str, pd.Series]:
    """
    Generate synthetic cointegrated price series for demo/testing.

    Construction:
      - One common I(1) random walk shared by all series
      - Each series = exp(base + beta_i * common_factor + stationary_residual)
      - Residuals are mean-reverting AR(1) processes with phi ~ 0.85
        so the spread is stationary and EG/Johansen tests will pass.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    prices: Dict[str, pd.Series] = {}

    # Common I(1) random walk
    common_factor = np.cumsum(rng.normal(0, 0.012, n_days))

    # AR(1) mean-reverting residuals per symbol
    phi = 0.85  # strong mean-reversion so EG passes easily
    sigma_eps = 0.003

    for i, sym in enumerate(symbols):
        resid = np.zeros(n_days)
        for t in range(1, n_days):
            resid[t] = phi * resid[t - 1] + rng.normal(0, sigma_eps)
        log_p = 9.2 + (i * 0.3) + common_factor + resid
        prices[sym] = pd.Series(np.exp(log_p), index=dates, name=sym)

    return prices


if __name__ == "__main__":
    import json

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "BNBUSDT"),
        ("ETHUSDT", "SOLUSDT"),
    ]

    print("Generating synthetic price data for demo...")
    price_data = _generate_synthetic_prices(symbols, n_days=600)

    print("Running walk-forward backtest...")
    backtester = WalkForwardBacktester(
        initial_train_days=252,
        refit_days=63,
        z_entry=2.0,
        z_exit=0.5,
        z_stop=4.0,
        max_hold_bars=21,
        kelly_cap=0.25,
        target_daily_vol=0.01,
    )

    fold_results = backtester.run(price_data, pairs)

    print(f"\nCompleted {len(fold_results)} folds\n")
    print(f"{'Fold':>4}  {'Period':>25}  {'Sharpe':>7}  "
          f"{'MaxDD':>7}  {'Trades':>6}  {'WinRate':>7}")
    print("-" * 65)

    for f in fold_results:
        period = f"{f.test_start.date()} → {f.test_end.date()}"
        print(
            f"{f.fold_id:>4}  {period:>25}  "
            f"{f.sharpe:>7.2f}  {f.max_drawdown:>7.2%}  "
            f"{f.n_trades:>6}  {f.win_rate:>7.1%}"
        )

    all_returns = np.concatenate([f.daily_returns for f in fold_results])
    all_trades = [t for f in fold_results for t in f.trades]
    overall = compute_metrics(all_returns, all_trades)

    print("\n--- Overall Walk-Forward Performance ---")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")
