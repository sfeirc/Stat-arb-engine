"""
Kelly-inspired position sizing with regime weighting and risk controls.

Full Kelly is capped at kelly_cap of NAV to prevent catastrophic over-sizing.
Regime probability scales the Kelly fraction linearly.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingParams:
    """Parameters for the position sizing model."""
    kelly_cap: float = 0.25          # maximum Kelly fraction of NAV
    min_size: float = 0.01           # minimum position size (1% of NAV)
    max_position_frac: float = 0.20  # maximum single-position NAV fraction
    target_daily_vol: float = 0.01   # target daily portfolio volatility
    regime_scale_floor: float = 0.10 # minimum regime scale factor
    vol_lookback: int = 21           # bars for spread vol estimation
    risk_free_rate: float = 0.05     # annualised risk-free rate


class PositionSizer:
    """
    Computes target position sizes using a Kelly-inspired criterion.

    Kelly fraction = (mu / sigma^2) * regime_factor,  capped at kelly_cap.

    Where:
      mu           = estimated daily expected return of the spread trade
      sigma^2      = estimated variance of daily returns
      regime_factor = P(mean-reverting) from HMM, clipped to [floor, 1.0]
    """

    def __init__(self, params: Optional[SizingParams] = None) -> None:
        self.p = params or SizingParams()
        self._return_history: list = []

    # ------------------------------------------------------------------
    # Main sizing API
    # ------------------------------------------------------------------

    def compute_size(
        self,
        spread_vol: float,        # estimated spread volatility (same units as spread)
        z_score: float,           # current z-score
        regime_prob_mr: float,    # P(mean-reverting) from HMM
        portfolio_vol_target: Optional[float] = None,
    ) -> float:
        """
        Compute target position size as a fraction of NAV.

        Parameters
        ----------
        spread_vol      : standard deviation of recent spread observations
        z_score         : current spread z-score (used for expected return estimate)
        regime_prob_mr  : posterior probability of mean-reverting state
        portfolio_vol_target : override the default target_daily_vol

        Returns
        -------
        float in [0, max_position_frac]
        """
        vol_target = portfolio_vol_target or self.p.target_daily_vol

        # Expected mean-reversion profit: each z-score unit of spread reversion
        # yields approximately spread_vol per unit of size
        expected_return = abs(z_score) * spread_vol * 0.5  # conservative: 50% capture

        sigma_sq = spread_vol ** 2
        if sigma_sq < 1e-10:
            return 0.0

        # Kelly fraction (ignoring risk-free rate for simplicity)
        full_kelly = expected_return / sigma_sq

        # Regime scaling: linear interpolation between floor and 1.0
        regime_factor = float(np.clip(
            regime_prob_mr,
            self.p.regime_scale_floor,
            1.0,
        ))

        # Scaled Kelly
        kelly_frac = full_kelly * regime_factor * 0.5  # use half-Kelly for safety

        # Cap
        capped = float(np.clip(kelly_frac, self.p.min_size, self.p.kelly_cap))

        # Volatility scaling: adjust so that expected portfolio vol = target
        if spread_vol > 1e-10:
            vol_scale = vol_target / spread_vol
            capped = min(capped * vol_scale, self.p.max_position_frac)

        return float(np.clip(capped, 0.0, self.p.max_position_frac))

    def compute_size_batch(
        self,
        spread_vols: np.ndarray,
        z_scores: np.ndarray,
        regime_probs: np.ndarray,
    ) -> np.ndarray:
        """Vectorised version of compute_size over arrays of equal length."""
        n = len(z_scores)
        sizes = np.zeros(n)
        for t in range(n):
            sizes[t] = self.compute_size(
                float(spread_vols[t]),
                float(z_scores[t]),
                float(regime_probs[t]),
            )
        return sizes

    # ------------------------------------------------------------------
    # Portfolio-level capital allocation
    # ------------------------------------------------------------------

    @staticmethod
    def equal_risk_allocation(
        pair_vols: np.ndarray,
        total_risk_budget: float = 1.0,
    ) -> np.ndarray:
        """
        Allocate capital inversely proportional to volatility (risk-parity).

        Returns allocation weights that sum to 1.0, scaled so that the
        expected portfolio risk equals total_risk_budget.
        """
        if len(pair_vols) == 0:
            return np.array([])
        safe_vols = np.maximum(pair_vols, 1e-10)
        inv_vol = 1.0 / safe_vols
        weights = inv_vol / inv_vol.sum()
        return weights * total_risk_budget

    @staticmethod
    def marginal_kelly(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        kelly_cap: float = 0.25,
    ) -> np.ndarray:
        """
        Multi-asset Kelly: w* = Sigma^{-1} * mu, capped at kelly_cap.

        Parameters
        ----------
        expected_returns : (n,) expected return per pair
        cov_matrix       : (n, n) return covariance matrix

        Returns
        -------
        (n,) weight vector
        """
        n = len(expected_returns)
        reg_cov = cov_matrix + 1e-6 * np.eye(n)
        try:
            cov_inv = np.linalg.inv(reg_cov)
        except np.linalg.LinAlgError:
            return np.zeros(n)

        weights = cov_inv @ expected_returns
        # Cap each weight
        weights = np.clip(weights, -kelly_cap, kelly_cap)
        return weights
