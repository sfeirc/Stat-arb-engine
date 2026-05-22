"""
Transaction cost model for crypto (Binance) and equities.

Includes:
  - Exchange commission (maker/taker)
  - Linear market impact (Almgren-Chriss)
  - Bid-ask spread proxy
  - Square-root impact for equities
"""
from __future__ import annotations

import numpy as np


class CostModel:
    """Transaction cost model for crypto and equity instruments."""

    # --- Binance fee schedule (BNB discount tier 1) ---
    MAKER_FEE: float = 0.0002   # 0.02%
    TAKER_FEE: float = 0.0004   # 0.04%

    # --- Almgren-Chriss linear impact coefficient ---
    IMPACT_ETA: float = 0.1

    # --- Maximum single-leg impact cap ---
    MAX_IMPACT_BPS: float = 0.005   # 50 basis points

    # --- Equity parameters ---
    EQUITY_COMMISSION_PER_SHARE: float = 0.001   # $0.001/share DMA
    EQUITY_HALF_SPREAD_BPS: float = 0.0005       # 5 bps proxy
    EQUITY_SQRT_IMPACT_COEF: float = 0.10

    def crypto_round_trip_cost(
        self,
        trade_size_frac: float,
        adv: float,
        use_maker: bool = False,
    ) -> float:
        """
        Round-trip transaction cost as a fraction of trade value.

        Parameters
        ----------
        trade_size_frac : trade size expressed as fraction of portfolio NAV
        adv             : 30-day average daily volume in USDT
        use_maker       : if True, use maker fee (limit orders)

        Returns
        -------
        float — total round-trip cost fraction (both legs combined)
        """
        fee = self.MAKER_FEE if use_maker else self.TAKER_FEE
        commission = 2.0 * fee

        # Linear market impact
        impact_per_leg = self.IMPACT_ETA * abs(trade_size_frac) / max(adv, 1.0)
        impact_per_leg = min(impact_per_leg, self.MAX_IMPACT_BPS)
        impact = 2.0 * impact_per_leg

        return commission + impact

    def crypto_single_leg_cost(
        self,
        trade_size_frac: float,
        adv: float,
        use_maker: bool = False,
    ) -> float:
        """Single-leg transaction cost (for incremental fill modelling)."""
        fee = self.MAKER_FEE if use_maker else self.TAKER_FEE
        impact = self.IMPACT_ETA * abs(trade_size_frac) / max(adv, 1.0)
        return fee + min(impact, self.MAX_IMPACT_BPS)

    def equity_round_trip_cost(
        self,
        price: float,
        shares: float,
        adv_shares: float,
    ) -> float:
        """
        Round-trip cost for equity instruments.

        Parameters
        ----------
        price      : current mid price ($)
        shares     : number of shares traded
        adv_shares : 30-day average daily volume in shares

        Returns
        -------
        float — total cost in dollars (both legs)
        """
        # DMA commission: $0.001/share each way
        commission = 2.0 * self.EQUITY_COMMISSION_PER_SHARE * abs(shares)

        # Bid-ask spread: 5bps each way → 10bps round-trip
        spread_cost = 2.0 * self.EQUITY_HALF_SPREAD_BPS * abs(shares) * price

        # Square-root market impact (Almgren-Chriss)
        volume_frac = abs(shares) / max(adv_shares, 1.0)
        impact_per_leg = (
            self.EQUITY_SQRT_IMPACT_COEF
            * float(np.sqrt(volume_frac))
            * 0.15   # volatility proxy
            * price
            * abs(shares)
        )
        impact = 2.0 * impact_per_leg

        return commission + spread_cost + impact

    def estimated_breakeven_z(
        self,
        total_cost_frac: float,
        spread_vol_frac: float,
    ) -> float:
        """
        Minimum |z_entry| needed to break even after costs.

        z_break = total_cost / (spread_vol per unit of z)
        """
        if spread_vol_frac < 1e-10:
            return float("inf")
        return total_cost_frac / spread_vol_frac

    def net_pnl(
        self,
        gross_pnl_frac: float,
        trade_size_frac: float,
        adv: float,
        n_legs: int = 2,
    ) -> float:
        """
        Compute net PnL after subtracting crypto transaction costs.

        Parameters
        ----------
        gross_pnl_frac  : gross PnL as fraction of portfolio
        trade_size_frac : trade size fraction
        adv             : average daily volume (USDT)
        n_legs          : number of pair legs (default 2)

        Returns
        -------
        net PnL fraction
        """
        cost = self.crypto_round_trip_cost(trade_size_frac, adv)
        return gross_pnl_frac - cost * n_legs / 2  # already counted per pair
