"""
Z-score signal generation and entry/exit logic.

Signal pipeline
---------------
  1. Kalman spread → rolling z-score
  2. HMM regime posterior → size_factor [0, 1]
  3. Z-score thresholds → entry / exit / stop-loss signals
  4. Kelly-inspired sizing → target notional
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SignalParams:
    """Threshold parameters for signal generation."""
    z_entry: float = 2.0        # enter trade at |z| > z_entry
    z_exit: float = 0.5         # exit at |z| < z_exit (mean-reversion)
    z_stop: float = 4.0         # stop-loss at |z| > z_stop
    z_window: int = 63          # lookback for z-score normalisation
    regime_cutoff: float = 0.40  # min P(mean-revert) to allow entry
    max_hold_bars: int = 21     # maximum position holding period


@dataclass
class Signal:
    """One bar's trading signal for a pair."""
    timestamp: pd.Timestamp
    pair: str
    side: int               # +1 long spread, -1 short spread, 0 flat
    z_score: float
    size_factor: float      # [0, 1] from regime
    action: str             # "entry", "exit", "stop", "hold", "flat"


class ZScoreSignalGenerator:
    """
    Generates entry/exit signals for one pair using a rolling z-score
    computed on Kalman filter spread observations.
    """

    def __init__(self, params: Optional[SignalParams] = None) -> None:
        self.p = params or SignalParams()
        self._position: int = 0          # current held position side
        self._entry_bar: int = -1        # bar index when position opened
        self._entry_z: float = 0.0
        self._spread_buf: list = []      # rolling buffer of spreads

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state (use between folds / pairs)."""
        self._position = 0
        self._entry_bar = -1
        self._entry_z = 0.0
        self._spread_buf = []

    def step(
        self,
        bar_idx: int,
        timestamp: pd.Timestamp,
        pair: str,
        kalman_spread: float,
        regime_prob_mr: float,  # P(state=MEAN_REVERTING)
    ) -> Signal:
        """
        Process one bar and return a Signal.

        Parameters
        ----------
        bar_idx         : integer bar index (for hold-bar counting)
        timestamp       : bar timestamp
        pair            : pair identifier string
        kalman_spread   : Kalman-filtered spread value
        regime_prob_mr  : posterior probability of mean-reverting state
        """
        self._spread_buf.append(kalman_spread)
        z = self._compute_z(kalman_spread)
        size_factor = float(np.clip(regime_prob_mr, 0.0, 1.0))

        if self._position != 0:
            return self._check_exit(bar_idx, timestamp, pair, z, size_factor)

        return self._check_entry(bar_idx, timestamp, pair, z, size_factor)

    def generate_series(
        self,
        timestamps: pd.DatetimeIndex,
        pair: str,
        spreads: np.ndarray,
        regime_probs: np.ndarray,
    ) -> list[Signal]:
        """Run signal generation over a full array of spreads."""
        self.reset()
        signals = []
        for t, (ts, sp, rp) in enumerate(zip(timestamps, spreads, regime_probs)):
            sig = self.step(t, ts, pair, float(sp), float(rp))
            signals.append(sig)
        return signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_z(self, spread: float) -> float:
        """Rolling z-score using the last z_window observations."""
        buf = self._spread_buf
        window = min(len(buf), self.p.z_window)
        if window < 5:
            return 0.0
        recent = buf[-window:]
        mu = float(np.mean(recent))
        sig = float(np.std(recent)) + 1e-8
        return (spread - mu) / sig

    def _check_exit(
        self,
        bar_idx: int,
        timestamp: pd.Timestamp,
        pair: str,
        z: float,
        size_factor: float,
    ) -> Signal:
        hold = bar_idx - self._entry_bar

        if z * self._position < -abs(self.p.z_stop):  # stop-loss
            action = "stop"
            self._position = 0
        elif abs(z) < self.p.z_exit:                  # mean-reversion
            action = "exit"
            self._position = 0
        elif hold >= self.p.max_hold_bars:             # time stop
            action = "exit"
            self._position = 0
        elif size_factor < self.p.regime_cutoff:       # regime breakdown
            action = "exit"
            self._position = 0
        else:
            action = "hold"

        return Signal(
            timestamp=timestamp,
            pair=pair,
            side=self._position,
            z_score=z,
            size_factor=size_factor,
            action=action,
        )

    def _check_entry(
        self,
        bar_idx: int,
        timestamp: pd.Timestamp,
        pair: str,
        z: float,
        size_factor: float,
    ) -> Signal:
        if size_factor < self.p.regime_cutoff:
            return Signal(timestamp=timestamp, pair=pair, side=0,
                          z_score=z, size_factor=size_factor, action="flat")

        if z < -self.p.z_entry:
            self._position = 1
            self._entry_bar = bar_idx
            self._entry_z = z
            return Signal(timestamp=timestamp, pair=pair, side=1,
                          z_score=z, size_factor=size_factor, action="entry")

        if z > self.p.z_entry:
            self._position = -1
            self._entry_bar = bar_idx
            self._entry_z = z
            return Signal(timestamp=timestamp, pair=pair, side=-1,
                          z_score=z, size_factor=size_factor, action="entry")

        return Signal(timestamp=timestamp, pair=pair, side=0,
                      z_score=z, size_factor=size_factor, action="flat")
