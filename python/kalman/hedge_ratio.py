"""
Kalman filter for time-varying hedge ratio with EM noise estimation.

State-space model:
  Observation:  y_t = [x_t, 1] * [beta_t, alpha_t]' + eps_t,   eps_t ~ N(0, R)
  Transition:   [beta_t, alpha_t]' = [beta_{t-1}, alpha_{t-1}]' + eta_t, eta_t ~ N(0, Q)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalmanState:
    beta_hat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0])
    )  # [beta, alpha]
    P: np.ndarray = field(default_factory=lambda: np.eye(2) * 1.0)
    spread_history: list = field(default_factory=list)
    innovation_var: list = field(default_factory=list)


class KalmanHedgeRatio:
    """
    Kalman filter estimating a time-varying hedge ratio (beta) and intercept (alpha).

    Parameters
    ----------
    sigma_beta  : process noise std for beta_t
    sigma_alpha : process noise std for alpha_t
    """

    def __init__(
        self,
        sigma_beta: float = 1e-4,
        sigma_alpha: float = 1e-4,
    ) -> None:
        self.Q = np.diag([sigma_beta ** 2, sigma_alpha ** 2])
        self.R: float = 0.001
        self.state = KalmanState()

    # ------------------------------------------------------------------
    # Core update step
    # ------------------------------------------------------------------

    def update(
        self,
        y: float,
        x: float,
        rolling_resid_var: Optional[float] = None,
    ) -> dict:
        """Process one observation.  Returns spread and posterior state."""
        if rolling_resid_var is not None:
            self.R = max(rolling_resid_var, 1e-8)

        # Predict
        beta_pred = self.state.beta_hat.copy()
        P_pred = self.state.P + self.Q

        # Observation vector H = [x, 1]
        H = np.array([x, 1.0])

        # Innovation
        y_hat = float(H @ beta_pred)
        innovation = y - y_hat
        S = float(H @ P_pred @ H) + self.R

        # Kalman gain
        K_gain = P_pred @ H / S

        # Update
        self.state.beta_hat = beta_pred + K_gain * innovation
        I_KH = np.eye(2) - np.outer(K_gain, H)
        self.state.P = I_KH @ P_pred

        spread = innovation
        self.state.spread_history.append(spread)
        self.state.innovation_var.append(innovation ** 2)

        return {
            "beta": float(self.state.beta_hat[0]),
            "alpha": float(self.state.beta_hat[1]),
            "spread": spread,
            "innovation_var": S,
            "P": self.state.P.copy(),
        }

    # ------------------------------------------------------------------
    # Batch filter over full series
    # ------------------------------------------------------------------

    def fit_series(self, y: np.ndarray, x: np.ndarray) -> dict:
        """Run Kalman filter over full series.  Returns time series of state."""
        n = len(y)
        betas = np.zeros(n)
        alphas = np.zeros(n)
        spreads = np.zeros(n)
        window = 21

        for t in range(n):
            if t >= window and len(self.state.spread_history) >= window:
                recent = self.state.spread_history[-window:]
                rolling_var = float(np.var(recent))
            else:
                rolling_var = None

            result = self.update(float(y[t]), float(x[t]), rolling_var)
            betas[t] = result["beta"]
            alphas[t] = result["alpha"]
            spreads[t] = result["spread"]

        return {"betas": betas, "alphas": alphas, "spreads": spreads}

    # ------------------------------------------------------------------
    # EM noise estimation (Rauch-Tung-Striebel smoother)
    # ------------------------------------------------------------------

    def em_estimate_noise(
        self,
        y: np.ndarray,
        x: np.ndarray,
        window: int = 252,
        n_iter: int = 10,
    ) -> tuple[np.ndarray, float]:
        """
        EM algorithm to re-estimate Q and R from trailing *window* observations.

        Runs *n_iter* EM iterations.  Returns (Q_diag_new, R_new).
        """
        n = min(len(y), window)
        y_w = y[-n:]
        x_w = x[-n:]

        Q_em = self.Q.copy()
        R_em = self.R

        for _ in range(n_iter):
            # --- E-step: forward Kalman pass ---
            beta_fwd = np.zeros((n, 2))
            P_fwd = np.zeros((n, 2, 2))
            beta_t = np.array([1.0, 0.0])
            P_t = np.eye(2)
            innovations = np.zeros(n)

            for t in range(n):
                P_pred = P_t + Q_em
                H = np.array([x_w[t], 1.0])
                innov = float(y_w[t]) - float(H @ beta_t)
                S = float(H @ P_pred @ H) + R_em
                K = P_pred @ H / S
                beta_t = beta_t + K * innov
                P_t = (np.eye(2) - np.outer(K, H)) @ P_pred
                beta_fwd[t] = beta_t
                P_fwd[t] = P_t
                innovations[t] = innov

            # --- E-step: backward RTS smoother ---
            beta_smooth = np.zeros((n, 2))
            P_smooth = np.zeros((n, 2, 2))
            beta_smooth[-1] = beta_fwd[-1]
            P_smooth[-1] = P_fwd[-1]

            for t in range(n - 2, -1, -1):
                P_pred = P_fwd[t] + Q_em
                try:
                    P_pred_inv = np.linalg.inv(P_pred + 1e-10 * np.eye(2))
                except np.linalg.LinAlgError:
                    P_pred_inv = np.eye(2)
                G = P_fwd[t] @ P_pred_inv
                beta_smooth[t] = beta_fwd[t] + G @ (beta_smooth[t + 1] - beta_fwd[t])
                P_smooth[t] = P_fwd[t] + G @ (P_smooth[t + 1] - P_pred) @ G.T

            # --- M-step: update R ---
            R_em = float(np.mean(innovations ** 2))
            R_em = max(R_em, 1e-8)

            # --- M-step: update Q ---
            diffs = np.diff(beta_smooth, axis=0)
            if len(diffs) > 1:
                Q_new_full = np.cov(diffs.T)
                Q_em = np.diag([max(Q_new_full[0, 0], 1e-8), max(Q_new_full[1, 1], 1e-8)])
            else:
                Q_em = np.diag([1e-4, 1e-4])

        return np.diag(Q_em), R_em

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset state for reuse with a new pair."""
        self.state = KalmanState()
