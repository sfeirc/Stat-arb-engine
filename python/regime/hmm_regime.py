"""
2-state Hidden Markov Model for market regime detection.
Implements Baum-Welch EM from scratch using numpy only.

States
------
  0 : MEAN-REVERTING  — trade with full Kelly size
  1 : TRENDING / BROKEN — reduce to 25% size (size_factor = 0.25)

Features (5-dimensional)
------------------------
  0 : spread_vol_21d        — rolling 21-bar standard deviation of spread
  1 : spread_autocorr_lag1  — rolling 21-bar lag-1 autocorrelation of spread
  2 : market_vol_proxy      — same as spread_vol_21d (can be overridden)
  3 : abs_z_score           — absolute z-score of the spread
  4 : kalman_innov_var      — rolling 21-bar variance of Kalman innovations
"""
from __future__ import annotations

import numpy as np


class HMMRegimeDetector:
    """
    2-state Gaussian-emission HMM fitted with Baum-Welch.

    Parameters
    ----------
    n_states   : number of hidden states (default 2)
    n_features : dimensionality of the observation vector (default 5)
    """

    def __init__(self, n_states: int = 2, n_features: int = 5) -> None:
        self.n_states = n_states
        self.n_features = n_features

        # Transition matrix A[i,j] = P(s_t=j | s_{t-1}=i)
        self.A = np.array([[0.95, 0.05],
                           [0.10, 0.90]], dtype=np.float64)

        # Emission parameters: Gaussian N(means[s], covs[s]) per state
        self.means = np.zeros((n_states, n_features), dtype=np.float64)
        self.covs = np.array(
            [np.eye(n_features) for _ in range(n_states)], dtype=np.float64
        )

        # Initial state distribution
        self.pi = np.array([0.7, 0.3], dtype=np.float64)

    # ------------------------------------------------------------------
    # Emission probability
    # ------------------------------------------------------------------

    def _log_emission(self, obs: np.ndarray, state: int) -> np.ndarray:
        """
        Log-probability of observations under Gaussian emission for *state*.

        obs : (T, n_features) or (n_features,)
        Returns scalar or (T,) array.
        """
        mu = self.means[state]
        cov = self.covs[state] + 1e-6 * np.eye(self.n_features)
        try:
            inv_cov = np.linalg.inv(cov)
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                raise np.linalg.LinAlgError("Non-positive definite")
        except np.linalg.LinAlgError:
            inv_cov = np.eye(self.n_features)
            log_det = 0.0

        const = -0.5 * (self.n_features * np.log(2 * np.pi) + log_det)

        if obs.ndim == 1:
            diff = obs - mu
            mahal = float(diff @ inv_cov @ diff)
            return const - 0.5 * mahal
        else:
            diff = obs - mu          # (T, n_features)
            # mahal[t] = diff[t] @ inv_cov @ diff[t]
            mahal = np.einsum("ti,ij,tj->t", diff, inv_cov, diff)
            return const - 0.5 * mahal

    # ------------------------------------------------------------------
    # Forward algorithm (log-scaled)
    # ------------------------------------------------------------------

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Forward algorithm.

        Parameters
        ----------
        obs : (T, n_features)

        Returns
        -------
        alpha      : (T, n_states) scaled forward probabilities
        scale      : (T,) normalisation constants
        log_likelihood : float
        """
        T = len(obs)
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)

        for s in range(self.n_states):
            log_e = self._log_emission(obs[0], s)
            alpha[0, s] = self.pi[s] * np.exp(float(log_e))

        scale[0] = alpha[0].sum()
        alpha[0] /= max(scale[0], 1e-300)

        for t in range(1, T):
            for s in range(self.n_states):
                log_e = self._log_emission(obs[t], s)
                alpha[t, s] = np.exp(float(log_e)) * (alpha[t - 1] @ self.A[:, s])
            scale[t] = alpha[t].sum()
            alpha[t] /= max(scale[t], 1e-300)

        log_likelihood = float(np.sum(np.log(np.maximum(scale, 1e-300))))
        return alpha, scale, log_likelihood

    # ------------------------------------------------------------------
    # Backward algorithm
    # ------------------------------------------------------------------

    def backward(
        self, obs: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        """Backward algorithm.  Returns beta (T, n_states)."""
        T = len(obs)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0

        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                acc = 0.0
                for s2 in range(self.n_states):
                    log_e = self._log_emission(obs[t + 1], s2)
                    acc += self.A[s, s2] * np.exp(float(log_e)) * beta[t + 1, s2]
                beta[t, s] = acc
            denom = max(scale[t + 1], 1e-300)
            beta[t] /= denom

        return beta

    # ------------------------------------------------------------------
    # Baum-Welch EM
    # ------------------------------------------------------------------

    def baum_welch(
        self,
        obs: np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-6,
    ) -> float:
        """
        Baum-Welch EM to fit HMM parameters.

        Parameters
        ----------
        obs    : (T, n_features)
        n_iter : maximum EM iterations
        tol    : convergence tolerance on log-likelihood

        Returns
        -------
        Final log-likelihood.
        """
        prev_ll = -np.inf
        T = len(obs)

        for _iteration in range(n_iter):
            alpha, scale, log_ll = self.forward(obs)
            beta = self.backward(obs, scale)

            # Posterior state probabilities: gamma[t, s] = P(s_t=s | obs)
            gamma = alpha * beta
            row_sums = gamma.sum(axis=1, keepdims=True)
            gamma /= np.maximum(row_sums, 1e-300)

            # xi[t, i, j] = P(s_t=i, s_{t+1}=j | obs)
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        log_e = self._log_emission(obs[t + 1], j)
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.A[i, j]
                            * np.exp(float(log_e))
                            * beta[t + 1, j]
                        )
                xi[t] /= max(xi[t].sum(), 1e-300)

            # --- M-step ---
            self.pi = gamma[0] / max(gamma[0].sum(), 1e-300)

            for i in range(self.n_states):
                denom = max(xi[:, i, :].sum(), 1e-300)
                self.A[i] = xi[:, i, :].sum(axis=0) / denom
                self.A[i] /= self.A[i].sum()

            for s in range(self.n_states):
                w = gamma[:, s]
                w_sum = max(w.sum(), 1e-300)
                self.means[s] = (w[:, np.newaxis] * obs).sum(axis=0) / w_sum
                diff = obs - self.means[s]
                cov_s = (
                    w[:, np.newaxis, np.newaxis]
                    * np.einsum("ti,tj->tij", diff, diff)
                ).sum(axis=0) / w_sum
                self.covs[s] = cov_s + 1e-4 * np.eye(self.n_features)

            if abs(log_ll - prev_ll) < tol:
                break
            prev_ll = log_ll

        return log_ll

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_regime(self, obs: np.ndarray) -> np.ndarray:
        """
        Return P(state=0 | obs_{1..t}) for each t — posterior probability
        of being in the mean-reverting state.

        obs : (T, n_features)
        Returns : (T,) array in [0, 1]
        """
        alpha, _scale, _ll = self.forward(obs)
        return alpha[:, 0]

    def viterbi(self, obs: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding — most likely state sequence.

        obs : (T, n_features)
        Returns : (T,) integer array of state labels.
        """
        T = len(obs)
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        for s in range(self.n_states):
            log_delta[0, s] = (
                np.log(max(self.pi[s], 1e-300))
                + float(self._log_emission(obs[0], s))
            )

        for t in range(1, T):
            for s in range(self.n_states):
                log_e = float(self._log_emission(obs[t], s))
                vals = log_delta[t - 1] + np.log(np.maximum(self.A[:, s], 1e-300))
                psi[t, s] = int(np.argmax(vals))
                log_delta[t, s] = vals[psi[t, s]] + log_e

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(log_delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, obs: np.ndarray) -> float:
        """
        Fit HMM using Baum-Welch.

        Initialises means with random sampling from obs, then runs EM.
        Returns final log-likelihood.
        """
        n = len(obs)
        if n < self.n_states:
            raise ValueError(f"Need at least {self.n_states} observations to fit HMM")

        # K-means-like initialisation: random seed points
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n, self.n_states, replace=False)
        self.means = obs[idx].copy()

        # Initialise covariances from global variance
        global_var = np.var(obs, axis=0)
        for s in range(self.n_states):
            self.covs[s] = np.diag(np.maximum(global_var, 1e-4))

        return self.baum_welch(obs)
