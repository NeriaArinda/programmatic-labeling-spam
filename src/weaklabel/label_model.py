"""Combine noisy LF votes into probabilistic labels.

`MajorityVote` is the obvious baseline. `LabelModel` learns, without any ground
truth, how accurate each LF is (separately for the SPAM and HAM classes) using
Expectation-Maximisation in the style of Dawid & Skene (1979). LFs that agree with
the consensus get more weight; LFs that disagree get less.
"""

from __future__ import annotations

import numpy as np

from . import ABSTAIN


class MajorityVote:
    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        spam = (L == 1).sum(axis=1)
        ham = (L == 0).sum(axis=1)
        total = spam + ham
        p_spam = np.where(total > 0, spam / np.maximum(total, 1), 0.5)
        return np.column_stack([1 - p_spam, p_spam])


class LabelModel:
    """Binary Dawid-Skene label model.

    Parameters learned:
      * ``prior_``      - P(y = 1)
      * ``accuracy_``   - shape (n_lfs, 2): P(LF votes y | y, LF did not abstain)
      * ``propensity_`` - shape (n_lfs, 2): P(LF does not abstain | y)
    """

    def __init__(self, n_iter: int = 100, tol: float = 1e-6, smoothing: float = 1.0):
        self.n_iter = n_iter
        self.tol = tol
        self.smoothing = smoothing

    def fit(self, L: np.ndarray) -> "LabelModel":
        L = np.asarray(L)
        q = MajorityVote().predict_proba(L)[:, 1]  # initialise from majority vote
        prev = -np.inf
        for self.n_iter_ in range(1, self.n_iter + 1):
            self._m_step(L, q)
            q, log_lik = self._e_step(L)
            if abs(log_lik - prev) < self.tol:
                break
            prev = log_lik
        self.log_likelihood_ = log_lik
        return self

    def _m_step(self, L: np.ndarray, q: np.ndarray) -> None:
        s = self.smoothing
        w = np.column_stack([1 - q, q])  # responsibility for y=0, y=1
        self.prior_ = float(np.clip(q.mean(), 0.01, 0.99))
        n_lfs = L.shape[1]
        self.accuracy_ = np.zeros((n_lfs, 2))
        self.propensity_ = np.zeros((n_lfs, 2))
        for j in range(n_lfs):
            voted = L[:, j] != ABSTAIN
            for y in (0, 1):
                weight_y = w[:, y]
                votes = weight_y[voted].sum()
                correct = weight_y[voted & (L[:, j] == y)].sum()
                self.accuracy_[j, y] = (correct + s) / (votes + 2 * s)
                self.propensity_[j, y] = (votes + s) / (weight_y.sum() + 2 * s)

    def _e_step(self, L: np.ndarray) -> tuple[np.ndarray, float]:
        log_p = np.zeros((L.shape[0], 2))
        log_p[:, 0] = np.log(1 - self.prior_)
        log_p[:, 1] = np.log(self.prior_)
        for j in range(L.shape[1]):
            vote = L[:, j]
            for y in (0, 1):
                acc = self.accuracy_[j, y]
                prop = self.propensity_[j, y]
                term = np.where(
                    vote == ABSTAIN,
                    np.log(1 - prop),
                    np.log(prop) + np.where(vote == y, np.log(acc), np.log(1 - acc)),
                )
                log_p[:, y] += term
        log_norm = np.logaddexp(log_p[:, 0], log_p[:, 1])
        return np.exp(log_p[:, 1] - log_norm), float(log_norm.sum())

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        p_spam, _ = self._e_step(np.asarray(L))
        return np.column_stack([1 - p_spam, p_spam])

    def predict(self, L: np.ndarray) -> np.ndarray:
        return (self.predict_proba(L)[:, 1] >= 0.5).astype(int)

    def weights(self, names: list[str]) -> list[dict]:
        """Learned accuracy of each LF on the class it votes for."""
        return [
            {"lf": n, "acc_when_spam": round(float(a[1]), 3), "acc_when_ham": round(float(a[0]), 3)}
            for n, a in zip(names, self.accuracy_)
        ]
