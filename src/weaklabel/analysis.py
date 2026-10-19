"""Diagnostics for labeling functions: how often they fire, overlap, conflict, and are right."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ABSTAIN


def lf_summary(L: np.ndarray, names: list[str], y: np.ndarray | None = None) -> pd.DataFrame:
    voted = L != ABSTAIN
    n_votes = voted.sum(axis=1)
    rows = []
    for j, name in enumerate(names):
        col = L[:, j]
        fired = voted[:, j]
        polarity = sorted({int(v) for v in col[fired]})
        others = np.delete(L, j, axis=1)
        conflicts = fired & ((others != ABSTAIN) & (others != col[:, None])).any(axis=1)
        row = {
            "lf": name,
            "polarity": polarity,
            "coverage": fired.mean(),
            "overlaps": (fired & (n_votes > 1)).mean(),
            "conflicts": conflicts.mean(),
        }
        if y is not None:
            row["accuracy"] = (col[fired] == y[fired]).mean() if fired.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("lf").round(3)


def total_coverage(L: np.ndarray) -> float:
    """Share of comments where at least one LF voted."""
    return float((L != ABSTAIN).any(axis=1).mean())
