"""Labeling functions (LFs): small, noisy heuristics that vote SPAM, HAM or ABSTAIN.

Each LF looks at one comment and either votes or abstains. None of them needs to be
accurate on its own; the label model combines them and learns how much to trust each.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from . import ABSTAIN, HAM, SPAM

LabelingFunction = Callable[[str], int]
REGISTRY: dict[str, LabelingFunction] = {}


def lf(fn: LabelingFunction) -> LabelingFunction:
    REGISTRY[fn.__name__] = fn
    return fn


URL = re.compile(r"(https?://|www\.|\.com\b|\.ly/|bit\.ly)", re.I)
SELF_PROMO = re.compile(r"\b(my|our)\s+(channel|video|vids?|page|music|songs?|band|account)\b", re.I)
CALL_TO_ACTION = re.compile(r"\b(subscribe|sub\s?4\s?sub|check\s+(it\s+)?out|please\s+(like|share|visit)|follow\s+me)\b", re.I)
MONEY = re.compile(r"(\$\d|\bfree\b|\bearn\b|\bmoney\b|\bgift\s?card|\bwin\b|\bprize\b|\bcash\b)", re.I)
SONG_TALK = re.compile(r"\b(song|beat|voice|lyrics|music video|dance|chorus|singer|best|love (this|it|her|him))\b", re.I)
NOSTALGIA = re.compile(r"\b(20\d\d|years? ago|still (listening|watching)|who('?s| is) (here|watching|listening))\b", re.I)
VIEWS = re.compile(r"\b(billion|million|views?)\b", re.I)


@lf
def has_link(text: str) -> int:
    return SPAM if URL.search(text) else ABSTAIN


@lf
def self_promotion(text: str) -> int:
    return SPAM if SELF_PROMO.search(text) else ABSTAIN


@lf
def call_to_action(text: str) -> int:
    return SPAM if CALL_TO_ACTION.search(text) else ABSTAIN


@lf
def money_or_prizes(text: str) -> int:
    return SPAM if MONEY.search(text) else ABSTAIN


@lf
def shouting(text: str) -> int:
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return SPAM
    return ABSTAIN


@lf
def very_long(text: str) -> int:
    return SPAM if len(text.split()) > 60 else ABSTAIN


@lf
def short_comment(text: str) -> int:
    return HAM if len(text.split()) <= 5 and not URL.search(text) else ABSTAIN


@lf
def talks_about_song(text: str) -> int:
    return HAM if SONG_TALK.search(text) and not URL.search(text) else ABSTAIN


@lf
def nostalgia(text: str) -> int:
    return HAM if NOSTALGIA.search(text) else ABSTAIN


@lf
def view_count_talk(text: str) -> int:
    return HAM if VIEWS.search(text) and not SELF_PROMO.search(text) else ABSTAIN


def apply(texts: pd.Series | list[str], lfs: dict[str, LabelingFunction] | None = None) -> np.ndarray:
    """Return the label matrix L with shape (n_comments, n_lfs)."""
    lfs = lfs or REGISTRY
    texts = list(texts)
    L = np.full((len(texts), len(lfs)), ABSTAIN, dtype=int)
    for j, fn in enumerate(lfs.values()):
        for i, text in enumerate(texts):
            L[i, j] = fn(text if isinstance(text, str) else "")
    return L
