"""Load the YouTube Spam Collection (five CSVs, one per music video)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

VIDEOS = ["Psy", "KatyPerry", "LMFAO", "Eminem", "Shakira"]


def _video_name(stem: str) -> str:
    """'Youtube01-Psy' or 'youtube01_psy' -> 'Psy'."""
    name = re.split(r"[-_]", stem, maxsplit=1)[-1]
    for video in VIDEOS:
        if video.lower() == name.lower():
            return video
    return name


def load_folder(folder: str | Path) -> pd.DataFrame:
    # The UCI original uses "Youtube01-Psy.csv"; some Kaggle mirrors use "youtube01_psy.csv".
    paths = sorted(p for p in Path(folder).glob("*.csv") if re.match(r"youtube0\d[-_]", p.name, re.I))
    frames = []
    for path in paths:
        df = pd.read_csv(path).rename(columns=str.lower).rename(columns={"tag": "class"})
        df["video"] = _video_name(path.stem)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no Youtube0X-<video>.csv files in {folder}")
    df = pd.concat(frames, ignore_index=True)
    df = df[["comment_id", "author", "content", "class", "video"]].copy()
    # UCI files use 1/0; the Kaggle re-upload uses true/false.
    df["class"] = df["class"].map(lambda v: 1 if str(v).strip().lower() in ("1", "true") else 0)
    df["content"] = df["content"].fillna("").astype(str)
    return df.drop_duplicates(subset="comment_id").reset_index(drop=True)


def split(df: pd.DataFrame, test_video: str = "Shakira", dev_size: int = 200, seed: int = 0):
    """Train on four videos, test on an unseen one.

    * ``train`` - labels are hidden; only the LFs see this text
    * ``dev``   - a small hand-labelled sample for writing/checking LFs
    * ``test``  - every comment from ``test_video``, never touched during development
    """
    test = df[df["video"] == test_video].reset_index(drop=True)
    rest = df[df["video"] != test_video].sample(frac=1, random_state=seed).reset_index(drop=True)
    dev = rest.iloc[:dev_size].reset_index(drop=True)
    train = rest.iloc[dev_size:].reset_index(drop=True)
    return train, dev, test
