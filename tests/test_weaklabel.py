import numpy as np
import pandas as pd
import pytest

from weaklabel import ABSTAIN, HAM, SPAM
from weaklabel import labeling_functions as lfs
from weaklabel.analysis import lf_summary, total_coverage
from weaklabel.label_model import LabelModel, MajorityVote
from weaklabel.pipeline import run


@pytest.mark.parametrize(
    "fn, text, expected",
    [
        (lfs.has_link, "visit http://spam.example now", SPAM),
        (lfs.has_link, "great song", ABSTAIN),
        (lfs.self_promotion, "please watch my channel", SPAM),
        (lfs.call_to_action, "Subscribe for more!", SPAM),
        (lfs.money_or_prizes, "win a free gift card", SPAM),
        (lfs.shouting, "THIS IS THE BEST CHANNEL EVER GO NOW", SPAM),
        (lfs.short_comment, "love it", HAM),
        (lfs.short_comment, "www.x.com", ABSTAIN),
        (lfs.talks_about_song, "her voice is amazing", HAM),
        (lfs.nostalgia, "who is still listening in 2024?", HAM),
    ],
)
def test_labeling_functions(fn, text, expected):
    assert fn(text) == expected


def test_apply_shape_and_values():
    L = lfs.apply(["check out my channel http://x.com", "nice song", None])
    assert L.shape == (3, len(lfs.REGISTRY))
    assert set(np.unique(L)) <= {ABSTAIN, HAM, SPAM}
    assert (L[2] == ABSTAIN).sum() >= len(lfs.REGISTRY) - 1


def simulate(n=3000, seed=0):
    """Ground truth plus 3 good LFs and 1 adversarial LF that is right only 30% of the time."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    cols = []
    for acc, cov in [(0.9, 0.6), (0.85, 0.5), (0.8, 0.7), (0.3, 0.6)]:
        fires = rng.random(n) < cov
        correct = rng.random(n) < acc
        col = np.where(correct, y, 1 - y)
        cols.append(np.where(fires, col, ABSTAIN))
    return np.column_stack(cols), y


def test_label_model_learns_lf_quality():
    L, y = simulate()
    model = LabelModel().fit(L)
    learned = model.accuracy_.mean(axis=1)
    assert learned[0] > 0.8
    assert learned[3] < 0.45  # spots the bad LF without seeing any labels
    assert abs(model.prior_ - 0.4) < 0.08


def test_label_model_beats_majority_vote():
    L, y = simulate()
    lm_acc = (LabelModel().fit(L).predict(L) == y).mean()
    mv_acc = ((MajorityVote().predict_proba(L)[:, 1] > 0.5).astype(int) == y).mean()
    assert lm_acc > mv_acc


def test_majority_vote_ties_and_abstains():
    L = np.array([[1, 0], [-1, -1], [1, 1]])
    p = MajorityVote().predict_proba(L)[:, 1]
    assert list(p) == [0.5, 0.5, 1.0]


def test_lf_summary():
    L = np.array([[1, -1], [1, 0], [-1, 0]])
    table = lf_summary(L, ["a", "b"], np.array([1, 0, 0]))
    assert table.loc["a", "coverage"] == round(2 / 3, 3)
    assert table.loc["a", "conflicts"] == round(1 / 3, 3)
    assert table.loc["b", "accuracy"] == 1.0
    assert total_coverage(L) == 1.0


def test_pipeline_runs_end_to_end(tmp_path):
    spam = ["check out my channel http://x.com", "subscribe to my channel please", "free money win now www.x.com",
            "visit my page for free gift card", "please like and share my video"]
    ham = ["love this song", "her voice is amazing", "who is still listening in 2024",
           "best song ever", "this beat is great"]
    rows = []
    for v, video in enumerate(["Psy", "KatyPerry", "LMFAO", "Eminem", "Shakira"]):
        for i in range(60):
            text = (spam if i % 2 else ham)[i % 5] + f" {i}"
            rows.append({"COMMENT_ID": f"{video}{i}", "AUTHOR": "a", "DATE": "", "CONTENT": text, "CLASS": i % 2})
        pd.DataFrame(rows[-60:]).to_csv(tmp_path / f"Youtube0{v + 1}-{video}.csv", index=False)

    summary = run(tmp_path, tmp_path / "out")
    assert summary["results"]["classifier_on_label_model_labels"]["f1"] > 0.9
    assert (tmp_path / "out" / "results.json").exists()
