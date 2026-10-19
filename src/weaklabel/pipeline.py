"""End-to-end experiment: weak labels -> classifier -> compare against baselines.

    weak-label data/ --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline

from . import labeling_functions as lfs
from .analysis import lf_summary, total_coverage
from .data import load_folder, split
from .label_model import LabelModel, MajorityVote


def text_classifier() -> Pipeline:
    """Word + character n-grams so the model generalises past the LF keywords."""
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0)),
    ])


def fit_on_soft_labels(texts, p_spam: np.ndarray, min_confidence: float = 0.0) -> Pipeline:
    """Train on probabilistic labels by duplicating each row once per class, weighted."""
    keep = np.abs(p_spam - 0.5) * 2 >= min_confidence
    texts = np.asarray(texts)[keep]
    p = p_spam[keep]
    X = np.concatenate([texts, texts])
    y = np.concatenate([np.ones(len(texts), int), np.zeros(len(texts), int)])
    w = np.concatenate([p, 1 - p])
    model = text_classifier()
    model.fit(X, y, clf__sample_weight=w)
    return model


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def run(folder: Path, out: Path, test_video: str = "Shakira") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train, dev, test = split(load_folder(folder), test_video=test_video)
    names = list(lfs.REGISTRY)

    L_train = lfs.apply(train["content"])
    L_dev = lfs.apply(dev["content"])
    L_test = lfs.apply(test["content"])

    label_model = LabelModel().fit(L_train)
    mv = MajorityVote()

    results = {}
    # 1. Votes alone (no classifier). Uncovered comments default to HAM.
    results["majority_vote_only"] = scores(test["class"], (mv.predict_proba(L_test)[:, 1] > 0.5).astype(int))
    results["label_model_only"] = scores(test["class"], label_model.predict(L_test))

    # 2. Classifiers trained on weak labels - no hand labels used.
    for name, model in (("majority_vote", mv), ("label_model", label_model)):
        p = model.predict_proba(L_train)[:, 1]
        clf = fit_on_soft_labels(train["content"], p)
        results[f"classifier_on_{name}_labels"] = scores(test["class"], clf.predict(test["content"]))

    # 3. Upper bound: the same classifier trained on the real hand labels.
    supervised = text_classifier().fit(train["content"], train["class"])
    results["classifier_on_gold_labels (upper bound)"] = scores(test["class"], supervised.predict(test["content"]))

    summary = {
        "data": {"train": len(train), "dev": len(dev), "test": len(test), "test_video": test_video,
                 "train_spam_rate_hidden": round(float(train["class"].mean()), 3)},
        "coverage": {"train": round(total_coverage(L_train), 3), "test": round(total_coverage(L_test), 3)},
        "label_model": {"iterations": label_model.n_iter_, "learned_spam_prior": round(label_model.prior_, 3),
                        "lf_weights": label_model.weights(names)},
        "results": results,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    dev_table = lf_summary(L_dev, names, dev["class"].to_numpy())
    (out / "lf_summary_dev.md").write_text(dev_table.to_markdown(), encoding="utf-8")

    print("Labeling functions on the dev set:\n")
    print(dev_table.to_string())
    print(f"\nLF coverage: train {summary['coverage']['train']:.0%}, test {summary['coverage']['test']:.0%}")
    print(f"\nTest set: {len(test)} comments from the unseen {test_video} video\n")
    width = max(map(len, results))
    print(f"{'method':{width}}  accuracy  precision  recall   f1")
    for method, s in results.items():
        print(f"{method:{width}}  {s['accuracy']:.3f}     {s['precision']:.3f}      {s['recall']:.3f}    {s['f1']:.3f}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with the five Youtube0*-*.csv files")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--test-video", default="Shakira")
    args = parser.parse_args(argv)
    run(args.folder, args.out, args.test_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
