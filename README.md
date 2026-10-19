# Programmatic Labeling: a Spam Classifier Without Hand Labels

![tests](https://github.com/Neriarinda/programmatic-labeling-spam/actions/workflows/ci.yml/badge.svg)

Labelling training data by hand is slow and expensive. This project trains a YouTube comment spam classifier
**without using a single hand label for training**. Ten short heuristic rules (*labeling functions*) vote on each comment,
a **label model fitted with Expectation-Maximisation** learns how much to trust each rule, and its probabilistic labels train
a text classifier.

**Stack:** Python · NumPy · scikit-learn · pandas · pytest · GitHub Actions

## Results

Tested on **369 comments from a music video the system never saw** (Shakira), after training on the other four videos:

| Method | Hand labels used for training | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Majority vote of the rules | none | 0.878 | 0.985 | 0.753 | 0.853 |
| Label model alone | none | 0.938 | 0.987 | 0.879 | 0.930 |
| Classifier trained on majority-vote labels | none | 0.924 | 0.974 | 0.862 | 0.915 |
| **Classifier trained on label-model labels** | **none** | **0.949** | 0.975 | **0.914** | **0.944** |
| Same classifier trained on the real labels (upper bound) | 1,384 | 0.938 | 0.993 | 0.874 | 0.930 |

**Takeaways:**

1. **The learned label model beats a simple vote** (F1 0.930 vs 0.853). Weighting rules by their estimated accuracy matters.
2. **Training a classifier on top adds more** (0.944). It learns word and character patterns the rules never mention,
   so it generalises beyond them.
3. **It matches the fully supervised model.** With one small test set, the 0.014 gap in its favour is within noise,
   so the fair claim is "as good as hand labels", not "better".

## How it works

```
comments ──► 10 labeling functions ──► label matrix L (n × 10, values SPAM / HAM / ABSTAIN)
                                              │
                                              ▼
                               Label model (EM, Dawid–Skene style)
                               learns each rule's accuracy per class + the spam rate
                                              │  P(spam | votes)
                                              ▼
              TF-IDF (word 1–2 grams + char 3–5 grams) + logistic regression,
              trained on the probabilistic labels via sample weights
```

**Labeling functions** ([`labeling_functions.py`](src/weaklabel/labeling_functions.py)) are tiny and noisy on purpose:
links, "check out my channel", "subscribe", money and prize words, ALL CAPS, very long posts → spam;
short comments, talk about the song, "who's still listening in 2024?" → ham. On the 200-comment dev set they range from 46% to 100% accurate.

**Label model** ([`label_model.py`](src/weaklabel/label_model.py)) is written from scratch in NumPy. It models, for each rule and each class,
how often the rule fires and how often it is right, and fits these with Expectation-Maximisation
(starting from majority vote). A unit test checks that it spots a deliberately bad rule (30% accurate)
**without seeing any labels**, and that it beats majority vote on simulated data.

**Diagnostics** ([`analysis.py`](src/weaklabel/analysis.py)) report each rule's coverage, overlap, conflict rate and dev-set accuracy.
This is the loop you iterate on when writing rules (`results/lf_summary_dev.md`). The rules together cover 82% of training comments.

## Run it

```bash
pip install -e ".[dev]"
weak-label data --out results     # prints the rule diagnostics and the results table, writes results/results.json
pytest -q                         # 16 tests
```

## Data

[YouTube Spam Collection](https://www.kaggle.com/datasets/prashant111/youtube-spam-collection) on Kaggle, originally from the
UCI Machine Learning Repository (Alberto, Lochter & Almeida, *TubeSpam: Comment Spam Filtering on YouTube*, ICMLA 2015), licensed CC BY 4.0.
It contains 1,956 comments from five popular music videos and is included in `data/`. The loader accepts both the UCI and Kaggle file names.

---

Built by Neria Fay Arinda, with help from Claude (Anthropic) as a coding assistant.
