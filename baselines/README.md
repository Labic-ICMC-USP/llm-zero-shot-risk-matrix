# Baselines

Non-LLM baselines for the paper's severity results. They are a comparison
point against the LLM-based pipeline in [`../LLM-BASED/`](../LLM-BASED/README.md),
so they live in their own folder.

## Convention

- One subfolder per baseline method, each self-contained: its own `run.py`
  and its own `results/` output. There is no shared code between methods.
- Ground truth and the instance population are read from
  `../LLM-BASED/results/<source>/<model>/*_scored.csv` via a relative path,
  using plain pandas (no import from the `LLM-BASED` Python package). The
  folder layout (`baselines/` next to `LLM-BASED/`) must therefore be kept.
- Metrics are computed with `sklearn.metrics.classification_report`, the
  same way `LLM-BASED/src/evaluation/metrics.py` does it internally.

## Methods

- `majority_class/`: always predicts each category's most common severity
  level (from the `victim_severity` ground truth).
- `keyword_rules/`: assigns severity with a small cascading set of
  injury-related keywords found in the incident description (highest level
  first). It defaults to level 1 when no keyword matches.
- `tfidf_logreg/`: TF-IDF features (unigrams and bigrams, English stop words
  removed, vocabulary capped at 10,000 terms) with a class-balanced logistic
  regression, evaluated with stratified cross-validation within each
  category. It uses 5 folds, or as many as the size of the rarest severity
  level when that is smaller. The vectorizer is refit on each fold's training
  split. It writes the out-of-fold predictions that the significance script
  `LLM-BASED/experiments/run_significance_baseline.py` reads.

## Running

From the repository root (the scripts resolve their paths from their own
location, so the working directory does not matter), with the environment
from the main [README](../README.md) active:

```bash
python baselines/majority_class/run.py
python baselines/keyword_rules/run.py
python baselines/tfidf_logreg/run.py
```

Outputs are written to each method's `results/` folder (included in this
repository).
