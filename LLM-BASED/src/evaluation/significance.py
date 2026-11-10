"""
Paired bootstrap significance testing for comparing two models' (or a
model vs. a baseline's) predictions on the same instances -- confidence
intervals and two-sided p-values for a metric difference, as requested by
STIL Reviews 3 and 4 ("confidence intervals, statistical significance
tests").

Not wired into evaluate.py's automatic per-category loop: this is invoked
on demand for specific comparisons (see experiments/run_significance.py),
not computed for every category/model pair by default.

Resampling is stratified by category and matches the granularity at which
the paper's headline numbers are computed: each category is resampled
independently (same size as that category, with replacement), the metric
is recomputed per category for both sides being compared, and the two
per-category results are averaged into a single number the same way
visualize.py's mean-of-per-category figures are -- not a single metric
pooled over all instances regardless of category. This keeps the
confidence interval attached to the same quantity that is actually
reported, rather than a different (if related) pooled statistic.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def paired_bootstrap_by_category(
    df: pd.DataFrame,
    category_col: str,
    true_col: str,
    pred_col_a: str,
    pred_col_b: str,
    labels: list[int],
    metric_fn: Callable[[np.ndarray, np.ndarray, list[int]], float] = f1_macro,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    df must have one row per instance, with a category column and the
    true label plus both models' predictions already aligned (e.g. via
    metrics._paired_columns's report_id merge before calling this).

    Returns observed metric for each side, their difference, a
    percentile confidence interval on the difference, a two-sided
    bootstrap p-value against the null of no difference, and the win
    rate (fraction of resamples where A > B) -- easier to read than the
    p-value alone.
    """
    rng = np.random.default_rng(seed)

    cat_arrays = [
        (
            sub[true_col].to_numpy(),
            sub[pred_col_a].to_numpy(),
            sub[pred_col_b].to_numpy(),
        )
        for _, sub in df.groupby(category_col)
    ]

    observed_a = float(np.mean([metric_fn(t, a, labels) for t, a, b in cat_arrays]))
    observed_b = float(np.mean([metric_fn(t, b, labels) for t, a, b in cat_arrays]))
    observed_diff = observed_a - observed_b

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        fa, fb = [], []
        for t, a, b in cat_arrays:
            idx = rng.integers(0, len(t), size=len(t))
            fa.append(metric_fn(t[idx], a[idx], labels))
            fb.append(metric_fn(t[idx], b[idx], labels))
        diffs[i] = float(np.mean(fa)) - float(np.mean(fb))

    alpha = 1 - ci
    ci_low, ci_high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])

    # Two-sided bootstrap p-value: recenter the resampled diffs on 0 (the
    # null of no difference) and see how often a value at least as
    # extreme as the observed one would occur under that null.
    centered = diffs - diffs.mean()
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_diff)))

    return {
        "metric_a": observed_a,
        "metric_b": observed_b,
        "observed_diff": observed_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "win_rate_a": float(np.mean(diffs > 0)),
        "n_resamples": n_resamples,
        "n_categories": len(cat_arrays),
    }
