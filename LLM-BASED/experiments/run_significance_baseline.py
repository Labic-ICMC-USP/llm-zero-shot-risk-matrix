"""
Same paired-bootstrap comparison as run_significance.py, but against the
TF-IDF + Logistic Regression baseline (baselines/tfidf_logreg/) instead of
between the two evaluated LLMs. Only covers the 12 categories where that
baseline could be evaluated (see the Baselines section of the paper).

Reads the baseline's out-of-fold predictions via a relative path into the
sibling baselines/ folder -- a plain CSV read, not a cross-project Python
import (see baselines/README.md for why that's kept separate).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from evaluation.significance import paired_bootstrap_by_category  # noqa: E402
from evaluation.metrics import SEVERITY_LEVELS  # noqa: E402

SOURCE = "saferproducts_gov"
RESULTS = Path("results") / SOURCE
BASELINE_PREDICTIONS = (
    Path(__file__).resolve().parents[2]
    / "baselines" / "tfidf_logreg" / "results"
    / f"{SOURCE}_tfidf_logreg_predictions.csv"
)


def load_model(model: str) -> pd.DataFrame:
    paths = sorted((RESULTS / model).glob("*.csv"))
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=["report_id", "severity_level"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def compare(name: str, merged: pd.DataFrame, pred_col_llm: str) -> None:
    result = paired_bootstrap_by_category(
        merged,
        category_col="category",
        true_col="victim_severity",
        pred_col_a=pred_col_llm,
        pred_col_b="pred_tfidf",
        labels=SEVERITY_LEVELS,
        n_resamples=2000,
        ci=0.95,
        seed=42,
    )
    print(f"\n{name} vs TF-IDF+LR (12 comparable categories)")
    print(f"  {name:13s}: {result['metric_a']:.3f}")
    print(f"  TF-IDF+LR    : {result['metric_b']:.3f}")
    print(f"  diff         : {result['observed_diff']:.3f}")
    print(f"  95% CI       : [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
    print(f"  p-value      : {result['p_value']:.4f}")
    print(f"  win rate ({name} > TF-IDF+LR): {result['win_rate_a']:.3f}")


def main() -> None:
    if not BASELINE_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"{BASELINE_PREDICTIONS} not found -- run baselines/tfidf_logreg/run.py first."
        )
    tfidf = pd.read_csv(BASELINE_PREDICTIONS)  # category, report_id, victim_severity, pred_tfidf

    llama = load_model("llama-4-scout").rename(columns={"severity_level": "pred_llama"})
    mistral = load_model("mistral-nemo").rename(columns={"severity_level": "pred_mistral"})

    merged = tfidf.merge(llama, on="report_id", how="inner").merge(mistral, on="report_id", how="inner")
    assert len(merged) == len(tfidf), "report_id mismatch between baseline and LLM predictions"

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("llama", "both"):
        compare("llama-4-scout", merged, "pred_llama")
    if which in ("mistral", "both"):
        compare("mistral-nemo", merged, "pred_mistral")


if __name__ == "__main__":
    main()
