"""
Computes the paired-bootstrap significance numbers cited in the paper
(paired bootstrap, llama-4-scout vs. mistral-nemo).

Invoked on demand, not part of the automatic evaluate.py/visualize.py
pipeline -- see src/evaluation/significance.py's module docstring.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from evaluation.significance import paired_bootstrap_by_category  # noqa: E402
from evaluation.metrics import SEVERITY_LEVELS  # noqa: E402

SOURCE = "saferproducts_gov"
RESULTS = Path("results") / SOURCE


def load_model(model: str) -> pd.DataFrame:
    paths = sorted((RESULTS / model).glob("*.csv"))
    dfs = []
    for p in paths:
        df = pd.read_csv(p, usecols=["report_id", "victim_severity", "severity_level"])
        df["category"] = p.stem.removesuffix("_scored")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    llama = load_model("llama-4-scout").rename(columns={"severity_level": "pred_llama"})
    mistral = load_model("mistral-nemo").rename(columns={"severity_level": "pred_mistral"})

    merged = llama.merge(
        mistral[["report_id", "pred_mistral"]], on="report_id", how="inner"
    )
    assert len(merged) == len(llama) == len(mistral), "report_id mismatch between models"

    result = paired_bootstrap_by_category(
        merged,
        category_col="category",
        true_col="victim_severity",
        pred_col_a="pred_llama",
        pred_col_b="pred_mistral",
        labels=SEVERITY_LEVELS,
        n_resamples=2000,
        ci=0.95,
        seed=42,
    )

    print("Severity F1-macro, llama-4-scout vs mistral-nemo (vs. ground truth)")
    print(f"  llama-4-scout : {result['metric_a']:.3f}")
    print(f"  mistral-nemo  : {result['metric_b']:.3f}")
    print(f"  diff          : {result['observed_diff']:.3f}")
    print(f"  95% CI        : [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
    print(f"  p-value       : {result['p_value']:.4f}")
    print(f"  win rate (llama > mistral across resamples): {result['win_rate_a']:.3f}")
    print(f"  n_categories={result['n_categories']}  n_resamples={result['n_resamples']}")


if __name__ == "__main__":
    main()
