"""
Majority-class baseline: for each product category, always predicts the
most common severity level (from victim_severity ground truth). Standard
sanity-check baseline -- if the LLM-based approach can't beat this, F1
numbers alone don't mean much.

Reads ground truth from LLM-BASED's already-scored results (same instance
population the evaluated LLM models were run on), via a relative path --
no import from the LLM-BASED package, see ../README.md for why.

Does not touch anything under LLM-BASED/ -- writes only to ./results/ here.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

SOURCE = "saferproducts_gov"
# Any evaluated model works as the source of ground truth + instance
# population -- victim_severity and report_id don't depend on which model
# scored the row. llama-4-scout picked arbitrarily.
REF_MODEL = "llama-4-scout"
SEVERITY_LEVELS = [1, 2, 3, 4]

HERE = Path(__file__).resolve().parent
LLM_BASED_DIR = HERE.parents[1] / "LLM-BASED"
SCORED_DIR = LLM_BASED_DIR / "results" / SOURCE / REF_MODEL
OUT_DIR = HERE / "results"


def main() -> None:
    paths = sorted(SCORED_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No scored CSVs found at {SCORED_DIR}")

    rows = []
    for path in paths:
        category = path.stem.removesuffix("_scored")
        df = pd.read_csv(path, usecols=["report_id", "victim_severity"])

        y_true = df["victim_severity"]
        majority = y_true.mode().iloc[0]
        y_pred = [majority] * len(df)

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0, labels=SEVERITY_LEVELS
        )
        f1_macro = report["macro avg"]["f1-score"]

        rows.append(
            {
                "category": category,
                "n": len(df),
                "majority_class": majority,
                "f1_macro": f1_macro,
            }
        )
        print(f"[{category}] n={len(df):5d}  majority={majority}  F1(macro)={f1_macro:.3f}")

    result = pd.DataFrame(rows).sort_values("category")
    mean_f1 = result["f1_macro"].mean()
    print(f"\nMean F1(macro) across categories: {mean_f1:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{SOURCE}_majority_class.csv"
    result.to_csv(out_path, index=False)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
