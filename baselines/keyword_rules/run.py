"""
Keyword/rule-based severity baseline: cascading keyword match over the
incident description text, highest severity first. No injury-related
keyword found -> defaults to level 1 (no incident, no injury).

Same conventions as ../majority_class/run.py -- reads ground truth from
LLM-BASED's already-scored results via a relative path (no package import),
computes metrics directly with sklearn. See ../README.md.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

SOURCE = "saferproducts_gov"
REF_MODEL = "llama-4-scout"  # arbitrary -- any evaluated model has the same ground truth/instances
SEVERITY_LEVELS = [1, 2, 3, 4]

HERE = Path(__file__).resolve().parent
LLM_BASED_DIR = HERE.parents[1] / "LLM-BASED"
SCORED_DIR = LLM_BASED_DIR / "results" / SOURCE / REF_MODEL
OUT_DIR = HERE / "results"

# Cascading priority: checked in this order, first match wins. Level 1 has
# no keyword list -- it's the default when nothing else matches.
KEYWORDS = {
    4: ["died", "death", "deceased", "fatality", "fatal", "killed", "passed away"],
    3: ["injury", "injured", "hurt", "wound", "wounded", "burn", "burned",
        "bruise", "bruised", "cut", "laceration", "fracture", "hospitalized",
        "hospital", "bleeding", "bled", "sprain"],
    2: ["malfunction", "malfunctioned", "defect", "defective", "broke", "broken",
        "failed", "failure", "recall", "recalled", "hazard", "smoke",
        "overheated", "overheating", "sparked", "shock", "leak", "leaked",
        "fell apart", "stopped working"],
}

# Longest phrases first within each level, so e.g. "fell apart" is tried
# before a hypothetical shorter overlapping term.
for level in KEYWORDS:
    KEYWORDS[level].sort(key=len, reverse=True)
    KEYWORDS[level] = [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in KEYWORDS[level]]


def classify(text: str) -> int:
    if not isinstance(text, str):
        return 1
    for level in (4, 3, 2):
        if any(pattern.search(text) for pattern in KEYWORDS[level]):
            return level
    return 1


def main() -> None:
    paths = sorted(SCORED_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No scored CSVs found at {SCORED_DIR}")

    rows = []
    for path in paths:
        category = path.stem.removesuffix("_scored")
        df = pd.read_csv(path, usecols=["report_id", "incident_description", "victim_severity"])

        y_true = df["victim_severity"]
        y_pred = df["incident_description"].apply(classify)

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0, labels=SEVERITY_LEVELS
        )
        f1_macro = report["macro avg"]["f1-score"]

        rows.append({"category": category, "n": len(df), "f1_macro": f1_macro})
        print(f"[{category}] n={len(df):5d}  F1(macro)={f1_macro:.3f}")

    result = pd.DataFrame(rows).sort_values("category")
    mean_f1 = result["f1_macro"].mean()
    print(f"\nMean F1(macro) across categories: {mean_f1:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{SOURCE}_keyword_rules.csv"
    result.to_csv(out_path, index=False)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
