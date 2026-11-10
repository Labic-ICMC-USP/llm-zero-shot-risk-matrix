"""
TF-IDF + Logistic Regression baseline: 5-fold stratified CV per category
(fewer folds if a category's rarest class doesn't have 5 members), TF-IDF
features (unigrams+bigrams, English stopwords removed, capped vocabulary),
class_weight="balanced" to account for severity class imbalance. The
vectorizer is refit on each fold's training split only, to avoid leaking
test-fold vocabulary into training.

Same conventions as ../majority_class/run.py -- reads ground truth from
LLM-BASED's already-scored results via a relative path (no package import).
See ../README.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

SOURCE = "saferproducts_gov"
REF_MODEL = "llama-4-scout"  # arbitrary -- any evaluated model has the same ground truth/instances
SEVERITY_LEVELS = [1, 2, 3, 4]
N_SPLITS = 5
SEED = 42

HERE = Path(__file__).resolve().parent
LLM_BASED_DIR = HERE.parents[1] / "LLM-BASED"
SCORED_DIR = LLM_BASED_DIR / "results" / SOURCE / REF_MODEL
OUT_DIR = HERE / "results"


def make_pipeline_pieces():
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    return vectorizer, clf


def evaluate_category(df: pd.DataFrame) -> tuple[float, int, pd.DataFrame | None]:
    X_text = df["incident_description"].fillna("")
    y = df["victim_severity"].to_numpy()
    report_ids = df["report_id"].to_numpy()

    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = min(N_SPLITS, min_class_count)
    if n_splits < 2:
        return float("nan"), n_splits, None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_f1s = []
    oof_report_ids, oof_true, oof_pred = [], [], []
    for train_idx, test_idx in skf.split(X_text, y):
        vectorizer, clf = make_pipeline_pieces()
        X_train = vectorizer.fit_transform(X_text.iloc[train_idx])
        X_test = vectorizer.transform(X_text.iloc[test_idx])
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)

        report = classification_report(
            y[test_idx], y_pred, output_dict=True, zero_division=0, labels=SEVERITY_LEVELS
        )
        fold_f1s.append(report["macro avg"]["f1-score"])

        # Out-of-fold predictions: each instance is predicted exactly once,
        # by a model that never saw it during training -- needed for the
        # per-instance significance comparison against the LLMs later.
        oof_report_ids.extend(report_ids[test_idx])
        oof_true.extend(y[test_idx])
        oof_pred.extend(y_pred)

    oof_df = pd.DataFrame(
        {"report_id": oof_report_ids, "victim_severity": oof_true, "pred_tfidf": oof_pred}
    )
    return float(np.mean(fold_f1s)), n_splits, oof_df


def main() -> None:
    paths = sorted(SCORED_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No scored CSVs found at {SCORED_DIR}")

    rows = []
    oof_frames = []
    for path in paths:
        category = path.stem.removesuffix("_scored")
        df = pd.read_csv(path, usecols=["report_id", "incident_description", "victim_severity"])

        f1_macro, n_splits, oof_df = evaluate_category(df)
        rows.append({"category": category, "n": len(df), "n_splits": n_splits, "f1_macro": f1_macro})
        if np.isnan(f1_macro):
            print(f"[{category}] n={len(df):5d}  SKIPPED (rarest class has <2 samples)")
        else:
            print(f"[{category}] n={len(df):5d}  folds={n_splits}  F1(macro)={f1_macro:.3f}")
            oof_df.insert(0, "category", category)
            oof_frames.append(oof_df)

    result = pd.DataFrame(rows).sort_values("category")
    mean_f1 = result["f1_macro"].mean(skipna=True)
    print(f"\nMean F1(macro) across categories: {mean_f1:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{SOURCE}_tfidf_logreg.csv"
    result.to_csv(out_path, index=False)
    print(f"Written to {out_path}")

    predictions = pd.concat(oof_frames, ignore_index=True)
    predictions_path = OUT_DIR / f"{SOURCE}_tfidf_logreg_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    print(f"Out-of-fold predictions written to {predictions_path}")


if __name__ == "__main__":
    main()
