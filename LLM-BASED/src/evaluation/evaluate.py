"""
Evaluates a model's scored results for a source, category by category,
writing to results/<source>/<model>/evaluations/<category>/.

risk_matrix.csv lives directly there -- it's just this model's own
predictions, the same regardless of what you compare it against.
Everything that DOES depend on the comparison target lives one level
deeper, in a vs_<target>/ subfolder (see metrics.comparison_target), so
running with --ref and running in ground-truth mode never overwrite
each other. Ground-truth mode (no --ref) has one shape only, so its
files sit directly in vs_ground_truth/. --ref mode can produce two
different comparisons, so vs_<ref>/ splits further into normal/
(per-variable metrics, full-ref risk_diff) and, if this source's ground
truth covers exactly one axis, mixed/ (ground truth for that axis,
--ref for the other) -- see metrics.comparison_subdir.
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation import metrics
from utils.source_config import load_source_config

RESULTS_DIR = Path("results")

SEVERITY_COL = "severity_level"
LIKELIHOOD_COL = "occurrence_likelihood"

# (variable name, predicted column, ordinal levels) -- the two things
# functions below need to treat severity/likelihood generically.
VARIABLES = [
    ("severity", SEVERITY_COL, metrics.SEVERITY_LEVELS),
    ("likelihood", LIKELIHOOD_COL, metrics.LIKELIHOOD_LEVELS),
]


def _scored_categories(source: str, model: str) -> dict[str, pd.DataFrame]:
    model_dir = RESULTS_DIR / source / model
    if not model_dir.exists():
        raise FileNotFoundError(f"No results found at {model_dir}.")
    paths = sorted(model_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No scored CSVs found in {model_dir}.")
    return {p.stem: pd.read_csv(p) for p in paths}


def evaluate_model(source: str, model: str, ref: Optional[str] = None) -> None:
    categories = _scored_categories(source, model)
    ref_categories = _scored_categories(source, ref) if ref else None
    target = metrics.comparison_target(ref)

    gt_columns: dict[str, str] = load_source_config(source).get("ground_truth_columns") or {}

    total = len(categories)
    for i, (cat, df) in enumerate(categories.items(), start=1):
        cat_dir = RESULTS_DIR / source / model / "evaluations" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        rm = metrics.compute_risk_matrix(df, SEVERITY_COL, LIKELIHOOD_COL)
        rm.reset_index().to_csv(cat_dir / "risk_matrix.csv", index=False)

        # Everything below depends on the comparison target, so it lives in
        # its own subfolder -- clearing only that subfolder (not cat_dir)
        # means switching between --ref and ground-truth mode never wipes
        # out the other mode's results.
        target_dir = cat_dir / target
        shutil.rmtree(target_dir, ignore_errors=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        ref_df = ref_categories.get(cat) if ref_categories is not None else None
        if ref_categories is not None and ref_df is None:
            raise FileNotFoundError(
                f"Category '{cat}' scored for '{model}' but not for ref '{ref}' -- "
                f"run the reference model first."
            )

        # Ground-truth mode has one shape, so its files sit directly in
        # target_dir; --ref mode has two (normal/, mixed/), so its files
        # go one level deeper -- see metrics.comparison_subdir.
        if ref_df is not None:
            out_dir = target_dir / metrics.comparison_subdir(ref, mixed=False)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = target_dir

        for name, pred_col, levels in VARIABLES:
            if ref_df is not None:
                ref_column = None  # same column name, read from ref_df
            else:
                ref_column = gt_columns.get(name)
                if ref_column is None:
                    print(f"[{cat}] no {'reference run' if ref else name + ' ground truth'} "
                          f"for {name} -- skipping metrics/confusion.")
                    continue
                if ref_column not in df.columns:
                    raise ValueError(
                        f"[{cat}] source '{source}' config points {name} ground truth to "
                        f"column '{ref_column}', but it isn't in this scored file's columns."
                    )

            suffix = "sev" if name == "severity" else "like"

            m = metrics.compute_basic_metrics(
                df, pred_col, labels=levels, ref_df=ref_df, ref_column=ref_column
            )
            m.to_csv(out_dir / f"metrics_{suffix}.csv")

            cm = metrics.compute_confusion(
                df, pred_col, levels, ref_df=ref_df, ref_column=ref_column
            )
            cm.to_csv(out_dir / f"confusion_{suffix}.csv")

        # A risk-matrix diff needs both variables from the SAME target.
        if ref_df is not None:
            ref_rm = metrics.compute_risk_matrix(ref_df, SEVERITY_COL, LIKELIHOOD_COL)
            diff = metrics.compute_matrix_diff(rm, ref_rm)
            diff.reset_index().to_csv(out_dir / "risk_diff.csv", index=False)

            # Ground truth covers some but not all axes: also build a
            # "best available" matrix (ground truth where it exists, ref
            # otherwise), in the mixed/ subfolder. If it covers none,
            # there's nothing to mix in; if it covers both, the full
            # ground-truth matrix is already produced by the no-ref run
            # -- no need to repeat it here.
            covered = sum(1 for v in ("severity", "likelihood") if v in gt_columns)
            if covered == 1:
                mixed_dir = target_dir / metrics.comparison_subdir(ref, mixed=True)
                mixed_dir.mkdir(parents=True, exist_ok=True)
                mixed_rm = metrics.compute_mixed_risk_matrix(
                    df, ref_df, gt_columns, SEVERITY_COL, LIKELIHOOD_COL
                )
                mixed_rm.reset_index().to_csv(mixed_dir / "risk_matrix.csv", index=False)
                mixed_diff = metrics.compute_matrix_diff(rm, mixed_rm)
                mixed_diff.reset_index().to_csv(mixed_dir / "risk_diff.csv", index=False)
        elif "severity" in gt_columns and "likelihood" in gt_columns:
            gt_rm = metrics.compute_risk_matrix(
                df, gt_columns["severity"], gt_columns["likelihood"]
            )
            gt_rm.reset_index().to_csv(out_dir / "risk_matrix_ground_truth.csv", index=False)
            diff = metrics.compute_matrix_diff(rm, gt_rm)
            diff.reset_index().to_csv(out_dir / "risk_diff.csv", index=False)

        print(f"[{cat}] done ({i}/{total}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a model's scored results for a source, category by category."
    )
    parser.add_argument(
        "--source", required=True, help="Source identifier, e.g. 'saferproducts_gov'."
    )
    parser.add_argument(
        "--model", required=True, help="Model folder name under results/<source>/."
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="Another model's folder name under results/<source>/ to compare against. "
        "If omitted, compares against this source's ground-truth labels "
        "(config/sources/<source>.yaml's ground_truth_columns), if any.",
    )
    args = parser.parse_args()
    evaluate_model(source=args.source, model=args.model, ref=args.ref)


if __name__ == "__main__":
    main()
