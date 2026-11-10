"""
Plots comparing models' results for a source, built from evaluate.py's
outputs. Two kinds, saved as PNGs instead of shown:

- confusion matrices are per model/category, so they're saved next to
  the CSV they visualize (evaluations/<category>/confusion_*.png).
- everything else (F1, ordinal error, matrix MAE, divergence, top-k
  overlap) compares several models on one chart, so it isn't any one
  model's file -- saved under results/<source>/comparisons/<target>/,
  where <target> is vs_<ref_model> or vs_ground_truth (see
  comparison_dir), matching evaluate.py's two comparison modes.

The three matrix-level plots (MAE, divergence, top-k overlap) mirror
evaluate.py's normal/ vs mixed/ split under vs_<ref>/ (see
metrics.comparison_subdir): normal/ is the full-ref comparison, mixed/
the ground-truth+ref one, when it exists for this source. F1 and
ordinal error never have a mixed variant, so their plots stay flat.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation import metrics
from utils.source_config import load_source_config

RESULTS_DIR = Path("results")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# Bar-chart plots (F1, ordinal error, matrix MAE, divergence, top-k overlap)
# get squeezed hard when embedded at a single IEEE column's \linewidth
# (~3.5in) from a figsize this wide (12in) -- roughly a 0.3x shrink, so
# their fonts need to start ~3x larger than the confusion-matrix/heatmap
# defaults above to still read at print size. Only these functions opt
# into this via `with plt.rc_context(BAR_RC):` -- confusion matrices and
# any other non-bar-chart plot keep the smaller defaults untouched.
BAR_RC = {
    "font.size": 26,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
}

# Short display names for the bar-chart x-axis (category folder names are
# long/snake_case and, rotated at BAR_RC's larger font, run into each other).
# Full names are in Table 2 -- figure captions should note these are
# abbreviated. Falls back to a title-cased, space-joined version of the
# stem for anything not listed here.
CATEGORY_LABELS = {
    "baby": "Baby",
    "clothing_and_accessories": "Clothing & Accessories",
    "containers_and_packaging": "Containers & Packaging",
    "drywall": "Drywall",
    "electronics": "Electronics",
    "fuel_lighters_and_fireworks": "Fuel, Lighters & Fireworks",
    "furniture_furnishings_and_decorations": "Furniture & Decor",
    "hobby": "Hobby",
    "home_maintenance_and_structures": "Home Maintenance",
    "kitchen": "Kitchen",
    "personal_care": "Personal Care",
    "products_at_public_facilities": "Public Facilities",
    "sports_and_recreation": "Sports & Recreation",
    "toys_and_children": "Toys & Children",
    "yard_and_garden": "Yard & Garden",
}


def _display_category(cat: str) -> str:
    """Short axis label for a category folder name (e.g. 'baby_scored')."""
    # Folder names keep the commas of the original category names
    # ("fuel,_lighters_and_fireworks"); CATEGORY_LABELS' keys don't.
    stem = cat.replace("_scored", "").replace(",", "")
    return CATEGORY_LABELS.get(stem, stem.replace("_", " ").title())


CATEGORY_TICK_FONTSIZE = 16  # only the x-axis category names -- see CATEGORY_LABELS


def _place_legend(ax, loc: str = "auto") -> None:
    """Legend placement for the bar charts, as one row (models side by side).
    "auto" tries, in order, keeping the first that hides nothing:
      1. inside, upper left / upper right -- must cover no bar top;
      2. inside, lower left / lower right (over the bars' bases) -- must not
         hide where any bar ends;
      3. otherwise outside, just above the axes (right-aligned), with the
         title pushed up to make room.
    Any other value is passed through as a regular matplotlib loc."""
    ncol = max(len(ax.get_legend_handles_labels()[0]), 1)
    if loc != "auto":
        ax.legend(loc=loc, ncol=ncol)
        return
    fig = ax.figure
    bars = [b for b in ax.patches if b.get_height() > 0]

    def extent(leg):
        fig.canvas.draw()
        return leg.get_window_extent().transformed(ax.transData.inverted())

    def in_x(b, bb):
        return b.get_x() < bb.x1 and b.get_x() + b.get_width() > bb.x0

    for cand in ("upper left", "upper right"):
        bb = extent(ax.legend(loc=cand, ncol=ncol))
        if not any(in_x(b, bb) and b.get_y() + b.get_height() > bb.y0 for b in bars):
            return
    for cand in ("lower left", "lower right"):
        bb = extent(ax.legend(loc=cand, ncol=ncol))
        if not any(in_x(b, bb) and b.get_y() + b.get_height() < bb.y1 for b in bars):
            return

    leg = ax.legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=ncol, borderaxespad=0)
    fig.canvas.draw()
    leg_h_pts = leg.get_window_extent().height * 72 / fig.dpi
    ax.set_title(ax.get_title(), pad=leg_h_pts + 15)


def _mean_title(df: pd.DataFrame, value_col: str, label: str, model_col: str = "model") -> str:
    """'Mean <label> — model_a: 0.123, model_b: 0.456', the whole title now
    that the redundant descriptive text (already in the LaTeX caption) has
    been dropped -- no more leading '\\n' or wrapping parens, since this is
    the only line rather than a suffix appended to one."""
    means = df.groupby(model_col)[value_col].mean()
    mean_str = ", ".join(f"{m}: {v:.3f}" for m, v in means.items())
    return f"Mean {label} — {mean_str}"


def _savefig(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / name, bbox_inches="tight")
    plt.close()


def _categories(source: str, model: str) -> list[str]:
    eval_dir = RESULTS_DIR / source / model / "evaluations"
    return sorted(p.name for p in eval_dir.iterdir() if p.is_dir())


def _target_dir(
    source: str, model: str, category: str, ref_model: Optional[str], mixed: bool = False
) -> Path:
    """Where a specific comparison's files actually live: flat under
    vs_ground_truth/ (no variants there), or under vs_<ref>/normal/ or
    vs_<ref>/mixed/ for --ref runs (see metrics.comparison_subdir)."""
    target = metrics.comparison_target(ref_model)
    base = RESULTS_DIR / source / model / "evaluations" / category / target
    if ref_model is None:
        return base
    return base / metrics.comparison_subdir(ref_model, mixed=mixed)


def _risk_matrix_path(
    source: str, model: str, ref_model: Optional[str], category: str, mixed: bool = False
) -> Path:
    """The comparison target's risk matrix -- a ref model's own risk_matrix.csv
    (target-independent, so not nested), this model's own ground-truth-derived
    one (flat under vs_ground_truth/), or, if mixed, the ground-truth+ref
    matrix evaluate.py writes under vs_<ref_model>/mixed/."""
    if mixed:
        return _target_dir(source, model, category, ref_model, mixed=True) / "risk_matrix.csv"
    if ref_model is not None:
        return RESULTS_DIR / source / ref_model / "evaluations" / category / "risk_matrix.csv"
    return _target_dir(source, model, category, None) / "risk_matrix_ground_truth.csv"


def comparison_flat_dir(source: str, ref_model: Optional[str]) -> Path:
    """Where F1 and ordinal-error plots are saved -- always flat, since
    neither has a mixed-matrix counterpart."""
    return RESULTS_DIR / source / "comparisons" / metrics.comparison_target(ref_model)


def comparison_dir(source: str, ref_model: Optional[str], mixed: bool = False) -> Path:
    """Where the matrix-level comparison plots (MAE, divergence, top-k
    overlap) are saved: normal/ or mixed/ under vs_<ref>/ when a ref
    model is given, else vs_ground_truth/ flat (no variants there)."""
    base = comparison_flat_dir(source, ref_model)
    if ref_model is None:
        return base
    return base / metrics.comparison_subdir(ref_model, mixed=mixed)


# --------------------------------- F1 per category ---------------------------------

def load_f1(
    source: str, model: str, variable: str, ref_model: Optional[str], avg_type: str = "macro"
) -> pd.DataFrame:
    suffix = "sev" if variable == "severity" else "like"
    avg_label = f"{avg_type} avg"

    rows = []
    for cat in _categories(source, model):
        path = _target_dir(source, model, cat, ref_model) / f"metrics_{suffix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if avg_label not in df.index:
            continue
        rows.append({
            "category": _display_category(cat),
            "f1": df.loc[avg_label]["f1-score"],
            "model": model,
        })
    return pd.DataFrame(rows)


def plot_f1_comparison(
    source: str,
    models: list[str],
    variable: str,
    ref_model: Optional[str],
    avg_type: str = "macro",
    legend_loc: str = "auto",
    out_dir: Optional[Path] = None,
) -> None:
    all_f1 = pd.concat([load_f1(source, m, variable, ref_model, avg_type) for m in models])
    if all_f1.empty:
        print(f"[{source}] no metrics_{'sev' if variable == 'severity' else 'like'}.csv found -- skipping F1 plot.")
        return

    with plt.rc_context(BAR_RC):
        plt.figure(figsize=(12, 5))
        sns.barplot(data=all_f1, x="category", y="f1", hue="model")
        ax = plt.gca()
        plt.title(_mean_title(all_f1, "f1", "F1"), pad=40)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=45, ha="right", fontsize=CATEGORY_TICK_FONTSIZE)
        ax.tick_params(axis="x", length=8)
        plt.ylim(0, 1)
        _place_legend(ax, legend_loc)

        if out_dir:
            _savefig(out_dir, f"f1_{variable}_{avg_type}.png")
        else:
            plt.show()


# --------------------------------- Confusion matrices ---------------------------------

def plot_confusion(
    source: str, model: str, variable: str, ref_model: Optional[str], show: bool = False
) -> None:
    """Saves each category's confusion matrix next to the CSV it visualizes
    (evaluations/<category>/vs_<target>/[normal/]confusion_{sev,like}.png)
    -- unlike every other plot here, this one belongs to a single
    model/category, not a cross-model comparison, but it's still specific
    to a comparison target (a confusion matrix is always against something)."""
    suffix = "sev" if variable == "severity" else "like"
    for cat in _categories(source, model):
        out_dir = _target_dir(source, model, cat, ref_model)
        path = out_dir / f"confusion_{suffix}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)

        plt.figure(figsize=(6, 5))
        sns.heatmap(df, annot=True, fmt="g", cmap="Reds")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{model} - {cat.replace('_scored', '')} ({variable})")

        if show:
            plt.show()
        else:
            _savefig(out_dir, f"confusion_{suffix}.png")


# --------------------------------- Ordinal error ---------------------------------

def plot_ordinal_error(
    source: str,
    models: list[str],
    ref_model: Optional[str],
    variable: str,
    legend_loc: str = "auto",
    out_dir: Optional[Path] = None,
) -> None:
    pred_col = "severity_level" if variable == "severity" else "occurrence_likelihood"
    categories = _categories(source, models[0])

    results = []
    for cat in categories:
        cat_stem = _display_category(cat)
        for model in models:
            if model == ref_model:
                continue
            df = pd.read_csv(RESULTS_DIR / source / model / f"{cat}.csv")

            if ref_model is not None:
                ref_df = pd.read_csv(RESULTS_DIR / source / ref_model / f"{cat}.csv")
                errors = metrics.compute_ordinal_error(df, pred_col, ref_df=ref_df)
            else:
                gt_columns = load_source_config(source).get("ground_truth_columns") or {}
                gt_col = gt_columns.get(variable)
                if gt_col is None or gt_col not in df.columns:
                    continue
                errors = metrics.compute_ordinal_error(df, pred_col, ref_column=gt_col)

            results.append({"category": cat_stem, "model": model, "ordinal_error": errors.mean()})

    if not results:
        print(f"[{source}] no comparison target available for ordinal error ({variable}).")
        return

    df = pd.DataFrame(results)
    with plt.rc_context(BAR_RC):
        plt.figure(figsize=(12, 5))
        sns.barplot(data=df, x="category", y="ordinal_error", hue="model")
        ax = plt.gca()
        plt.title(_mean_title(df, "ordinal_error", "error"), pad=40)
        plt.xticks(rotation=45, ha="right", fontsize=CATEGORY_TICK_FONTSIZE)
        ax.tick_params(axis="x", length=8)
        plt.ylim(0, df["ordinal_error"].max() * 1.1)
        plt.xlabel("")
        plt.ylabel("")
        sns.despine()
        _place_legend(ax, legend_loc)

        if out_dir:
            _savefig(out_dir, f"ordinal_error_{variable}.png")
        else:
            plt.show()


# --------------------------------- Matrix-level MAE ---------------------------------

def plot_matrix_level_mae(
    source: str,
    models: list[str],
    ref_model: Optional[str],
    out_dir: Optional[Path] = None,
    mixed: bool = False,
) -> None:
    results = []
    for model in models:
        for cat in _categories(source, model):
            diff_path = _target_dir(source, model, cat, ref_model, mixed=mixed) / "risk_diff.csv"
            if not diff_path.exists():
                continue
            diff = pd.read_csv(diff_path, index_col=0)
            # Normalize by n (records in this category) so categories of
            # different sizes are comparable: MAE becomes a proportion.
            n = _matrix_n(source, model, cat)
            if not n:
                continue
            results.append({
                "category": _display_category(cat),
                "model": model,
                "mae": metrics.matrix_cell_mae(diff) / n,
            })

    if not results:
        print(f"[{source}] no risk_diff.csv found -- skipping{' mixed' if mixed else ''} matrix MAE plot.")
        return

    df = pd.DataFrame(results)
    categories_order = df["category"].unique()
    x = np.arange(len(categories_order))
    width = 0.8 / max(len(df["model"].unique()), 1)

    with plt.rc_context(BAR_RC):
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        for i, model in enumerate(df["model"].unique()):
            values = df[df["model"] == model].set_index("category").reindex(categories_order)["mae"].values
            plt.bar(x + i * width, values, width, label=model, zorder=2)

        plt.xticks(
            x + width * (len(df["model"].unique()) - 1) / 2, categories_order,
            rotation=45, ha="right", fontsize=CATEGORY_TICK_FONTSIZE,
        )
        ax.tick_params(axis="x", length=8)
        plt.ylabel("")
        plt.xlabel("")
        plt.title(_mean_title(df, "mae", "MAE"), pad=40)
        _place_legend(ax)

        if out_dir:
            _savefig(out_dir, "matrix_level_mae.png")
        else:
            plt.show()


# --------------------------------- Marginal divergence ---------------------------------

def _load_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def _matrix_n(source: str, model: str, category: str) -> float:
    """Number of records in a category (sum of the model's own risk matrix),
    used to normalize MAE / divergence into proportions. 0 if missing."""
    path = RESULTS_DIR / source / model / "evaluations" / category / "risk_matrix.csv"
    if not path.exists():
        return 0
    return float(_load_matrix(path).values.sum())


def plot_row_column_divergence(
    source: str,
    models: list[str],
    ref_model: Optional[str],
    out_dir: Optional[Path] = None,
    mixed: bool = False,
) -> None:
    results = []
    for model in models:
        if model == ref_model:
            continue
        for cat in _categories(source, model):
            pred_path = RESULTS_DIR / source / model / "evaluations" / cat / "risk_matrix.csv"
            ref_path = _risk_matrix_path(source, model, ref_model, cat, mixed=mixed)
            if not pred_path.exists() or not ref_path.exists():
                continue

            pred_mat = _load_matrix(pred_path)
            n = float(pred_mat.values.sum())
            if not n:
                continue
            row_div, col_div = metrics.marginal_divergence(_load_matrix(ref_path), pred_mat)
            # Normalize by n so the divergence is a proportion of the records.
            row_div, col_div = row_div / n, col_div / n
            cat_stem = _display_category(cat)
            results.append({"category": cat_stem, "model": model, "type": "Severity (rows)", "divergence": row_div})
            results.append({"category": cat_stem, "model": model, "type": "Likelihood (columns)", "divergence": col_div})

    if not results:
        print(f"[{source}] no{' mixed' if mixed else ''} comparison target available for marginal divergence.")
        return

    df = pd.DataFrame(results)
    types = list(df["type"].unique())
    with plt.rc_context(BAR_RC):
        # Stacked (one row per type), not side-by-side: side-by-side doubles the
        # figure width but not the height, so after scaling to the same
        # \linewidth as every other figure everything (fonts included) ends up
        # visibly smaller. Stacking keeps the width -- and thus the effective
        # font size -- the same as the single-panel figures.
        fig, axes = plt.subplots(len(types), 1, figsize=(12, 5 * len(types)))
        if len(types) == 1:
            axes = [axes]

        for ax, t in zip(axes, types):
            sub = df[df["type"] == t]
            categories = sub["category"].unique()
            sub_models = sub["model"].unique()
            x = np.arange(len(categories))
            width = 0.8 / len(sub_models)

            for i, model in enumerate(sub_models):
                values = [
                    sub[(sub["category"] == c) & (sub["model"] == model)]["divergence"].values[0]
                    for c in categories
                ]
                ax.bar(x + i * width, values, width, label=model, zorder=2)

            ax.set_xticks(x + width * (len(sub_models) - 1) / 2)
            ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=CATEGORY_TICK_FONTSIZE)
            ax.tick_params(axis="x", length=8)
            # "Distribution Divergence (mixed reference) --" dropped: it's
            # redundant with the LaTeX caption. `t` ("Severity (rows)" /
            # "Likelihood (columns)") stays -- it's the only thing telling
            # the two stacked panels apart, not just decorative.
            ax.set_title(t + "\n" + _mean_title(sub, "divergence", "divergence"), pad=40)
            # Shortened from "Mean Absolute Difference": at BAR_RC's larger
            # axes.labelsize, the full-length rotated label was taller than
            # the (already vertically squeezed, two-panel) axes, overflowing
            # into the title above it.
            ax.set_ylabel("Mean Abs. Diff. / n", fontsize=18)

        # tight_layout gives up here (warns and falls back to too-tight
        # spacing). subplots_adjust with an explicit hspace avoids the
        # overlap reliably -- now that titles dropped the redundant
        # descriptive line, they're short enough that a much smaller gap
        # than before is enough (a large one just left a big blank band).
        fig.subplots_adjust(hspace=2.4, top=0.94, bottom=0.1, left=0.12, right=0.97)

        # One legend for both panels, outside them, just above the top title.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.canvas.draw()
        title_top = axes[0].title.get_window_extent(fig.canvas.get_renderer()) \
            .transformed(fig.transFigure.inverted()).y1
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, title_top + 0.01))

        if out_dir:
            _savefig(out_dir, "divergence.png")
        else:
            plt.show()


# --------------------------------- Top-k overlap ---------------------------------

def plot_topk_overlap(
    source: str,
    models: list[str],
    ref_model: Optional[str],
    k: int = 5,
    out_dir: Optional[Path] = None,
    mixed: bool = False,
) -> None:
    results = []
    for model in models:
        if model == ref_model:
            continue
        for cat in _categories(source, model):
            pred_path = RESULTS_DIR / source / model / "evaluations" / cat / "risk_matrix.csv"
            ref_path = _risk_matrix_path(source, model, ref_model, cat, mixed=mixed)
            if not pred_path.exists() or not ref_path.exists():
                continue

            overlap = metrics.topk_overlap(_load_matrix(ref_path), _load_matrix(pred_path), k)
            results.append({"category": _display_category(cat), "model": model, "overlap": overlap})

    if not results:
        print(f"[{source}] no{' mixed' if mixed else ''} comparison target available for top-k overlap.")
        return

    df = pd.DataFrame(results)
    with plt.rc_context(BAR_RC):
        plt.figure(figsize=(12, 5))
        sns.barplot(data=df, x="category", y="overlap", hue="model")
        ax = plt.gca()
        plt.ylim(0, 1)
        plt.xlabel("")
        plt.ylabel("")
        plt.title(_mean_title(df, "overlap", "overlap"), pad=40)
        plt.xticks(rotation=45, ha="right", fontsize=CATEGORY_TICK_FONTSIZE)
        ax.tick_params(axis="x", length=8)
        sns.despine()
        _place_legend(ax)

        if out_dir:
            _savefig(out_dir, f"topk_overlap_{k}.png")
        else:
            plt.show()


def generate_all(source: str, models: list[str], ref_model: Optional[str], k: int = 5) -> None:
    flat_out = comparison_flat_dir(source, ref_model)
    for variable in ("severity", "likelihood"):
        for avg_type in ("macro", "weighted"):
            plot_f1_comparison(source, models, variable, ref_model, avg_type, out_dir=flat_out)
        for model in models:
            plot_confusion(source, model, variable, ref_model)
        plot_ordinal_error(source, models, ref_model, variable, out_dir=flat_out)

    normal_out = comparison_dir(source, ref_model)
    plot_matrix_level_mae(source, models, ref_model, out_dir=normal_out)
    plot_row_column_divergence(source, models, ref_model, out_dir=normal_out)
    plot_topk_overlap(source, models, ref_model, k=k, out_dir=normal_out)

    if ref_model is not None:
        # No-ops (with a skip message) if this source has no mixed matrix.
        mixed_out = comparison_dir(source, ref_model, mixed=True)
        plot_matrix_level_mae(source, models, ref_model, out_dir=mixed_out, mixed=True)
        plot_row_column_divergence(source, models, ref_model, out_dir=mixed_out, mixed=True)
        plot_topk_overlap(source, models, ref_model, k=k, out_dir=mixed_out, mixed=True)

    print(f"Saved comparison plots to {flat_out} (and its normal/, mixed/ subfolders).")
    print("Saved confusion matrices next to each model's evaluations/<category>/ CSVs.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from a source's evaluation outputs."
    )
    parser.add_argument("--source", required=True, help="Source identifier, e.g. 'saferproducts_gov'.")
    parser.add_argument("--models", required=True, nargs="+", help="Model folder names under results/<source>/.")
    parser.add_argument(
        "--ref",
        default=None,
        help="Model folder name to use as the comparison target. If omitted, uses each "
        "model's ground-truth-derived outputs (see src/evaluation/evaluate.py).",
    )
    parser.add_argument("--k", type=int, default=5, help="k for the top-k risk-cell overlap plot.")
    args = parser.parse_args()
    generate_all(source=args.source, models=args.models, ref_model=args.ref, k=args.k)


if __name__ == "__main__":
    main()