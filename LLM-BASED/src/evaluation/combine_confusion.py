"""
Combine several confusion matrices (the confusion_*.csv files written by
evaluate.py) into ONE figure, as a grid, to save space in the paper.

The CSVs are given in row-major order. Each row of the grid is meant to be
one variable / comparison target (e.g. severity vs. ground truth, likelihood
vs. a reference model) and each column one model. To keep the rows visibly
distinct, every row gets:
  - a bold label on the left (--row-labels),
  - its own colormap (--cmaps, default Reds, Blues, Greens, ...),
  - its own color scale, shared by the panels in that row so the models can
    be compared, and its own colorbar.
Panels are lettered (a), (b), ... and titled with their column label.

Example (Baby: severity vs. ground truth / likelihood vs. DeepSeek-V3):

    python src/evaluation/combine_confusion.py \
        --csvs results/saferproducts_gov/mistral-nemo/evaluations/baby_scored/vs_ground_truth/confusion_sev.csv \
               results/saferproducts_gov/llama-4-scout/evaluations/baby_scored/vs_ground_truth/confusion_sev.csv \
               results/saferproducts_gov/mistral-nemo/evaluations/baby_scored/vs_deepseek-chat-v3-0324/normal/confusion_like.csv \
               results/saferproducts_gov/llama-4-scout/evaluations/baby_scored/vs_deepseek-chat-v3-0324/normal/confusion_like.csv \
        --row-labels "Severity (vs. ground truth)" "Likelihood (vs. DeepSeek-V3)" \
        --col-labels mistral-nemo llama-4-scout \
        --out results/saferproducts_gov/comparisons/confusion_baby_combined.png
"""

import argparse
import string
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_CMAPS = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]


def combine_confusion(
    csvs: list[Path],
    row_labels: list[str],
    col_labels: list[str],
    out: Path,
    cmaps: list[str] | None = None,
    panel_size: float = 4.2,
    font_size: int = 14,
) -> None:
    nrows, ncols = len(row_labels), len(col_labels)
    if len(csvs) != nrows * ncols:
        raise ValueError(
            f"got {len(csvs)} CSVs, but {nrows} row labels x {ncols} column labels "
            f"= {nrows * ncols} panels"
        )
    cmaps = cmaps or DEFAULT_CMAPS
    mats = [pd.read_csv(p, index_col=0) for p in csvs]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols + 1.5, panel_size * nrows),
        squeeze=False, constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.15, w_pad=0.1)

    letters = iter(string.ascii_lowercase)
    for r in range(nrows):
        row_mats = mats[r * ncols:(r + 1) * ncols]
        vmax = max(m.values.max() for m in row_mats)
        cmap = cmaps[r % len(cmaps)]

        for c, df in enumerate(row_mats):
            ax = axes[r, c]
            sns.heatmap(
                df, annot=True, fmt="g", cmap=cmap, vmin=0, vmax=vmax, ax=ax,
                square=True, cbar=False, linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": font_size - 2},
            )
            ax.set_title(f"({next(letters)}) {col_labels[c]}", fontsize=font_size)
            ax.set_xlabel("Predicted", fontsize=font_size - 1)
            ax.set_ylabel("True" if c == 0 else "", fontsize=font_size - 1)
            ax.tick_params(labelsize=font_size - 2, length=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # Row label, bold, to the left of the row's first panel.
        axes[r, 0].annotate(
            row_labels[r], xy=(-0.32, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=font_size + 1, fontweight="bold",
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, ax=list(axes[r, :]), shrink=0.85, pad=0.02)
        cb.set_label("Count", fontsize=font_size - 1)
        cb.ax.tick_params(labelsize=font_size - 3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine confusion-matrix CSVs into one grid figure "
        "(rows = variable/target, columns = model)."
    )
    parser.add_argument("--csvs", required=True, nargs="+", type=Path,
                        help="confusion_*.csv files, row-major (row 1's panels, then row 2's, ...).")
    parser.add_argument("--row-labels", required=True, nargs="+",
                        help="One label per row, e.g. 'Severity (vs. ground truth)'.")
    parser.add_argument("--col-labels", required=True, nargs="+",
                        help="One label per column, e.g. the model names.")
    parser.add_argument("--out", required=True, type=Path, help="Output image path (.png/.pdf).")
    parser.add_argument("--cmaps", nargs="+", default=None,
                        help=f"One colormap per row (default: {' '.join(DEFAULT_CMAPS[:2])} ...).")
    args = parser.parse_args()
    combine_confusion(args.csvs, args.row_labels, args.col_labels, args.out, args.cmaps)


if __name__ == "__main__":
    main()
