"""
Comparison metrics for a source's scored results -- one variable at a
time (severity or likelihood) for the compute_basic_metrics/confusion/
ordinal_error family, or the joint severity x likelihood matrix for
the compute_risk_matrix family.

Compares against either another model's run (pass ref_df) or this
source's ground truth, already a column in df (pass ref_column, no
ref_df -- see evaluate.py). Always merged by id_column, never row
position: risk_scorer.py's ThreadPoolExecutor writes rows in
completion order, not input order.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ID_COLUMN = "report_id"
SEVERITY_LEVELS = [1, 2, 3, 4]
LIKELIHOOD_LEVELS = [1, 2, 3, 4, 5]


def comparison_target(ref: Optional[str]) -> str:
    """Folder-safe name for a comparison target, shared by evaluate.py and
    visualize.py so both always agree on where a given mode's outputs live."""
    return f"vs_{ref}" if ref else "vs_ground_truth"


def comparison_subdir(ref: Optional[str], mixed: bool = False) -> str:
    """Subfolder name, within a comparison_target() dir, for one mode's
    files. --ref runs split into normal/ (per-variable metrics, full-ref
    risk_diff) and mixed/ (ground-truth+ref risk_diff, when it exists);
    ground-truth mode (ref=None) has no variants, so its files stay
    directly in the target dir -- callers should skip this for that case."""
    return "mixed" if mixed else "normal"


def _paired_columns(
    df: pd.DataFrame,
    column: str,
    ref_df: Optional[pd.DataFrame],
    ref_column: Optional[str],
    id_column: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Resolves (y_true, y_pred) for a single variable, always aligned by
    `id_column` rather than row position -- see module docstring.
    """
    if ref_df is not None:
        true_col = ref_column or column
        left = df[[id_column, column]].rename(columns={column: "_pred"})
        right = ref_df[[id_column, true_col]].rename(columns={true_col: "_true"})
        merged = left.merge(right, on=id_column, how="inner")
        if merged.empty:
            raise ValueError(
                f"No overlapping '{id_column}' values between df and ref_df -- "
                "nothing to compare."
            )
        return merged["_true"], merged["_pred"]

    if not ref_column or ref_column not in df.columns:
        raise ValueError(
            f"No ground-truth column available for '{column}' -- pass ref_df, "
            f"or provide a valid ref_column already present in df."
        )
    return df[ref_column], df[column]


def compute_basic_metrics(
    df: pd.DataFrame,
    column: str,
    labels: Optional[list[int]] = None,
    id_column: str = ID_COLUMN,
    ref_df: Optional[pd.DataFrame] = None,
    ref_column: Optional[str] = None,
) -> pd.DataFrame:
    """Per-class precision/recall/F1 (sklearn's classification_report) for one variable."""
    y_true, y_pred = _paired_columns(df, column, ref_df, ref_column, id_column)
    if labels is None:
        labels = sorted(y_true.unique())
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0, labels=labels
    )
    return pd.DataFrame(report).T


def compute_confusion(
    df: pd.DataFrame,
    column: str,
    labels: list[int],
    id_column: str = ID_COLUMN,
    ref_df: Optional[pd.DataFrame] = None,
    ref_column: Optional[str] = None,
) -> pd.DataFrame:
    """Confusion matrix (sklearn) for one variable, indexed/columned by `labels`."""
    y_true, y_pred = _paired_columns(df, column, ref_df, ref_column, id_column)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def compute_ordinal_error(
    df: pd.DataFrame,
    column: str,
    id_column: str = ID_COLUMN,
    ref_df: Optional[pd.DataFrame] = None,
    ref_column: Optional[str] = None,
) -> pd.Series:
    """Per-row |true - pred| for an ordinal variable (severity or likelihood)."""
    y_true, y_pred = _paired_columns(df, column, ref_df, ref_column, id_column)
    return (y_true.astype(int) - y_pred.astype(int)).abs()


def compute_risk_matrix(
    df: pd.DataFrame,
    severity_col: str,
    likelihood_col: str,
    id_column: str = ID_COLUMN,
    severity_levels: list[int] = SEVERITY_LEVELS,
    likelihood_levels: list[int] = LIKELIHOOD_LEVELS,
) -> pd.DataFrame:
    """
    Counts of incidents per (severity, likelihood) cell -- a risk matrix.
    `severity_col`/`likelihood_col` can be predicted columns
    (severity_level/occurrence_likelihood) or ground-truth columns,
    depending on what the caller passes in.
    """
    mat = pd.pivot_table(
        df,
        values=id_column,
        index=severity_col,
        columns=likelihood_col,
        aggfunc="count",
        fill_value=0,
    )
    return mat.reindex(index=severity_levels, columns=likelihood_levels, fill_value=0)


def compute_matrix_diff(mat_a: pd.DataFrame, mat_b: pd.DataFrame) -> pd.DataFrame:
    """Cell-wise absolute difference between two risk matrices."""
    return (mat_a - mat_b).abs()


def _resolve_axis(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    own_col: str,
    gt_col: Optional[str],
    id_column: str,
) -> pd.Series:
    """One axis (severity or likelihood) for the mixed matrix: this
    source's ground truth column if given, otherwise ref_df's own_col
    merged in by id_column, aligned to df's row order."""
    if gt_col:
        return df[gt_col]
    merged = df[[id_column]].merge(ref_df[[id_column, own_col]], on=id_column, how="left")
    return merged[own_col]


def compute_mixed_risk_matrix(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    gt_columns: dict[str, str],
    severity_col: str,
    likelihood_col: str,
    id_column: str = ID_COLUMN,
    severity_levels: list[int] = SEVERITY_LEVELS,
    likelihood_levels: list[int] = LIKELIHOOD_LEVELS,
) -> pd.DataFrame:
    """
    Best-available reference matrix: for each axis, uses this source's
    ground truth if gt_columns has it, otherwise falls back to ref_df's
    own prediction for that axis. Only meaningful when gt_columns covers
    some but not all axes -- callers should skip this otherwise (see
    evaluate.py).
    """
    sev = _resolve_axis(df, ref_df, severity_col, gt_columns.get("severity"), id_column)
    like = _resolve_axis(df, ref_df, likelihood_col, gt_columns.get("likelihood"), id_column)
    tmp = pd.DataFrame({
        id_column: df[id_column].values,
        severity_col: sev.values,
        likelihood_col: like.values,
    })
    mat = pd.pivot_table(
        tmp, values=id_column, index=severity_col, columns=likelihood_col, aggfunc="count", fill_value=0
    )
    return mat.reindex(index=severity_levels, columns=likelihood_levels, fill_value=0)


def matrix_cell_mae(diff_matrix: pd.DataFrame) -> float:
    """Mean absolute error across every cell of a risk-matrix diff."""
    return float(diff_matrix.values.mean())


def marginal_divergence(ref_mat: pd.DataFrame, pred_mat: pd.DataFrame) -> tuple[float, float]:
    """
    Mean absolute difference between the two matrices' row sums
    (severity marginal) and column sums (likelihood marginal).
    """
    ref_row, pred_row = ref_mat.sum(axis=1), pred_mat.sum(axis=1)
    ref_col, pred_col = ref_mat.sum(axis=0), pred_mat.sum(axis=0)
    row_div = float(np.mean(np.abs(ref_row - pred_row)))
    col_div = float(np.mean(np.abs(ref_col - pred_col)))
    return row_div, col_div


def topk_overlap(ref_mat: pd.DataFrame, pred_mat: pd.DataFrame, k: int) -> float:
    """Fraction of the top-k highest-count cells shared by both matrices."""
    ref_flat, pred_flat = ref_mat.values.flatten(), pred_mat.values.flatten()
    ref_topk = set(np.argsort(ref_flat)[-k:])
    pred_topk = set(np.argsort(pred_flat)[-k:])
    return len(ref_topk & pred_topk) / k
