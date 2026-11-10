"""
Scans every scored CSV in the folder of the config's output_path (e.g.
results/<source>/<model>/ or results/<source>/<model>_graph/) and re-runs
the LLM scoring only for rows flagged as scoring_failed, replacing them in
place -- without touching successfully scored rows.

Uses the same config as the scorer, so predict_likelihood should match the
run being fixed (false for a _graph folder).
"""

from pathlib import Path

import pandas as pd

from risk_scorer.risk_scorer import RiskAssessor, load_config

CONFIG_PATH = Path("config/config.yaml")
TMP_DIR = Path("tmp_fix_rows")

FAILED_COL = "scoring_failed"


def fix_error_rows(scored_path: Path, assessor: RiskAssessor) -> None:
    """
    Detects rows with scoring_failed=True, re-runs assessment ONLY for
    them, and overwrites those rows in place in the scored CSV.
    """
    df = pd.read_csv(scored_path)

    if FAILED_COL not in df.columns:
        print(f"[{scored_path.stem}] No '{FAILED_COL}' column -- skipping (older file?).")
        return

    error_mask = df[FAILED_COL] == True
    error_indices = df.index[error_mask]

    if len(error_indices) == 0:
        print(f"[{scored_path.stem}] No error rows.")
        return

    print(f"[{scored_path.stem}] Error rows: {len(error_indices)} of {len(df)}.")

    tmp_input = TMP_DIR / f"{scored_path.stem}.csv"
    tmp_output = TMP_DIR / f"{scored_path.stem}_fixed.csv"
    df.loc[error_indices].to_csv(tmp_input, index=False)

    new_output_path = assessor.process_csv(input_path=tmp_input, output_path=tmp_output)
    new_df = pd.read_csv(new_output_path)

    # Align columns explicitly before a positional assignment, rather than
    # relying on both frames happening to already be in the same column
    # order (that held before, but silently breaks if it ever doesn't).
    new_df = new_df[df.columns]
    df.loc[error_indices] = new_df.values

    df.to_csv(scored_path, index=False)
    print(f"[{scored_path.stem}] Fixed {len(error_indices)} rows -> {scored_path}")

    tmp_input.unlink(missing_ok=True)
    new_output_path.unlink(missing_ok=True)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    assessor = RiskAssessor(cfg)
    # Same folder the scorer writes to, so it also covers _graph (severity-only)
    # runs and any source -- not just results/saferproducts_gov/<model>/.
    results_dir = Path(cfg.output_path).parent

    scored_paths = sorted(results_dir.glob("*_scored.csv"))
    if not scored_paths:
        print(f"No scored CSVs found in {results_dir}.")
        return

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    for scored_path in scored_paths:
        fix_error_rows(scored_path, assessor)

    TMP_DIR.rmdir()


if __name__ == "__main__":
    main()