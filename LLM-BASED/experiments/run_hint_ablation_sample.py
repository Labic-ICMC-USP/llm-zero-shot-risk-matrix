"""
Hint ablation on a sample: scores the same per-category sample twice,
without hints (reference_mode off) and with hints (reference_mode on).

Sample, per category: categories with <= CAP rows go in whole; otherwise
up to PER_LEVEL rows of each severity level are taken first, and the rest
of the CAP is filled with a stratified (proportional) draw from the
remaining rows. Fixed SEED, so the same sample is rebuilt on every run --
the sample files are temporary and deleted at the end.

With hints, the reference severity distribution comes from the FULL
category file (reference_df), not from the sample.

Model, endpoint and temperature come from config/config.yaml (same in both
variants); its input_path/output_path are ignored. Outputs:
  results/<source>/<model>_sample/<category>_scored.csv        (no hints)
  results/<source>/<model>_sample_hints/<category>_scored.csv  (hints)

Resumable: rerunning skips rows already in the output CSVs.

Usage (from LLM-BASED/):
  python experiments/run_hint_ablation_sample.py
"""

import tempfile
from pathlib import Path

import pandas as pd

from risk_scorer.risk_scorer import load_config, RiskAssessor

CONFIG_PATH = Path("config/config.yaml")
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

SEED = 42
CAP = 100
PER_LEVEL = 10


def _proportional_counts(counts: pd.Series, n: int) -> pd.Series:
    """Splits n across groups proportionally to counts (largest remainder)."""
    quotas = counts * n / counts.sum()
    alloc = quotas.astype(int)
    remainder = n - alloc.sum()
    order = (quotas - alloc).sort_values(ascending=False).index[:remainder]
    alloc[order] += 1
    return alloc


def sample_category(df: pd.DataFrame, severity_col: str) -> pd.DataFrame:
    if len(df) <= CAP:
        return df

    base = pd.concat([
        g.sample(min(len(g), PER_LEVEL), random_state=SEED)
        for _, g in df.groupby(severity_col)
    ])

    rest = df.drop(base.index)
    alloc = _proportional_counts(rest[severity_col].value_counts(), CAP - len(base))
    fill = pd.concat([
        g.sample(int(alloc[level]), random_state=SEED)
        for level, g in rest.groupby(severity_col)
        if alloc.get(level, 0) > 0
    ])

    return pd.concat([base, fill])


def run() -> None:
    cfg = load_config(CONFIG_PATH)
    severity_col = cfg.severity_level_column
    model_dir_name = cfg.model_name.split("/")[-1]
    full_paths = sorted((DATA_DIR / cfg.source).glob("*.parquet"))

    variants = [
        (f"{model_dir_name}_sample", False),
        (f"{model_dir_name}_sample_hints", True),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Build all samples once, shared by both variants.
        full_dfs = {}
        for path in full_paths:
            full_df = pd.read_parquet(path)
            sample = sample_category(full_df, severity_col)
            sample.to_parquet(tmp_dir / path.name)
            full_dfs[path.stem] = full_df
            print(f"[{path.stem}] sample: {len(sample)} of {len(full_df)} rows")

        for folder, reference_mode in variants:
            assessor = RiskAssessor(cfg.model_copy(update={"reference_mode": reference_mode}))
            out_dir = RESULTS_DIR / cfg.source / folder
            print(f"\n=== {folder} (hints={'on' if reference_mode else 'off'}) ===")

            for path in full_paths:
                print(f"\n[{path.stem}]")
                assessor.process_csv(
                    input_path=tmp_dir / path.name,
                    output_path=out_dir / f"{path.stem}_scored.csv",
                    reference_df=full_dfs[path.stem],
                )

    print("\nDone.")


if __name__ == "__main__":
    run()
