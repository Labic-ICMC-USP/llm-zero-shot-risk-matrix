"""
Initial cleaning of a raw incident reports dataset.

Reads the raw dataset for a given source (see config/sources/<source>.yaml),
selects and renames the relevant columns, and saves the result to
data/interim/<source>/incident_reports_clean.csv.

It reads from 'input_path' if the source config sets one, or from the raw
file otherwise.
"""

import argparse
from pathlib import Path

import pandas as pd

from utils.source_config import load_source_config

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")


def clean_incidents(source: str) -> pd.DataFrame:
    config = load_source_config(source)

    input_path = config.get("input_path")
    if input_path:
        df = pd.read_csv(input_path, low_memory=False)
    else:
        raw_path = RAW_DIR / source / config["raw_file"]
        df = pd.read_csv(
            raw_path,
            encoding=config.get("encoding", "utf-8"),
            skiprows=config.get("skiprows", 0),
            usecols=config["columns_keep"],
            low_memory=False,
        )

    df = df.rename(columns=config["columns_rename"])

    drop_empty_column = config.get("drop_empty_column")
    if drop_empty_column:
        n_before = len(df)
        is_empty = df[drop_empty_column].isna() | (df[drop_empty_column].astype(str).str.strip() == "")
        df = df[~is_empty]
        print(f"Dropped {n_before - len(df)} rows with empty '{drop_empty_column}' ({len(df)} remaining).")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean a raw incident reports dataset for a given source."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source identifier, e.g. 'saferproducts_gov' "
        "(must match a file in config/sources/).",
    )
    args = parser.parse_args()

    df = clean_incidents(args.source)

    output_dir = INTERIM_DIR / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "incident_reports_clean.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()