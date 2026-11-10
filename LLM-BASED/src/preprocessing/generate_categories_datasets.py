"""
Splits a source's mapped incident reports into one Parquet file per
product category.

Reads data/interim/<source>/incident_reports_mapped.csv, drops the
category column (redundant once split), and saves each category as a
separate Parquet file in data/processed/<source>/.
"""

import argparse
from pathlib import Path

import pandas as pd

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def save_category_datasets(
    input_csv: Path,
    output_dir: Path,
    category_column: str = "product_category",
) -> None:
    """
    Generate per-category Parquet files containing only relevant columns.
    """
    # incident_reports_mapped.csv is written with pandas' default utf-8
    # encoding by map_severity.py, so it's read back as utf-8 here.
    df = pd.read_csv(input_csv, encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)

    categories = df[category_column].unique()

    for cat in categories:
        df_cat = df[df[category_column] == cat]
        df_cat = df_cat.drop(columns=[category_column])

        safe_name = cat.lower().replace(" ", "_").replace("&", "and")
        parquet_path = output_dir / f"{safe_name}.parquet"

        df_cat.to_parquet(parquet_path, index=False)

        print(f"Saved {cat}: {len(df_cat)} rows -> {parquet_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a source's mapped dataset into per-category Parquet files."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source identifier, e.g. 'saferproducts_gov'.",
    )
    args = parser.parse_args()

    input_csv = INTERIM_DIR / args.source / "incident_reports_mapped.csv"
    output_dir = PROCESSED_DIR / args.source

    save_category_datasets(input_csv, output_dir)


if __name__ == "__main__":
    main()