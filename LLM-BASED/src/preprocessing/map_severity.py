"""
Maps victim severity descriptions to a numeric severity scale, for a
given source.

Reads data/interim/<source>/incident_reports_clean.csv, applies the
source's severity mapping (see config/sources/<source>.yaml), and
saves the result to data/interim/<source>/incident_reports_mapped.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from utils.source_config import load_source_config

INTERIM_DIR = Path("data/interim")
SEVERITY_COLUMN = "victim_severity"


def map_severity(
    df: pd.DataFrame, source: str, severity_column: str = SEVERITY_COLUMN
) -> pd.DataFrame:
    """
    Applies the source's severity mapping to the dataframe.

    Exact matches are mapped directly; any value containing the
    source's injury keyword (and not already an exact match) is
    mapped to the injury severity level. Unmapped values (e.g.
    'Unspecified') are discarded.

    If the source config has no 'severity_map' (no self-reported
    ground-truth label for this source), the dataframe is returned
    unchanged -- a passthrough, not an error.
    """
    config = load_source_config(source)
    severity_map = config.get("severity_map")

    if severity_map is None:
        print(f"[{source}] No severity_map configured -- passthrough (no mapping applied).")
        return df

    injury_keyword = config["injury_keyword"]

    def map_value(val):
        if pd.isna(val):
            return None
        if val in severity_map:
            return severity_map[val]
        if injury_keyword in val:
            return severity_map[injury_keyword]
        return None  # discards 'Unspecified' and others

    n_before = len(df)
    df[severity_column] = df[severity_column].apply(map_value)
    df = df.dropna(subset=[severity_column])
    n_after = len(df)
    print(f"Discarded {n_before - n_after} rows with unmapped severity ({n_after} remaining).")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map victim severity to a numeric scale for a given source."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source identifier, e.g. 'saferproducts_gov' "
        "(must match a file in config/sources/).",
    )
    args = parser.parse_args()

    source_dir = INTERIM_DIR / args.source
    input_path = source_dir / "incident_reports_clean.csv"
    output_path = source_dir / "incident_reports_mapped.csv"

    # incident_reports_clean.csv is written with pandas' default utf-8
    # encoding by clean_incidents.py, so it's read back as utf-8 here --
    # this is independent of the source's raw-file encoding in the config.
    df = pd.read_csv(input_path, encoding="utf-8")
    df = map_severity(df, args.source)

    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()