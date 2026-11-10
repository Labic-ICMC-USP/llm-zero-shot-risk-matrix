"""
Loads per-source preprocessing configuration from config/sources/<source>.yaml.
"""

from pathlib import Path

import yaml

SOURCES_CONFIG_DIR = Path("config/sources")


def load_source_config(source: str) -> dict:
    config_path = SOURCES_CONFIG_DIR / f"{source}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found for source '{source}' at {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)