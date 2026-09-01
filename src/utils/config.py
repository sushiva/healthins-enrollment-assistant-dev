"""Config loading for the formulary RAG pipeline.

Ported from notebooks/formulary/02-Basic_RAG_Formulary.ipynb §1 — same
config.yaml, same keys. Paths in config.yaml are project-root-relative; the
notebooks prepend "../../" because they run from notebooks/formulary/, this
module resolves them against the actual project root instead.
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a config.yaml path (e.g. config['paths']['raw_data']) against the project root."""
    return PROJECT_ROOT / relative_path
