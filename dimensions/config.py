"""Load and validate YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.is_absolute():
        path = _CONFIGS_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dimension_config(dimension: str) -> dict[str, Any]:
    """Load ``configs/<dimension>.yaml``."""
    return load_yaml(_CONFIGS_DIR / f"{dimension}.yaml")


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load a model config YAML (absolute path or relative to configs/)."""
    return load_yaml(path)
