"""Experiment configuration for refusal personalization experiments.

Conditions are loaded from the existing per-dimension YAML files under
``dimensions/configs``. Benchmark paths point at grouped JSONL directories
under ``data/benchmarks``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DATA_DIR = REPO_ROOT / "data" / "benchmarks"

DIMENSIONS = ("knowledge", "intent", "emotion", "belief")


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static metadata for one top-level benchmark metric."""

    id: str
    label: str
    path: Path
    metric_module: str
    max_tokens: int = 4096
    protocol: str = "mixed"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dimension_configs() -> dict[str, dict[str, Any]]:
    """Load the four dimension YAML configs keyed by dimension name."""
    configs: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        cfg = _load_yaml(CONFIG_DIR / f"{dimension}.yaml")
        if cfg.get("dimension") != dimension:
            raise ValueError(f"{dimension}.yaml has dimension={cfg.get('dimension')!r}")
        configs[dimension] = cfg
    return configs


DIMENSION_CONFIGS = load_dimension_configs()
BASE_SYSTEM_PROMPT = DIMENSION_CONFIGS["knowledge"]["base_system_prompt"].strip()


def build_conditions() -> list[dict[str, Any]]:
    """Build the shared baseline plus all non-baseline conditions from YAML."""
    conditions: list[dict[str, Any]] = [
        {
            "id": "baseline",
            "dimension": None,
            "category": None,
            "level": "baseline",
            "level_label": "Baseline",
            "level_description": None,
            "ordinal": 0,
        }
    ]

    for dimension, cfg in DIMENSION_CONFIGS.items():
        for category in cfg.get("categories", []):
            category_key = category["key"]
            for level in category.get("levels", []):
                level_key = level["key"]
                cond = {
                    "id": f"{dimension}_{category_key}_{level_key}",
                    "dimension": dimension,
                    "category": category_key,
                    "category_label": category.get("label", category_key),
                    "level": level_key,
                    "level_label": level.get("label", level_key),
                    "level_description": level.get("description"),
                    "ordinal": level.get("ordinal"),
                }
                for optional_key in ("valence", "arousal"):
                    if optional_key in level:
                        cond[optional_key] = level[optional_key]
                conditions.append(cond)
    return conditions


CONDITIONS = build_conditions()
CONDITIONS_BY_ID = {condition["id"]: condition for condition in CONDITIONS}

ALL_BENCHMARKS = ["refusal"]

BENCHMARK_SCOPE: dict[str | None, list[str]] = {
    "knowledge": ALL_BENCHMARKS,
    "intent": ALL_BENCHMARKS,
    "emotion": ALL_BENCHMARKS,
    "belief": ALL_BENCHMARKS,
    None: ALL_BENCHMARKS,
}

BENCHMARK_SPECS: dict[str, BenchmarkSpec] = {
    "refusal": BenchmarkSpec(
        id="refusal",
        label="Refusal",
        path=DATA_DIR / "refusal",
        metric_module="dimensions.metrics.refusal_metric",
    ),
}

BENCHMARK_PATHS = {key: spec.path for key, spec in BENCHMARK_SPECS.items()}
BENCHMARK_METRIC_MODULE = {
    key: spec.metric_module for key, spec in BENCHMARK_SPECS.items()
}
BENCHMARK_MAX_TOKENS = {
    key: spec.max_tokens for key, spec in BENCHMARK_SPECS.items()
}

DEFAULT_GEN_KWARGS: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}

BENCHMARK_GEN_KWARGS = {
    key: dict(DEFAULT_GEN_KWARGS) for key in BENCHMARK_SPECS
}


def benchmark_paths_for_metadata() -> dict[str, str]:
    """Return benchmark ids and configured paths for metadata.json."""
    return {key: str(path) for key, path in BENCHMARK_PATHS.items()}
