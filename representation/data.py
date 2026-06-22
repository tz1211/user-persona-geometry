from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from dimensions.experiment_config import BENCHMARK_PATHS, CONDITIONS, CONDITIONS_BY_ID

REFUSAL_BENCHMARK_ID = "refusal"
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.7


def model_results_dir(output_dir: str | Path, model_id: str) -> Path:
    """Return the model-specific vector root, preserving existing model-id path style."""
    return Path(output_dir) / model_id


def load_jsonl_file(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_source_file"] = str(path)
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def load_benchmark_records(
    benchmark_id: str = REFUSAL_BENCHMARK_ID,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load benchmark records from the configured JSONL file or directory."""
    path = BENCHMARK_PATHS[benchmark_id]
    if path.is_file():
        return load_jsonl_file(path, limit=limit)
    if not path.is_dir():
        raise FileNotFoundError(f"No benchmark data found at {path}")

    records: list[dict[str, Any]] = []
    for jsonl_path in sorted(path.glob("*.jsonl")):
        remaining = None if limit is None else limit - len(records)
        if remaining is not None and remaining <= 0:
            break
        records.extend(load_jsonl_file(jsonl_path, limit=remaining))
    return records


def prompt_id(record: dict[str, Any]) -> str:
    value = record.get("id", record.get("key"))
    if value is None:
        raise ValueError(f"Benchmark record has no id/key: {record}")
    return str(value)


def prompt_stratum(record: dict[str, Any]) -> str:
    """Return the stratification group used for train/held-out prompt splits."""
    if record.get("subtask_id"):
        return str(record["subtask_id"])
    if record.get("source"):
        return str(record["source"])
    source_file = record.get("_source_file")
    if source_file:
        return Path(source_file).stem
    return "unknown"


def make_stratified_prompt_split(
    records: list[dict[str, Any]],
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Create a deterministic split over prompts, stratified by benchmark source."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    strata_by_prompt: dict[str, str] = {}
    for record in records:
        pid = prompt_id(record)
        stratum = prompt_stratum(record)
        by_stratum[stratum].append(pid)
        strata_by_prompt[pid] = stratum

    rng = random.Random(seed)
    train_ids: list[str] = []
    heldout_ids: list[str] = []
    counts: dict[str, dict[str, int]] = {}

    for stratum, ids in sorted(by_stratum.items()):
        unique_ids = sorted(set(ids))
        rng.shuffle(unique_ids)
        if len(unique_ids) == 1:
            n_train = 1
        else:
            n_train = round(len(unique_ids) * train_fraction)
            n_train = min(max(1, n_train), len(unique_ids) - 1)
        train_part = sorted(unique_ids[:n_train])
        heldout_part = sorted(unique_ids[n_train:])
        train_ids.extend(train_part)
        heldout_ids.extend(heldout_part)
        counts[stratum] = {
            "total": len(unique_ids),
            "train": len(train_part),
            "heldout": len(heldout_part),
        }

    return {
        "benchmark_id": REFUSAL_BENCHMARK_ID,
        "seed": seed,
        "train_fraction": train_fraction,
        "train_prompt_ids": sorted(train_ids),
        "heldout_prompt_ids": sorted(heldout_ids),
        "strata_by_prompt_id": strata_by_prompt,
        "stratum_counts": counts,
    }


def write_split(model_root: Path, split: dict[str, Any]) -> Path:
    split_dir = model_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{REFUSAL_BENCHMARK_ID}_split_seed_{split['seed']}.json"
    path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_split(model_root: Path, split_seed: int = DEFAULT_SPLIT_SEED) -> dict[str, Any]:
    path = model_root / "splits" / f"{REFUSAL_BENCHMARK_ID}_split_seed_{split_seed}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def non_baseline_conditions() -> list[dict[str, Any]]:
    return [condition for condition in CONDITIONS if condition["id"] != "baseline"]


def conditions_by_id() -> dict[str, dict[str, Any]]:
    return CONDITIONS_BY_ID
