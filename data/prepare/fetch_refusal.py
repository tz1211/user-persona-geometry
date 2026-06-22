"""Prepare Refusal benchmark data.

Outputs JSONL files under ``data/benchmarks/refusal/``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path
from typing import Iterable

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "refusal"
SEED = 42


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(rows) <= n:
        return rows
    return rng.sample(rows, n)


def prepare_harmbench(rng: random.Random) -> int:
    url = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
        "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for i, row in enumerate(reader):
        prompt = row.get("Behavior") or row.get("behavior") or row.get("prompt")
        if not prompt:
            continue
        functional = (
            row.get("FunctionalCategory")
            or row.get("functional_category")
            or row.get("Functional Category")
            or ""
        ).lower()
        if functional and "standard" not in functional:
            continue
        rows.append({
            "id": f"harmbench_{i:04d}",
            "benchmark_id": "refusal",
            "subtask_id": "harmbench",
            "prompt": prompt,
            "split": "should_refuse",
            "expected_behavior": "should_refuse",
            "source": "centerforaisafety/HarmBench",
            "category": row.get("SemanticCategory") or row.get("semantic_category"),
        })
    sample = _sample(rows, 400, rng)
    _write_jsonl(OUTPUT_DIR / "harmbench.jsonl", sample)
    return len(sample)


def prepare_orbench(rng: random.Random) -> int:
    from datasets import load_dataset

    def load_first_available(configs: list[str]):
        last_error: Exception | None = None
        for dataset_name in ("bench-llm/or-bench", "bench-llms/or-bench"):
            for config in configs:
                try:
                    return dataset_name, load_dataset(dataset_name, config, split="train")
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(f"Could not load OR-Bench configs {configs}") from last_error

    written = 0
    hard_source, hard = load_first_available(["or-bench-hard-1k", "or-bench-hard"])
    hard_rows = []
    for i, row in enumerate(hard):
        hard_rows.append({
            "id": f"orbench_hard_{i:04d}",
            "benchmark_id": "refusal",
            "subtask_id": "orbench_hard",
            "prompt": row.get("prompt") or row.get("question"),
            "split": "should_not_refuse",
            "expected_behavior": "should_not_refuse",
            "source": hard_source,
            "category": row.get("category"),
        })
    hard_rows = [row for row in hard_rows if row["prompt"]]
    hard_sample = _sample(hard_rows, 500, rng)
    _write_jsonl(OUTPUT_DIR / "orbench_hard.jsonl", hard_sample)
    written += len(hard_sample)

    toxic_source, toxic = load_first_available([
        "or-bench-toxic-all",
        "or-bench-toxic",
        "or-bench-toxic-600",
    ])
    toxic_rows = []
    for i, row in enumerate(toxic):
        toxic_rows.append({
            "id": f"orbench_toxic_{i:04d}",
            "benchmark_id": "refusal",
            "subtask_id": "orbench_toxic",
            "prompt": row.get("prompt") or row.get("question"),
            "split": "should_refuse",
            "expected_behavior": "should_refuse",
            "source": toxic_source,
            "category": row.get("category"),
        })
    toxic_rows = [row for row in toxic_rows if row["prompt"]]
    toxic_sample = _sample(toxic_rows, 300, rng)
    _write_jsonl(OUTPUT_DIR / "orbench_toxic.jsonl", toxic_sample)
    written += len(toxic_sample)
    return written


def prepare_xstest(rng: random.Random) -> int:
    from datasets import load_dataset

    ds = load_dataset("Paul/XSTest", split="train")
    safe_rows = []
    for i, row in enumerate(ds):
        label = str(row.get("label", "")).lower()
        prompt = row.get("prompt")
        if not prompt:
            continue
        record = {
            "id": f"xstest_{i:04d}",
            "benchmark_id": "refusal",
            "prompt": prompt,
            "source": "Paul/XSTest",
            "category": row.get("type"),
            "note": row.get("note"),
        }
        if label == "safe":
            safe_rows.append({
                **record,
                "subtask_id": "xstest_safe",
                "split": "should_not_refuse",
                "expected_behavior": "should_not_refuse",
            })

    safe_sample = _sample(safe_rows, 250, rng)
    _write_jsonl(OUTPUT_DIR / "xstest_safe.jsonl", safe_sample)
    return len(safe_sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts = {
        "harmbench": prepare_harmbench(rng),
        "orbench": prepare_orbench(rng),
        "xstest": prepare_xstest(rng),
    }
    total = sum(counts.values())
    print(f"Wrote refusal data to {OUTPUT_DIR}: {counts} total={total}", flush=True)


if __name__ == "__main__":
    main()
