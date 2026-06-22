"""Generate model responses for behavioral modulation benchmarks.

The runner loads conditions from ``dimensions/configs/*.yaml`` and benchmark
records from ``data/benchmarks/<metric>/*.jsonl``. It writes raw generations
only; metric and judge evaluation happens in ``dimensions/evaluate_results.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from dimensions.experiment_config import (
    BASE_SYSTEM_PROMPT,
    BENCHMARK_GEN_KWARGS,
    BENCHMARK_MAX_TOKENS,
    BENCHMARK_PATHS,
    BENCHMARK_SCOPE,
    CONDITIONS,
    benchmark_paths_for_metadata,
)
from dimensions.inference import GenerationResult, InferenceEngine


def _load_jsonl_file(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
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


def _load_benchmark_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load benchmark records from a JSONL file or a directory of JSONL files."""
    if path.is_file():
        return _load_jsonl_file(path, limit=limit)
    if not path.is_dir():
        return []

    records: list[dict[str, Any]] = []
    for jsonl_path in sorted(path.glob("*.jsonl")):
        remaining = None if limit is None else limit - len(records)
        if remaining is not None and remaining <= 0:
            break
        records.extend(_load_jsonl_file(jsonl_path, limit=remaining))
    return records


def _build_system_prompt(level_description: str | None) -> str:
    if level_description:
        return f"{BASE_SYSTEM_PROMPT}\n{level_description.strip()}"
    return BASE_SYSTEM_PROMPT


def _build_single_turn_messages(record: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        messages = list(record["messages"])
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}, *messages]
        return messages
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": record["prompt"]},
    ]


def _second_turn_prompt(record: dict[str, Any]) -> str:
    for key in ("challenge_prompt", "rebuttal_prompt", "followup_prompt", "turn_2_prompt"):
        value = record.get(key)
        if value:
            return str(value)
    return "Are you sure?"


def _result_to_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "raw_response": result.raw_text,
        "cleaned_response": result.response_content,
        "think_content": result.thinking_content,
        "thinking_tokens": result.thinking_tokens,
        "response_tokens": result.response_tokens,
        "total_tokens": result.thinking_tokens + result.response_tokens,
    }


def _make_sample(
    *,
    record: dict[str, Any],
    condition: dict[str, Any],
    benchmark_id: str,
    system_prompt: str,
    model_id: str,
    seed: int,
    messages: list[dict[str, str]],
    result: GenerationResult,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_id = str(record.get("id", record.get("key", "")))
    payload = {
        "condition_id": condition["id"],
        "dimension": condition["dimension"] or "baseline",
        "category": condition.get("category"),
        "level": condition["level"],
        "level_label": condition.get("level_label"),
        "ordinal": condition.get("ordinal"),
        "benchmark_id": benchmark_id,
        "subtask_id": record.get("subtask_id"),
        "prompt_id": prompt_id,
        "prompt": record.get("prompt", ""),
        "messages": messages,
        "system_prompt": system_prompt,
        "record": record,
        "model_id": model_id,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(_result_to_dict(result))
    if extra:
        payload.update(extra)
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_single_turn_records(
    *,
    engine: InferenceEngine,
    records: list[dict[str, Any]],
    condition: dict[str, Any],
    benchmark_id: str,
    system_prompt: str,
    seed: int,
) -> list[dict[str, Any]]:
    messages_batch = [
        _build_single_turn_messages(record, system_prompt) for record in records
    ]
    results = engine.generate_chat_detailed(
        messages_batch,
        BENCHMARK_GEN_KWARGS[benchmark_id],
        BENCHMARK_MAX_TOKENS[benchmark_id],
    )
    return [
        _make_sample(
            record=record,
            condition=condition,
            benchmark_id=benchmark_id,
            system_prompt=system_prompt,
            model_id=engine.model_id,
            seed=seed,
            messages=messages,
            result=result,
        )
        for record, messages, result in zip(records, messages_batch, results)
    ]


def _run_two_turn_records(
    *,
    engine: InferenceEngine,
    records: list[dict[str, Any]],
    condition: dict[str, Any],
    benchmark_id: str,
    system_prompt: str,
    seed: int,
) -> list[dict[str, Any]]:
    first_messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["prompt"]},
        ]
        for record in records
    ]
    first_results = engine.generate_chat_detailed(
        first_messages,
        BENCHMARK_GEN_KWARGS[benchmark_id],
        BENCHMARK_MAX_TOKENS[benchmark_id],
    )

    second_messages = []
    for record, first in zip(records, first_results):
        second_messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": first.response_content},
                {"role": "user", "content": _second_turn_prompt(record)},
            ]
        )

    second_results = engine.generate_chat_detailed(
        second_messages,
        BENCHMARK_GEN_KWARGS[benchmark_id],
        BENCHMARK_MAX_TOKENS[benchmark_id],
    )

    samples: list[dict[str, Any]] = []
    for record, messages, first, second in zip(
        records, second_messages, first_results, second_results
    ):
        samples.append(
            _make_sample(
                record=record,
                condition=condition,
                benchmark_id=benchmark_id,
                system_prompt=system_prompt,
                model_id=engine.model_id,
                seed=seed,
                messages=messages,
                result=second,
                extra={
                    "turn_1_raw_response": first.raw_text,
                    "turn_1_response": first.response_content,
                    "turn_1_think_content": first.thinking_content,
                    "turn_1_thinking_tokens": first.thinking_tokens,
                    "turn_1_response_tokens": first.response_tokens,
                    "turn_2_raw_response": second.raw_text,
                    "turn_2_response": second.response_content,
                    "turn_2_think_content": second.thinking_content,
                    "turn_2_thinking_tokens": second.thinking_tokens,
                    "turn_2_response_tokens": second.response_tokens,
                },
            )
        )
    return samples


def _run_records(
    *,
    engine: InferenceEngine,
    records: list[dict[str, Any]],
    condition: dict[str, Any],
    benchmark_id: str,
    system_prompt: str,
    seed: int,
) -> list[dict[str, Any]]:
    two_turn = [
        record for record in records
        if record.get("turn_protocol") == "two_turn_challenge"
    ]
    single_turn = [
        record for record in records
        if record.get("turn_protocol") != "two_turn_challenge"
    ]
    samples: list[dict[str, Any]] = []
    if single_turn:
        samples.extend(
            _run_single_turn_records(
                engine=engine,
                records=single_turn,
                condition=condition,
                benchmark_id=benchmark_id,
                system_prompt=system_prompt,
                seed=seed,
            )
        )
    if two_turn:
        samples.extend(
            _run_two_turn_records(
                engine=engine,
                records=two_turn,
                condition=condition,
                benchmark_id=benchmark_id,
                system_prompt=system_prompt,
                seed=seed,
            )
        )
    return samples


def run_experiment(
    *,
    engine: InferenceEngine,
    output_dir: Path,
    conditions_filter: list[str] | None = None,
    benchmarks_filter: list[str] | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> None:
    try:
        from tqdm import tqdm
        progress = tqdm
    except ImportError:
        def progress(iterable, **_kwargs):
            return iterable

    conditions = [
        condition for condition in CONDITIONS
        if conditions_filter is None or condition["id"] in conditions_filter
    ]

    for condition in progress(conditions, desc="Conditions"):
        system_prompt = _build_system_prompt(condition["level_description"])
        applicable = BENCHMARK_SCOPE[condition["dimension"]]
        benchmarks = [
            benchmark_id for benchmark_id in applicable
            if benchmarks_filter is None or benchmark_id in benchmarks_filter
        ]

        for benchmark_id in progress(benchmarks, desc=condition["id"], leave=False):
            records = _load_benchmark_records(BENCHMARK_PATHS[benchmark_id], limit=limit)
            if not records:
                print(
                    f"[SKIP] {condition['id']}/{benchmark_id}: no data at "
                    f"{BENCHMARK_PATHS[benchmark_id]}",
                    flush=True,
                )
                continue

            out_path = (
                output_dir / "generations" / condition["id"] /
                benchmark_id / "samples.jsonl"
            )
            print(
                f"[RUN] {condition['id']}/{benchmark_id}: {len(records)} records",
                flush=True,
            )
            samples = _run_records(
                engine=engine,
                records=records,
                condition=condition,
                benchmark_id=benchmark_id,
                system_prompt=system_prompt,
                seed=seed,
            )
            _write_jsonl(out_path, samples)
            print(f"[DONE] wrote {len(samples)} samples to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate behavioral modulation benchmark responses.",
    )
    parser.add_argument(
        "--model-config",
        default="dimensions/configs/models/qwen3_4b.yaml",
        help="Path to model YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/behavioral",
        help="Root output directory.",
    )
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_cfg_path = Path(args.model_config)
    if not model_cfg_path.is_absolute():
        model_cfg_path = repo_root / model_cfg_path
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    model_cfg["seed"] = args.seed

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_dir = output_root / model_cfg["model_id"] / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_config": str(model_cfg_path),
        "model_id": model_cfg["model_id"],
        "seed": args.seed,
        "limit": args.limit,
        "conditions": args.conditions,
        "benchmarks": args.benchmarks,
        "benchmark_paths": benchmark_paths_for_metadata(),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    engine = InferenceEngine(model_cfg)
    run_experiment(
        engine=engine,
        output_dir=output_dir,
        conditions_filter=args.conditions,
        benchmarks_filter=args.benchmarks,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"Generation complete. Results in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
