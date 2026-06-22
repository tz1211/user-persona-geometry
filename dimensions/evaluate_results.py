"""Evaluate generated behavioral modulation benchmark responses."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from dimensions.experiment_config import BENCHMARK_METRIC_MODULE
from dimensions.llm_judge import DEFAULT_JUDGE_MODEL, JudgeConfig, build_judge


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _numeric_values(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, bool):
                values.setdefault(key, []).append(float(value))
            elif isinstance(value, (int, float)) and value is not None:
                values.setdefault(key, []).append(float(value))
    return values


def _aggregate(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"n": len(eval_rows)}
    for key, vals in _numeric_values(eval_rows).items():
        if not vals:
            continue
        aggregate[f"{key}_mean"] = statistics.mean(vals)
        aggregate[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return aggregate


def _evaluate_samples(
    *,
    generation_path: Path,
    metric_module: str,
    limit: int | None,
    judge,
) -> list[dict[str, Any]]:
    module = importlib.import_module(metric_module)
    samples = _load_jsonl(generation_path, limit=limit)
    if hasattr(module, "evaluate_samples"):
        metric_rows = module.evaluate_samples(samples, judge=judge)
    else:
        evaluate_sample = getattr(module, "evaluate_sample")
        metric_rows = [evaluate_sample(sample) for sample in samples]

    rows: list[dict[str, Any]] = []
    for sample, metrics in zip(samples, metric_rows):
        record = sample.get("record", {})
        rows.append({
            "condition_id": sample.get("condition_id"),
            "dimension": sample.get("dimension"),
            "category": sample.get("category"),
            "level": sample.get("level"),
            "benchmark_id": sample.get("benchmark_id"),
            "subtask_id": sample.get("subtask_id"),
            "prompt_id": sample.get("prompt_id"),
            "expected_behavior": (
                record.get("expected_behavior")
                or record.get("expected_behaviour")
                or sample.get("expected_behavior")
                or sample.get("expected_behaviour")
            ),
            "model_id": sample.get("model_id"),
            "seed": sample.get("seed"),
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    return rows


def _write_summary(results_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for agg_path in sorted((results_dir / "aggregates").glob("*/*/*.json")):
        data = json.loads(agg_path.read_text(encoding="utf-8"))
        base = {
            "condition_id": data.get("condition_id", ""),
            "dimension": data.get("dimension", ""),
            "category": data.get("category", ""),
            "level": data.get("level", ""),
            "benchmark_id": data.get("benchmark_id", ""),
            "model_id": data.get("model_id", ""),
            "seed": data.get("seed", ""),
        }
        for metric, value in data.get("metrics", {}).items():
            rows.append({**base, "metric": metric, "value": value})

    if not rows:
        return

    summary_path = results_dir / "summary.csv"
    fieldnames = [
        "condition_id",
        "dimension",
        "category",
        "level",
        "benchmark_id",
        "model_id",
        "seed",
        "metric",
        "value",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_results(
    *,
    results_dir: Path,
    conditions_filter: list[str] | None = None,
    benchmarks_filter: list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    judge_config: JudgeConfig | None = None,
) -> None:
    generation_root = results_dir / "generations"
    if not generation_root.is_dir():
        raise FileNotFoundError(f"No generations directory found at {generation_root}")
    judge = build_judge(judge_config or JudgeConfig())

    for samples_path in sorted(generation_root.glob("*/*/samples.jsonl")):
        condition_id = samples_path.parents[1].name
        benchmark_id = samples_path.parents[0].name
        if conditions_filter is not None and condition_id not in conditions_filter:
            continue
        if benchmarks_filter is not None and benchmark_id not in benchmarks_filter:
            continue
        if benchmark_id not in BENCHMARK_METRIC_MODULE:
            print(f"[SKIP] {condition_id}/{benchmark_id}: no metric module", flush=True)
            continue

        eval_path = (
            results_dir / "evaluations" / condition_id /
            benchmark_id / f"{benchmark_id}.jsonl"
        )
        if eval_path.exists() and not force:
            print(f"[SKIP] {eval_path} exists; use --force to recompute", flush=True)
            continue

        rows = _evaluate_samples(
            generation_path=samples_path,
            metric_module=BENCHMARK_METRIC_MODULE[benchmark_id],
            limit=limit,
            judge=judge,
        )
        _write_jsonl(eval_path, rows)

        first = rows[0] if rows else {}
        aggregate = {
            "condition_id": condition_id,
            "dimension": first.get("dimension"),
            "category": first.get("category"),
            "level": first.get("level"),
            "benchmark_id": benchmark_id,
            "model_id": first.get("model_id"),
            "seed": first.get("seed"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": _aggregate(rows),
        }
        aggregate_path = (
            results_dir / "aggregates" / condition_id / benchmark_id / "aggregate.json"
        )
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[DONE] evaluated {condition_id}/{benchmark_id}", flush=True)

    _write_summary(results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated behavioral modulation benchmark responses.",
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--judge-backend",
        default="vllm",
        choices=[
            "vllm",
            "openai",
            "api",
            "openai-compatible",
            "anthropic",
            "gemini",
            "google",
            "none",
        ],
        help="LLM judge backend. Use 'none' for local heuristic-only evaluation.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-api-base-url", default=None)
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-api-concurrency", type=int, default=32)
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--judge-dtype", default="bfloat16")
    parser.add_argument("--judge-batch-size", type=int, default=8)
    args = parser.parse_args()

    evaluate_results(
        results_dir=Path(args.results_dir),
        conditions_filter=args.conditions,
        benchmarks_filter=args.benchmarks,
        limit=args.limit,
        force=args.force,
        judge_config=JudgeConfig(
            backend=args.judge_backend,
            model=args.judge_model,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
            api_base_url=args.judge_api_base_url,
            api_key_env=args.judge_api_key_env,
            api_concurrency=args.judge_api_concurrency,
            gpu_memory_utilization=args.judge_gpu_memory_utilization,
            dtype=args.judge_dtype,
            batch_size=args.judge_batch_size,
        ),
    )


if __name__ == "__main__":
    main()
