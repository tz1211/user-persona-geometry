"""Activation steering for Section 6.3.

This runner generates baseline refusal responses while adding a saved
user-attribute contrast vector to the residual stream during response decoding.
It then evaluates the generated samples with the existing benchmark metric.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import importlib
import json
import math
import os
import statistics
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from dimensions.evaluate_results import _load_jsonl, _write_jsonl
from dimensions.experiment_config import (
    BENCHMARK_GEN_KWARGS,
    BENCHMARK_MAX_TOKENS,
    BENCHMARK_METRIC_MODULE,
    CONDITIONS_BY_ID,
    benchmark_paths_for_metadata,
)
from dimensions.inference import GenerationResult, _split_think_tokens
from dimensions.llm_judge import DEFAULT_JUDGE_MODEL, JudgeConfig, build_judge
from dimensions.postprocess import strip_think_tags
from representation.data import (
    DEFAULT_SPLIT_SEED,
    load_benchmark_records,
    load_jsonl_file,
    load_split,
    prompt_id,
)
from representation.extract_activations import load_model_and_tokenizer
from representation.prompts import build_system_prompt, render_with_positions
from representation.vector_specs import VECTOR_SPECS, specs_by_name

DIMENSION_VECTORS = {
    dimension: tuple(spec.name for spec in VECTOR_SPECS if spec.dimension == dimension)
    for dimension in dict.fromkeys(spec.dimension for spec in VECTOR_SPECS)
}
DEFAULT_VECTORS = DIMENSION_VECTORS["knowledge"]
DEFAULT_COEFFICIENTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
DEFAULT_BENCHMARK = "refusal"
DEFAULT_DIMENSION = "knowledge"
DEFAULT_RECORD_SPLIT = "all"


def _batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _coef_label(coefficient: float) -> str:
    return format(float(coefficient), "g")


class ResponseOnlySteerer:
    """Add a vector to decoder-block output only during cached response decode."""

    def __init__(
        self,
        model: Any,
        *,
        layer: int,
        vector: torch.Tensor,
        coefficient: float,
    ) -> None:
        self.model = model
        self.layer = layer
        self.vector = vector
        self.coefficient = float(coefficient)
        self._handle: Any = None

    def __enter__(self) -> "ResponseOnlySteerer":
        layers = self.model.model.layers
        if not 0 <= self.layer < len(layers):
            raise IndexError(f"Layer {self.layer} out of range for {len(layers)} layers")
        if self.coefficient != 0.0:
            self._handle = layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            return output

        direction = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden + self.coefficient * direction.view(1, 1, -1)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered


def _load_vector(
    *,
    vectors_dir: Path,
    model_id: str,
    dimension: str,
    vector_name: str,
    position: str,
    layer: int,
) -> torch.Tensor:
    path = vectors_dir / model_id / dimension / f"{vector_name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing steering vector artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if int(artifact.get("layer", layer)) != layer:
        raise ValueError(
            f"{path} was built at layer {artifact.get('layer')}, but --layer={layer}"
        )
    positions = artifact.get("positions", {})
    if position not in positions:
        raise KeyError(f"{path} has no position {position!r}; available: {sorted(positions)}")
    vector = positions[position].float()
    if vector.ndim != 1:
        raise ValueError(f"{path}:{position} expected a 1D vector, got shape {tuple(vector.shape)}")
    return vector


def _generation_kwargs(benchmark_id: str) -> dict[str, Any]:
    return dict(BENCHMARK_GEN_KWARGS[benchmark_id])


def _load_records_from_override(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
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


def _select_records(
    *,
    records: list[dict[str, Any]],
    record_split: str,
    split_root: Path,
    split_seed: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    if record_split == "all":
        selected = records
    else:
        split = load_split(split_root, split_seed=split_seed)
        key = f"{record_split}_prompt_ids"
        if key not in split:
            raise KeyError(f"Split file has no {key!r}; available keys: {sorted(split)}")
        allowed = set(split[key])
        selected = [record for record in records if prompt_id(record) in allowed]

    if limit is not None:
        selected = selected[:limit]
    return selected


def _resolve_vectors(dimension: str, vector_names: list[str] | None) -> list[str]:
    if dimension not in DIMENSION_VECTORS:
        raise KeyError(
            f"Unknown steering dimension {dimension!r}; available: {sorted(DIMENSION_VECTORS)}"
        )
    resolved = list(DIMENSION_VECTORS[dimension] if vector_names is None else vector_names)
    specs = specs_by_name()
    bad = [
        name
        for name in resolved
        if name not in specs or specs[name].dimension != dimension
    ]
    if bad:
        raise ValueError(
            f"Vectors {bad} do not belong to dimension {dimension!r}; "
            f"valid vectors: {list(DIMENSION_VECTORS[dimension])}"
        )
    return resolved


def _perplexity_stats(log_probs: list[float]) -> dict[str, float | int | None]:
    if not log_probs:
        return {"nll": None, "perplexity": None, "scored_tokens": 0}
    nll = -statistics.mean(log_probs)
    perplexity = math.exp(nll) if nll < 709 else float("inf")
    return {
        "nll": float(nll),
        "perplexity": float(perplexity),
        "scored_tokens": len(log_probs),
    }


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    rendered_texts: list[str],
    gen_kwargs: dict[str, Any],
    max_new_tokens: int,
) -> list[GenerationResult]:
    encoded = tokenizer(
        rendered_texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to(_input_device(model))

    temperature = float(gen_kwargs.get("temperature", 0.6) or 0.0)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "top_p": gen_kwargs.get("top_p", 0.95),
        "top_k": gen_kwargs.get("top_k", 20),
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if temperature > 0.0:
        generate_kwargs["temperature"] = temperature

    outputs = model.generate(**encoded, **generate_kwargs)
    input_width = encoded["input_ids"].shape[1]
    results: list[GenerationResult] = []
    special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    for row_idx, row in enumerate(outputs.sequences[:, input_width:]):
        step_ids = row.detach().cpu().tolist()
        step_log_probs: list[float] = []
        for step_idx, score in enumerate(outputs.scores):
            if step_idx >= len(step_ids):
                break
            token_id = int(step_ids[step_idx])
            if token_id in special_ids:
                continue
            log_prob = torch.log_softmax(score[row_idx].float(), dim=-1)[token_id]
            step_log_probs.append(float(log_prob.detach().cpu().item()))

        ids = [int(tok) for tok in step_ids if int(tok) not in special_ids]
        raw = tokenizer.decode(ids, skip_special_tokens=False)
        thinking_count, response_count = _split_think_tokens(ids)
        cleaned, think_content = strip_think_tags(raw)
        result = GenerationResult(
            raw_text=raw,
            thinking_content=think_content,
            response_content=cleaned,
            thinking_tokens=thinking_count,
            response_tokens=response_count,
        )
        completion_stats = _perplexity_stats(step_log_probs)
        response_stats = _perplexity_stats(step_log_probs[thinking_count:])
        result.completion_nll = completion_stats["nll"]
        result.completion_perplexity = completion_stats["perplexity"]
        result.completion_scored_tokens = completion_stats["scored_tokens"]
        result.response_nll = response_stats["nll"]
        result.response_perplexity = response_stats["perplexity"]
        result.response_scored_tokens = response_stats["scored_tokens"]
        results.append(result)
    return results


def _result_to_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "raw_response": result.raw_text,
        "cleaned_response": result.response_content,
        "think_content": result.thinking_content,
        "thinking_tokens": result.thinking_tokens,
        "response_tokens": result.response_tokens,
        "total_tokens": result.thinking_tokens + result.response_tokens,
        "completion_nll": getattr(result, "completion_nll", None),
        "completion_perplexity": getattr(result, "completion_perplexity", None),
        "completion_scored_tokens": getattr(result, "completion_scored_tokens", 0),
        "response_nll": getattr(result, "response_nll", None),
        "response_perplexity": getattr(result, "response_perplexity", None),
        "response_scored_tokens": getattr(result, "response_scored_tokens", 0),
    }


def _make_sample(
    *,
    record: dict[str, Any],
    rendered: Any,
    benchmark_id: str,
    model_id: str,
    seed: int,
    result: GenerationResult,
    steering_dimension: str,
    vector_name: str,
    coefficient: float,
    layer: int,
    steering_position: str,
    steering_mode: str,
) -> dict[str, Any]:
    condition = CONDITIONS_BY_ID["baseline"]
    system_prompt = build_system_prompt(condition.get("level_description"))
    sample = {
        "condition_id": "baseline",
        "dimension": "baseline",
        "category": None,
        "level": "baseline",
        "level_label": condition.get("level_label"),
        "ordinal": condition.get("ordinal"),
        "benchmark_id": benchmark_id,
        "subtask_id": record.get("subtask_id"),
        "prompt_id": prompt_id(record),
        "prompt": record.get("prompt", ""),
        "messages": rendered.messages,
        "system_prompt": system_prompt,
        "record": record,
        "model_id": model_id,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steering_dimension": steering_dimension,
        "vector_name": vector_name,
        "coefficient": coefficient,
        "steering_layer": layer,
        "steering_position": steering_position,
        "steering_mode": steering_mode,
    }
    sample.update(_result_to_dict(result))
    return sample


@torch.no_grad()
def generate_steered_samples(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_id: str,
    seed: int,
    benchmark_id: str,
    steering_dimension: str,
    vector_name: str,
    vector: torch.Tensor,
    coefficient: float,
    layer: int,
    steering_position: str,
    steering_mode: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    if steering_mode != "response_only":
        raise ValueError(f"Unsupported steering mode: {steering_mode}")

    condition = CONDITIONS_BY_ID["baseline"]
    samples: list[dict[str, Any]] = []
    gen_kwargs = _generation_kwargs(benchmark_id)
    max_new_tokens = BENCHMARK_MAX_TOKENS[benchmark_id]

    for record_batch in _batches(records, batch_size):
        rendered_batch = [
            render_with_positions(tokenizer=tokenizer, record=record, condition=condition)
            for record in record_batch
        ]
        with ResponseOnlySteerer(
            model,
            layer=layer,
            vector=vector,
            coefficient=coefficient,
        ):
            results = _generate_batch(
                model=model,
                tokenizer=tokenizer,
                rendered_texts=[rendered.text for rendered in rendered_batch],
                gen_kwargs=gen_kwargs,
                max_new_tokens=max_new_tokens,
            )
        for record, rendered, result in zip(record_batch, rendered_batch, results):
            samples.append(
                _make_sample(
                    record=record,
                    rendered=rendered,
                    benchmark_id=benchmark_id,
                    model_id=model_id,
                    seed=seed,
                    result=result,
                    steering_dimension=steering_dimension,
                    vector_name=vector_name,
                    coefficient=coefficient,
                    layer=layer,
                    steering_position=steering_position,
                    steering_mode=steering_mode,
                )
            )
    return samples


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
    for key, values in _numeric_values(eval_rows).items():
        aggregate[f"{key}_mean"] = statistics.mean(values)
        aggregate[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return aggregate


def _evaluate_samples(
    *,
    samples_path: Path,
    eval_path: Path,
    aggregate_path: Path,
    benchmark_id: str,
    judge: Any,
    limit: int | None,
    force: bool,
) -> None:
    if eval_path.exists() and aggregate_path.exists() and not force:
        print(f"[SKIP] {eval_path} exists; use --force-eval to recompute", flush=True)
        return

    metric_module = BENCHMARK_METRIC_MODULE[benchmark_id]
    module = importlib.import_module(metric_module)
    samples = _load_jsonl(samples_path, limit=limit)
    metric_rows = module.evaluate_samples(samples, judge=judge)

    rows: list[dict[str, Any]] = []
    for sample, metrics in zip(samples, metric_rows):
        metrics = dict(metrics)
        for key in (
            "completion_nll",
            "completion_perplexity",
            "completion_scored_tokens",
            "response_nll",
            "response_perplexity",
            "response_scored_tokens",
        ):
            if key in sample:
                metrics[key] = sample.get(key)
        rows.append(
            {
                "condition_id": sample.get("condition_id"),
                "dimension": sample.get("dimension"),
                "category": sample.get("category"),
                "level": sample.get("level"),
                "benchmark_id": sample.get("benchmark_id"),
                "subtask_id": sample.get("subtask_id"),
                "prompt_id": sample.get("prompt_id"),
                "model_id": sample.get("model_id"),
                "seed": sample.get("seed"),
                "steering_dimension": sample.get("steering_dimension"),
                "vector_name": sample.get("vector_name"),
                "coefficient": sample.get("coefficient"),
                "steering_layer": sample.get("steering_layer"),
                "steering_position": sample.get("steering_position"),
                "steering_mode": sample.get("steering_mode"),
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    _write_jsonl(eval_path, rows)
    first = rows[0] if rows else {}
    aggregate = {
        "condition_id": "baseline",
        "dimension": first.get("dimension"),
        "category": first.get("category"),
        "level": first.get("level"),
        "benchmark_id": benchmark_id,
        "model_id": first.get("model_id"),
        "seed": first.get("seed"),
        "steering_dimension": first.get("steering_dimension"),
        "vector_name": first.get("vector_name"),
        "coefficient": first.get("coefficient"),
        "steering_layer": first.get("steering_layer"),
        "steering_position": first.get("steering_position"),
        "steering_mode": first.get("steering_mode"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": _aggregate(rows),
    }
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[DONE] evaluated {eval_path}", flush=True)


def _write_summary(model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    lock_path = model_root / "summary.lock"
    with open(lock_path, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows: list[dict[str, Any]] = []
        for agg_path in sorted(model_root.glob("seed_*/aggregates/*/*/coef_*/*/aggregate.json")):
            data = json.loads(agg_path.read_text(encoding="utf-8"))
            base = {
                "model_id": data.get("model_id", ""),
                "seed": data.get("seed", ""),
                "benchmark_id": data.get("benchmark_id", ""),
                "steering_dimension": data.get("steering_dimension", ""),
                "vector_name": data.get("vector_name", ""),
                "coefficient": data.get("coefficient", ""),
                "steering_layer": data.get("steering_layer", ""),
                "steering_position": data.get("steering_position", ""),
                "steering_mode": data.get("steering_mode", ""),
            }
            for metric, value in data.get("metrics", {}).items():
                rows.append({**base, "metric": metric, "value": value})

        if not rows:
            return
        path = model_root / "summary.csv"
        tmp_path = model_root / f".summary.{os.getpid()}.tmp"
        fieldnames = [
            "model_id",
            "seed",
            "benchmark_id",
            "steering_dimension",
            "vector_name",
            "coefficient",
            "steering_layer",
            "steering_position",
            "steering_mode",
            "metric",
            "value",
        ]
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_path.replace(path)


def _combo_paths(
    *,
    seed_root: Path,
    dimension: str,
    vector_name: str,
    coefficient: float,
    benchmark_id: str,
) -> tuple[Path, Path, Path]:
    coef_dir = f"coef_{_coef_label(coefficient)}"
    samples_path = (
        seed_root / "generations" / dimension / vector_name /
        coef_dir / benchmark_id / "samples.jsonl"
    )
    eval_path = (
        seed_root / "evaluations" / dimension / vector_name /
        coef_dir / benchmark_id / f"{benchmark_id}.jsonl"
    )
    aggregate_path = (
        seed_root / "aggregates" / dimension / vector_name /
        coef_dir / benchmark_id / "aggregate.json"
    )
    return samples_path, eval_path, aggregate_path


def run_steering(
    *,
    model_cfg: dict[str, Any],
    model_config_path: Path,
    vectors_dir: Path,
    output_dir: Path,
    benchmark_data_dir: Path | None,
    benchmark_id: str,
    dimension: str,
    vector_names: list[str] | None,
    coefficients: list[float],
    layer: int,
    steering_position: str,
    steering_mode: str,
    batch_size: int,
    limit: int | None,
    record_split: str,
    split_seed: int,
    seed: int,
    overwrite: bool,
    skip_generation: bool,
    skip_eval: bool,
    force_eval: bool,
    judge_config: JudgeConfig,
) -> None:
    if benchmark_id not in BENCHMARK_METRIC_MODULE:
        raise ValueError(f"No metric module configured for benchmark {benchmark_id!r}")
    vector_names = _resolve_vectors(dimension, vector_names)

    model_id = model_cfg["model_id"]
    model_root = output_dir / model_id
    seed_root = model_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_config": str(model_config_path),
        "model_id": model_id,
        "seed": seed,
        "limit": limit,
        "record_split": record_split,
        "split_seed": split_seed,
        "benchmark_id": benchmark_id,
        "steering_dimension": dimension,
        "vectors": vector_names,
        "coefficients": coefficients,
        "steering_layer": layer,
        "steering_position": steering_position,
        "steering_mode": steering_mode,
        "benchmark_paths": {
            **benchmark_paths_for_metadata(),
            **({benchmark_id: str(benchmark_data_dir)} if benchmark_data_dir else {}),
        },
        "started": datetime.now(timezone.utc).isoformat(),
    }
    metadata_dir = seed_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_name = (
        f"{dimension}_{'-'.join(vector_names)}_"
        f"{'-'.join(_coef_label(coef) for coef in coefficients)}.json"
    )
    (metadata_dir / metadata_name).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_records = (
        _load_records_from_override(benchmark_data_dir, limit=None)
        if benchmark_data_dir is not None
        else load_benchmark_records(benchmark_id, limit=None)
    )
    split_root = vectors_dir / model_id
    records = _select_records(
        records=all_records,
        record_split=record_split,
        split_root=split_root,
        split_seed=split_seed,
        limit=limit,
    )
    if not records:
        raise ValueError(
            f"No records loaded for benchmark {benchmark_id} with record_split={record_split}"
        )
    vectors = {
        vector_name: _load_vector(
            vectors_dir=vectors_dir,
            model_id=model_id,
            dimension=dimension,
            vector_name=vector_name,
            position=steering_position,
            layer=layer,
        )
        for vector_name in vector_names
    }

    if not skip_generation:
        try:
            from transformers import set_seed

            set_seed(seed)
        except ImportError:
            torch.manual_seed(seed)
        model_cfg = dict(model_cfg)
        model_cfg["seed"] = seed
        model, tokenizer = load_model_and_tokenizer(model_cfg)
        tokenizer.padding_side = "left"

        for vector_name in vector_names:
            for coefficient in coefficients:
                samples_path, _eval_path, _aggregate_path = _combo_paths(
                    seed_root=seed_root,
                    dimension=dimension,
                    vector_name=vector_name,
                    coefficient=coefficient,
                    benchmark_id=benchmark_id,
                )
                if samples_path.exists() and not overwrite:
                    print(f"[SKIP] {samples_path} exists; use --overwrite to regenerate", flush=True)
                    continue
                print(
                    f"[RUN] {vector_name} coef={_coef_label(coefficient)} "
                    f"{benchmark_id}: {len(records)} records",
                    flush=True,
                )
                samples = generate_steered_samples(
                    model=model,
                    tokenizer=tokenizer,
                    records=records,
                    model_id=model_id,
                    seed=seed,
                    benchmark_id=benchmark_id,
                    steering_dimension=dimension,
                    vector_name=vector_name,
                    vector=vectors[vector_name],
                    coefficient=coefficient,
                    layer=layer,
                    steering_position=steering_position,
                    steering_mode=steering_mode,
                    batch_size=batch_size,
                )
                _write_jsonl(samples_path, samples)
                print(f"[DONE] wrote {len(samples)} samples to {samples_path}", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not skip_eval:
        judge = build_judge(judge_config)
        for vector_name in vector_names:
            for coefficient in coefficients:
                samples_path, eval_path, aggregate_path = _combo_paths(
                    seed_root=seed_root,
                    dimension=dimension,
                    vector_name=vector_name,
                    coefficient=coefficient,
                    benchmark_id=benchmark_id,
                )
                if not samples_path.exists():
                    raise FileNotFoundError(f"Missing generation file for evaluation: {samples_path}")
                _evaluate_samples(
                    samples_path=samples_path,
                    eval_path=eval_path,
                    aggregate_path=aggregate_path,
                    benchmark_id=benchmark_id,
                    judge=judge,
                    limit=None,
                    force=force_eval,
                )
        _write_summary(model_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate response-only user-attribute vector steering.",
    )
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--vectors-dir", default="results/user_attr_vectors")
    parser.add_argument("--output-dir", default="results/steering")
    parser.add_argument(
        "--benchmark-data-dir",
        default=None,
        help="Optional JSONL file or directory overriding the configured benchmark data path.",
    )
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--dimension",
        default=DEFAULT_DIMENSION,
        choices=sorted(DIMENSION_VECTORS),
        help="Steering vector dimension to load.",
    )
    parser.add_argument(
        "--vectors",
        nargs="+",
        default=None,
        help="Vector names to run. Defaults to all vectors for --dimension.",
    )
    parser.add_argument(
        "--coefficients",
        nargs="+",
        type=float,
        default=list(DEFAULT_COEFFICIENTS),
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--steering-position", default="P4")
    parser.add_argument(
        "--steering-mode",
        default="response_only",
        choices=["response_only"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--record-split",
        default=DEFAULT_RECORD_SPLIT,
        choices=["all", "train", "heldout"],
        help="Benchmark records to run. train/heldout use the saved representation split.",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
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
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-api-base-url", default=None)
    parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-api-concurrency", type=int, default=32)
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--judge-dtype", default="bfloat16")
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--judge-max-model-len", type=int, default=16384)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_config_path = Path(args.model_config)
    if not model_config_path.is_absolute():
        model_config_path = repo_root / model_config_path
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    vectors_dir = Path(args.vectors_dir)
    if not vectors_dir.is_absolute():
        vectors_dir = repo_root / vectors_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    benchmark_data_dir = None
    if args.benchmark_data_dir is not None:
        benchmark_data_dir = Path(args.benchmark_data_dir)
        if not benchmark_data_dir.is_absolute():
            benchmark_data_dir = repo_root / benchmark_data_dir

    run_steering(
        model_cfg=model_cfg,
        model_config_path=model_config_path,
        vectors_dir=vectors_dir,
        output_dir=output_dir,
        benchmark_data_dir=benchmark_data_dir,
        benchmark_id=args.benchmark,
        dimension=args.dimension,
        vector_names=None if args.vectors is None else list(args.vectors),
        coefficients=list(args.coefficients),
        layer=args.layer,
        steering_position=args.steering_position,
        steering_mode=args.steering_mode,
        batch_size=args.batch_size,
        limit=args.limit,
        record_split=args.record_split,
        split_seed=args.split_seed,
        seed=args.seed,
        overwrite=args.overwrite,
        skip_generation=args.skip_generation,
        skip_eval=args.skip_eval,
        force_eval=args.force_eval,
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
            max_model_len=args.judge_max_model_len,
        ),
    )


if __name__ == "__main__":
    main()
