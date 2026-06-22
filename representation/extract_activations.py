from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from dimensions.experiment_config import CONDITIONS
from representation.data import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_FRACTION,
    load_benchmark_records,
    make_stratified_prompt_split,
    model_results_dir,
    prompt_id,
    prompt_stratum,
    write_split,
)
from representation.prompts import RenderedPrompt, render_with_positions
from representation.math_utils import cos_sim


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower(), torch.bfloat16)


def _batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


class ResidPostCapture:
    """Capture residual stream after a decoder block via a forward hook."""

    def __init__(self, model: Any, layer: int) -> None:
        self.model = model
        self.layer = layer
        self.hidden: torch.Tensor | None = None
        self._handle: Any = None

    def __enter__(self) -> "ResidPostCapture":
        layers = self.model.model.layers
        if not 0 <= self.layer < len(layers):
            raise IndexError(f"Layer {self.layer} out of range for {len(layers)} layers")
        self._handle = layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._handle.remove()

    def _hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self.hidden = hidden.detach()


def load_model_and_tokenizer(model_cfg: dict[str, Any]) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_torch_dtype(model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _row_token_ids(encoded: Any, row_idx: int) -> torch.Tensor:
    """Input ids for one row with right-padding stripped (real tokens only)."""
    ids = encoded["input_ids"][row_idx]
    if "attention_mask" in encoded:
        length = int(encoded["attention_mask"][row_idx].sum().item())
        return ids[:length].detach().cpu()
    return ids.detach().cpu()


def _assert_batch_row_matches_rendered(
    encoded: Any,
    row_idx: int,
    rendered: RenderedPrompt,
    *,
    condition_id: str,
    prompt_id: str,
) -> None:
    """Ensure forward-pass tokenization matches what position math used.

    Batched tokenization can differ from a standalone ``tokenizer(text)`` call in
    edge cases; all P1–P4 indices must refer to the same rows the model saw.
    """
    row_ids = _row_token_ids(encoded, row_idx)
    expected = torch.tensor(rendered.token_ids, dtype=row_ids.dtype)
    if row_ids.shape != expected.shape or not torch.equal(row_ids, expected):
        raise ValueError(
            f"Tokenization mismatch for condition={condition_id} prompt={prompt_id}: "
            f"batched forward has {row_ids.numel()} tokens, standalone render has "
            f"{expected.numel()}. Indices P1-P4 must be recomputed against the batched IDs."
        )


def _static_positions_match(
    ref: torch.Tensor,
    candidate: torch.Tensor,
    *,
    cos_threshold: float = 0.99,
    atol_fallback: float = 0.02,
) -> bool:
    """True if two activations are the same causal-prefix state within GPU/BF16 noise."""
    if torch.allclose(ref, candidate, atol=atol_fallback, rtol=1e-3):
        return True
    return bool(cos_sim(ref, candidate) >= cos_threshold)


def _minimal_prompt_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id(record),
        "stratum": prompt_stratum(record),
        "subtask_id": record.get("subtask_id"),
        "split": record.get("split"),
        "expected_behavior": record.get("expected_behavior"),
        "source": record.get("source"),
        "source_file": record.get("_source_file"),
    }


@torch.no_grad()
def extract_condition_activations(
    *,
    model: Any,
    tokenizer: Any,
    condition: dict[str, Any],
    records: list[dict[str, Any]],
    layer: int,
    batch_size: int,
) -> dict[str, Any]:
    """Extract P1-P4 activations for one condition over all refusal prompts."""
    prompt_ids: list[str] = []
    prompt_metadata: list[dict[str, Any]] = []
    position_indices: list[dict[str, int | None]] = []
    static_vectors: dict[str, torch.Tensor | None] = {"P1": None, "P2": None}
    per_prompt_vectors: dict[str, list[torch.Tensor]] = {"P3": [], "P4": []}

    for record_batch in _batches(records, batch_size):
        rendered_batch = [
            render_with_positions(tokenizer=tokenizer, record=record, condition=condition)
            for record in record_batch
        ]
        encoded = tokenizer(
            [rendered.text for rendered in rendered_batch],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(_input_device(model))

        with ResidPostCapture(model, layer) as capture:
            model(**encoded, use_cache=False)
        if capture.hidden is None:
            raise RuntimeError("Forward hook did not capture hidden states")
        hidden = capture.hidden.float().cpu()

        for row_idx, (record, rendered) in enumerate(zip(record_batch, rendered_batch)):
            pid = prompt_id(record)
            prompt_ids.append(pid)
            prompt_metadata.append(_minimal_prompt_metadata(record))
            position_indices.append(rendered.positions)

            _assert_batch_row_matches_rendered(
                encoded,
                row_idx,
                rendered,
                condition_id=condition["id"],
                prompt_id=pid,
            )
            seq_len = int(encoded["attention_mask"][row_idx].sum().item())

            for pos in ("P1", "P2"):
                token_idx = rendered.positions[pos]
                if token_idx is None:
                    continue
                if token_idx >= seq_len:
                    raise ValueError(
                        f"{pos}: token_idx={token_idx} >= seq_len={seq_len} for "
                        f"condition={condition['id']} prompt={pid}"
                    )
                vector = hidden[row_idx, token_idx].clone()
                if static_vectors[pos] is None:
                    static_vectors[pos] = vector
                elif not _static_positions_match(static_vectors[pos], vector):
                    raise ValueError(
                        f"{pos} was expected to be deterministic for {condition['id']}, "
                        f"but prompt {pid} produced a different activation. "
                        f"(If tokenizer alignment passes, tighten deterministic kernels or "
                        f"use batch-size 1; cos_sim="
                        f"{float(cos_sim(static_vectors[pos], vector)):.6f})"
                    )

            for pos in ("P3", "P4"):
                token_idx = rendered.positions[pos]
                if token_idx is None:
                    raise ValueError(f"{pos} missing for prompt {pid}")
                if token_idx >= seq_len:
                    raise ValueError(
                        f"{pos}: token_idx={token_idx} >= seq_len={seq_len} for "
                        f"condition={condition['id']} prompt={pid}"
                    )
                per_prompt_vectors[pos].append(hidden[row_idx, token_idx].clone())

    return {
        "condition_id": condition["id"],
        "dimension": condition["dimension"] or "baseline",
        "category": condition.get("category"),
        "level": condition.get("level"),
        "layer": layer,
        "activation_type": "resid_post",
        "prompt_ids": prompt_ids,
        "prompt_metadata": prompt_metadata,
        "position_indices": position_indices,
        "positions": {
            "P1": static_vectors["P1"],
            "P2": static_vectors["P2"],
            "P3": torch.stack(per_prompt_vectors["P3"], dim=0),
            "P4": torch.stack(per_prompt_vectors["P4"], dim=0),
        },
    }


def extract_all_activations(
    *,
    model_cfg: dict[str, Any],
    output_dir: Path,
    layer: int = 20,
    batch_size: int = 4,
    limit: int | None = None,
    conditions_filter: list[str] | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    overwrite: bool = False,
) -> Path:
    model_id = model_cfg["model_id"]
    model_root = model_results_dir(output_dir, model_id)
    activation_dir = model_root / "activations" / f"layer_{layer}"
    activation_dir.mkdir(parents=True, exist_ok=True)

    records = load_benchmark_records(limit=limit)
    split = make_stratified_prompt_split(
        records,
        train_fraction=train_fraction,
        seed=split_seed,
    )
    split_path = write_split(model_root, split)

    metadata = {
        "model_id": model_id,
        "layer": layer,
        "activation_type": "resid_post",
        "benchmark_id": split["benchmark_id"],
        "split_path": str(split_path),
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "limit": limit,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected_conditions = [
        condition for condition in CONDITIONS
        if conditions_filter is None or condition["id"] in conditions_filter
    ]
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    for condition in selected_conditions:
        out_path = activation_dir / f"{condition['id']}.pt"
        if out_path.exists() and not overwrite:
            print(f"[SKIP] {condition['id']}: {out_path} exists", flush=True)
            continue
        print(f"[EXTRACT] {condition['id']}: {len(records)} prompts", flush=True)
        payload = extract_condition_activations(
            model=model,
            tokenizer=tokenizer,
            condition=condition,
            records=records,
            layer=layer,
            batch_size=batch_size,
        )
        torch.save(payload, out_path)
        print(f"[DONE] wrote {out_path}", flush=True)
    return activation_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract P1-P4 user-attribute activations.")
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--output-dir", default="results/user_attr_vectors")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    extract_all_activations(
        model_cfg=model_cfg,
        output_dir=Path(args.output_dir),
        layer=args.layer,
        batch_size=args.batch_size,
        limit=args.limit,
        conditions_filter=args.conditions,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
