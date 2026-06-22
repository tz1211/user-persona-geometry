from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from representation.data import DEFAULT_SPLIT_SEED, load_split, model_results_dir
from representation.vector_specs import POSITIONS, VECTOR_SPECS, ContrastSpec


def _load_activation(model_root: Path, layer: int, condition_id: str) -> dict[str, Any]:
    path = model_root / "activations" / f"layer_{layer}" / f"{condition_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing activation cache: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _prompt_index(payload: dict[str, Any]) -> dict[str, int]:
    return {pid: idx for idx, pid in enumerate(payload["prompt_ids"])}


def _condition_position_mean(
    payload: dict[str, Any],
    position: str,
    train_prompt_ids: set[str],
) -> torch.Tensor:
    value = payload["positions"][position]
    if value is None:
        raise ValueError(f"{payload['condition_id']} has no activation for {position}")
    if position in ("P1", "P2"):
        return value.float()

    index = _prompt_index(payload)
    row_indices = [index[pid] for pid in sorted(train_prompt_ids) if pid in index]
    if not row_indices:
        raise ValueError(
            f"No train prompts available for {payload['condition_id']} at {position}"
        )
    return value[row_indices].float().mean(dim=0)


def _mean_for_conditions(
    activations: dict[str, dict[str, Any]],
    condition_ids: tuple[str, ...],
    position: str,
    train_prompt_ids: set[str],
) -> torch.Tensor:
    means = [
        _condition_position_mean(activations[condition_id], position, train_prompt_ids)
        for condition_id in condition_ids
    ]
    return torch.stack(means, dim=0).mean(dim=0)


def _contrast_vector(
    activations: dict[str, dict[str, Any]],
    spec: ContrastSpec,
    position: str,
    train_prompt_ids: set[str],
) -> torch.Tensor:
    pos = _mean_for_conditions(
        activations,
        spec.positive_conditions,
        position,
        train_prompt_ids,
    )
    neg = _mean_for_conditions(
        activations,
        spec.negative_conditions,
        position,
        train_prompt_ids,
    )
    return pos - neg


def _required_condition_ids() -> set[str]:
    ids: set[str] = set()
    for spec in VECTOR_SPECS:
        ids.update(spec.positive_conditions)
        ids.update(spec.negative_conditions)
    return ids


def build_vectors(
    *,
    model_id: str,
    output_dir: Path,
    layer: int = 20,
    split_seed: int = DEFAULT_SPLIT_SEED,
    overwrite: bool = False,
) -> list[Path]:
    model_root = model_results_dir(output_dir, model_id)
    split = load_split(model_root, split_seed=split_seed)
    train_prompt_ids = set(split["train_prompt_ids"])
    activations = {
        condition_id: _load_activation(model_root, layer, condition_id)
        for condition_id in sorted(_required_condition_ids())
    }

    written: list[Path] = []

    for spec in VECTOR_SPECS:
        vectors_by_position = {
            position: _contrast_vector(activations, spec, position, train_prompt_ids)
            for position in POSITIONS
        }
        written.append(_write_vector_file(
            model_root=model_root,
            model_id=model_id,
            layer=layer,
            split=split,
            spec=spec,
            vectors_by_position=vectors_by_position,
            overwrite=overwrite,
        ))

    return written


def _write_vector_file(
    *,
    model_root: Path,
    model_id: str,
    layer: int,
    split: dict[str, Any],
    spec: ContrastSpec,
    vectors_by_position: dict[str, torch.Tensor],
    overwrite: bool,
) -> Path:
    out_dir = model_root / spec.dimension
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.pt"
    if path.exists() and not overwrite:
        print(f"[SKIP] {path} exists", flush=True)
        return path

    payload = {
        "vector_name": spec.name,
        "dimension": spec.dimension,
        "vector_type": spec.vector_type,
        "layer": layer,
        "activation_type": "resid_post",
        "positions": vectors_by_position,
        "formula": spec.formula,
        "source_conditions": {
            "positive": list(spec.positive_conditions),
            "negative": list(spec.negative_conditions),
        },
        "split": {
            "benchmark_id": split["benchmark_id"],
            "seed": split["seed"],
            "train_fraction": split["train_fraction"],
            "n_train_prompts": len(split["train_prompt_ids"]),
            "n_heldout_prompts": len(split["heldout_prompt_ids"]),
        },
        "model_id": model_id,
    }
    torch.save(payload, path)
    print(f"[VECTOR] wrote {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Section 6.2 user-attribute vectors.")
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--output-dir", default="results/user_attr_vectors")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    build_vectors(
        model_id=model_cfg["model_id"],
        output_dir=Path(args.output_dir),
        layer=args.layer,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
