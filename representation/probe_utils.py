from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from dimensions.experiment_config import CONDITIONS_BY_ID
from representation.vector_specs import VECTOR_SPECS, ContrastSpec, specs_by_name

ASSISTANT_MARKER = "<|im_start|>assistant"
DEFAULT_COEFFICIENTS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
DEFAULT_SEEDS = (0, 42, 67, 69, 420)
DEFAULT_POSITION = "P4"
DEFAULT_LAYER = 20
REPO_ROOT = Path(__file__).resolve().parents[1]


def _input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _coef_label(coefficient: float) -> str:
    return format(float(coefficient), "g")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_vector(
    *,
    vectors_dir: Path,
    model_id: str,
    spec: ContrastSpec,
    position: str,
    layer: int,
) -> torch.Tensor:
    path = vectors_dir / model_id / spec.dimension / f"{spec.name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing vector artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if int(artifact.get("layer", layer)) != layer:
        raise ValueError(
            f"{path} was built at layer {artifact.get('layer')}, but --layer={layer}"
        )
    positions = artifact.get("positions", {})
    if position not in positions:
        raise KeyError(f"{path} has no position {position!r}; available={sorted(positions)}")
    vector = positions[position].float()
    if vector.ndim != 1:
        raise ValueError(f"{path}:{position} expected 1D vector, got {tuple(vector.shape)}")
    return vector


def _condition_trait_text(condition_id: str) -> dict[str, Any]:
    condition = CONDITIONS_BY_ID[condition_id]
    description = str(condition.get("level_description") or condition.get("level_label") or "")
    description = " ".join(description.strip().split()).rstrip(".")
    lower = description.lower()
    if lower.startswith("the user "):
        trait = "someone who " + description[len("The user "):]
    elif lower.startswith("user "):
        trait = "someone who " + description[len("User "):]
    else:
        trait = description[:1].lower() + description[1:]
    return {
        "condition_id": condition_id,
        "dimension": condition.get("dimension"),
        "category": condition.get("category"),
        "level": condition.get("level"),
        "level_label": condition.get("level_label"),
        "level_description": description,
        "trait_text": trait,
    }


def _assistant_span(tokenizer: Any, rendered: str) -> dict[str, int]:
    start_char = rendered.find(ASSISTANT_MARKER)
    if start_char < 0:
        raise ValueError(f"Rendered prompt does not contain {ASSISTANT_MARKER!r}: {rendered!r}")

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        prefix_ids = tokenizer(rendered[:start_char], add_special_tokens=False)["input_ids"]
        start_token = len(prefix_ids)
    else:
        candidates = [
            idx for idx, (start, end) in enumerate(offsets)
            if end > start and end > start_char
        ]
        if not candidates:
            raise ValueError("Could not resolve assistant token span")
        start_token = candidates[0]
    return {
        "assistant_start_char": start_char,
        "assistant_start_token": start_token,
        "assistant_end_token": len(input_ids) - 1,
        "n_prompt_tokens": len(input_ids),
    }


def _single_token_candidates(tokenizer: Any, variants: tuple[str, ...]) -> dict[int, str]:
    candidates: dict[int, str] = {}
    for text in variants:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            candidates[int(ids[0])] = text
    return candidates


class AssistantPrefillSteerer:
    """Add a steering vector to the assistant-side prompt prefix during prefill."""

    def __init__(
        self,
        model: Any,
        *,
        layer: int,
        vector: torch.Tensor,
        coefficient: float,
        start_token: int,
        end_token: int,
    ) -> None:
        self.model = model
        self.layer = layer
        self.vector = vector
        self.coefficient = float(coefficient)
        self.start_token = start_token
        self.end_token = end_token
        self._handle: Any = None

    def __enter__(self) -> AssistantPrefillSteerer:
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
        if hidden.ndim != 3:
            return output
        direction = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden.clone()
        steered[:, self.start_token:self.end_token + 1, :] += (
            self.coefficient * direction.view(1, 1, -1)
        )
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered


def _iter_specs(vector_names: list[str] | None) -> Iterator[ContrastSpec]:
    by_name = specs_by_name()
    if vector_names is None:
        yield from VECTOR_SPECS
        return
    for name in vector_names:
        if name not in by_name:
            raise KeyError(f"Unknown vector {name!r}; available={sorted(by_name)}")
        yield by_name[name]
