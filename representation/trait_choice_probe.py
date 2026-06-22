"""Forced-choice steering probe for user-attribute vectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from representation.extract_activations import load_model_and_tokenizer
from representation.steering import ResponseOnlySteerer
from representation.probe_utils import (
    ASSISTANT_MARKER,
    DEFAULT_COEFFICIENTS,
    DEFAULT_LAYER,
    DEFAULT_POSITION,
    DEFAULT_SEEDS,
    REPO_ROOT,
    AssistantPrefillSteerer,
    _assistant_span,
    _coef_label,
    _condition_trait_text,
    _input_device,
    _iter_specs,
    _load_jsonl,
    _load_vector,
    _single_token_candidates,
    _write_jsonl,
)
from representation.vector_specs import ContrastSpec

PROMPT_MODE = "completion_sentence"
STEERING_MODE = "completion_final_token"
CHOICE_FILENAME = "choice_probe.jsonl"
STEERING_TARGET = "decode"


def _choice_candidates(tokenizer: Any) -> tuple[dict[int, str], dict[int, str]]:
    a_tokens = _single_token_candidates(
        tokenizer,
        ("A", " A", "A.", " A.", "\nA", "\nA.", "a", " a"),
    )
    b_tokens = _single_token_candidates(
        tokenizer,
        ("B", " B", "B.", " B.", "\nB", "\nB.", "b", " b"),
    )
    if not a_tokens or not b_tokens:
        raise ValueError(
            f"Could not build single-token A/B candidates: A={a_tokens}, B={b_tokens}"
        )
    overlap = set(a_tokens) & set(b_tokens)
    if overlap:
        raise ValueError(f"A/B candidate token overlap: {overlap}")
    return a_tokens, b_tokens


def _render_choice_prompt(
    tokenizer: Any,
    *,
    option_a: str,
    option_b: str,
    prompt_mode: str,
) -> tuple[str, str]:
    if prompt_mode == "completion_sentence":
        prompt = (
            "Which best describes the user?\n\n"
            f"A. {option_a}\n"
            f"B. {option_b}\n\n"
            "The user is better described by option"
        )
        rendered = prompt
    elif prompt_mode == "chat_ab":
        prompt = (
            "Which best describes the user?\n\n"
            f"A. {option_a}\n"
            f"B. {option_b}\n\n"
            "Answer A or B only."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            raise TypeError(
                "tokenizer.apply_chat_template does not accept enable_thinking=False; "
                "use a Qwen3 tokenizer or update transformers/tokenizer files."
            ) from exc
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return prompt, rendered


def _score_token_span(
    tokenizer: Any,
    rendered: str,
    *,
    prompt_mode: str,
) -> dict[str, int | None]:
    if prompt_mode == "chat_ab":
        span = _assistant_span(tokenizer, rendered)
        span["score_token"] = span["assistant_end_token"]
        return span

    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    n_prompt_tokens = int(encoded["input_ids"].shape[-1])
    if n_prompt_tokens <= 0:
        raise ValueError("Rendered prompt produced no tokens")
    return {
        "assistant_start_char": None,
        "assistant_start_token": None,
        "assistant_end_token": None,
        "score_token": n_prompt_tokens - 1,
        "n_prompt_tokens": n_prompt_tokens,
    }


def _choice_prompt_rows(spec: ContrastSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for neg_id in spec.negative_conditions:
        for pos_id in spec.positive_conditions:
            neg = _condition_trait_text(neg_id)
            pos = _condition_trait_text(pos_id)
            neg["source_polarity"] = "negative"
            pos["source_polarity"] = "positive"
            pair_id = f"{neg_id}__vs__{pos_id}"
            rows.append({
                "pair_id": pair_id,
                "order": "l1_a_l2_b",
                "option_a": neg,
                "option_b": pos,
                "positive_option": "B",
            })
    return rows


@torch.no_grad()
def _score_choice_prompt(
    *,
    model: Any,
    tokenizer: Any,
    rendered: str,
    span: dict[str, int | None],
    vector: torch.Tensor,
    coefficient: float,
    layer: int,
    a_tokens: dict[int, str],
    b_tokens: dict[int, str],
    positive_option: str,
    steering_target: str,
) -> dict[str, Any]:
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(_input_device(model))
    final_token = span["score_token"]
    if steering_target == "prefill":
        if final_token is None:
            raise ValueError(f"Missing score token for span: {span}")
        with AssistantPrefillSteerer(
            model,
            layer=layer,
            vector=vector,
            coefficient=coefficient,
            start_token=final_token,
            end_token=final_token,
        ):
            logits = model(**encoded, use_cache=False).logits[0, -1].float()
    elif steering_target == "decode":
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if input_ids.shape[1] < 1:
            raise ValueError("Prompt tokenization produced empty input_ids")
        if input_ids.shape[1] == 1:
            prefix_ids = input_ids
            step_ids = input_ids
            prefix_mask = attention_mask
        else:
            prefix_ids = input_ids[:, :-1]
            step_ids = input_ids[:, -1:]
            prefix_mask = attention_mask[:, :-1] if attention_mask is not None else None
        prefill = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
            return_dict=True,
        )
        with ResponseOnlySteerer(
            model,
            layer=layer,
            vector=vector,
            coefficient=coefficient,
        ):
            step = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                past_key_values=prefill.past_key_values,
                use_cache=False,
                return_dict=True,
            )
        logits = step.logits[0, -1].float()
    else:
        raise ValueError(f"Unknown steering target: {steering_target}")
    probs = torch.softmax(logits, dim=-1)
    a_mass = probs[list(a_tokens)].sum()
    b_mass = probs[list(b_tokens)].sum()
    denom = (a_mass + b_mass).clamp_min(1e-30)
    positive_mass = b_mass if positive_option == "B" else a_mass
    return {
        "a_mass": float(a_mass.detach().cpu().item()),
        "b_mass": float(b_mass.detach().cpu().item()),
        "p_positive": float((positive_mass / denom).detach().cpu().item()),
    }


def _row_path(
    *,
    seed_root: Path,
    dimension: str,
    vector_name: str,
    coefficient: float,
) -> Path:
    return (
        seed_root / "logits" / dimension / vector_name /
        f"coef_{_coef_label(coefficient)}" / CHOICE_FILENAME
    )


def _aggregate_rows(model_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(model_root.glob(f"seed_*/logits/*/*/coef_*/{CHOICE_FILENAME}")):
        rows.extend(_load_jsonl(path))
    return rows


def _write_aggregate_csv(model_root: Path) -> Path | None:
    rows = _aggregate_rows(model_root)
    if not rows:
        return None
    out_path = model_root / "aggregates" / "choice_probe_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "seed",
        "dimension",
        "vector_name",
        "coefficient",
        "order",
        "pair_id",
        "positive_option",
        "negative_condition_id",
        "positive_condition_id",
        "negative_level_label",
        "positive_level_label",
        "negative_trait_text",
        "positive_trait_text",
        "p_positive",
        "a_mass",
        "b_mass",
        "prompt_mode",
        "steering_layer",
        "steering_position",
        "steering_mode",
        "score_token",
        "assistant_start_token",
        "assistant_end_token",
        "n_prompt_tokens",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[AGG] wrote {out_path}", flush=True)
    return out_path


def _print_token_check(
    *,
    tokenizer: Any,
    a_tokens: dict[int, str],
    b_tokens: dict[int, str],
    specs: list[ContrastSpec],
) -> None:
    print("Single-token A candidates:", flush=True)
    for token_id, text in sorted(a_tokens.items()):
        print(f"  {token_id}: {text!r} -> {tokenizer.decode([token_id])!r}", flush=True)
    print("Single-token B candidates:", flush=True)
    for token_id, text in sorted(b_tokens.items()):
        print(f"  {token_id}: {text!r} -> {tokenizer.decode([token_id])!r}", flush=True)
    if specs:
        row = _choice_prompt_rows(specs[0])[0]
        prompt, rendered = _render_choice_prompt(
            tokenizer,
            option_a=row["option_a"]["level_description"],
            option_b=row["option_b"]["level_description"],
            prompt_mode=PROMPT_MODE,
        )
        span = _score_token_span(tokenizer, rendered, prompt_mode=PROMPT_MODE)
        print("Example prompt:", flush=True)
        print(prompt, flush=True)
        print("Example rendered prompt:", flush=True)
        print(rendered, flush=True)
        print(f"Assistant marker present: {ASSISTANT_MARKER in rendered}", flush=True)
        print(f"Score token span: {span}", flush=True)


def run_probe(
    *,
    model_cfg: dict[str, Any],
    model_config_path: Path,
    vectors_dir: Path,
    output_dir: Path,
    vector_names: list[str] | None,
    coefficients: list[float],
    seeds: list[int],
    layer: int,
    position: str,
    overwrite: bool,
    dry_run_token_check: bool,
    prompt_mode: str,
    steering_target: str,
) -> None:
    model_id = model_cfg["model_id"]
    specs = list(_iter_specs(vector_names))
    model_root = output_dir / model_id
    model_root.mkdir(parents=True, exist_ok=True)

    if dry_run_token_check:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        a_tokens, b_tokens = _choice_candidates(tokenizer)
        _print_token_check(
            tokenizer=tokenizer,
            a_tokens=a_tokens,
            b_tokens=b_tokens,
            specs=specs,
        )
        return

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    tokenizer.padding_side = "right"
    a_tokens, b_tokens = _choice_candidates(tokenizer)

    metadata = {
        "model_config": str(model_config_path),
        "model_id": model_id,
        "vectors": [spec.name for spec in specs],
        "coefficients": coefficients,
        "seeds": seeds,
        "steering_layer": layer,
        "steering_position": position,
        "prompt_mode": prompt_mode,
        "steering_mode": (
            f"{STEERING_MODE}_{steering_target}"
            if prompt_mode == PROMPT_MODE
            else f"final_assistant_token_{steering_target}"
        ),
        "steering_target": steering_target,
        "a_token_ids": sorted(a_tokens),
        "b_token_ids": sorted(b_tokens),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    (model_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vectors = {
        spec.name: _load_vector(
            vectors_dir=vectors_dir,
            model_id=model_id,
            spec=spec,
            position=position,
            layer=layer,
        )
        for spec in specs
    }

    for seed in seeds:
        try:
            from transformers import set_seed

            set_seed(seed)
        except ImportError:
            torch.manual_seed(seed)
        seed_root = model_root / f"seed_{seed}"

        for spec in specs:
            prompt_rows = _choice_prompt_rows(spec)
            for coefficient in coefficients:
                out_path = _row_path(
                    seed_root=seed_root,
                    dimension=spec.dimension,
                    vector_name=spec.name,
                    coefficient=coefficient,
                )
                if out_path.exists() and not overwrite:
                    print(f"[SKIP] {out_path} exists; use --overwrite to recompute", flush=True)
                    continue

                rows: list[dict[str, Any]] = []
                for choice in prompt_rows:
                    negative = (
                        choice["option_a"]
                        if choice["option_a"]["source_polarity"] == "negative"
                        else choice["option_b"]
                    )
                    positive = (
                        choice["option_a"]
                        if choice["option_a"]["source_polarity"] == "positive"
                        else choice["option_b"]
                    )
                    prompt, rendered = _render_choice_prompt(
                        tokenizer,
                        option_a=choice["option_a"]["level_description"],
                        option_b=choice["option_b"]["level_description"],
                        prompt_mode=prompt_mode,
                    )
                    span = _score_token_span(tokenizer, rendered, prompt_mode=prompt_mode)
                    steering_mode = (
                        f"{STEERING_MODE}_{steering_target}"
                        if prompt_mode == PROMPT_MODE
                        else f"final_assistant_token_{steering_target}"
                    )
                    scores = _score_choice_prompt(
                        model=model,
                        tokenizer=tokenizer,
                        rendered=rendered,
                        span=span,
                        vector=vectors[spec.name],
                        coefficient=coefficient,
                        layer=layer,
                        a_tokens=a_tokens,
                        b_tokens=b_tokens,
                        positive_option=choice["positive_option"],
                        steering_target=steering_target,
                    )
                    rows.append({
                        "model_id": model_id,
                        "seed": seed,
                        "dimension": spec.dimension,
                        "vector_name": spec.name,
                        "coefficient": coefficient,
                        "order": choice["order"],
                        "pair_id": choice["pair_id"],
                        "positive_option": choice["positive_option"],
                        "option_a_condition_id": choice["option_a"]["condition_id"],
                        "option_b_condition_id": choice["option_b"]["condition_id"],
                        "option_a_text": choice["option_a"]["level_description"],
                        "option_b_text": choice["option_b"]["level_description"],
                        "negative_condition_id": negative["condition_id"],
                        "positive_condition_id": positive["condition_id"],
                        "negative_level_label": negative["level_label"],
                        "positive_level_label": positive["level_label"],
                        "negative_trait_text": negative["trait_text"],
                        "positive_trait_text": positive["trait_text"],
                        "prompt": prompt,
                        "rendered_prompt": rendered,
                        "a_token_ids": sorted(a_tokens),
                        "b_token_ids": sorted(b_tokens),
                        "a_token_strings": {str(k): v for k, v in a_tokens.items()},
                        "b_token_strings": {str(k): v for k, v in b_tokens.items()},
                        "prompt_mode": prompt_mode,
                        "steering_layer": layer,
                        "steering_position": position,
                        "steering_mode": steering_mode,
                        "steering_target": steering_target,
                        "assistant_start_char": span["assistant_start_char"],
                        "assistant_start_token": span["assistant_start_token"],
                        "assistant_end_token": span["assistant_end_token"],
                        "score_token": span["score_token"],
                        "n_prompt_tokens": span["n_prompt_tokens"],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **scores,
                    })

                _write_jsonl(out_path, rows)
                print(f"[DONE] wrote {out_path}", flush=True)

    _write_aggregate_csv(model_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-token A/B choice steering probe.")
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--vectors-dir", default="results/user_attr_vectors")
    parser.add_argument("--output-dir", default="results/trait_choice_probe")
    parser.add_argument("--vectors", nargs="+", default=None)
    parser.add_argument(
        "--coefficients",
        nargs="+",
        type=float,
        default=list(DEFAULT_COEFFICIENTS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--steering-position", default=DEFAULT_POSITION)
    parser.add_argument(
        "--prompt-mode",
        choices=["completion_sentence", "chat_ab"],
        default=PROMPT_MODE,
        help=(
            "completion_sentence uses a plain prompt ending with "
            "'The user is better described by option'; chat_ab keeps the old "
            "Qwen chat-template A/B answer prompt."
        ),
    )
    parser.add_argument(
        "--steering-target",
        choices=["prefill", "decode"],
        default=STEERING_TARGET,
        help=(
            "prefill steers at the chosen prompt token; decode steers the single-token "
            "decode step that produces A/B logits."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run-token-check", action="store_true")
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    if not model_config_path.is_absolute():
        model_config_path = REPO_ROOT / model_config_path
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    vectors_dir = Path(args.vectors_dir)
    if not vectors_dir.is_absolute():
        vectors_dir = REPO_ROOT / vectors_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    run_probe(
        model_cfg=model_cfg,
        model_config_path=model_config_path,
        vectors_dir=vectors_dir,
        output_dir=output_dir,
        vector_names=args.vectors,
        coefficients=list(args.coefficients),
        seeds=list(args.seeds),
        layer=args.layer,
        position=args.steering_position,
        overwrite=args.overwrite,
        dry_run_token_check=args.dry_run_token_check,
        prompt_mode=args.prompt_mode,
        steering_target=args.steering_target,
    )


if __name__ == "__main__":
    main()
