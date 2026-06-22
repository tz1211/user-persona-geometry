"""Section 6.3.2 — projection analysis.

Computes per-prompt projection scalars ``h · v_hat`` for every constructed
user-attribute vector at chosen ``(build_pos, eval_pos)``, joins with refusal
labels, and caches three tidy artifacts under
``results/user_attr_vectors/{model_id}/projection/<eval-pos>_onto_<build-pos>/``:

- ``projections_held_out.parquet`` — long-form per-prompt rows.
- ``condition_summary.parquet`` — per-condition aggregates (drives Plot B).
- ``binned_summary.parquet`` — projection-decile aggregates (drives Plot C).

Plus a ``projection_summary.json`` with sanity-check stats and CSV mirrors of
the parquets for quick inspection.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from dimensions.experiment_config import CONDITIONS_BY_ID
from representation.data import DEFAULT_SPLIT_SEED, load_split, model_results_dir
from representation.math_utils import normalize, scalar_projection
from representation.vector_specs import POSITIONS, VECTOR_SPECS, ContrastSpec

DEFAULT_BUILD_POS = "P4"
DEFAULT_EVAL_POS = "P4"
DEFAULT_N_BINS = 10
REFUSAL_BENCHMARK_ID = "refusal"
REFUSAL_STYLE_VALUES = ("not_refused", "soft", "partial", "hard")


# ── IO ────────────────────────────────────────────────────────────────────────

def _load_vector(model_root: Path, spec: ContrastSpec, build_pos: str) -> torch.Tensor:
    path = model_root / spec.dimension / f"{spec.name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing vector artifact: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    positions = payload.get("positions") or {}
    if build_pos not in positions:
        raise ValueError(f"{path} has no vector at {build_pos}")
    return positions[build_pos].float()


def _load_activation(model_root: Path, layer: int, condition_id: str) -> dict[str, Any]:
    path = model_root / "activations" / f"layer_{layer}" / f"{condition_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing activation cache: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_refusal_labels(
    behavioral_root: Path,
    model_id: str,
    condition_ids: Iterable[str],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Return a (condition_id, seed, prompt_id) -> metrics lookup."""
    model_dir = behavioral_root / model_id
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not model_dir.exists():
        raise FileNotFoundError(f"No behavioral results under {model_dir}")
    cond_set = set(condition_ids)
    for seed_dir in sorted(model_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        eval_dir = seed_dir / "evaluations"
        if not eval_dir.exists():
            continue
        for cond_dir in sorted(eval_dir.iterdir()):
            if cond_dir.name not in cond_set:
                continue
            jsonl = cond_dir / REFUSAL_BENCHMARK_ID / f"{REFUSAL_BENCHMARK_ID}.jsonl"
            if not jsonl.exists():
                continue
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = rec.get("prompt_id")
                    if not pid:
                        continue
                    out[(cond_dir.name, seed, pid)] = rec.get("metrics", {})
    return out


# ── Per-condition projection ─────────────────────────────────────────────────

def _condition_metadata(condition_id: str) -> dict[str, Any]:
    cond = CONDITIONS_BY_ID.get(condition_id, {})
    return {
        "cond_dimension": cond.get("dimension") or "baseline",
        "cond_category": cond.get("category") or "_none_",
        "cond_level": cond.get("level") or "baseline",
    }


def _project_condition(
    activation: dict[str, Any],
    v_hat: torch.Tensor,
    eval_pos: str,
    heldout_ids: set[str],
) -> tuple[list[str], list[str], np.ndarray]:
    """Return (prompt_ids, subtask_ids, projection_scalars) for held-out rows."""
    activations = activation["positions"][eval_pos]
    if activations is None:
        raise ValueError(f"{activation['condition_id']} has no activation at {eval_pos}")
    if eval_pos in ("P1", "P2"):
        raise ValueError(
            f"eval_pos={eval_pos!r} is degenerate (one point per condition); "
            "projection analysis requires P3 or P4."
        )
    prompt_ids: list[str] = list(activation["prompt_ids"])
    metas: list[dict[str, Any]] = activation.get("prompt_metadata") or [{}] * len(prompt_ids)
    rows = [i for i, pid in enumerate(prompt_ids) if pid in heldout_ids]
    if not rows:
        return [], [], np.empty(0, dtype=np.float32)

    h = activations[rows].float()
    proj = scalar_projection(h, v_hat).cpu().numpy().astype(np.float32)
    pids_out = [prompt_ids[i] for i in rows]
    subtasks = [str(metas[i].get("subtask_id") or metas[i].get("stratum") or "") for i in rows]
    return pids_out, subtasks, proj


def _required_condition_ids(specs: Iterable[ContrastSpec]) -> set[str]:
    ids: set[str] = set()
    for spec in specs:
        ids.update(_conditions_for_vector(spec))
    return ids


def _conditions_for_vector(spec: ContrastSpec) -> list[str]:
    """Conditions over which to compute projections for a given vector.

    Every vector projects all non-baseline conditions in its parent dimension
    plus baseline. This lets sub-category vectors test both own-pair separation
    and whether the same axis generalises to sibling sub-categories. Emotion
    planned contrasts likewise cover all four circumplex quadrants.
    """
    ids = [
        cid
        for cid, cond in CONDITIONS_BY_ID.items()
        if cond.get("dimension") == spec.dimension
    ]
    if "baseline" not in ids:
        ids.append("baseline")
    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


# ── Aggregation ───────────────────────────────────────────────────────────────

def _per_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["vector_name", "vector_kind", "dimension", "build_pos", "eval_pos",
            "condition_id", "cond_dimension", "cond_category", "cond_level", "seed"]
    for grp_keys, grp in df.groupby(keys, dropna=False):
        proj = grp["projection"].astype(float).to_numpy()
        n_prompts = int(grp["prompt_id"].nunique())
        row: dict[str, Any] = dict(zip(keys, grp_keys))
        row["n_prompts"] = n_prompts
        row["projection_mean"] = float(np.mean(proj)) if proj.size else float("nan")
        row["projection_std"] = float(np.std(proj, ddof=1)) if proj.size > 1 else 0.0
        row["projection_median"] = float(np.median(proj)) if proj.size else float("nan")
        for metric in ("llm_refused", "keyword_refused"):
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna().astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{metric}_n"] = int(vals.size)
        for metric in ("response_tokens", "thinking_tokens"):
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna().astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
        styles = grp["refusal_style"].fillna("not_refused")
        if len(styles):
            counts = styles.value_counts(normalize=True)
            for s in REFUSAL_STYLE_VALUES:
                row[f"style_{s}"] = float(counts.get(s, 0.0))
        else:
            for s in REFUSAL_STYLE_VALUES:
                row[f"style_{s}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _binned_summary(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["vector_name", "vector_kind", "dimension", "build_pos", "eval_pos"]
    for grp_keys, grp in df.groupby(keys, dropna=False):
        sub = grp.copy()
        proj = sub["projection"].to_numpy().astype(float)
        if proj.size == 0:
            continue
        # Quantile bins — use unique edges to avoid duplicate-edge errors.
        try:
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.unique(np.quantile(proj, quantiles))
            if edges.size < 2:
                edges = np.array([proj.min(), proj.max() + 1e-9])
            bin_idx = np.clip(np.digitize(proj, edges[1:-1], right=False), 0, edges.size - 2)
        except Exception:
            edges = np.array([proj.min(), proj.max() + 1e-9])
            bin_idx = np.zeros_like(proj, dtype=int)
        sub["bin"] = bin_idx
        for bin_id, bin_grp in sub.groupby("bin"):
            row: dict[str, Any] = dict(zip(keys, grp_keys))
            row["bin"] = int(bin_id)
            row["n_total"] = int(len(bin_grp))
            row["projection_min"] = float(edges[int(bin_id)])
            row["projection_max"] = float(edges[int(bin_id) + 1])
            row["projection_center"] = float(bin_grp["projection"].mean())
            for metric in ("llm_refused", "keyword_refused"):
                vals = pd.to_numeric(bin_grp[metric], errors="coerce").dropna().astype(float).to_numpy()
                row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
                row[f"{metric}_n"] = int(vals.size)
            for metric in ("response_tokens", "thinking_tokens"):
                vals = pd.to_numeric(bin_grp[metric], errors="coerce").dropna().astype(float).to_numpy()
                row[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            styles = bin_grp["refusal_style"].fillna("not_refused")
            counts = styles.value_counts(normalize=True)
            for s in REFUSAL_STYLE_VALUES:
                row[f"style_{s}"] = float(counts.get(s, 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


# ── Sanity summary ────────────────────────────────────────────────────────────

def _sanity_summary(
    df: pd.DataFrame,
    *,
    model_id: str,
    layer: int,
    build_pos: str,
    eval_pos: str,
    n_heldout: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model_id": model_id,
        "layer": layer,
        "build_pos": build_pos,
        "eval_pos": eval_pos,
        "n_heldout_prompts": n_heldout,
        "n_rows": int(len(df)),
        "vectors": {},
    }
    spec_by_name = {spec.name: spec for spec in VECTOR_SPECS}
    for vname, grp in df.groupby("vector_name"):
        info: dict[str, Any] = {
            "kind": spec_by_name[vname].vector_type,
            "n_rows": int(len(grp)),
            "n_conditions": int(grp["condition_id"].nunique()),
            "projection_mean_by_condition": {},
        }
        for cid, cgrp in grp.groupby("condition_id"):
            info["projection_mean_by_condition"][cid] = float(cgrp["projection"].mean())

        llm = pd.to_numeric(grp["llm_refused"], errors="coerce")
        mask = llm.notna()
        if mask.sum() >= 5 and grp.loc[mask, "projection"].std() > 0:
            rho, pval = stats.spearmanr(
                grp.loc[mask, "projection"], llm[mask].astype(float)
            )
            info["spearman_proj_vs_llm_refused"] = {
                "rho": float(rho),
                "p_value": float(pval),
                "n": int(mask.sum()),
            }
        summary["vectors"][vname] = info
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_projections(
    *,
    model_id: str,
    output_dir: Path,
    behavioral_root: Path,
    layer: int = 20,
    build_pos: str = DEFAULT_BUILD_POS,
    eval_pos: str = DEFAULT_EVAL_POS,
    split_seed: int = DEFAULT_SPLIT_SEED,
    n_bins: int = DEFAULT_N_BINS,
    overwrite: bool = False,
) -> Path:
    if build_pos not in POSITIONS:
        raise ValueError(f"build_pos must be one of {POSITIONS}; got {build_pos!r}")
    if eval_pos not in ("P3", "P4"):
        raise ValueError(f"eval_pos must be P3 or P4; got {eval_pos!r}")

    model_root = model_results_dir(output_dir, model_id)
    out_dir = model_root / "projection" / f"{eval_pos}_onto_{build_pos}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "projections_held_out.parquet"
    cond_path = out_dir / "condition_summary.parquet"
    bin_path = out_dir / "binned_summary.parquet"
    summary_path = out_dir / "projection_summary.json"
    if parquet_path.exists() and not overwrite:
        print(f"[SKIP] {parquet_path} exists (use --overwrite to regenerate)", flush=True)
        return out_dir

    split = load_split(model_root, split_seed=split_seed)
    heldout_ids = set(split["heldout_prompt_ids"])
    print(
        f"[PROJ] model={model_id} layer={layer} build={build_pos} eval={eval_pos} "
        f"n_heldout={len(heldout_ids)}",
        flush=True,
    )

    needed_conditions: set[str] = set()
    for spec in VECTOR_SPECS:
        needed_conditions.update(_conditions_for_vector(spec))

    # Load refusal labels for those conditions across all seeds.
    labels = _load_refusal_labels(behavioral_root, model_id, needed_conditions)
    seeds_seen = sorted({seed for (_, seed, _) in labels.keys()})
    print(f"[PROJ] loaded {len(labels)} label rows across seeds={seeds_seen}", flush=True)

    rows: list[dict[str, Any]] = []
    for spec in VECTOR_SPECS:
        v = _load_vector(model_root, spec, build_pos)
        v_hat = normalize(v)
        start_rows = len(rows)
        for cid in _conditions_for_vector(spec):
            activation = _load_activation(model_root, layer, cid)
            pids, subtasks, projs = _project_condition(
                activation, v_hat, eval_pos, heldout_ids
            )
            if not pids:
                continue
            cmeta = _condition_metadata(cid)
            for pid, subtask, proj in zip(pids, subtasks, projs):
                seed_metrics = [
                    (seed, labels[(cid, seed, pid)])
                    for seed in seeds_seen
                    if (cid, seed, pid) in labels
                ]
                if not seed_metrics:
                    continue
                for seed, m in seed_metrics:
                    rows.append({
                        "vector_name": spec.name,
                        "vector_kind": spec.vector_type,
                        "dimension": spec.dimension,
                        "build_pos": build_pos,
                        "eval_pos": eval_pos,
                        "prompt_id": pid,
                        "subtask_id": subtask,
                        "condition_id": cid,
                        **cmeta,
                        "seed": int(seed),
                        "projection": float(proj),
                        "llm_refused": _to_bool(m.get("llm_refused")),
                        "keyword_refused": _to_bool(m.get("keyword_refused")),
                        "refusal_style": (m.get("refusal_style") or "not_refused"),
                        "response_tokens": m.get("response_tokens"),
                        "thinking_tokens": m.get("thinking_tokens"),
                        "correct_refusal_behavior": _to_bool(m.get("correct_refusal_behavior")),
                    })
        print(f"[PROJ] {spec.name}: rows={len(rows) - start_rows}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No projection rows produced — check splits and label availability.")

    df["projection"] = df["projection"].astype(float)
    for col in ("llm_refused", "keyword_refused", "correct_refusal_behavior"):
        df[col] = df[col].astype("Float64")  # nullable float for downstream means

    df.to_parquet(parquet_path, index=False)
    df.to_csv(parquet_path.with_suffix(".csv"), index=False)
    print(f"[PROJ] wrote {parquet_path} ({len(df):,} rows)", flush=True)

    cond_df = _per_condition_summary(df)
    cond_df.to_parquet(cond_path, index=False)
    cond_df.to_csv(cond_path.with_suffix(".csv"), index=False)
    print(f"[PROJ] wrote {cond_path} ({len(cond_df):,} rows)", flush=True)

    bin_df = _binned_summary(df, n_bins=n_bins)
    bin_df.to_parquet(bin_path, index=False)
    bin_df.to_csv(bin_path.with_suffix(".csv"), index=False)
    print(f"[PROJ] wrote {bin_path} ({len(bin_df):,} rows)", flush=True)

    summary = _sanity_summary(
        df,
        model_id=model_id,
        layer=layer,
        build_pos=build_pos,
        eval_pos=eval_pos,
        n_heldout=len(heldout_ids),
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PROJ] wrote {summary_path}", flush=True)

    _print_sanity(summary)
    return out_dir


def _to_bool(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.lower() == "true")
    return None


def _print_sanity(summary: dict[str, Any]) -> None:
    print("\n[SANITY] Spearman ρ(projection, llm_refused) by vector:", flush=True)
    for vname, info in summary["vectors"].items():
        sp = info.get("spearman_proj_vs_llm_refused")
        if sp is not None:
            print(
                f"  {vname:>12s}  ρ={sp['rho']:+.3f}  p={sp['p_value']:.2e}  n={sp['n']}",
                flush=True,
            )
        else:
            print(f"  {vname:>12s}  (insufficient data)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Section 6.3.2 projection artifacts (Plots A/B/C inputs).",
    )
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--vectors-dir", default="results/user_attr_vectors")
    parser.add_argument("--behavioral-dir", default="results/behavioral")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--build-pos", default=DEFAULT_BUILD_POS, choices=list(POSITIONS))
    parser.add_argument("--eval-pos", default=DEFAULT_EVAL_POS, choices=["P3", "P4"])
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Reserved for forward-compat; projection is CPU-only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    compute_projections(
        model_id=model_cfg["model_id"],
        output_dir=Path(args.vectors_dir),
        behavioral_root=Path(args.behavioral_dir),
        layer=args.layer,
        build_pos=args.build_pos,
        eval_pos=args.eval_pos,
        split_seed=args.split_seed,
        n_bins=args.n_bins,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
