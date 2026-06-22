"""Visualise refusal behaviour across the four behavioral dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams["ps.useafm"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.unicode_minus"] = False

__all__ = [
    "DIMENSIONS",
    "DIM_LABELS",
    "LEVEL_ORDERS",
    "CATEGORY_ORDERS",
    "STYLE_ORDER",
    "STYLE_COLORS",
    "STYLE_LABELS",
    "PALETTE",
    "BAR_COLOR",
    "load_data",
    "agg_metric",
    "agg_style",
    "_is_multicat",
    "_level_order",
    "_cat_order",
    "_lbl",
    "_baseline",
    "_pct_fmt",
    "draw_line_panel",
    "draw_bar_panel",
    "draw_emotion_heatmap",
    "draw_emotion_style_heatmap",
    "draw_style_multicat",
    "draw_style_simple",
    "make_over_refusal_keyword_fig",
    "make_under_refusal_keyword_fig",
    "make_over_refusal_judge_fig",
    "make_under_refusal_judge_fig",
    "make_refusal_errors_keyword_fig",
    "make_refusal_errors_judge_fig",
]


# ── Constants ────────────────────────────────────────────────────────────────

DIMENSIONS = ["knowledge", "intent", "emotion", "belief"]
DIM_LABELS = {"belief": "Belief", "knowledge": "Knowledge", "intent": "Intent", "emotion": "Emotion"}
REQUIRED_SEEDS = [0, 42, 67, 69, 420]

# Explicit orderings for named-level and multi-category dimensions.
LEVEL_ORDERS: dict[str, list[str]] = {
    "emotion": ["high_positive", "low_positive", "high_negative", "low_negative"],
}
EMOTION_AXES: dict[str, tuple[str, str]] = {
    "high_positive": ("Positive", "High"),
    "low_positive": ("Positive", "Low"),
    "high_negative": ("Negative", "High"),
    "low_negative": ("Negative", "Low"),
}
EMOTION_VALENCE_ORDER = ["Positive", "Negative"]
EMOTION_AROUSAL_ORDER = ["Low", "High"]
CATEGORY_ORDERS: dict[str, list[str]] = {
    "knowledge": ["formal", "professional", "self_directed"],
    "intent": ["curiosity", "educational", "professional", "institutional"],
    "belief": ["empirical", "normative", "conspiratorial"],
}
# Human-readable sub-category labels (keys are CSV ``category`` slugs).
CATEGORY_DISPLAY_LABELS: dict[str, dict[str, str]] = {
    "knowledge": {
        "formal": "Educational",
        "professional": "Professional",
        "self_directed": "Autodidactic",
    },
}
LEVEL_LABELS: dict[str, dict[str, str]] = {
    "knowledge": {"l1": "Novice", "l2": "Expert"},
    "intent": {"l1": "Weak", "l2": "Strong"},
    "belief": {"l1": "Weak", "l2": "Strong"},
}

STYLE_ORDER = ["not_refused", "soft", "partial", "hard"]
STYLE_COLORS = {
    "not_refused": "#c7c7c7",
    "soft": "#ff7f0e",
    "partial": "#1f77b4",
    "hard": "#d62728",
}
STYLE_LABELS = {"not_refused": "Not refused", "soft": "Soft", "partial": "Partial", "hard": "Hard"}

PALETTE = sns.color_palette("colorblind")
BAR_COLOR = PALETTE[0]

FONT_TICK = 14
FONT_AXIS = 14
FONT_PANEL_TITLE = 16
FONT_SUPTITLE = 18
FONT_LEGEND = 10
FONT_HEATMAP_ANNOT = 12

# Refusal-style figure only (``make_style_fig``): readability vs. other refusal plots.
STYLE_FIG_FONT_TICK = 16
STYLE_FIG_FONT_AXIS = 16
STYLE_FIG_FONT_PANEL = 18
STYLE_FIG_FONT_SUPTITLE = 20
STYLE_FIG_EMOTION_ANNOT = 12
STYLE_FIG_LEGEND = 15
STYLE_FIG_2X2_FIGSIZE = (17.5, 13.5)
STYLE_FIG_2X2_HEIGHT_RATIOS = (1.05, 1.38)
STYLE_FIG_2X2_WIDTH_RATIOS = (1.32, 1.0)
STYLE_FIG_1X4_FIGSIZE = (19.0, 6.0)
STYLE_FIG_1X4_WIDTH_RATIOS = (1.0, 1.0, 1.55, 1.0)
STYLE_FIG_MULTICAT_XTICK_ROTATION = 35

EXPECTED_BEHAVIORS = {"should_refuse", "should_not_refuse"}
EXPECTED_BEHAVIOR_JOIN_KEYS = [
    "condition_id",
    "benchmark_id",
    "subtask_id",
    "prompt_id",
    "seed",
]


def _set_theme_like_trait_choice() -> None:
    """Match ``plot_trait_choice_probe`` seaborn theme while preserving font rc."""
    sns.set_theme(style="white", context="paper")


# ── Data loading ──────────────────────────────────────────────────────────────

def _expected_condition_ids() -> list[str]:
    condition_ids = ["baseline"]
    for dim in ("knowledge", "intent", "belief"):
        for cat in CATEGORY_ORDERS[dim]:
            condition_ids.extend(f"{dim}_{cat}_{level}" for level in ("l1", "l2"))
    condition_ids.extend(f"emotion_circumplex_{level}" for level in LEVEL_ORDERS["emotion"])
    return condition_ids


def _validate_required_files(
    model_path: Path,
    benchmark_id: str,
    required_seeds: list[int],
    condition_ids: list[str],
) -> None:
    missing: list[str] = []
    for seed in required_seeds:
        seed_dir = model_path / f"seed_{seed}"
        for condition_id in condition_ids:
            eval_file = (
                seed_dir
                / "evaluations"
                / condition_id
                / benchmark_id
                / f"{benchmark_id}.jsonl"
            )
            if not eval_file.exists():
                missing.append(str(eval_file))

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:80])
        extra = "" if len(missing) <= 80 else f"\n  ... and {len(missing) - 80} more"
        raise FileNotFoundError(
            "Missing required refusal evaluation files for the balanced "
            f"seed set {required_seeds}:\n{preview}{extra}"
        )


def _expected_behavior_from_sample(sample: dict) -> str | None:
    record = sample.get("record", {})
    value = (
        sample.get("expected_behavior")
        or sample.get("expected_behaviour")
        or record.get("expected_behavior")
        or record.get("expected_behaviour")
        or record.get("split")
    )
    if value in EXPECTED_BEHAVIORS:
        return value
    return None


def _join_key(row: dict) -> tuple:
    return tuple(row.get(key) for key in EXPECTED_BEHAVIOR_JOIN_KEYS)


def _load_expected_behavior_lookup(
    seed_dir: Path,
    benchmark_id: str,
    condition_ids: list[str],
) -> dict[tuple, str]:
    lookup: dict[tuple, str] = {}
    for condition_id in condition_ids:
        samples_file = seed_dir / "generations" / condition_id / benchmark_id / "samples.jsonl"
        if not samples_file.exists():
            continue
        with open(samples_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                expected_behavior = _expected_behavior_from_sample(sample)
                if expected_behavior is None:
                    continue
                lookup[_join_key(sample)] = expected_behavior
    return lookup


def _normalise_expected_behavior(df: pd.DataFrame) -> pd.DataFrame:
    if "expected_behavior" not in df.columns:
        df["expected_behavior"] = np.nan
    if "expected_behaviour" in df.columns:
        df["expected_behavior"] = df["expected_behavior"].fillna(df["expected_behaviour"])
    df["expected_behavior"] = df["expected_behavior"].where(
        df["expected_behavior"].isin(EXPECTED_BEHAVIORS),
        np.nan,
    )

    missing = df["expected_behavior"].isna()
    if missing.any():
        preview_cols = [
            col for col in EXPECTED_BEHAVIOR_JOIN_KEYS
            if col in df.columns
        ]
        preview = df.loc[missing, preview_cols].head(5).to_dict("records")
        raise ValueError(
            "Could not determine expected_behavior for "
            f"{int(missing.sum())} refusal rows. "
            "Re-run evaluation with the updated evaluator or make sure generation "
            f"samples.jsonl files exist beside evaluations. Example missing keys: {preview}"
        )

    observed = set(df["expected_behavior"].dropna().unique())
    if not EXPECTED_BEHAVIORS.issubset(observed):
        raise ValueError(
            "Refusal error plots require both should_refuse and should_not_refuse "
            f"rows; observed {sorted(observed)}."
        )
    return df


def _as_float_metric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _add_refusal_error_metrics(df: pd.DataFrame) -> pd.DataFrame:
    should_refuse = df["expected_behavior"] == "should_refuse"
    should_not_refuse = df["expected_behavior"] == "should_not_refuse"

    if "keyword_refused" in df.columns:
        keyword = _as_float_metric(df["keyword_refused"])
        df["keyword_over_refusal"] = keyword.where(should_not_refuse, np.nan)
        df["keyword_under_refusal"] = (1.0 - keyword).where(should_refuse, np.nan)

    if "llm_refused" in df.columns:
        llm = _as_float_metric(df["llm_refused"])
        df["llm_over_refusal"] = llm.where(should_not_refuse, np.nan)
        df["llm_under_refusal"] = (1.0 - llm).where(should_refuse, np.nan)

    return df


def load_data(
    results_dir: str,
    model_id: str,
    benchmark_id: str = "refusal",
    required_seeds: list[int] | None = None,
) -> pd.DataFrame:
    if required_seeds is None:
        required_seeds = REQUIRED_SEEDS
    records: list[dict] = []
    model_path = Path(results_dir) / model_id
    condition_ids = _expected_condition_ids()
    _validate_required_files(model_path, benchmark_id, required_seeds, condition_ids)

    for seed in required_seeds:
        seed_dir = model_path / f"seed_{seed}"
        eval_dir = seed_dir / "evaluations"
        expected_behavior_lookup = _load_expected_behavior_lookup(
            seed_dir,
            benchmark_id,
            condition_ids,
        )
        for condition_id in condition_ids:
            condition_dir = eval_dir / condition_id
            eval_file = condition_dir / benchmark_id / f"{benchmark_id}.jsonl"
            with open(eval_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    metrics = rec.pop("metrics", {})
                    rec.update(metrics)
                    rec["seed"] = seed
                    rec["expected_behavior"] = (
                        rec.get("expected_behavior")
                        or rec.get("expected_behaviour")
                        or expected_behavior_lookup.get(_join_key(rec))
                    )
                    records.append(rec)

    df = pd.DataFrame(records)
    df = _normalise_expected_behavior(df)
    df = _add_refusal_error_metrics(df)
    df["dimension"] = df["dimension"].fillna("baseline")
    df["category"] = df["category"].fillna("_none_")
    df["level"] = df["level"].fillna("baseline")
    df["refusal_style"] = df["refusal_style"].fillna("not_refused")
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def agg_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-condition mean with 95% CI across seeds (t-distribution, df=n-1)."""
    seed_means = (
        df.groupby(["dimension", "category", "level", "seed"])[metric]
        .mean()
        .reset_index(name="seed_mean")
    )
    rows = []
    for (dim, cat, lvl), grp in seed_means.groupby(["dimension", "category", "level"]):
        vals = grp["seed_mean"].values
        n = len(vals)
        mean = float(vals.mean())
        if n > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(n))
            lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        else:
            lo, hi = mean, mean  # no CI with a single seed
        rows.append({
            "dimension": dim, "category": cat, "level": lvl,
            "mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "n_seeds": n,
        })
    return pd.DataFrame(rows)


def agg_style(df: pd.DataFrame) -> pd.DataFrame:
    """Refusal style proportions per condition, pooled across seeds."""
    rows = []
    for (dim, cat, lvl), grp in df.groupby(["dimension", "category", "level"]):
        counts = grp["refusal_style"].value_counts(normalize=True)
        for style in STYLE_ORDER:
            rows.append({
                "dimension": dim, "category": cat, "level": lvl,
                "style": style, "proportion": float(counts.get(style, 0.0)),
            })
    return pd.DataFrame(rows)


def agg_style_ci_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """95% CI (t-interval, df=n-1) for each style proportion, across seeds.

    Per seed, proportions are computed within each (dimension, category, level) cell,
    matching how ``agg_metric`` treats seeds.
    """
    per_seed_rows: list[dict] = []
    for (dim, cat, lvl, seed), grp in df.groupby(
        ["dimension", "category", "level", "seed"], sort=False,
    ):
        counts = grp["refusal_style"].value_counts(normalize=True)
        for style in STYLE_ORDER:
            per_seed_rows.append({
                "dimension": dim,
                "category": cat,
                "level": lvl,
                "seed": seed,
                "style": style,
                "proportion": float(counts.get(style, 0.0)),
            })
    per_seed = pd.DataFrame(per_seed_rows)
    out_rows: list[dict] = []
    for (dim, cat, lvl, style), grp in per_seed.groupby(
        ["dimension", "category", "level", "style"], sort=False,
    ):
        vals = grp["proportion"].values.astype(float)
        n = len(vals)
        mean = float(np.mean(vals))
        if n > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(n))
            lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
            lo = float(np.clip(lo, 0.0, 1.0))
            hi = float(np.clip(hi, 0.0, 1.0))
        else:
            lo, hi = mean, mean
        out_rows.append({
            "dimension": dim,
            "category": cat,
            "level": lvl,
            "style": style,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_seeds": n,
        })
    return pd.DataFrame(out_rows)


def print_refusal_style_stats(df: pd.DataFrame) -> None:
    """Print refusal-style pooled proportions (figure-aligned) and 95% CIs across seeds."""
    style_df = agg_style(df)
    ci_df = agg_style_ci_across_seeds(df)
    if ci_df.empty:
        ci_lo_w = ci_hi_w = None
        n_seeds_idx: pd.Series | None = None
    else:
        ci_lo_w = (
            ci_df.pivot(index=["dimension", "category", "level"], columns="style", values="ci_lo")
            .reindex(columns=STYLE_ORDER)
        )
        ci_hi_w = (
            ci_df.pivot(index=["dimension", "category", "level"], columns="style", values="ci_hi")
            .reindex(columns=STYLE_ORDER)
        )
        n_seeds_idx = (
            ci_df.drop_duplicates(subset=["dimension", "category", "level"])
            .set_index(["dimension", "category", "level"])["n_seeds"]
        )
    col_names = [STYLE_LABELS[s] for s in STYLE_ORDER]
    col_w = 26

    def _fmt_cell(p: float, lo: float, hi: float, n_s: int) -> str:
        if n_s <= 1:
            return f"{p:5.1%} (no CI, n=1)".center(col_w)
        return f"{p:5.1%} [{lo:4.1%}, {hi:4.1%}]".center(col_w)

    line_w = max(100, 44 + col_w * len(STYLE_ORDER))

    print("\n" + "=" * line_w)
    print(
        "Refusal style: pooled proportion per cell (matches style figure); "
        "95% CI = t-interval across seeds on per-seed proportions within that cell."
    )
    print("Within each cell, pooled proportions sum to 100%.")
    print("=" * line_w)

    for dim in DIMENSIONS:
        sub = style_df[style_df["dimension"] == dim]
        if sub.empty:
            continue

        wide = (
            sub.pivot(index=["category", "level"], columns="style", values="proportion")
            .reindex(columns=STYLE_ORDER)
            .fillna(0.0)
        )
        multicat = _is_multicat(df, dim)
        ordered: list[tuple[str, str]] = []
        cats_loop = _cat_order(style_df, dim) if multicat else sorted(sub["category"].unique())
        for cat in cats_loop:
            for lvl in _level_order(style_df, dim):
                key = (cat, lvl)
                if key in wide.index and key not in ordered:
                    ordered.append(key)
        for key in wide.index:
            cat_k, lvl_k = key[0], key[1]
            if (cat_k, lvl_k) not in ordered:
                ordered.append((cat_k, lvl_k))

        print(f"\n{DIM_LABELS[dim]}  ({dim})")
        print("-" * line_w)
        hdr_pad = f"{'Category':<22} {'Level':<22}" if multicat else f"{'Level':<22}"
        header = hdr_pad + "".join(f"{h:^{col_w}}" for h in col_names)
        print(header)
        for cat, lvl in ordered:
            if (cat, lvl) not in wide.index:
                continue
            row = wide.loc[(cat, lvl)]
            lvl_disp = _lbl(lvl).replace("\n", " / ")
            key = (dim, cat, lvl)
            n_s = int(n_seeds_idx[key]) if n_seeds_idx is not None and key in n_seeds_idx.index else 1
            cells = []
            for s in STYLE_ORDER:
                p = float(row[s])
                if ci_lo_w is not None and ci_hi_w is not None and key in ci_lo_w.index:
                    lo = float(ci_lo_w.loc[key, s])
                    hi = float(ci_hi_w.loc[key, s])
                    cells.append(_fmt_cell(p, lo, hi, n_s))
                else:
                    cells.append(_fmt_cell(p, p, p, n_s))
            block = "".join(cells)
            if multicat:
                cat_disp = _cat_lbl(dim, cat)
                print(f"{cat_disp:<22} {lvl_disp:<22}{block}")
            else:
                print(f"{lvl_disp:<22}{block}")

    print("\n" + "=" * line_w + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_multicat(df: pd.DataFrame, dim: str) -> bool:
    return len(df[df["dimension"] == dim]["category"].unique()) > 1


def _level_order(data: pd.DataFrame, dim: str) -> list[str]:
    if dim in LEVEL_ORDERS:
        present = set(data[data["dimension"] == dim]["level"].unique())
        return [lv for lv in LEVEL_ORDERS[dim] if lv in present]
    lvls = sorted(data[data["dimension"] == dim]["level"].unique())
    return sorted(lvls, key=lambda lv: int(lv[1:]) if lv.startswith("l") and lv[1:].isdigit() else 999)


def _cat_order(data: pd.DataFrame, dim: str) -> list[str]:
    present = set(data[data["dimension"] == dim]["category"].unique())
    if dim in CATEGORY_ORDERS:
        ordered = [cat for cat in CATEGORY_ORDERS[dim] if cat in present]
        extras = sorted(present - set(ordered))
        return ordered + extras
    return sorted(present)


def _lbl(level: str) -> str:
    if level.startswith("l") and level[1:].isdigit():
        return f"L{level[1:]}"
    return {
        "high_positive": "Positive\nHigh",
        "low_positive": "Positive\nLow",
        "high_negative": "Negative\nHigh",
        "low_negative": "Negative\nLow",
    }.get(level, level.replace("_", "\n").title())


def _dim_level_lbl(dim: str, level: str) -> str:
    return LEVEL_LABELS.get(dim, {}).get(level, _lbl(level))


def _cat_lbl(dim: str, cat: str) -> str:
    """Pretty label for a sub-category slug (dimension-specific overrides)."""
    return CATEGORY_DISPLAY_LABELS.get(dim, {}).get(cat, cat.replace("_", " ").title())


def _baseline(agg: pd.DataFrame) -> float:
    row = agg[agg["dimension"] == "baseline"]
    return float(row["mean"].mean()) if not row.empty else float("nan")


def _pct_fmt(y: float, _) -> str:
    return f"{y:.0%}"


def _limit_y_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=4))


def _make_subplot_grid(layout: str, *, row_height: float = 4.0) -> tuple[plt.Figure, np.ndarray]:
    if layout == "2x2":
        return plt.subplots(2, 2, figsize=(10, row_height * 2))
    return plt.subplots(1, 4, figsize=(15, row_height))


def _make_refusal_error_grid(layout: str) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    if layout == "2x2":
        fig, axes = plt.subplots(4, 2, figsize=(11, 14))
        return fig, axes[:2, :].ravel(), axes[2:, :].ravel()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    return fig, axes[0, :], axes[1, :]


# ── Panel drawing ─────────────────────────────────────────────────────────────

def draw_line_panel(ax: plt.Axes, agg: pd.DataFrame, dim: str, baseline: float) -> None:
    cats = _cat_order(agg, dim)
    level_order = _level_order(agg, dim)
    x = np.arange(len(level_order))

    for i, cat in enumerate(cats):
        sub = (
            agg[(agg["dimension"] == dim) & (agg["category"] == cat)]
            .set_index("level")
            .reindex(level_order)
        )
        ys = sub["mean"].values.astype(float)
        lo = sub["ci_lo"].values.astype(float)
        hi = sub["ci_hi"].values.astype(float)
        n = sub["n_seeds"].fillna(1).values.astype(int)

        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, ys, marker="o", color=color, linewidth=1.8, markersize=5,
                label=_cat_lbl(dim, cat))
        if np.any(n > 1) and np.all(np.isfinite(lo)):
            ax.fill_between(x, lo, hi, alpha=0.18, color=color)

    if not np.isnan(baseline):
        ax.axhline(baseline, color="dimgray", linestyle="--", linewidth=1.0, label="Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels([_dim_level_lbl(dim, lv) for lv in level_order], fontsize=FONT_TICK)
    ax.set_xlim(-0.4, len(level_order) - 0.6)


def draw_bar_panel(ax: plt.Axes, agg: pd.DataFrame, dim: str, baseline: float) -> None:
    level_order = _level_order(agg, dim)
    sub = agg[agg["dimension"] == dim].set_index("level").reindex(level_order)
    ys = sub["mean"].values.astype(float)
    lo = sub["ci_lo"].values.astype(float)
    hi = sub["ci_hi"].values.astype(float)
    n = sub["n_seeds"].fillna(1).values.astype(int)

    x = np.arange(len(level_order))
    has_ci = bool(np.any(n > 1)) and np.all(np.isfinite(lo)) and not np.allclose(lo, ys)
    yerr = np.array([ys - lo, hi - ys]) if has_ci else None

    ax.bar(x, ys, color=BAR_COLOR, alpha=0.85, width=0.55,
           yerr=yerr, capsize=4, error_kw={"linewidth": 1.2})

    if not np.isnan(baseline):
        ax.axhline(baseline, color="dimgray", linestyle="--", linewidth=1.0, label="Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_dim_level_lbl(dim, lv) for lv in level_order],
        rotation=20,
        ha="right",
        fontsize=FONT_TICK,
    )


def _emotion_grid() -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=EMOTION_VALENCE_ORDER, columns=EMOTION_AROUSAL_ORDER)


def _emotion_metric_matrix(agg: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = _emotion_grid()
    sub = agg[agg["dimension"] == "emotion"].set_index("level")
    for level, (valence, arousal) in EMOTION_AXES.items():
        if level in sub.index:
            matrix.loc[valence, arousal] = float(sub.loc[level, value_col])
    return matrix


def _emotion_cell_annotation(
    mean_v: float,
    lo_v: float,
    hi_v: float,
    n_seeds: float,
    *,
    percent: bool,
) -> str:
    """Two-line heatmap label: pooled mean, then seed-based CI as (lower, upper)."""
    if percent:
        mean_txt = f"{mean_v:.0%}"
        if not np.isfinite(mean_v):
            return mean_txt
        if not np.isfinite(lo_v) or not np.isfinite(hi_v):
            return mean_txt
        n_int = int(n_seeds) if np.isfinite(n_seeds) else 1
        if n_int <= 1 or np.allclose(lo_v, hi_v):
            return mean_txt
        ci_txt = f"({lo_v:.0%}, {hi_v:.0%})"
    else:
        mean_txt = f"{mean_v:.0f}"
        if not np.isfinite(mean_v):
            return mean_txt
        if not np.isfinite(lo_v) or not np.isfinite(hi_v):
            return mean_txt
        n_int = int(n_seeds) if np.isfinite(n_seeds) else 1
        if n_int <= 1 or np.allclose(lo_v, hi_v):
            return mean_txt
        ci_txt = f"({lo_v:.0f}, {hi_v:.0f})"
    return f"{mean_txt}\n{ci_txt}"


def _emotion_annotation_matrix(agg: pd.DataFrame, *, percent: bool) -> pd.DataFrame:
    annot = pd.DataFrame("", index=EMOTION_VALENCE_ORDER, columns=EMOTION_AROUSAL_ORDER)
    sub = agg[agg["dimension"] == "emotion"].set_index("level")
    for level, (valence, arousal) in EMOTION_AXES.items():
        if level not in sub.index:
            continue
        row = sub.loc[level]
        mean_v = float(row["mean"])
        lo_v = float(row["ci_lo"])
        hi_v = float(row["ci_hi"])
        n_seeds = float(row.get("n_seeds", 1))
        annot.loc[valence, arousal] = _emotion_cell_annotation(
            mean_v, lo_v, hi_v, n_seeds, percent=percent,
        )
    return annot


def draw_emotion_heatmap(
    ax: plt.Axes,
    agg: pd.DataFrame,
    baseline: float,
    *,
    percent: bool,
) -> None:
    matrix = _emotion_metric_matrix(agg, "mean")
    annotations = _emotion_annotation_matrix(agg, percent=percent)
    sns.heatmap(
        matrix,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="rocket_r",
        vmin=0 if percent else None,
        vmax=1 if percent else None,
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": FONT_HEATMAP_ANNOT},
    )
    ax.set_xlabel("Arousal", fontsize=FONT_AXIS)
    ax.set_ylabel("Valence", fontsize=FONT_AXIS)
    ax.tick_params(axis="both", labelsize=FONT_TICK, length=0)
    if not np.isnan(baseline):
        base_text = f"Baseline: {baseline:.0%}" if percent else f"Baseline: {baseline:.0f}"
        ax.text(0.5, -0.16, base_text, ha="center", va="top",
                fontsize=FONT_TICK, color="dimgray", transform=ax.transAxes)


def draw_style_multicat(ax: plt.Axes, style_df: pd.DataFrame, dim: str) -> None:
    """Flat x-axis: sub-category groups separated by gaps, levels as bars within each group.

    Layout example for belief (3 cats × 2 levels):
      [Emp L1][Emp L2]  [Nor L1][Nor L2]  [Con L1][Con L2]
    Each bar is 100% stacked by refusal style.
    Sub-category names are annotated below the group.
    """
    cats = _cat_order(style_df, dim)
    level_order = _level_order(style_df, dim)
    n_levels = len(level_order)
    gap = 0.8  # extra space between sub-category groups

    # Build flat x positions: each cat block starts after the previous + gap
    x_positions: list[float] = []
    group_centers: list[float] = []
    cursor = 0.0
    for ci, cat in enumerate(cats):
        if ci > 0:
            cursor += gap
        start = cursor
        for _ in range(n_levels):
            x_positions.append(cursor)
            cursor += 1.0
        group_centers.append((start + cursor - 1) / 2)

    bar_w = 0.72
    flat_idx = 0
    for ci, cat in enumerate(cats):
        sub = style_df[(style_df["dimension"] == dim) & (style_df["category"] == cat)]
        piv = (
            sub.set_index(["level", "style"])["proportion"]
            .unstack(fill_value=0)
            .reindex(level_order, fill_value=0)
            .reindex(columns=STYLE_ORDER, fill_value=0)
        )
        bottoms = np.zeros(n_levels)
        xs = x_positions[flat_idx : flat_idx + n_levels]
        for style in STYLE_ORDER:
            vals = piv[style].values
            ax.bar(xs, vals, bar_w, bottom=bottoms, color=STYLE_COLORS[style])
            bottoms += vals
        flat_idx += n_levels

    # Level tick labels (L1, L2, ...)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_lbl(lv) for lv in level_order * len(cats)],
        fontsize=STYLE_FIG_FONT_TICK,
        rotation=STYLE_FIG_MULTICAT_XTICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )

    # Sub-category group labels below the bars
    for ci, (cat, cx) in enumerate(zip(cats, group_centers)):
        ax.text(cx, -0.22, _cat_lbl(dim, cat),
                ha="center", va="top", fontsize=STYLE_FIG_FONT_TICK, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Light vertical dividers between groups
    for ci in range(1, len(cats)):
        div_x = (x_positions[(ci) * n_levels - 1] + x_positions[(ci) * n_levels]) / 2
        ax.axvline(div_x, color="lightgray", linewidth=0.8, zorder=0)

    total_width = x_positions[-1] + 1
    ax.set_xlim(-0.5, total_width - 0.5)


def draw_style_simple(ax: plt.Axes, style_df: pd.DataFrame, dim: str) -> None:
    level_order = _level_order(style_df, dim)
    sub = style_df[style_df["dimension"] == dim]
    piv = (
        sub.set_index(["level", "style"])["proportion"]
        .unstack(fill_value=0)
        .reindex(level_order, fill_value=0)
        .reindex(columns=STYLE_ORDER, fill_value=0)
    )
    x = np.arange(len(level_order))
    bottoms = np.zeros(len(level_order))
    for style in STYLE_ORDER:
        vals = piv[style].values
        ax.bar(x, vals, 0.55, bottom=bottoms, color=STYLE_COLORS[style])
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_lbl(lv) for lv in level_order],
        rotation=20,
        ha="right",
        fontsize=STYLE_FIG_FONT_TICK,
    )


def draw_emotion_style_heatmap(ax: plt.Axes, style_df: pd.DataFrame) -> None:
    hard = _emotion_grid()
    annotations = pd.DataFrame("", index=EMOTION_VALENCE_ORDER, columns=EMOTION_AROUSAL_ORDER)
    sub = (
        style_df[style_df["dimension"] == "emotion"]
        .set_index(["level", "style"])["proportion"]
        .unstack(fill_value=0)
        .reindex(columns=STYLE_ORDER, fill_value=0)
    )
    for level, (valence, arousal) in EMOTION_AXES.items():
        if level not in sub.index:
            continue
        row = sub.loc[level]
        hard.loc[valence, arousal] = float(row["hard"])
        annotations.loc[valence, arousal] = "\n".join(
            f"{STYLE_LABELS[style]} {row[style]:.0%}" for style in STYLE_ORDER
        )

    sns.heatmap(
        hard,
        ax=ax,
        annot=annotations,
        fmt="",
        cmap="Reds",
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": STYLE_FIG_EMOTION_ANNOT},
    )
    ax.set_xlabel("Arousal", fontsize=STYLE_FIG_FONT_AXIS)
    ax.set_ylabel("Valence", fontsize=STYLE_FIG_FONT_AXIS)
    ax.tick_params(axis="both", labelsize=STYLE_FIG_FONT_TICK, length=0)


# ── Figure builders ───────────────────────────────────────────────────────────

def _draw_metric_panel_row(
    axes: np.ndarray,
    df: pd.DataFrame,
    *,
    metric: str,
    row_ylabel: str,
    value_ylabel: str = "Refusal Rate",
    show_column_titles: bool = True,
) -> None:
    agg = agg_metric(df, metric)
    base = _baseline(agg)
    axes_flat = axes.ravel()
    for col, dim in enumerate(DIMENSIONS):
        ax = axes_flat[col]
        if dim == "emotion":
            draw_emotion_heatmap(ax, agg, base, percent=True)
        elif _is_multicat(df, dim):
            draw_line_panel(ax, agg, dim, base)
            ax.legend(fontsize=FONT_LEGEND, loc="best", framealpha=0.7)
        else:
            draw_bar_panel(ax, agg, dim, base)
            ax.legend(fontsize=FONT_LEGEND, loc="best", framealpha=0.7)

        if dim != "emotion":
            _limit_y_ticks(ax)
            ax.yaxis.set_major_formatter(_pct_fmt)
            ax.tick_params(axis="both", labelsize=FONT_TICK)
        if show_column_titles:
            ax.set_title(DIM_LABELS[dim], fontsize=FONT_PANEL_TITLE, fontweight="bold")
        if col == 0:
            label = f"{row_ylabel}\n{value_ylabel}" if row_ylabel else value_ylabel
            ax.set_ylabel(label, fontsize=FONT_AXIS)
        sns.despine(ax=ax)


def _make_refusal_rate_metric_fig(
    df: pd.DataFrame,
    *,
    metric: str,
    row_ylabel: str,
    suptitle: str,
    value_ylabel: str = "Refusal Rate",
    layout: str = "1x4",
    show_title: bool = True,
) -> plt.Figure:
    fig, axes = _make_subplot_grid(layout, row_height=4.0)
    _draw_metric_panel_row(
        axes,
        df,
        metric=metric,
        row_ylabel=row_ylabel,
        value_ylabel=value_ylabel,
    )
    if show_title:
        fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_refusal_rate_judge_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_rate_metric_fig(
        df,
        metric="llm_refused",
        row_ylabel="LLM Judge",
        suptitle="Refusal Rate (LLM Judge) by Behavioral Dimension",
        layout=layout,
        show_title=show_title,
    )


def make_refusal_rate_keyword_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_rate_metric_fig(
        df,
        metric="keyword_refused",
        row_ylabel="Keyword",
        suptitle="Refusal Rate (Keyword) by Behavioral Dimension",
        layout=layout,
        show_title=show_title,
    )


def make_over_refusal_keyword_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_rate_metric_fig(
        df,
        metric="keyword_over_refusal",
        row_ylabel="Keyword",
        value_ylabel="Over-Refusal Rate",
        suptitle="Over-Refusal Rate: Refused When Should Not Refuse",
        layout=layout,
        show_title=show_title,
    )


def make_under_refusal_keyword_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_rate_metric_fig(
        df,
        metric="keyword_under_refusal",
        row_ylabel="Keyword",
        value_ylabel="Under-Refusal Rate",
        suptitle="Under-Refusal Rate: Did Not Refuse When Should Refuse",
        layout=layout,
        show_title=show_title,
    )


def make_over_refusal_judge_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_rate_metric_fig(
        df,
        metric="llm_over_refusal",
        row_ylabel="",
        value_ylabel="Over-Refusal Rate",
        suptitle="Over-Refusal Rate: Refused When Should Not Refuse",
        layout=layout,
        show_title=show_title,
    )


def make_under_refusal_judge_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_rate_metric_fig(
        df,
        metric="llm_under_refusal",
        row_ylabel="",
        value_ylabel="Under-Refusal Rate",
        suptitle="Under-Refusal Rate: Did Not Refuse When Should Refuse",
        layout=layout,
        show_title=show_title,
    )


def _make_refusal_errors_fig(
    df: pd.DataFrame,
    *,
    refused_metric_prefix: str,
    row_ylabel: str,
    suptitle: str,
    layout: str = "1x4",
    show_title: bool = True,
) -> plt.Figure:
    fig, over_axes, under_axes = _make_refusal_error_grid(layout)
    _draw_metric_panel_row(
        over_axes,
        df,
        metric=f"{refused_metric_prefix}_over_refusal",
        row_ylabel=row_ylabel,
        value_ylabel="Over-Refusal",
        show_column_titles=True,
    )
    _draw_metric_panel_row(
        under_axes,
        df,
        metric=f"{refused_metric_prefix}_under_refusal",
        row_ylabel=row_ylabel,
        value_ylabel="Under-Refusal",
        show_column_titles=False,
    )

    if show_title:
        fig.suptitle(suptitle, fontsize=FONT_SUPTITLE, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.4)
    return fig


def make_refusal_errors_keyword_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_errors_fig(
        df,
        refused_metric_prefix="keyword",
        row_ylabel="",
        suptitle="Refusal Errors (Keyword) by Behavioral Dimension",
        layout=layout,
        show_title=show_title,
    )


def make_refusal_errors_judge_fig(
    df: pd.DataFrame, layout: str = "1x4", show_title: bool = True
) -> plt.Figure:
    return _make_refusal_errors_fig(
        df,
        refused_metric_prefix="llm",
        row_ylabel="",
        suptitle="Refusal Errors (LLM Judge) by Behavioral Dimension",
        layout=layout,
        show_title=show_title,
    )


def make_style_fig(df: pd.DataFrame, layout: str = "1x4", show_title: bool = True) -> plt.Figure:
    style_df = agg_style(df)
    if layout == "2x2":
        fig = plt.figure(figsize=STYLE_FIG_2X2_FIGSIZE)
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=STYLE_FIG_2X2_HEIGHT_RATIOS,
            width_ratios=STYLE_FIG_2X2_WIDTH_RATIOS,
        )
        axes = np.array(
            [
                [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            ]
        )
    else:
        fig = plt.figure(figsize=STYLE_FIG_1X4_FIGSIZE)
        gs = fig.add_gridspec(1, 4, width_ratios=STYLE_FIG_1X4_WIDTH_RATIOS)
        axes = np.array([fig.add_subplot(gs[0, i]) for i in range(4)])
    axes_flat = axes.ravel()

    for col, dim in enumerate(DIMENSIONS):
        ax = axes_flat[col]
        if dim == "emotion":
            draw_emotion_style_heatmap(ax, style_df)
        elif _is_multicat(df, dim):
            draw_style_multicat(ax, style_df, dim)
        else:
            draw_style_simple(ax, style_df, dim)

        ax.set_title(DIM_LABELS[dim], fontsize=STYLE_FIG_FONT_PANEL, fontweight="bold")
        if dim != "emotion":
            ax.set_ylim(0, 1.02)
            _limit_y_ticks(ax)
            ax.yaxis.set_major_formatter(_pct_fmt)
            ax.tick_params(axis="both", labelsize=STYLE_FIG_FONT_TICK)
        if col == 0:
            ax.set_ylabel("Proportion", fontsize=STYLE_FIG_FONT_AXIS)
        sns.despine(ax=ax)

    style_handles = [
        mpatches.Patch(color=STYLE_COLORS[s], label=STYLE_LABELS[s]) for s in STYLE_ORDER
    ]
    fig.legend(
        handles=style_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=STYLE_FIG_LEGEND,
        title="Refusal Style",
        title_fontsize=STYLE_FIG_LEGEND,
    )

    if show_title:
        fig.suptitle(
            "Refusal Style by Behavioral Dimension",
            fontsize=STYLE_FIG_FONT_SUPTITLE,
            fontweight="bold",
        )
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    return fig


def make_verbosity_fig(df: pd.DataFrame, layout: str = "1x4", show_title: bool = True) -> plt.Figure:
    agg = agg_metric(df, "response_tokens")
    base = _baseline(agg)

    fig, axes = _make_subplot_grid(layout, row_height=4.0)
    axes_flat = axes.ravel()

    for col, dim in enumerate(DIMENSIONS):
        ax = axes_flat[col]
        if dim == "emotion":
            draw_emotion_heatmap(ax, agg, base, percent=False)
        elif _is_multicat(df, dim):
            draw_line_panel(ax, agg, dim, base)
            ax.legend(fontsize=FONT_LEGEND, loc="best", framealpha=0.7)
        else:
            draw_bar_panel(ax, agg, dim, base)
            ax.legend(fontsize=FONT_LEGEND, loc="best", framealpha=0.7)

        if dim != "emotion":
            _limit_y_ticks(ax)
            ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.set_title(DIM_LABELS[dim], fontsize=FONT_PANEL_TITLE, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Response Tokens", fontsize=FONT_AXIS)
        sns.despine(ax=ax)

    if show_title:
        fig.suptitle(
            "Response Verbosity by Behavioral Dimension",
            fontsize=FONT_SUPTITLE,
            fontweight="bold",
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _has_metric_values(df: pd.DataFrame, metric: str) -> bool:
    return metric in df.columns and df[metric].notna().any()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot refusal analysis figures.")
    parser.add_argument("--results-dir", default="results/behavioral")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=REQUIRED_SEEDS,
        help="Required balanced seed set for refusal plots.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Output format for figures (default: png). Use 'both' to write .png and .pdf.",
    )
    parser.add_argument(
        "--layout",
        choices=("1x4", "2x2"),
        default="1x4",
        help="Panel layout for the four dimensions (default: 1x4).",
    )
    parser.add_argument(
        "--no-plot-title",
        action="store_true",
        help="Disable figure-level plot titles (suptitles).",
    )
    parser.add_argument(
        "--refusal-style-only",
        action="store_true",
        help="Render only refusal_style and force a 2x2 panel layout.",
    )
    args = parser.parse_args()
    _set_theme_like_trait_choice()

    out = (
        Path(args.output_dir)
        if args.output_dir
        else Path("figs/behavioral") / args.model_id / "refusal"
    )

    print("Loading data...")
    df = load_data(args.results_dir, args.model_id, required_seeds=args.seeds)
    n_seeds = df["seed"].nunique()
    n_conds = df["condition_id"].nunique() if "condition_id" in df.columns else "?"
    print(f"  {len(df):,} samples | {n_seeds} seed(s) | {n_conds} condition(s)")
    expected_counts = df["expected_behavior"].value_counts().to_dict()
    print(f"  expected_behavior counts: {expected_counts}")
    if n_seeds == 1:
        print("  Note: single seed — error bars will be omitted")

    out.mkdir(parents=True, exist_ok=True)
    extensions: list[str] = []
    if args.format in ("png", "both"):
        extensions.append("png")
    if args.format in ("pdf", "both"):
        extensions.append("pdf")

    if args.refusal_style_only:
        figures = [("refusal_style", make_style_fig)]
    else:
        figures = [
            ("refusal_rate_judge", make_refusal_rate_judge_fig),
            ("refusal_rate_keyword", make_refusal_rate_keyword_fig),
            ("refusal_errors_keyword", make_refusal_errors_keyword_fig),
            ("refusal_style", make_style_fig),
            ("refusal_verbosity", make_verbosity_fig),
        ]
        if _has_metric_values(df, "llm_refused"):
            figures.extend([
                ("refusal_errors_judge", make_refusal_errors_judge_fig),
            ])
        else:
            print("Skipping LLM judge error plots: llm_refused has no non-null values")

    for name, builder in figures:
        print(f"Generating {name} ({', '.join(extensions)}) ...")
        if name == "refusal_style":
            print_refusal_style_stats(df)
        layout = "2x2" if args.refusal_style_only else args.layout
        fig = builder(df, layout=layout, show_title=not args.no_plot_title)
        for ext in extensions:
            dest = out / f"{name}.{ext}"
            if ext == "png":
                fig.savefig(dest, dpi=150, bbox_inches="tight")
            else:
                fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
