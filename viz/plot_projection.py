"""Section 6.3.2 — projection figures.

Consumes the parquets written by ``representation.projection``:
- ``projections_held_out.parquet`` (long-form per-prompt rows)
- ``condition_summary.parquet`` (per-condition aggregates)
- ``binned_summary.parquet`` (projection-decile aggregates)

Outputs to ``figs/projection/<eval-pos>_onto_<build-pos>/{dist,scatter,binned}/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.ticker import MaxNLocator

mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams["ps.useafm"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.unicode_minus"] = False

from representation.vector_specs import VECTOR_SPECS, ContrastSpec
from viz.plot_refusal import (
    CATEGORY_ORDERS,
    DIM_LABELS,
    DIMENSIONS,
    LEVEL_ORDERS,
    PALETTE,
    _lbl,
    _pct_fmt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_POS = "P4"
DEFAULT_EVAL_POS = "P4"
MAX_AXIS_TICKS = 4


def _set_theme_like_trait_choice() -> None:
    """Match ``plot_trait_choice_probe`` seaborn theme while preserving font rc."""
    sns.set_theme(style="white", context="paper")

# Typography controls (edit these to tune figure text globally).
DIST_TICK_FONTSIZE = 9
DIST_AXIS_LABEL_FONTSIZE = 10
DIST_VECTOR_TITLE_FONTSIZE = 12
DIST_GROUP_LABEL_FONTSIZE = 9
DIST_DIM_TITLE_FONTSIZE = 14
DIST_SUPTITLE_FONTSIZE = 16

SCATTER_ANNOTATION_FONTSIZE = 8
SCATTER_AXIS_LABEL_FONTSIZE = 10
SCATTER_VECTOR_TITLE_FONTSIZE = 12
SCATTER_LEGEND_FONTSIZE = 8.5
SCATTER_TICK_FONTSIZE = 9
SCATTER_SUPTITLE_FONTSIZE = 14

BINNED_PANEL_TITLE_FONTSIZE = 20
BINNED_AXIS_LABEL_FONTSIZE = 18
BINNED_LEGEND_FONTSIZE = 12
BINNED_TICK_FONTSIZE = 14
BINNED_YLABEL_FONTSIZE = 18
BINNED_SUPTITLE_FONTSIZE = 20

# Line-width controls (edit these to tune stroke thickness globally).
DIST_VIOLIN_EDGE_LINEWIDTH = 1.1
DIST_VIOLIN_MEDIAN_LINEWIDTH = 1.8
DIST_ZERO_LINE_LINEWIDTH = 1.4

SCATTER_SERIES_LINEWIDTH = 2.0
SCATTER_ERRORBAR_LINEWIDTH = 1.7
SCATTER_ERRORBAR_CAPSIZE = 4
SCATTER_SOURCE_EDGE_LINEWIDTH = 1.8
SCATTER_NON_SOURCE_EDGE_LINEWIDTH = 1.1
SCATTER_BASELINE_LINEWIDTH = 1.6
SCATTER_SOURCE_DOT_SIZE = 58
SCATTER_NON_SOURCE_DOT_SIZE = 42

BINNED_SERIES_LINEWIDTH = 4.5
BINNED_DOT_SIZE = 8

VECTOR_LABELS = {
    "v_formal": "Educational",
    "v_experiential": "Professional",
    "v_autodidact": "Autodidactic",
    "v_curiosity": "Curiosity",
    "v_educational": "Educational",
    "v_professional": "Professional",
    "v_institutional": "Institutional",
    "v_valence": "Valence",
    "v_arousal": "Arousal",
    "v_empirical": "Empirical",
    "v_normative": "Normative",
    "v_conspiratorial": "Conspiratorial",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vectors_for_dim(dim: str) -> list[str]:
    return [spec.name for spec in VECTOR_SPECS if spec.dimension == dim]


def _spec_by_name() -> dict[str, ContrastSpec]:
    return {spec.name: spec for spec in VECTOR_SPECS}


def _level_order_local(df: pd.DataFrame, dim: str) -> list[str]:
    if dim in LEVEL_ORDERS:
        present = set(df[df["cond_dimension"] == dim]["cond_level"].unique())
        return [lv for lv in LEVEL_ORDERS[dim] if lv in present]
    levels = sorted(df[df["cond_dimension"] == dim]["cond_level"].unique())
    return sorted(levels, key=lambda lv: int(lv[1:]) if lv.startswith("l") and lv[1:].isdigit() else 999)


def _cat_order_local(df: pd.DataFrame, dim: str) -> list[str]:
    present = {
        c for c in df[df["cond_dimension"] == dim]["cond_category"].unique()
        if c and c != "_none_"
    }
    if dim in CATEGORY_ORDERS:
        ordered = [cat for cat in CATEGORY_ORDERS[dim] if cat in present]
        extras = sorted(present - set(ordered))
        return ordered + extras
    return sorted(present)


def _agg_seed_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-condition mean ± 95% t-CI across seeds for a refusal metric."""
    seed_means = (
        df.groupby(["vector_name", "cond_dimension", "cond_category", "cond_level", "seed"])[metric]
        .mean()
        .reset_index(name="seed_mean")
    )
    rows = []
    for (vname, dim, cat, lvl), grp in seed_means.groupby(
        ["vector_name", "cond_dimension", "cond_category", "cond_level"]
    ):
        vals = grp["seed_mean"].dropna().to_numpy()
        n = len(vals)
        if n == 0:
            mean = lo = hi = float("nan")
        else:
            mean = float(vals.mean())
            if n > 1:
                sem = float(vals.std(ddof=1) / np.sqrt(n))
                lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
            else:
                lo, hi = mean, mean
        rows.append({
            "vector_name": vname,
            "cond_dimension": dim,
            "cond_category": cat,
            "cond_level": lvl,
            "mean": mean,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_seeds": n,
        })
    return pd.DataFrame(rows)


def _baseline_value(df: pd.DataFrame, vector_name: str, metric: str) -> float:
    sub = df[(df["vector_name"] == vector_name) & (df["cond_dimension"] == "baseline")]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub[metric], errors="coerce").dropna().mean())


def _projection_mean(cond_df: pd.DataFrame, vector_name: str, condition_id: str) -> float:
    sub = cond_df[
        (cond_df["vector_name"] == vector_name)
        & (cond_df["condition_id"] == condition_id)
    ]
    if sub.empty:
        return float("nan")
    return float(sub["projection_mean"].mean())


def _condition_label(dim: str, cat: str, level: str) -> str:
    if dim == "baseline":
        return "Base"
    if dim == "emotion":
        return _lbl(level).replace("\n", " ")
    if cat and cat != "_none_":
        return f"{cat.replace('_', ' ').title()} {_lbl(level)}"
    return _lbl(level)


def _condition_color_maps(triples: list[tuple[str, str, str, str]]) -> tuple[dict[str, tuple], dict[str, tuple]]:
    cats: list[str] = []
    levels: list[str] = []
    for _, dim, cat, level in triples:
        if dim == "baseline":
            continue
        if dim == "emotion":
            if level not in levels:
                levels.append(level)
        elif cat not in cats:
            cats.append(cat)
    cat_colors = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(cats)}
    level_colors = {level: PALETTE[i % len(PALETTE)] for i, level in enumerate(levels)}
    return cat_colors, level_colors


def _vector_label(name: str) -> str:
    return VECTOR_LABELS.get(name, name)


# ── Per-condition projection distributions ───────────────────────────────────

def _conditions_in_view(
    prompt_df: pd.DataFrame,
    vector_name: str,
    *,
    include_baseline: bool = True,
) -> list[tuple[str, str, str, str]]:
    """Return ordered (condition_id, cond_dimension, cond_category, cond_level)
    tuples for the conditions present under one vector. Multicat first, then
    baseline appended at the end."""
    sub = prompt_df[prompt_df["vector_name"] == vector_name]
    triples = (
        sub[["condition_id", "cond_dimension", "cond_category", "cond_level"]]
        .drop_duplicates()
    )
    out = []
    for dim in DIMENSIONS:
        dim_rows = triples[triples["cond_dimension"] == dim]
        if dim_rows.empty:
            continue
        cats = _cat_order_local(dim_rows, dim) or ["_none_"]
        lvls = _level_order_local(dim_rows, dim)
        for cat in cats:
            for lvl in lvls:
                row = dim_rows[(dim_rows["cond_category"] == cat) & (dim_rows["cond_level"] == lvl)]
                if not row.empty:
                    out.append((str(row.iloc[0]["condition_id"]), dim, cat, lvl))
    base = triples[triples["cond_dimension"] == "baseline"]
    if include_baseline and not base.empty:
        out.append((str(base.iloc[0]["condition_id"]), "baseline", "_none_", "baseline"))
    return out


def _draw_distribution(
    ax: plt.Axes,
    prompt_df: pd.DataFrame,
    vector_name: str,
    *,
    show_title: bool = True,
) -> None:
    """Violin distributions across conditions present for this vector."""
    sub = prompt_df[prompt_df["vector_name"] == vector_name]
    # One row per (prompt_id, condition_id) — projections do not vary with seed.
    sub = sub.drop_duplicates(subset=["prompt_id", "condition_id"])

    triples = _conditions_in_view(sub, vector_name, include_baseline=False)
    if not triples:
        ax.set_visible(False)
        return

    cats_seen = [t[2] for t in triples if t[1] != "baseline" and t[2] != "_none_"]
    seen: list[str] = []
    for c in cats_seen:
        if c not in seen:
            seen.append(c)
    cat_palette = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(seen)}
    baseline_color = "#888888"

    positions: list[float] = []
    cursor = 0.0
    last_cat: str | None = None
    data: list[np.ndarray] = []
    colors: list[tuple] = []
    tick_labels: list[str] = []
    for cid, dim, cat, lvl in triples:
        if last_cat is not None and cat != last_cat:
            cursor += 0.6
        last_cat = cat
        positions.append(cursor)
        cursor += 1.0
        vals = sub[sub["condition_id"] == cid]["projection"].astype(float).to_numpy()
        data.append(vals)
        if dim == "baseline":
            colors.append(baseline_color)
            tick_labels.append("Base")
        else:
            colors.append(cat_palette.get(cat, PALETTE[0]))
            tick_labels.append(_lbl(lvl))

    parts = ax.violinplot(
        data, positions=positions, widths=0.78, showmeans=False, showmedians=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("dimgray")
        body.set_linewidth(DIST_VIOLIN_EDGE_LINEWIDTH)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(DIST_VIOLIN_MEDIAN_LINEWIDTH)

    ax.axhline(0.0, color="lightgray", linestyle=":", linewidth=DIST_ZERO_LINE_LINEWIDTH, zorder=0)
    if len(positions) > MAX_AXIS_TICKS:
        tick_idx = np.linspace(0, len(positions) - 1, MAX_AXIS_TICKS, dtype=int)
        tick_idx = sorted(set(tick_idx.tolist()))
        shown_positions = [positions[i] for i in tick_idx]
        shown_labels = [tick_labels[i] for i in tick_idx]
    else:
        shown_positions = positions
        shown_labels = tick_labels
    ax.set_xticks(shown_positions)
    ax.set_xticklabels(shown_labels, fontsize=DIST_TICK_FONTSIZE, rotation=0)
    ax.set_xlim(min(positions) - 0.6, max(positions) + 0.6)
    ax.set_ylabel("h · v̂", fontsize=DIST_AXIS_LABEL_FONTSIZE)
    if show_title:
        ax.set_title(vector_name, fontsize=DIST_VECTOR_TITLE_FONTSIZE, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_AXIS_TICKS))
    sns.despine(ax=ax)

    # Sub-category group labels under multicat panels
    if len(seen) > 1:
        groups: dict[str, list[float]] = {}
        for pos, (_, dim, cat, _) in zip(positions, triples):
            if dim == "baseline":
                continue
            groups.setdefault(cat, []).append(pos)
        for cat, xs in groups.items():
            cx = (min(xs) + max(xs)) / 2
            ax.text(cx, -0.10, cat.replace("_", " ").title(),
                    ha="center", va="top", fontsize=DIST_GROUP_LABEL_FONTSIZE, fontweight="bold",
                    transform=ax.get_xaxis_transform())


def make_distribution_fig(prompt_df: pd.DataFrame, *, show_title: bool = True) -> plt.Figure:
    max_rows = max(len(_vectors_for_dim(dim)) for dim in DIMENSIONS)
    fig, axes = plt.subplots(
        max_rows,
        len(DIMENSIONS),
        figsize=(5.3 * len(DIMENSIONS), 3.0 * max_rows),
        squeeze=False,
    )
    for col, dim in enumerate(DIMENSIONS):
        vectors = _vectors_for_dim(dim)
        if show_title:
            axes[0, col].set_title(DIM_LABELS[dim], fontsize=DIST_DIM_TITLE_FONTSIZE, fontweight="bold", pad=18)
        for row, vname in enumerate(vectors):
            ax = axes[row, col]
            _draw_distribution(ax, prompt_df, vname, show_title=show_title)
        for row in range(len(vectors), max_rows):
            axes[row, col].set_visible(False)
    if show_title:
        fig.suptitle("Projection Distributions", fontsize=DIST_SUPTITLE_FONTSIZE, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    else:
        fig.tight_layout()
    return fig


# ── Per-condition projection vs. refusal ─────────────────────────────────────

def _draw_scatter_panel(
    ax: plt.Axes,
    cond_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    vector_name: str,
    metric: str,
    *,
    y_label: str,
    pct_y: bool,
    show_title: bool = True,
) -> None:
    spec = _spec_by_name()[vector_name]
    dim = spec.dimension
    source_conditions = set(spec.positive_conditions) | set(spec.negative_conditions)
    agg = _agg_seed_metric(prompt_df, metric)
    agg = agg[agg["vector_name"] == vector_name]
    if agg.empty:
        ax.set_visible(False)
        return

    triples = _conditions_in_view(prompt_df, vector_name)
    cat_colors, level_colors = _condition_color_maps(triples)
    point_rows: list[dict[str, object]] = []
    for cid, cond_dim, cat, level in triples:
        row = agg[
            (agg["cond_dimension"] == cond_dim)
            & (agg["cond_category"] == cat)
            & (agg["cond_level"] == level)
        ]
        if row.empty:
            continue
        x = _projection_mean(cond_df, vector_name, cid)
        y = float(row.iloc[0]["mean"])
        lo = float(row.iloc[0]["ci_lo"])
        hi = float(row.iloc[0]["ci_hi"])
        if np.isnan(x) or np.isnan(y):
            continue
        if cond_dim == "baseline":
            color = "#888888"
            legend_label = _condition_label(cond_dim, cat, level)
        elif dim == "emotion":
            color = level_colors.get(level, PALETTE[0])
            legend_label = _condition_label(cond_dim, cat, level)
        else:
            color = cat_colors.get(cat, PALETTE[0])
            legend_label = _condition_label(cond_dim, cat, level)
        point_rows.append({
            "condition_id": cid,
            "dim": cond_dim,
            "cat": cat,
            "level": level,
            "x": x,
            "y": y,
            "lo": lo,
            "hi": hi,
            "color": color,
            "legend_label": legend_label,
        })

    if not point_rows:
        ax.set_visible(False)
        return

    if dim != "emotion":
        for cat in _cat_order_local(prompt_df[prompt_df["vector_name"] == vector_name], dim):
            cat_points = [
                row for row in point_rows
                if row["cat"] == cat and row["dim"] != "baseline"
            ]
            if len(cat_points) < 2:
                continue
            cat_points = sorted(cat_points, key=lambda row: str(row["level"]))
            ax.plot(
                [float(row["x"]) for row in cat_points],
                [float(row["y"]) for row in cat_points],
                color=cat_colors.get(cat, PALETTE[0]),
                linewidth=SCATTER_SERIES_LINEWIDTH,
                alpha=0.7,
                zorder=1,
            )

    labels_seen: set[str] = set()
    for row in point_rows:
        x = float(row["x"])
        y = float(row["y"])
        lo = float(row["lo"])
        hi = float(row["hi"])
        color = row["color"]
        cid = str(row["condition_id"])
        legend_label = str(row["legend_label"])
        show_label = legend_label if legend_label not in labels_seen else None
        labels_seen.add(legend_label)
        edgecolor = "black" if cid in source_conditions else "white"
        linewidth = SCATTER_SOURCE_EDGE_LINEWIDTH if cid in source_conditions else SCATTER_NON_SOURCE_EDGE_LINEWIDTH
        size = SCATTER_SOURCE_DOT_SIZE if cid in source_conditions else SCATTER_NON_SOURCE_DOT_SIZE

        if not np.isnan(lo) and not np.isnan(hi) and lo != hi:
            ax.errorbar(
                [x], [y], yerr=[[y - lo], [hi - y]],
                fmt="none", ecolor=color, elinewidth=SCATTER_ERRORBAR_LINEWIDTH, capsize=SCATTER_ERRORBAR_CAPSIZE,
                alpha=0.85, zorder=2,
            )
        ax.scatter(
            [x], [y], s=size, color=color, edgecolor=edgecolor,
            linewidth=linewidth, zorder=3, label=show_label,
        )
        ax.annotate(
            _condition_label(str(row["dim"]), str(row["cat"]), str(row["level"])),
            xy=(x, y), xytext=(4, 4),
            textcoords="offset points", fontsize=SCATTER_ANNOTATION_FONTSIZE, color="dimgray",
        )

    base = _baseline_value(prompt_df, vector_name, metric)
    if not np.isnan(base):
        ax.axhline(base, color="dimgray", linestyle="--", linewidth=SCATTER_BASELINE_LINEWIDTH)

    if pct_y:
        ax.yaxis.set_major_formatter(_pct_fmt)
    ax.set_xlabel(f"Mean projection on {vector_name}", fontsize=SCATTER_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=SCATTER_AXIS_LABEL_FONTSIZE)
    if show_title:
        ax.set_title(vector_name, fontsize=SCATTER_VECTOR_TITLE_FONTSIZE, fontweight="bold")
    ax.legend(fontsize=SCATTER_LEGEND_FONTSIZE, loc="best", framealpha=0.7)
    ax.tick_params(axis="both", labelsize=SCATTER_TICK_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_AXIS_TICKS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_AXIS_TICKS))
    sns.despine(ax=ax)


def make_scatter_figs(
    cond_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    metric: str,
    *,
    title: str,
    y_label: str,
    pct_y: bool,
    show_title: bool = True,
) -> dict[str, plt.Figure]:
    figs: dict[str, plt.Figure] = {}
    for dim in DIMENSIONS:
        vectors = _vectors_for_dim(dim)
        if not vectors:
            continue
        n = len(vectors)
        rows = 2 if n > 2 else 1
        cols = 2 if n > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(7.8 * cols, 3.8 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, vector_name in zip(axes, vectors):
            _draw_scatter_panel(
                ax, cond_df, prompt_df, vector_name,
                metric, y_label=y_label, pct_y=pct_y, show_title=show_title,
            )
        for ax in axes[len(vectors):]:
            ax.set_visible(False)
        if show_title:
            fig.suptitle(
                f"{title}: {DIM_LABELS[dim]}",
                fontsize=SCATTER_SUPTITLE_FONTSIZE,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
        else:
            fig.tight_layout()
        figs[dim] = fig
    return figs


# ── Projection-binned behaviour ───────────────────────────────────────────────

def make_binned_refusal_fig(
    bin_df: pd.DataFrame,
    eval_pos: str,
    build_pos: str,
    *,
    layout: str = "1x4",
    show_title: bool = True,
) -> plt.Figure:
    n = len(DIMENSIONS)
    if layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), sharey=True)
        axes_flat = axes.ravel()
    else:
        fig, axes = plt.subplots(1, n, figsize=(18, 4.3), sharey=True)
        axes_flat = np.atleast_1d(axes)
    for idx, dim in enumerate(DIMENSIONS):
        ax = axes_flat[idx]
        vectors = _vectors_for_dim(dim)
        for i, vector_name in enumerate(vectors):
            sub = bin_df[bin_df["vector_name"] == vector_name].sort_values("bin")
            if sub.empty:
                continue
            ax.plot(
                sub["projection_center"].astype(float).to_numpy(),
                sub["llm_refused_mean"].astype(float).to_numpy(),
                marker="o",
                linewidth=BINNED_SERIES_LINEWIDTH,
                markersize=BINNED_DOT_SIZE,
                color=PALETTE[i % len(PALETTE)],
                label=_vector_label(vector_name),
            )
        # Always label panels by behavioral dimension; suptitle is optional (--no-plot-title).
        ax.set_title(DIM_LABELS[dim], fontsize=BINNED_PANEL_TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel("Binned projection", fontsize=BINNED_AXIS_LABEL_FONTSIZE)
        ax.yaxis.set_major_formatter(_pct_fmt)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=BINNED_LEGEND_FONTSIZE, loc="best", framealpha=0.75)
        ax.tick_params(axis="both", labelsize=BINNED_TICK_FONTSIZE)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_AXIS_TICKS))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_AXIS_TICKS))
        sns.despine(ax=ax)
    axes_flat[0].set_ylabel("Refusal Rate", fontsize=BINNED_YLABEL_FONTSIZE)
    if show_title:
        fig.suptitle(f"Projection-Binned Refusal Rate ({eval_pos} onto {build_pos})", fontsize=BINNED_SUPTITLE_FONTSIZE, fontweight="bold", y=0.92)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def _filter_pos(df: pd.DataFrame, build_pos: str, eval_pos: str) -> pd.DataFrame:
    return df[(df["build_pos"] == build_pos) & (df["eval_pos"] == eval_pos)].copy()


def _save_fig(fig: plt.Figure, path: Path, formats: list[str], *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"  wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Section 6.3.2 projection figures.")
    parser.add_argument("--vectors-dir", default="results/user_attr_vectors")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Parent directory for figures (default: <repo>/figs/projection). "
            "Figures are written under <output-dir>/<eval-pos>_onto_<build-pos>/."
        ),
    )
    parser.add_argument("--build-pos", default=DEFAULT_BUILD_POS)
    parser.add_argument("--eval-pos", default=DEFAULT_EVAL_POS)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--binned-layout",
        choices=("1x4", "2x2"),
        default="1x4",
        help="Panel layout for projection-binned refusal figure (default: 1x4).",
    )
    parser.add_argument(
        "--no-plot-title",
        action="store_true",
        help=(
            "Omit figure suptitles and per-vector subplot titles. "
            "Binned plot still shows Knowledge / Intent / Emotion / Belief panel titles."
        ),
    )
    args = parser.parse_args()
    _set_theme_like_trait_choice()
    formats = list(dict.fromkeys(args.formats))

    pos_tag = f"{args.eval_pos}_onto_{args.build_pos}"
    proj_dir = Path(args.vectors_dir) / args.model_id / "projection" / pos_tag
    prompt_path = proj_dir / "projections_held_out.parquet"
    cond_path = proj_dir / "condition_summary.parquet"
    bin_path = proj_dir / "binned_summary.parquet"
    for p in (prompt_path, cond_path, bin_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}; run representation.projection first.")

    out_base = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "figs" / "projection")
    out_root = out_base / pos_tag
    dist_dir = out_root / "dist"
    scatter_dir = out_root / "scatter"
    binned_dir = out_root / "binned"
    for d in (dist_dir, scatter_dir, binned_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading projections from {proj_dir} ...", flush=True)
    prompt_df = _filter_pos(pd.read_parquet(prompt_path), args.build_pos, args.eval_pos)
    cond_df = _filter_pos(pd.read_parquet(cond_path), args.build_pos, args.eval_pos)
    bin_df = _filter_pos(pd.read_parquet(bin_path), args.build_pos, args.eval_pos)
    if prompt_df.empty:
        raise RuntimeError(
            f"No rows for build_pos={args.build_pos!r} eval_pos={args.eval_pos!r}; "
            "rerun representation.projection with these positions."
        )
    print(
        f"  prompt_rows={len(prompt_df):,}  cond_rows={len(cond_df):,}  bin_rows={len(bin_df):,}",
        flush=True,
    )

    show_title = not args.no_plot_title

    print("Projection distributions ...", flush=True)
    fig = make_distribution_fig(prompt_df, show_title=show_title)
    path = dist_dir / "projection_distributions.png"
    _save_fig(fig, path, formats)
    plt.close(fig)

    print("Projection vs. refusal ...", flush=True)
    plot_b_specs = [
        ("projection_vs_refusal_rate", "llm_refused", "LLM Refusal Rate", True,
         "Mean Projection vs. Mean LLM Refusal Rate"),
        ("projection_vs_keyword_rate", "keyword_refused", "Keyword Refusal Rate", True,
         "Mean Projection vs. Mean Keyword Refusal Rate"),
        ("projection_vs_verbosity", "response_tokens", "Mean response tokens", False,
         "Mean Projection vs. Mean Response Tokens"),
    ]
    for name, metric, ylabel, pct_y, title in plot_b_specs:
        for dim, fig in make_scatter_figs(
            cond_df, prompt_df, metric,
            title=title, y_label=ylabel, pct_y=pct_y, show_title=show_title,
        ).items():
            path = scatter_dir / f"{name}_{dim}.png"
            _save_fig(fig, path, formats)
            plt.close(fig)

    print("Projection-binned LLM refusal rate ...", flush=True)
    fig = make_binned_refusal_fig(
        bin_df, eval_pos=args.eval_pos, build_pos=args.build_pos, layout=args.binned_layout, show_title=show_title,
    )
    path = binned_dir / "projection_binned_llm_refusal_rate.png"
    _save_fig(fig, path, formats)
    plt.close(fig)

    print(f"\nAll figures saved under {out_root}/", flush=True)


if __name__ == "__main__":
    main()
