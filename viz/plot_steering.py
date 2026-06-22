"""Plot user-attribute vector steering effects on refusal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams["ps.useafm"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.unicode_minus"] = False

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from representation.vector_specs import VECTOR_SPECS
from viz.plot_refusal import DIM_LABELS, PALETTE, _pct_fmt

DIMENSION_VECTORS = {
    dimension: [spec.name for spec in VECTOR_SPECS if spec.dimension == dimension]
    for dimension in dict.fromkeys(spec.dimension for spec in VECTOR_SPECS)
}
DEFAULT_DIMENSIONS = list(DIMENSION_VECTORS)
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
DIMENSION_LABELS = {
    "knowledge": "Knowledge",
    "intent": "Intent",
    "emotion": "Emotion",
    "belief": "Belief",
}
MAX_AXIS_TICKS = 4

PANEL_TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 14
YLABEL_FONTSIZE = 18
SUPTITLE_FONTSIZE = 20
SERIES_LINEWIDTH = 4.5
DOT_SIZE = 8
ERRORBAR_LINEWIDTH = 1.7
ERRORBAR_CAPSIZE = 4
Y_AXIS_PAD_FRACTION = 0.08
Y_AXIS_MIN_SPAN = 0.08

REFUSAL_METRICS = ("llm_refused", "keyword_refused")
PERPLEXITY_METRICS = ("response_perplexity", "completion_perplexity")

# Edit this list to control which steering coefficients are plotted.
# Set to None to include every coefficient found under results-dir.
COEFFICIENT_FILTER: list[float] | None = [-2.0, -1.0, 0.0, 1.0, 2.0]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_evaluations(
    *,
    results_dir: Path,
    model_id: str,
    benchmark_id: str,
    dimensions: list[str],
    vectors: list[str] | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    model_root = results_dir / model_id
    requested_vectors = set(vectors) if vectors is not None else None
    for seed_dir in sorted(model_root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        for dimension in dimensions:
            dim_root = seed_dir / "evaluations" / dimension
            for path in sorted(dim_root.glob(f"*/coef_*/{benchmark_id}/{benchmark_id}.jsonl")):
                vector_name = path.parents[2].name
                if requested_vectors is not None and vector_name not in requested_vectors:
                    continue
                coef_dir = path.parents[1].name
                try:
                    coefficient = float(coef_dir.removeprefix("coef_"))
                except ValueError:
                    continue
                for row in _load_jsonl(path):
                    metrics = row.get("metrics", {})
                    out_row = {
                        "seed": seed,
                        "steering_dimension": row.get("steering_dimension", dimension),
                        "vector_name": row.get("vector_name", vector_name),
                        "coefficient": float(row.get("coefficient", coefficient)),
                        "prompt_id": row.get("prompt_id"),
                    }
                    for metric in (*REFUSAL_METRICS, *PERPLEXITY_METRICS):
                        out_row[metric] = metrics.get(metric)
                    rows.append(out_row)
    if not rows:
        raise FileNotFoundError(
            f"No steering evaluations found under {model_root}/seed_*/evaluations"
        )
    df = pd.DataFrame(rows)
    for col in (*REFUSAL_METRICS, *PERPLEXITY_METRICS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _seed_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df[metric].dropna().empty:
        raise ValueError(
            f"No non-null {metric} values found. For LLM refusal metrics, rerun "
            "steering evaluation with a judge backend other than 'none'."
        )
    seed_means = (
        df.groupby(["steering_dimension", "vector_name", "coefficient", "seed"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "seed_mean"})
    )
    rows = []
    for (dimension, vector_name, coefficient), grp in seed_means.groupby(
        ["steering_dimension", "vector_name", "coefficient"]
    ):
        vals = grp["seed_mean"].dropna().to_numpy(dtype=float)
        n = len(vals)
        mean = float(vals.mean()) if n else float("nan")
        if n > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(n))
            if sem > 0.0:
                lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
            else:
                lo = hi = mean
        else:
            lo = hi = mean
        rows.append(
            {
                "vector_name": vector_name,
                "steering_dimension": dimension,
                "coefficient": float(coefficient),
                "mean": mean,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_seeds": n,
            }
        )
    return pd.DataFrame(rows)


def _vector_label(name: str) -> str:
    return VECTOR_LABELS.get(name, name)


def _metric_label(metric: str) -> str:
    return {
        "llm_refused": "LLM Refusal Rate",
        "keyword_refused": "Keyword Refusal Rate",
        "response_perplexity": "Response Perplexity",
        "completion_perplexity": "Completion Perplexity",
    }.get(metric, metric.replace("_", " ").title())


def _set_theme_like_projection() -> None:
    sns.set_theme(style="white", context="paper")


def _make_axes(layout: str, *, sharey: bool) -> tuple[plt.Figure, np.ndarray]:
    n = len(DEFAULT_DIMENSIONS)
    if layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), sharey=sharey)
        axes_flat = np.asarray(axes).ravel()
    else:
        fig, axes = plt.subplots(1, n, figsize=(18, 4.3), sharey=sharey)
        axes_flat = np.atleast_1d(axes)
    return fig, axes_flat


def _metric_y_limits(summary: pd.DataFrame, metric: str) -> tuple[float, float]:
    lo = pd.to_numeric(summary["ci_lo"], errors="coerce").min()
    hi = pd.to_numeric(summary["ci_hi"], errors="coerce").max()
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = pd.to_numeric(summary["mean"], errors="coerce").min()
        hi = pd.to_numeric(summary["mean"], errors="coerce").max()
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0) if metric in REFUSAL_METRICS else (0.0, 1.0)

    span = max(float(hi - lo), Y_AXIS_MIN_SPAN)
    pad = span * Y_AXIS_PAD_FRACTION
    ymin = float(lo - pad)
    ymax = float(hi + pad)
    if metric in REFUSAL_METRICS:
        ymin = max(0.0, ymin)
        ymax = min(1.0, ymax)
        if ymax - ymin < Y_AXIS_MIN_SPAN:
            mid = (ymin + ymax) / 2.0
            ymin = max(0.0, mid - Y_AXIS_MIN_SPAN / 2.0)
            ymax = min(1.0, mid + Y_AXIS_MIN_SPAN / 2.0)
    return ymin, ymax


def _draw_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    dimension: str,
    vectors: list[str],
    metric: str,
    pct_y: bool,
    y_limits: tuple[float, float],
) -> None:
    legend_handles: list[Line2D] = []
    for i, vector_name in enumerate(vectors):
        sub = summary[
            (summary["steering_dimension"] == dimension)
            & (summary["vector_name"] == vector_name)
        ].sort_values("coefficient")
        if sub.empty:
            continue

        x = sub["coefficient"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        lo = sub["ci_lo"].to_numpy(dtype=float)
        hi = sub["ci_hi"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])
        color = PALETTE[i % len(PALETTE)]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=SERIES_LINEWIDTH,
            markersize=DOT_SIZE,
            color=color,
            ecolor=color,
            elinewidth=ERRORBAR_LINEWIDTH,
            capsize=ERRORBAR_CAPSIZE,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linewidth=SERIES_LINEWIDTH,
                markersize=DOT_SIZE,
                label=_vector_label(vector_name),
            )
        )

    ax.axvline(0.0, color="dimgray", linestyle="--", linewidth=1.6, alpha=0.75)
    ax.set_title(
        DIM_LABELS.get(dimension, DIMENSION_LABELS.get(dimension, dimension)),
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Steering coefficient", fontsize=AXIS_LABEL_FONTSIZE)
    if pct_y:
        ax.yaxis.set_major_formatter(_pct_fmt)
    ax.set_ylim(*y_limits)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE,
            loc="best",
            framealpha=0.75,
        )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_AXIS_TICKS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_AXIS_TICKS))
    sns.despine(ax=ax)


def plot_metric_vs_coefficient(
    *,
    df: pd.DataFrame,
    dimensions: list[str],
    vectors_by_dimension: dict[str, list[str]],
    metric: str,
    output_path: Path,
    formats: list[str],
    layout: str,
    show_title: bool,
) -> None:
    summary = _seed_summary(df, metric)
    _set_theme_like_projection()
    pct_y = metric in REFUSAL_METRICS
    sharey = pct_y
    y_limits = _metric_y_limits(summary, metric)
    fig, axes_flat = _make_axes(layout, sharey=sharey)

    for idx, dimension in enumerate(dimensions):
        ax = axes_flat[idx]
        _draw_metric_panel(
            ax,
            summary,
            dimension=dimension,
            vectors=vectors_by_dimension[dimension],
            metric=metric,
            pct_y=pct_y,
            y_limits=y_limits,
        )
    for ax in axes_flat[len(dimensions):]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel(_metric_label(metric), fontsize=YLABEL_FONTSIZE)
    if show_title:
        fig.suptitle(
            f"Steering: {_metric_label(metric)} vs. Coefficient",
            fontsize=SUPTITLE_FONTSIZE,
            fontweight="bold",
            y=0.92,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = output_path.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[FIG] wrote {out_path}", flush=True)
    plt.close(fig)


def _resolve_dimensions(names: list[str] | None) -> list[str]:
    if names is None:
        return DEFAULT_DIMENSIONS
    unknown = set(names) - set(DIMENSION_VECTORS)
    if unknown:
        raise KeyError(f"Unknown dimensions: {sorted(unknown)}")
    return [dimension for dimension in DEFAULT_DIMENSIONS if dimension in set(names)]


def _vectors_by_dimension(
    dimensions: list[str],
    vectors: list[str] | None,
) -> dict[str, list[str]]:
    if vectors is None:
        return {dimension: list(DIMENSION_VECTORS[dimension]) for dimension in dimensions}
    requested = set(vectors)
    known = {name for names in DIMENSION_VECTORS.values() for name in names}
    unknown = requested - known
    if unknown:
        raise KeyError(f"Unknown vectors: {sorted(unknown)}")
    return {
        dimension: [name for name in DIMENSION_VECTORS[dimension] if name in requested]
        for dimension in dimensions
    }


def _filter_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    if COEFFICIENT_FILTER is None:
        return df
    coef_vals = df["coefficient"].to_numpy(dtype=float)
    mask = np.zeros(len(df), dtype=bool)
    for coefficient in COEFFICIENT_FILTER:
        mask |= np.isclose(coef_vals, float(coefficient))
    out = df[mask].copy()
    if out.empty:
        raise ValueError(f"No rows found for COEFFICIENT_FILTER={COEFFICIENT_FILTER}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot steering refusal curves.")
    parser.add_argument("--results-dir", default="results/steering")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--benchmark", default="refusal")
    parser.add_argument("--dimensions", nargs="+", default=None)
    parser.add_argument("--vectors", nargs="+", default=None)
    parser.add_argument("--metric", default="llm_refused", choices=list(REFUSAL_METRICS))
    parser.add_argument(
        "--perplexity-metric",
        default="response_perplexity",
        choices=list(PERPLEXITY_METRICS),
    )
    parser.add_argument("--layout", default="1x4", choices=["1x4", "2x2"])
    parser.add_argument("--no-plot-title", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf"])
    parser.add_argument("--output-dir", default="figs/steering")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    dimensions = _resolve_dimensions(args.dimensions)
    vectors_by_dimension = _vectors_by_dimension(dimensions, args.vectors)
    dimensions = [dimension for dimension in dimensions if vectors_by_dimension[dimension]]
    if not dimensions:
        raise ValueError("No vectors selected for plotting")
    vector_filter = [
        vector
        for dimension in dimensions
        for vector in vectors_by_dimension[dimension]
    ]

    df = _read_evaluations(
        results_dir=results_dir,
        model_id=args.model_id,
        benchmark_id=args.benchmark,
        dimensions=dimensions,
        vectors=vector_filter,
    )
    df = _filter_coefficients(df)
    output_group = dimensions[0] if len(dimensions) == 1 else "all"
    show_title = not args.no_plot_title
    output_path = output_dir / args.model_id / output_group / "refusal_vs_coefficient.png"
    plot_metric_vs_coefficient(
        df=df,
        dimensions=dimensions,
        vectors_by_dimension=vectors_by_dimension,
        metric=args.metric,
        output_path=output_path,
        formats=args.formats,
        layout=args.layout,
        show_title=show_title,
    )
    perplexity_path = output_dir / args.model_id / output_group / "perplexity_vs_coefficient.png"
    plot_metric_vs_coefficient(
        df=df,
        dimensions=dimensions,
        vectors_by_dimension=vectors_by_dimension,
        metric=args.perplexity_metric,
        output_path=perplexity_path,
        formats=args.formats,
        layout=args.layout,
        show_title=show_title,
    )


if __name__ == "__main__":
    main()
