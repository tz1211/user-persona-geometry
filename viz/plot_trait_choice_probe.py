"""Plot forced-choice steering probe results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams["ps.useafm"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.unicode_minus"] = False

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from representation.vector_specs import VECTOR_SPECS
from viz.plot_refusal import PALETTE

REPO_ROOT = Path(__file__).resolve().parents[1]
DIMENSION_ORDER = ("knowledge", "intent", "emotion", "belief")
DIMENSION_SUBTITLE = {
    "knowledge": "Knowledge",
    "intent": "Intent",
    "emotion": "Emotion",
    "belief": "Belief",
}

# Presentation typography (both figures)
_FS_PANEL_TITLE = 20
_FS_SUPTITLE = 18
_FS_FIG_NOTE = 12
_FS_AXIS_LABEL = 18
_FS_TICK = 14
_FS_LEGEND = 12
_TICK_NBINS = 4

# Stroke/marker controls
CHOICE_LINEWIDTH = 1.7
CHOICE_MARKERSIZE = 5
CHOICE_REF_LINEWIDTH = 0.9

DELTA_LINEWIDTH = 4.5
DELTA_MARKERSIZE = 8
DELTA_REF_LINEWIDTH = 0.9


def _style_axes_ticks(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=_FS_TICK,
        length=6,
        width=1.0,
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=_TICK_NBINS, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=_TICK_NBINS, prune=None))


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


def _vector_to_dimension() -> dict[str, str]:
    return {spec.name: spec.dimension for spec in VECTOR_SPECS}


def _vectors_for_dimension(dim: str, allowed: set[str]) -> list[str]:
    return [spec.name for spec in VECTOR_SPECS if spec.dimension == dim and spec.name in allowed]


def _vector_order(vector_names: list[str] | None = None) -> list[str]:
    names = [spec.name for spec in VECTOR_SPECS]
    if vector_names is None:
        return names
    requested = set(vector_names)
    missing = requested - set(names)
    if missing:
        raise KeyError(f"Unknown vectors: {sorted(missing)}")
    return [name for name in names if name in requested]


def _load_summary(results_dir: Path, model_id: str) -> pd.DataFrame:
    path = results_dir / model_id / "aggregates" / "choice_probe_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {path}")
    df = pd.read_csv(path)
    df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce")
    df["p_positive"] = pd.to_numeric(df["p_positive"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df.dropna(subset=["coefficient", "p_positive", "seed"])


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    seed_means = (
        df.groupby(["vector_name", "coefficient", "seed"], as_index=False)["p_positive"]
        .mean()
        .rename(columns={"p_positive": "seed_mean"})
    )
    rows = []
    for (vector_name, coefficient), grp in seed_means.groupby(["vector_name", "coefficient"]):
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
        rows.append({
            "vector_name": vector_name,
            "coefficient": float(coefficient),
            "mean": mean,
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_seeds": n,
        })
    return pd.DataFrame(rows)


def _seed_stats(df: pd.DataFrame) -> tuple[int, list[int]]:
    seeds = sorted({int(s) for s in df["seed"].dropna().unique()})
    return len(seeds), seeds


def _filter_coefficients(df: pd.DataFrame, coefficients: list[float] | None) -> pd.DataFrame:
    if coefficients is None:
        return df
    coeffs = [float(c) for c in coefficients]
    coef_vals = df["coefficient"].to_numpy(dtype=float)
    mask = np.zeros(len(df), dtype=bool)
    for c in coeffs:
        mask |= np.isclose(coef_vals, c)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No rows found for coefficients={coeffs}")
    return out


def _save_fig(fig: plt.Figure, path: Path, formats: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[FIG] wrote {out_path}", flush=True)


def plot_choice_probability(
    *,
    df: pd.DataFrame,
    vector_names: list[str],
    output_path: Path,
    formats: list[str],
) -> None:
    subset = df[df["vector_name"].isin(vector_names)]
    summary = _summarise(subset)
    if summary.empty:
        raise ValueError("No rows available to plot")
    n_seeds_detected, seeds = _seed_stats(subset)

    sns.set_theme(style="white", context="paper")
    n = len(vector_names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.0 * rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, vector_name in zip(axes, vector_names):
        sub = summary[summary["vector_name"] == vector_name].sort_values("coefficient")
        if sub.empty:
            ax.set_visible(False)
            continue
        x = sub["coefficient"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        lo = sub["ci_lo"].to_numpy(dtype=float)
        hi = sub["ci_hi"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=CHOICE_LINEWIDTH,
            markersize=CHOICE_MARKERSIZE,
            color="#4C78A8",
        )
        ax.fill_between(x, lo, hi, color="#4C78A8", alpha=0.18, linewidth=0)
        ax.axhline(0.5, color="0.65", linestyle="--", linewidth=CHOICE_REF_LINEWIDTH)
        ax.axvline(0.0, color="0.6", linestyle=":", linewidth=CHOICE_REF_LINEWIDTH)
        ax.set_title(
            VECTOR_LABELS.get(vector_name, vector_name),
            fontsize=_FS_PANEL_TITLE,
            fontweight="bold",
        )
        ax.set_xlabel("Steering coefficient", fontsize=_FS_AXIS_LABEL)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(lambda value, _pos: f"{100 * value:.0f}%")
        _style_axes_ticks(ax)
        sns.despine(ax=ax)

    for ax in axes[n:]:
        ax.set_visible(False)
    for idx in range(0, len(axes), cols):
        # Avoid mathtext ($...$) because it pulls in embedded DejaVu fonts (Type 3).
        axes[idx].set_ylabel("P(c_hat = L2)", fontsize=_FS_AXIS_LABEL)

    fig.suptitle(
        "Forced-Choice L2 Probability vs. Steering Coefficient",
        fontsize=_FS_SUPTITLE,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.985,
        f"Mean with 95% CI across seeds (n={n_seeds_detected}; seeds={seeds})",
        ha="center",
        va="top",
        fontsize=_FS_FIG_NOTE,
        color="0.35",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, output_path, formats)
    plt.close(fig)


def _per_vector_probability(df: pd.DataFrame) -> pd.DataFrame:
    """Per vector: mean P(L2 choice) at each coefficient (mean over seeds and pairs)."""
    vec_to_dim = _vector_to_dimension()
    vc = df.groupby(["vector_name", "coefficient"], as_index=False)["p_positive"].mean()
    vc["dimension"] = vc["vector_name"].map(vec_to_dim)
    rows: list[dict[str, float | str]] = []
    for (vector_name, dimension), g in vc.groupby(["vector_name", "dimension"], dropna=False):
        if pd.isna(dimension):
            continue
        g = g.sort_values("coefficient")
        for _, row in g.iterrows():
            coef = float(row["coefficient"])
            p = float(row["p_positive"])
            rows.append({
                "vector_name": str(vector_name),
                "dimension": str(dimension),
                "coefficient": coef,
                "p_positive": p,
            })
    if not rows:
        return pd.DataFrame(columns=["vector_name", "dimension", "coefficient", "p_positive"])
    return pd.DataFrame(rows)


def plot_dimension_signed_delta_1x4(
    *,
    df: pd.DataFrame,
    vector_names: list[str],
    output_path: Path,
    formats: list[str],
) -> None:
    subset = df[df["vector_name"].isin(vector_names)]
    summary = _per_vector_probability(subset)
    if summary.empty:
        raise ValueError("No rows available for dimension probability plot")

    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.3), sharey=True)
    allowed = set(vector_names)

    y_lim = (-0.02, 1.02)

    for ax, dim in zip(axes, DIMENSION_ORDER):
        ax.set_title(
            DIMENSION_SUBTITLE[dim],
            fontsize=_FS_PANEL_TITLE,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel("Steering coefficient", fontsize=_FS_AXIS_LABEL)
        vectors_here = _vectors_for_dimension(dim, allowed)
        if not vectors_here:
            ax.set_visible(False)
            continue
        for i, vector_name in enumerate(vectors_here):
            sub = summary[
                (summary["dimension"] == dim) & (summary["vector_name"] == vector_name)
            ].sort_values("coefficient")
            if sub.empty:
                continue
            x = sub["coefficient"].to_numpy(dtype=float)
            y = sub["p_positive"].to_numpy(dtype=float)
            label = VECTOR_LABELS.get(vector_name, vector_name)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=DELTA_LINEWIDTH,
                markersize=DELTA_MARKERSIZE,
                color=PALETTE[i % len(PALETTE)],
                label=label,
            )
        ax.axvline(0.0, color="0.6", linestyle=":", linewidth=DELTA_REF_LINEWIDTH)
        ax.set_ylim(y_lim)
        ax.yaxis.set_major_formatter(lambda value, _pos: f"{100 * value:.0f}%")
        _style_axes_ticks(ax)
        ax.legend(fontsize=_FS_LEGEND, loc="best", framealpha=0.75)
        sns.despine(ax=ax)

    # Avoid mathtext ($...$) because it pulls in embedded DejaVu fonts (Type 3).
    axes[0].set_ylabel("P(c_hat = L2)", fontsize=_FS_AXIS_LABEL)
    fig.tight_layout()
    _save_fig(fig, output_path, formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot forced-choice steering probe.")
    parser.add_argument("--results-dir", default="results/trait_choice_probe")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--vectors", nargs="+", default=None)
    parser.add_argument(
        "--coefficients",
        nargs="+",
        type=float,
        default=None,
        help="Optional coefficient values to include in plots (e.g., --coefficients -2 -1 0 1 2).",
    )
    parser.add_argument("--output-dir", default="figs/trait_choice_probe")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf"],
        help="Figure formats to write.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    vector_names = _vector_order(args.vectors)
    df = _filter_coefficients(_load_summary(results_dir, args.model_id), args.coefficients)
    n_seeds_detected, seeds = _seed_stats(df)
    print(f"[INFO] Detected {n_seeds_detected} seeds: {seeds}", flush=True)
    output_path = output_dir / args.model_id / "l2_choice_probability_vs_coefficient.png"
    plot_choice_probability(
        df=df,
        vector_names=vector_names,
        output_path=output_path,
        formats=list(dict.fromkeys(args.formats)),
    )
    delta_path = output_dir / args.model_id / "l2_choice_delta_by_dimension.png"
    plot_dimension_signed_delta_1x4(
        df=df,
        vector_names=vector_names,
        output_path=delta_path,
        formats=list(dict.fromkeys(args.formats)),
    )


if __name__ == "__main__":
    main()
