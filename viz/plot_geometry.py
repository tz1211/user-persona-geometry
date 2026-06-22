"""P4-only geometry figures.

This script consumes the CSV outputs from ``representation.geometry`` and
renders the Section 6.3.1 plots for the P4 vectors only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams["ps.useafm"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.unicode_minus"] = False

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from representation.geometry import CROSS_DIMENSION_VECTORS, SUBCATEGORY_GROUPS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSITION = "P4"

# Typography for ``_plot_heatmap`` (cross-dimension + subcategory grids).
HEATMAP_TICK_FONTSIZE = 16
HEATMAP_TITLE_FONTSIZE = 24
HEATMAP_CELL_VALUE_FONTSIZE = 12
COLORBAR_TICK_FONTSIZE = 14
COLORBAR_LABEL_FONTSIZE = 16
CROSS_DIMENSION_BANNER_FONTSIZE = 18

# Bar plots in this module (`plot_intrinsic_dimension`).
BARPLOT_TICK_FONTSIZE = 12
BARPLOT_TITLE_FONTSIZE = 14
BARPLOT_AXIS_LABEL_FONTSIZE = 12

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

DIM_LABELS = {
    "knowledge": "Knowledge",
    "intent": "Intent",
    "emotion": "Emotion",
    "belief": "Belief",
}

CROSS_DIMENSION_BLOCK_DIMS = ("Knowledge", "Intent", "Emotion", "Belief")


def _cross_dimension_block_sizes() -> list[int]:
    return [
        len(SUBCATEGORY_GROUPS["knowledge"]),
        len(SUBCATEGORY_GROUPS["intent"]),
        2,  # v_valence, v_arousal
        len(SUBCATEGORY_GROUPS["belief"]),
    ]


def _annotate_cross_dimension_block_labels(ax: Axes, sizes: list[int], names: tuple[str, ...]) -> None:
    trans_y_axis = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    trans_x_axis = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    lo = 0.0
    for name, sz in zip(names, sizes, strict=True):
        y_center = lo + sz / 2.0
        ax.text(
            -0.30,
            y_center,
            name,
            transform=trans_y_axis,
            ha="right",
            va="center",
            rotation=90,
            fontsize=CROSS_DIMENSION_BANNER_FONTSIZE,
            fontweight="bold",
            clip_on=False,
        )
        x_center = lo + sz / 2.0
        ax.text(
            x_center,
            -0.20,
            name,
            transform=trans_x_axis,
            ha="center",
            va="top",
            fontsize=CROSS_DIMENSION_BANNER_FONTSIZE,
            fontweight="bold",
            clip_on=False,
        )
        lo += float(sz)


def _style_heatmap_extra_axes(fig: plt.Figure, main_ax: Axes) -> None:
    """Set tick + label fontsize on axes that are not the main heatmap (e.g. colorbar)."""
    for axis in fig.axes:
        if axis is main_ax:
            continue
        axis.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
        ylab = axis.get_ylabel()
        if ylab:
            axis.set_ylabel(ylab, fontsize=COLORBAR_LABEL_FONTSIZE)
        xlab = axis.get_xlabel()
        if xlab:
            axis.set_xlabel(xlab, fontsize=COLORBAR_LABEL_FONTSIZE)


def _model_geometry_root(vectors_dir: Path, model_id: str) -> Path:
    return vectors_dir / model_id / "geometry"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing geometry CSV: {path}. Run `python -m representation.geometry` first."
        )
    return pd.read_csv(path)


def _read_matrix_csv(path: Path) -> pd.DataFrame:
    """Load a square geometry matrix; index/columns stay ``v_*`` vector ids (unique)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing geometry matrix CSV: {path}. Run `python -m representation.geometry` first."
        )
    df = pd.read_csv(path, index_col=0)
    df.index = [str(idx) for idx in df.index]
    df.columns = [str(col) for col in df.columns]
    return df


def _cross_dimension_block_separator_positions() -> list[float]:
    """X/Y coordinates for lines between dimension blocks (matches ``CROSS_DIMENSION_VECTORS``)."""
    sizes = _cross_dimension_block_sizes()
    cum = 0
    positions: list[float] = []
    for s in sizes[:-1]:
        cum += s
        positions.append(float(cum))
    return positions


def _plot_heatmap(
    *,
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    figsize: tuple[float, float],
    formats: list[str],
    annot: bool = True,
    block_separator_positions: list[float] | None = None,
    block_banner_sizes: list[int] | None = None,
    block_banner_names: tuple[str, ...] | None = None,
    heatmap_ticklabels: list[str] | None = None,
    x_tick_positions_offset: float = 0.0,
    x_tick_horizontalalignment: str | None = None,
    show_title: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    heatmap_kw: dict = {
        "vmin": -1,
        "vmax": 1,
        "cmap": "vlag",
        "annot": annot,
        "fmt": ".2f",
        "square": True,
        "cbar_kws": {"label": "Cosine", "shrink": 0.78, "pad": 0.03},
        "ax": ax,
    }
    if annot:
        heatmap_kw["annot_kws"] = {"size": HEATMAP_CELL_VALUE_FONTSIZE}
    if block_separator_positions:
        heatmap_kw["linewidths"] = 0.35
        heatmap_kw["linecolor"] = "white"
    if heatmap_ticklabels is not None:
        heatmap_kw["xticklabels"] = heatmap_ticklabels
        heatmap_kw["yticklabels"] = heatmap_ticklabels
    sns.heatmap(matrix, **heatmap_kw)
    _style_heatmap_extra_axes(fig, ax)
    if block_separator_positions:
        for pos in block_separator_positions:
            ax.axhline(
                pos,
                color="0.2",
                linewidth=1.6,
                zorder=15,
                clip_on=False,
            )
            ax.axvline(
                pos,
                color="0.2",
                linewidth=1.6,
                zorder=15,
                clip_on=False,
            )
    if block_banner_sizes is not None and block_banner_names is not None:
        _annotate_cross_dimension_block_labels(ax, block_banner_sizes, block_banner_names)
    if show_title:
        ax.set_title(title, fontsize=HEATMAP_TITLE_FONTSIZE, fontweight="bold", pad=20)
    ncol = matrix.shape[1]
    if x_tick_positions_offset != 0.0 or x_tick_horizontalalignment is not None:
        x_labs = (
            list(heatmap_ticklabels)
            if heatmap_ticklabels is not None
            else [t.get_text() for t in ax.get_xticklabels()]
        )
        x_pos = np.arange(ncol, dtype=float) + 0.5 + x_tick_positions_offset
        ha = x_tick_horizontalalignment if x_tick_horizontalalignment is not None else "right"
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            x_labs,
            rotation=35,
            fontsize=HEATMAP_TICK_FONTSIZE,
            ha=ha,
            rotation_mode="anchor",
        )
        ax.tick_params(axis="y", labelrotation=0, labelsize=HEATMAP_TICK_FONTSIZE)
    else:
        ax.tick_params(axis="x", labelrotation=35, labelsize=HEATMAP_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelrotation=0, labelsize=HEATMAP_TICK_FONTSIZE)
    fig.tight_layout()
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[FIG] wrote {out_path}", flush=True)
    plt.close(fig)


def _subcategory_matrix(rows: pd.DataFrame, dimension: str) -> pd.DataFrame:
    names = SUBCATEGORY_GROUPS[dimension]
    labels = [VECTOR_LABELS.get(name, name) for name in names]
    matrix = pd.DataFrame(1.0, index=labels, columns=labels)
    sub = rows[(rows["dimension"] == dimension) & (rows["position"] == DEFAULT_POSITION)]
    if sub.empty:
        raise ValueError(f"No {DEFAULT_POSITION} subcategory collinearity rows for {dimension}")
    for row in sub.itertuples(index=False):
        left = VECTOR_LABELS.get(str(row.vector_a), str(row.vector_a))
        right = VECTOR_LABELS.get(str(row.vector_b), str(row.vector_b))
        matrix.loc[left, right] = float(row.cosine)
        matrix.loc[right, left] = float(row.cosine)
    return matrix


def plot_subcategory_heatmaps(
    *,
    geometry_root: Path,
    output_dir: Path,
    formats: list[str],
    show_title: bool = True,
) -> None:
    rows = _read_required_csv(geometry_root / "subcategory_collinearity.csv")
    for dimension in SUBCATEGORY_GROUPS:
        matrix = _subcategory_matrix(rows, dimension)
        _plot_heatmap(
            matrix=matrix,
            title=f"{DIM_LABELS.get(dimension, dimension.title())} Sub-Category Collinearity",
            path=output_dir / f"{dimension}_subcategory_collinearity.png",
            figsize=(4.6, 4.0),
            formats=formats,
            show_title=show_title,
        )


def plot_cross_dimension(
    *,
    geometry_root: Path,
    output_dir: Path,
    formats: list[str],
    show_title: bool = True,
) -> None:
    raw = _read_matrix_csv(
        geometry_root / "matrices" / f"cross_dimension_{DEFAULT_POSITION}.csv"
    )
    matrix = raw.loc[list(CROSS_DIMENSION_VECTORS), list(CROSS_DIMENSION_VECTORS)]
    display_labels = [VECTOR_LABELS.get(name, name) for name in CROSS_DIMENSION_VECTORS]
    _plot_heatmap(
        matrix=matrix,
        title="Pairwise Attribute Cosine Similarity",
        path=output_dir / "cross_dimension.png",
        figsize=(9.5, 8.2),
        formats=formats,
        block_separator_positions=_cross_dimension_block_separator_positions(),
        block_banner_sizes=_cross_dimension_block_sizes(),
        block_banner_names=CROSS_DIMENSION_BLOCK_DIMS,
        heatmap_ticklabels=display_labels,
        # Shift tick anchor slightly left of column centers (tune between ~-0.5 and 0).
        x_tick_positions_offset=-0.1,
        x_tick_horizontalalignment="right",
        show_title=show_title,
    )


def plot_intrinsic_dimension(
    *,
    geometry_root: Path,
    output_dir: Path,
    formats: list[str],
    show_title: bool = True,
) -> None:
    rows = _read_required_csv(geometry_root / "subspace_intrinsic_dimension.csv")
    sub = rows[rows["position"] == DEFAULT_POSITION].copy()
    if sub.empty:
        raise ValueError(f"No {DEFAULT_POSITION} intrinsic dimension rows found")
    sub["Dimension"] = sub["dimension"].map(lambda x: DIM_LABELS.get(str(x), str(x).title()))

    print("\nIntrinsic dimension values:")
    for row in sub.sort_values("dimension").itertuples(index=False):
        print(
            f"  {row.Dimension}: "
            f"participation_ratio={row.participation_ratio:.6f}, "
            f"effective_rank_entropy={row.effective_rank_entropy:.6f}, "
            f"pc1_var={row.pc1_var:.6f}, "
            f"pc2_var={row.pc2_var:.6f}, "
            f"pc3_var={row.pc3_var:.6f}, "
            f"top2_var={row.top2_var:.6f}",
            flush=True,
        )

    path = output_dir / "subspace_intrinsic_dimension.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.barplot(
        data=sub,
        x="Dimension",
        y="participation_ratio",
        order=[DIM_LABELS[d] for d in SUBCATEGORY_GROUPS],
        color="#4C78A8",
        ax=ax,
    )
    if show_title:
        ax.set_title("Subspace Intrinsic Dimension", fontsize=BARPLOT_TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(
        "Participation Ratio",
        fontsize=BARPLOT_AXIS_LABEL_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=BARPLOT_TICK_FONTSIZE)
    ax.set_ylim(0, max(1.0, float(sub["participation_ratio"].max()) * 1.15))
    sns.despine(ax=ax)
    fig.tight_layout()
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[FIG] wrote {out_path}", flush=True)
    plt.close(fig)


def plot_geometry(
    *,
    model_id: str,
    vectors_dir: Path,
    output_dir: Path,
    formats: list[str],
    show_title: bool = True,
) -> None:
    geometry_root = _model_geometry_root(vectors_dir, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    plot_subcategory_heatmaps(
        geometry_root=geometry_root,
        output_dir=output_dir,
        formats=formats,
        show_title=show_title,
    )
    plot_cross_dimension(
        geometry_root=geometry_root,
        output_dir=output_dir,
        formats=formats,
        show_title=show_title,
    )
    plot_intrinsic_dimension(
        geometry_root=geometry_root,
        output_dir=output_dir,
        formats=formats,
        show_title=show_title,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render P4-only geometry figures.")
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--vectors-dir", default="results/user_attr_vectors")
    parser.add_argument("--output-dir", default="figs/geometry/P4")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--no-plot-title",
        action="store_true",
        help="Omit figure titles on all geometry plots.",
    )
    args = parser.parse_args()

    model_id = args.model_id
    if model_id is None:
        model_config = Path(args.model_config)
        if not model_config.is_absolute():
            model_config = REPO_ROOT / model_config
        with open(model_config, "r", encoding="utf-8") as f:
            model_id = yaml.safe_load(f)["model_id"]

    vectors_dir = Path(args.vectors_dir)
    if not vectors_dir.is_absolute():
        vectors_dir = REPO_ROOT / vectors_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    plot_geometry(
        model_id=model_id,
        vectors_dir=vectors_dir,
        output_dir=output_dir,
        formats=list(dict.fromkeys(args.formats)),
        show_title=not args.no_plot_title,
    )


if __name__ == "__main__":
    main()
