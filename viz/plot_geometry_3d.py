"""3D cross-dimensional geometry figures.

This script is a companion to ``viz.plot_geometry``. It loads the same
user-attribute vector artifacts used to build the cross-dimensional cosine
heatmap, embeds the P4 directions into a shared 3D PCA basis, and renders a
single 3D scene with colored vectors for each behavioral dimension.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from scipy.optimize import minimize
from scipy.special import expit, logit
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams["ps.useafm"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["axes.unicode_minus"] = False

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from representation.geometry import (  # noqa: E402
    CROSS_DIMENSION_VECTORS,
    SUBCATEGORY_GROUPS,
    load_vector_artifacts,
)
from representation.vector_specs import POSITIONS, VECTOR_SPECS, ContrastSpec  # noqa: E402
from viz.plot_geometry import (  # noqa: E402
    DEFAULT_POSITION,
    DIM_LABELS,
    VECTOR_LABELS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DIMENSION_VECTORS: dict[str, list[str]] = {
    "knowledge": SUBCATEGORY_GROUPS["knowledge"],
    "intent": SUBCATEGORY_GROUPS["intent"],
    "emotion": ["v_valence", "v_arousal"],
    "belief": SUBCATEGORY_GROUPS["belief"],
}

PANEL_ORDER = ("knowledge", "intent", "emotion", "belief")
DIMENSION_LABELS = DIM_LABELS | {"emotion": "Emotion"}
DIMENSION_COLORS = {
    "knowledge": "#4C78A8",
    "intent": "#F58518",
    "emotion": "#54A24B",
    "belief": "#B279A2",
}

REFUSAL_MORE_COLOR = "#C44E52"
REFUSAL_LESS_COLOR = "#4C72B0"

# Figure typography (matplotlib + interactive HTML CSS stay loosely matched).
FONT_AXIS_LABEL = 12
FONT_AXIS_TICK = 12
FONT_VECTOR_LABEL = 10
FONT_LEGEND = 12
FONT_SUPTITLE = 17
FONT_HTML_TITLE_PX = 20
FONT_HTML_HINT_PX = 14
FONT_HTML_AXIS_LABEL_PX = 15
FONT_HTML_POINT_LABEL_PX = 16
FONT_HTML_LEGEND_PX = 14
# Fewer major ticks on 3-D axes (matplotlib default is fairly dense).
AXIS_TICK_MAX_BINS = 4
# Extra space between tick numbers and axis line (helps PC1/PC2 tick labels at the corner).
AXIS_TICK_PAD = 6
# mplot3d can misplace PC1 tick labels; a bit more pad on x helps alignment vs PC2/PC3.
AXIS_TICK_PAD_X = 9
# Space between numeric ticks and the PC1/PC2/PC3 titles (points; larger = farther out).
AXIS_LABEL_PAD_XY = 10
AXIS_LABEL_PAD_Z = 10
# Symmetric [-L, L] on each axis: L = max(extent * pad, min_halfspan), where extent is the
# max absolute x/y/z coordinate. Smaller pad tightens the cube so the scene fills more of it.
AXIS_LIMIT_PAD_FACTOR = 1.15
AXIS_LIMIT_MIN_HALFSPAN = 0.55
DEFAULT_ELEV = 10.0
DEFAULT_AZIM = -165.0

# Quiver (matplotlib) and SVG line width for attribute vectors.
VECTOR_ARROW_LINEWIDTH = 3
VECTOR_ARROW_HTML_STROKE_PX = 5.5
# Semi-transparent hull between vectors in the same dimension (matplotlib + HTML).
VECTOR_SUBSPACE_SHADE_ALPHA = 0.1
VECTOR_SUBSPACE_SHADE_HTML_OPACITY = 0.13
# Background grid on 3-D axis planes (lighter grey: higher ``AXIS_GRID_COLOR``, lower alpha).
AXIS_GRID_LINEWIDTH = 0.30
AXIS_GRID_COLOR = "0.88"
AXIS_GRID_ALPHA = 0.10
# Dashed PC guide lines in interactive SVG (lighter than former #999).
HTML_AXIS_GUIDE_COLOR = "#c6c6c6"
HTML_AXIS_GUIDE_OPACITY = 0.42

LABEL_OFFSETS_3D = {
    "v_formal": np.array([-0.17, 0.28, 0.01]),
    "v_experiential": np.array([0.11, -0.14, -0.10]),
    "v_autodidact": np.array([-0.03, 0.10, 0.12]),
    "v_curiosity": np.array([-0.06, 0.02, 0.08]),
    "v_educational": np.array([0.05, -0.05, 0.12]),
    "v_professional": np.array([-0.04, 0.04, 0.03]),
    "v_institutional": np.array([0.04, -0.04, 0.08]),
    "v_valence": np.array([-0.08, 0.04, 0.10]),
    "v_arousal": np.array([0.02, -0.08, -0.08]),
    "v_empirical": np.array([-0.02, -0.04, -0.08]),
    "v_normative": np.array([0.08, -0.08, 0.08]),
    "v_conspiratorial": np.array([0.04, -0.06, -0.08]),
}


@dataclass(frozen=True)
class Embedding3D:
    coordinates: pd.DataFrame
    explained_variance: tuple[float, float, float]
    prompt_projection_matrix: np.ndarray
    scale: float


@dataclass(frozen=True)
class PromptRefusalOverlay:
    vector_deltas: dict[str, float]
    normal: np.ndarray | None
    boundary_offset: float | None
    mean_refusal: float | None
    n_complete_rows: int
    source_path: Path


def _format_axis_coord(value: float, _pos: int) -> str:
    """Compact axis tick strings (-1 not -1.0) to reduce overlap between PC1/PC2 ticks in 3-D."""
    x = float(value)
    if abs(x) < 1e-9:
        return "0"
    r = round(x, 8)
    if abs(r - round(r)) < 1e-6:
        i = int(round(r))
        return str(i)
    text = f"{r:.3g}"
    return text


def _model_root(vectors_dir: Path, model_id: str) -> Path:
    return vectors_dir / model_id


def _spec_by_name() -> dict[str, ContrastSpec]:
    return {spec.name: spec for spec in VECTOR_SPECS}


def _load_position_vectors(
    *,
    model_root: Path,
    position: str,
    layer: int | None,
) -> dict[str, torch.Tensor]:
    if position not in POSITIONS:
        raise ValueError(f"--position must be one of {POSITIONS}; got {position!r}")
    artifacts = load_vector_artifacts(model_root=model_root, layer=layer, device="cpu")
    return {
        name: artifacts[name]["positions"][position].detach().float().cpu()
        for name in CROSS_DIMENSION_VECTORS
    }


def _embed_vectors_3d(vectors: dict[str, torch.Tensor]) -> Embedding3D:
    names = list(CROSS_DIMENSION_VECTORS)
    matrix = torch.stack([vectors[name] for name in names]).float()
    matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x = matrix.cpu().numpy()

    u, singular_values, vh = np.linalg.svd(x, full_matrices=False)
    basis = vh[:3]
    if basis.shape[0] < 3:
        basis = np.pad(basis, ((0, 3 - basis.shape[0]), (0, 0)))
    coords = x @ basis.T

    # Stabilize signs across runs: make the largest loading on each axis positive.
    signs = np.ones(3, dtype=float)
    for axis in range(3):
        idx = int(np.argmax(np.abs(coords[:, axis])))
        if coords[idx, axis] < 0:
            coords[:, axis] *= -1
            signs[axis] = -1.0

    energy = np.square(singular_values)
    ratios = energy / max(float(energy.sum()), 1e-12)
    explained = tuple(float(ratios[i]) if i < len(ratios) else 0.0 for i in range(3))

    norms = np.linalg.norm(coords, axis=1)
    scale = float(norms.max()) if norms.size else 1.0
    if scale > 0:
        coords = coords / scale

    prompt_projection_matrix = np.zeros((len(names), 3), dtype=np.float64)
    for axis in range(min(3, len(singular_values))):
        denom = float(singular_values[axis])
        if denom > 1e-12:
            prompt_projection_matrix[:, axis] = signs[axis] * u[:, axis] / denom
    if scale > 0:
        prompt_projection_matrix = prompt_projection_matrix / scale

    rows = []
    for name, coord in zip(names, coords, strict=True):
        rows.append({
            "vector_name": name,
            "label": VECTOR_LABELS.get(name, name),
            "dimension": _dimension_for_vector(name),
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
        })
    return Embedding3D(
        coordinates=pd.DataFrame(rows),
        explained_variance=explained,
        prompt_projection_matrix=prompt_projection_matrix,
        scale=scale,
    )


def _dimension_for_vector(vector_name: str) -> str:
    if vector_name in {"v_valence", "v_arousal"}:
        return "emotion"
    for dimension, names in SUBCATEGORY_GROUPS.items():
        if vector_name in names:
            return dimension
    return _spec_by_name()[vector_name].dimension


def _fit_logistic_region(
    z: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, float, float] | None:
    if z.shape[0] < 20 or len(np.unique(y)) < 2:
        return None

    mu = z.mean(axis=0)
    sigma = z.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    zs = (z - mu) / sigma

    y_mean = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    init = np.zeros(zs.shape[1] + 1, dtype=np.float64)
    init[0] = float(logit(y_mean))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = theta[0]
        beta = theta[1:]
        eta = intercept + zs @ beta
        loss = np.logaddexp(0.0, eta).mean() - np.mean(y * eta)
        loss += 0.5 * ridge * float(beta @ beta)
        residual = expit(eta) - y
        grad = np.empty_like(theta)
        grad[0] = float(residual.mean())
        grad[1:] = zs.T @ residual / len(y) + ridge * beta
        return float(loss), grad

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=init,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        return None

    theta = result.x
    normal = theta[1:] / sigma
    intercept = float(theta[0] - np.sum(theta[1:] * mu / sigma))
    boundary_offset = float(logit(y_mean) - intercept)
    return normal, boundary_offset, y_mean


def _projection_refusal_delta(group: pd.DataFrame, metric: str) -> float | None:
    sub = group[["projection", metric]].copy()
    sub["projection"] = pd.to_numeric(sub["projection"], errors="coerce")
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 20 or sub[metric].nunique() < 2:
        return None
    lo = float(sub["projection"].quantile(0.25))
    hi = float(sub["projection"].quantile(0.75))
    low_refusal = sub[sub["projection"] <= lo][metric]
    high_refusal = sub[sub["projection"] >= hi][metric]
    if low_refusal.empty or high_refusal.empty:
        return None
    return float(high_refusal.mean() - low_refusal.mean())


def _read_prompt_refusal_overlay(
    *,
    model_root: Path,
    build_pos: str,
    eval_pos: str,
    metric: str,
    embedding: Embedding3D,
) -> PromptRefusalOverlay | None:
    if metric == "none":
        return None

    path = (
        model_root
        / "projection"
        / f"{eval_pos}_onto_{build_pos}"
        / "projections_held_out.csv"
    )
    if not path.exists():
        print(f"[INFO] no refusal overlay: missing {path}", flush=True)
        return None

    df = pd.read_csv(path)
    if metric not in df.columns:
        print(f"[INFO] no refusal overlay: {path} lacks {metric}", flush=True)
        return None

    if "build_pos" in df.columns:
        df = df[df["build_pos"] == build_pos]
    if "eval_pos" in df.columns:
        df = df[df["eval_pos"] == eval_pos]
    if df.empty:
        print(
            f"[INFO] no refusal overlay: no rows for build_pos={build_pos}, eval_pos={eval_pos}",
            flush=True,
        )
        return None

    vector_deltas: dict[str, float] = {}
    for vector_name, group in df.groupby("vector_name"):
        if vector_name not in CROSS_DIMENSION_VECTORS:
            continue
        delta = _projection_refusal_delta(group, metric)
        if delta is not None:
            vector_deltas[vector_name] = delta

    keys = ["prompt_id", "subtask_id", "condition_id", "seed"]
    available_keys = [key for key in keys if key in df.columns]
    pivot = df.pivot_table(
        index=available_keys,
        columns="vector_name",
        values="projection",
        aggfunc="first",
    )
    y = df.groupby(available_keys)[metric].first()
    complete = pivot.reindex(columns=CROSS_DIMENSION_VECTORS).join(y.rename(metric))
    complete = complete.dropna(subset=list(CROSS_DIMENSION_VECTORS) + [metric])

    normal = None
    boundary_offset = None
    mean_refusal = None
    if not complete.empty:
        p = complete.loc[:, CROSS_DIMENSION_VECTORS].to_numpy(dtype=np.float64)
        z = p @ embedding.prompt_projection_matrix
        labels = complete[metric].to_numpy(dtype=np.float64)
        fit = _fit_logistic_region(z, labels)
        if fit is not None:
            normal, boundary_offset, mean_refusal = fit

    if vector_deltas:
        print(
            f"[INFO] refusal overlay uses prompt-level {metric}: "
            "top-quartile projection rate minus bottom-quartile projection rate",
            flush=True,
        )
    if normal is not None:
        print(
            f"[INFO] fitted 3D prompt refusal boundary from {len(complete)} complete projection rows",
            flush=True,
        )
    elif not complete.empty:
        print(
            "[INFO] skipped 3D prompt refusal boundary: complete projection rows "
            "do not contain enough label variation",
            flush=True,
        )

    return PromptRefusalOverlay(
        vector_deltas=vector_deltas,
        normal=normal,
        boundary_offset=boundary_offset,
        mean_refusal=mean_refusal,
        n_complete_rows=len(complete),
        source_path=path,
    )


def _draw_reference_axes(ax: plt.Axes, axis_limit: float) -> None:
    style = {"color": "0.78", "linewidth": 0.8, "linestyle": ":", "zorder": 0}
    ax.plot([-axis_limit, axis_limit], [0, 0], [0, 0], **style)
    ax.plot([0, 0], [-axis_limit, axis_limit], [0, 0], **style)
    ax.plot([0, 0], [0, 0], [-axis_limit, axis_limit], **style)


def _draw_arrow(
    ax: plt.Axes,
    xyz: np.ndarray,
    *,
    color: str,
    alpha: float,
    linewidth: float,
    arrow_length_ratio: float = 0.11,
) -> None:
    ax.quiver(
        0.0,
        0.0,
        0.0,
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        arrow_length_ratio=arrow_length_ratio,
        normalize=False,
    )


def _draw_subspace_wedge(
    ax: plt.Axes,
    endpoints: list[np.ndarray],
    *,
    color: str,
) -> None:
    if len(endpoints) < 2:
        return
    faces = [
        [np.zeros(3), endpoints[i], endpoints[j]]
        for i in range(len(endpoints))
        for j in range(i + 1, len(endpoints))
    ]
    poly = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=color,
        linewidths=0.35,
        alpha=VECTOR_SUBSPACE_SHADE_ALPHA,
    )
    ax.add_collection3d(poly)


def _draw_refusal_cue(
    ax: plt.Axes,
    xyz: np.ndarray,
    *,
    delta: float,
    max_abs_delta: float,
) -> None:
    if not np.isfinite(delta) or max_abs_delta <= 0:
        return
    high_side = xyz if delta >= 0 else -xyz
    low_side = -xyz if delta >= 0 else xyz
    size = 32.0 + 180.0 * min(abs(delta) / max_abs_delta, 1.0)
    ax.scatter(
        [high_side[0]],
        [high_side[1]],
        [high_side[2]],
        s=size,
        color=REFUSAL_MORE_COLOR,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.42,
        depthshade=False,
    )
    ax.scatter(
        [low_side[0]],
        [low_side[1]],
        [low_side[2]],
        s=size * 0.72,
        color=REFUSAL_LESS_COLOR,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.30,
        depthshade=False,
    )


def _orthonormal_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal_hat = normal / np.linalg.norm(normal)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(float(candidate @ normal_hat)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    e1 = candidate - float(candidate @ normal_hat) * normal_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal_hat, e1)
    return e1, e2


def _draw_refusal_region(
    ax: plt.Axes,
    *,
    normal: np.ndarray | None,
    boundary_offset: float | None,
    axis_limit: float,
) -> None:
    if normal is None or boundary_offset is None:
        return
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return

    normal_hat = normal / norm
    signed_distance = float(boundary_offset / norm)
    signed_distance = float(np.clip(signed_distance, -0.72 * axis_limit, 0.72 * axis_limit))
    center = normal_hat * signed_distance
    e1, e2 = _orthonormal_plane_basis(normal)

    grid = np.linspace(-axis_limit, axis_limit, 2)
    uu, vv = np.meshgrid(grid, grid)
    plane = center[:, None, None] + e1[:, None, None] * uu + e2[:, None, None] * vv
    ax.plot_surface(
        plane[0],
        plane[1],
        plane[2],
        color="0.25",
        alpha=0.10,
        linewidth=0,
        shade=False,
        zorder=0,
    )

    cue_len = axis_limit * 0.92
    high = normal_hat * cue_len
    low = -normal_hat * cue_len
    ax.scatter(
        [high[0]],
        [high[1]],
        [high[2]],
        s=240,
        color=REFUSAL_MORE_COLOR,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.18,
        depthshade=False,
    )
    ax.scatter(
        [low[0]],
        [low[1]],
        [low[2]],
        s=220,
        color=REFUSAL_LESS_COLOR,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.16,
        depthshade=False,
    )


def _refusal_plane_mesh(
    *,
    normal: np.ndarray | None,
    boundary_offset: float | None,
    axis_limit: float,
) -> np.ndarray | None:
    if normal is None or boundary_offset is None:
        return None
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        return None

    normal_hat = normal / norm
    signed_distance = float(boundary_offset / norm)
    signed_distance = float(np.clip(signed_distance, -0.72 * axis_limit, 0.72 * axis_limit))
    center = normal_hat * signed_distance
    e1, e2 = _orthonormal_plane_basis(normal)
    span = axis_limit * 1.20
    return np.array([
        center - span * e1 - span * e2,
        center + span * e1 - span * e2,
        center + span * e1 + span * e2,
        center - span * e1 + span * e2,
    ])


def _plot_all_dimensions_scene(
    ax: plt.Axes,
    coords: pd.DataFrame,
    *,
    axis_limit: float,
    refusal_overlay: PromptRefusalOverlay | None,
    max_abs_delta: float,
    elev: float,
    azim: float,
    show_vector_labels: bool = True,
) -> None:
    _draw_reference_axes(ax, axis_limit)

    for dimension in PANEL_ORDER:
        color = DIMENSION_COLORS[dimension]
        dim_vectors = set(DIMENSION_VECTORS[dimension])
        highlighted = coords[coords["vector_name"].isin(dim_vectors)]
        endpoints: list[np.ndarray] = []
        for row in highlighted.itertuples(index=False):
            endpoints.append(np.array([row.x, row.y, row.z], dtype=float))
        _draw_subspace_wedge(ax, endpoints, color=color)

    for dimension in PANEL_ORDER:
        color = DIMENSION_COLORS[dimension]
        dim_vectors = set(DIMENSION_VECTORS[dimension])
        highlighted = coords[coords["vector_name"].isin(dim_vectors)]
        for row in highlighted.itertuples(index=False):
            xyz = np.array([row.x, row.y, row.z], dtype=float)
            _draw_arrow(
                ax,
                xyz,
                color=color,
                alpha=0.94,
                linewidth=VECTOR_ARROW_LINEWIDTH,
            )
            ax.scatter(
                [row.x],
                [row.y],
                [row.z],
                s=34,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                depthshade=False,
            )
            if refusal_overlay is not None and row.vector_name in refusal_overlay.vector_deltas:
                _draw_refusal_cue(
                    ax,
                    xyz,
                    delta=refusal_overlay.vector_deltas[row.vector_name],
                    max_abs_delta=max_abs_delta,
                )
            label_xyz = xyz * 1.11 + LABEL_OFFSETS_3D.get(row.vector_name, np.zeros(3))
            if show_vector_labels:
                ax.text(
                    float(label_xyz[0]),
                    float(label_xyz[1]),
                    float(label_xyz[2]),
                    str(row.label),
                    fontsize=FONT_VECTOR_LABEL,
                    color="0.12",
                    ha="center",
                    va="center",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.42,
                        "pad": 0.4,
                    },
                )

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linewidth=AXIS_GRID_LINEWIDTH, color=AXIS_GRID_COLOR, alpha=AXIS_GRID_ALPHA)
    # One shared symmetric tick set for all axes (MaxNLocator per-axis can pick
    # mismatched x vs y vs z ticks on 3-D axes, which makes PC1 look "off").
    _helper = MaxNLocator(nbins=AXIS_TICK_MAX_BINS, min_n_ticks=3)
    tick_vals = np.asarray(_helper.tick_values(-axis_limit, axis_limit), dtype=float)
    tick_vals = tick_vals[(tick_vals >= -axis_limit - 1e-9) & (tick_vals <= axis_limit + 1e-9)]
    tick_vals = np.unique(tick_vals)
    tick_locator = FixedLocator(tick_vals)

    ax.tick_params(axis="x", labelsize=FONT_AXIS_TICK, pad=AXIS_TICK_PAD_X)
    ax.tick_params(axis="y", labelsize=FONT_AXIS_TICK, pad=AXIS_TICK_PAD)
    ax.tick_params(axis="z", labelsize=FONT_AXIS_TICK, pad=AXIS_TICK_PAD)
    for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis_obj.set_major_locator(tick_locator)
        axis_obj.get_offset_text().set_visible(False)

    compact_fmt = FuncFormatter(_format_axis_coord)
    ax.xaxis.set_major_formatter(compact_fmt)
    ax.yaxis.set_major_formatter(compact_fmt)
    ax.zaxis.set_major_formatter(compact_fmt)


def _plot_interactive_html(
    *,
    coords: pd.DataFrame,
    embedding: Embedding3D,
    refusal_overlay: PromptRefusalOverlay | None,
    output_dir: Path,
    position: str,
    axis_limit: float,
    max_abs_delta: float,
    show_title: bool,
    show_vector_labels: bool = True,
) -> None:
    ev = embedding.explained_variance
    scene: dict[str, object] = {
        "title": f"Cross-Dimension Attribute Geometry ({position})" if show_title else "",
        "axisLimit": axis_limit,
        "explainedVariance": ev,
        "dimensions": [],
        "refusal": {"more": [], "less": [], "plane": None},
    }

    scene["refusal"]["plane"] = None  # type: ignore[index]

    for dimension in PANEL_ORDER:
        color = DIMENSION_COLORS[dimension]
        dim_vectors = set(DIMENSION_VECTORS[dimension])
        highlighted = coords[coords["vector_name"].isin(dim_vectors)]

        vectors: list[dict[str, object]] = []
        points: list[list[float]] = []
        for row in highlighted.itertuples(index=False):
            xyz = [float(row.x), float(row.y), float(row.z)]
            offset = LABEL_OFFSETS_3D.get(row.vector_name, np.zeros(3)).tolist()
            points.append(xyz)
            vectors.append({"label": str(row.label), "xyz": xyz, "labelOffset": offset})

        triangles: list[list[list[float]]] = []
        if len(highlighted) >= 2:
            for left in range(len(points)):
                for right in range(left + 1, len(points)):
                    triangles.append([[0.0, 0.0, 0.0], points[left], points[right]])

        scene["dimensions"].append({  # type: ignore[union-attr]
            "name": DIMENSION_LABELS[dimension],
            "color": color,
            "vectors": vectors,
            "triangles": triangles,
        })

    if refusal_overlay is not None and refusal_overlay.vector_deltas and max_abs_delta > 0:
        more: list[dict[str, object]] = []
        less: list[dict[str, object]] = []
        for row in coords.itertuples(index=False):
            delta = refusal_overlay.vector_deltas.get(row.vector_name)
            if delta is None:
                continue
            xyz = np.array([row.x, row.y, row.z], dtype=float)
            more_xyz = xyz if delta >= 0 else -xyz
            less_xyz = -xyz if delta >= 0 else xyz
            radius = 7.0 + 23.0 * min(abs(delta) / max_abs_delta, 1.0)
            label = f"{row.label}: delta={delta:.3f}"
            more.append({"xyz": more_xyz.tolist(), "radius": radius, "label": label})
            less.append({"xyz": less_xyz.tolist(), "radius": radius * 0.85, "label": label})
        scene["refusal"]["more"] = more  # type: ignore[index]
        scene["refusal"]["less"] = less  # type: ignore[index]

    scene["showVectorLabels"] = show_vector_labels
    scene_json = json.dumps(scene)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{scene['title'] or 'Cross-Dimension Attribute Geometry'}</title>
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; color: #242424; background: #fff; }}
  #wrap {{ height: 100vh; display: flex; flex-direction: column; }}
  #bar {{ display: flex; align-items: center; gap: 14px; padding: 10px 14px; border-bottom: 1px solid #ddd; }}
  #title {{ font-weight: 700; font-size: {FONT_HTML_TITLE_PX}px; margin-right: auto; }}
  #hint {{ color: #666; font-size: {FONT_HTML_HINT_PX}px; }}
  button {{ border: 1px solid #bbb; background: #f7f7f7; border-radius: 5px; padding: 5px 9px; cursor: pointer; }}
  svg {{ flex: 1; width: 100%; min-height: 0; touch-action: none; cursor: grab; }}
  svg.dragging {{ cursor: grabbing; }}
  .axis {{ stroke: {HTML_AXIS_GUIDE_COLOR}; stroke-dasharray: 4 4; stroke-width: 1.3; opacity: {HTML_AXIS_GUIDE_OPACITY}; }}
  .label {{ font-size: {FONT_HTML_POINT_LABEL_PX}px; paint-order: stroke; stroke: white; stroke-width: 4px; stroke-linejoin: round; }}
  .axis-label {{ font-size: {FONT_HTML_AXIS_LABEL_PX}px; fill: #555; paint-order: stroke; stroke: white; stroke-width: 4px; }}
  .legend text {{ font-size: {FONT_HTML_LEGEND_PX}px; dominant-baseline: middle; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="bar">
    <div id="title"></div>
    <div id="hint">Drag to rotate · Wheel to zoom</div>
    <button id="reset">Reset view</button>
  </div>
  <svg id="scene" role="img"></svg>
</div>
<script>
const DATA = {scene_json};
const svg = document.getElementById('scene');
document.getElementById('title').textContent = DATA.title || 'Cross-Dimension Attribute Geometry';
let yaw = -2.25, pitch = 0.35, zoom = 260;
let dragging = false, lastX = 0, lastY = 0;

function rot(p) {{
  const x = p[0], y = p[1], z = p[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * y;
  const y1 = -sy * x + cy * y;
  const z1 = z;
  return [x1, cp * y1 - sp * z1, sp * y1 + cp * z1];
}}

function project(p, w, h) {{
  const r = rot(p);
  return {{x: w / 2 + zoom * r[0], y: h / 2 - zoom * r[1], depth: r[2]}};
}}

function clear() {{
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}}

function el(name, attrs = {{}}, parent = svg) {{
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent.appendChild(node);
  return node;
}}

function drawArrow(group, a, b, color, depth) {{
  el('line', {{x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: color, 'stroke-width': {VECTOR_ARROW_HTML_STROKE_PX}, 'stroke-linecap': 'round', opacity: 0.92}}, group);
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const len = 13;
  const p1 = [b.x, b.y];
  const p2 = [b.x - len * Math.cos(angle - 0.42), b.y - len * Math.sin(angle - 0.42)];
  const p3 = [b.x - len * Math.cos(angle + 0.42), b.y - len * Math.sin(angle + 0.42)];
  el('polygon', {{points: `${{p1[0]}},${{p1[1]}} ${{p2[0]}},${{p2[1]}} ${{p3[0]}},${{p3[1]}}`, fill: color, opacity: 0.95}}, group);
}}

function render() {{
  clear();
  const w = svg.clientWidth || 1000, h = svg.clientHeight || 720;
  svg.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`);
  const layers = [];

  const axes = [
    {{a: [-DATA.axisLimit,0,0], b: [DATA.axisLimit,0,0], label: `PC1 (${{Math.round(DATA.explainedVariance[0]*100)}}%)`}},
    {{a: [0,-DATA.axisLimit,0], b: [0,DATA.axisLimit,0], label: `PC2 (${{Math.round(DATA.explainedVariance[1]*100)}}%)`}},
    {{a: [0,0,-DATA.axisLimit], b: [0,0,DATA.axisLimit], label: `PC3 (${{Math.round(DATA.explainedVariance[2]*100)}}%)`}},
  ];
  for (const axis of axes) {{
    const a = project(axis.a, w, h), b = project(axis.b, w, h);
    layers.push({{depth: (a.depth+b.depth)/2 - 20, draw: g => {{
      el('line', {{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'axis'}}, g);
      el('text', {{x:b.x+6,y:b.y-6,class:'axis-label'}}, g).textContent = axis.label;
    }}}});
  }}

  const plane = DATA.refusal.plane;
  if (plane) {{
    const pp = plane.map(p => project(p, w, h));
    layers.push({{depth: pp.reduce((s,p)=>s+p.depth,0)/pp.length - 10, draw: g => {{
      el('polygon', {{points: pp.map(p => `${{p.x}},${{p.y}}`).join(' '), fill:'#777', opacity:'0.16'}}, g);
    }}}});
  }}

  for (const dim of DATA.dimensions) {{
    for (const tri of dim.triangles) {{
      const pts = tri.map(p => project(p, w, h));
      layers.push({{depth: pts.reduce((s,p)=>s+p.depth,0)/3 - 3, draw: g => {{
        el('polygon', {{points: pts.map(p => `${{p.x}},${{p.y}}`).join(' '), fill: dim.color, opacity: '{VECTOR_SUBSPACE_SHADE_HTML_OPACITY}', stroke: dim.color, 'stroke-width': 0.6, 'stroke-opacity': 0.22}}, g);
      }}}});
    }}
    for (const vec of dim.vectors) {{
      const a = project([0,0,0], w, h), b = project(vec.xyz, w, h);
      const labelPoint = vec.xyz.map((v, i) => v * 1.11 + (vec.labelOffset ? vec.labelOffset[i] : 0));
      const lp = project(labelPoint, w, h);
      layers.push({{depth: (a.depth+b.depth)/2, draw: g => {{
        drawArrow(g, a, b, dim.color);
        el('circle', {{cx:b.x,cy:b.y,r:5.5,fill:dim.color,stroke:'white','stroke-width':1.2}}, g);
        if (DATA.showVectorLabels) {{
          el('text', {{x:lp.x,y:lp.y,fill:'#222',class:'label','text-anchor':'middle'}}, g).textContent = vec.label;
        }}
      }}}});
    }}
  }}

  for (const marker of DATA.refusal.less) {{
    const p = project(marker.xyz, w, h);
    layers.push({{depth: p.depth + 0.5, draw: g => {{
      el('circle', {{cx:p.x,cy:p.y,r:marker.radius,fill:'{REFUSAL_LESS_COLOR}',opacity:'0.34',stroke:'white','stroke-width':1}}, g);
      el('title', {{}}, g).textContent = marker.label;
    }}}});
  }}
  for (const marker of DATA.refusal.more) {{
    const p = project(marker.xyz, w, h);
    layers.push({{depth: p.depth + 1, draw: g => {{
      el('circle', {{cx:p.x,cy:p.y,r:marker.radius,fill:'{REFUSAL_MORE_COLOR}',opacity:'0.42',stroke:'white','stroke-width':1}}, g);
      el('title', {{}}, g).textContent = marker.label;
    }}}});
  }}

  layers.sort((a,b) => a.depth - b.depth);
  for (const layer of layers) layer.draw(el('g'));

  // Center legend under the SVG (nominal legend width ~= 630px built below).
  const legendX = Math.max(10, (w - 630) / 2);
  const legend = el('g', {{class:'legend', transform:`translate(${{legendX}},${{h - 92}})`}});
  let lx = 0, ly = 0;
  for (const dim of DATA.dimensions) {{
    el('line', {{x1:lx,y1:ly,x2:lx+24,y2:ly,stroke:dim.color,'stroke-width':5,'stroke-linecap':'round'}}, legend);
    el('text', {{x:lx+32,y:ly,fill:'#222'}}, legend).textContent = dim.name;
    lx += 145;
  }}
  lx = 0; ly = 28;
  el('circle', {{cx:lx+9,cy:ly,r:9,fill:'{REFUSAL_MORE_COLOR}',opacity:'0.45'}}, legend);
  el('text', {{x:lx+28,y:ly,fill:'#222'}}, legend).textContent = 'More refusal';
  lx += 170;
  el('circle', {{cx:lx+9,cy:ly,r:9,fill:'{REFUSAL_LESS_COLOR}',opacity:'0.38'}}, legend);
  el('text', {{x:lx+28,y:ly,fill:'#222'}}, legend).textContent = 'Less refusal';
  lx += 170;
  el('circle', {{cx:lx+7,cy:ly,r:4,fill:'#555',opacity:'0.38'}}, legend);
  el('text', {{x:lx+22,y:ly,fill:'#222'}}, legend).textContent = 'Small gap';
  lx += 125;
  el('circle', {{cx:lx+11,cy:ly,r:11,fill:'#555',opacity:'0.38'}}, legend);
  el('text', {{x:lx+30,y:ly,fill:'#222'}}, legend).textContent = 'Large gap';
}}

svg.addEventListener('pointerdown', e => {{
  dragging = true; lastX = e.clientX; lastY = e.clientY; svg.classList.add('dragging'); svg.setPointerCapture(e.pointerId);
}});
svg.addEventListener('pointermove', e => {{
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.008;
  pitch += (e.clientY - lastY) * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  lastX = e.clientX; lastY = e.clientY;
  render();
}});
svg.addEventListener('pointerup', e => {{ dragging = false; svg.classList.remove('dragging'); }});
svg.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.001);
  zoom = Math.max(120, Math.min(620, zoom));
  render();
}}, {{passive:false}});
document.getElementById('reset').addEventListener('click', () => {{ yaw = -2.25; pitch = 0.35; zoom = 260; render(); }});
window.addEventListener('resize', render);
render();
</script>
</body>
</html>
"""

    out_path = output_dir / "cross_dimension_3d.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[FIG] wrote {out_path}", flush=True)


def plot_cross_dimension_3d(
    *,
    model_id: str,
    vectors_dir: Path,
    output_dir: Path,
    formats: list[str],
    position: str = DEFAULT_POSITION,
    layer: int | None = None,
    refusal_metric: str = "llm_refused",
    elev: float = DEFAULT_ELEV,
    azim: float = DEFAULT_AZIM,
    interactive_html: bool = True,
    show_title: bool = True,
    show_vector_labels: bool = True,
) -> None:
    model_root = _model_root(vectors_dir, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    vectors = _load_position_vectors(
        model_root=model_root,
        position=position,
        layer=layer,
    )
    embedding = _embed_vectors_3d(vectors)
    coords = embedding.coordinates
    refusal_overlay = _read_prompt_refusal_overlay(
        model_root=model_root,
        build_pos=position,
        eval_pos=position,
        metric=refusal_metric,
        embedding=embedding,
    )
    vector_deltas = refusal_overlay.vector_deltas if refusal_overlay is not None else {}
    max_abs_delta = max((abs(v) for v in vector_deltas.values()), default=0.0)

    extent = float(coords[["x", "y", "z"]].abs().to_numpy().max())
    axis_limit = max(AXIS_LIMIT_MIN_HALFSPAN, extent * AXIS_LIMIT_PAD_FACTOR)
    fig = plt.figure(figsize=(8.6, 7.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _plot_all_dimensions_scene(
        ax,
        coords,
        axis_limit=axis_limit,
        refusal_overlay=refusal_overlay,
        max_abs_delta=max_abs_delta,
        elev=elev,
        azim=azim,
        show_vector_labels=show_vector_labels,
    )

    ev = embedding.explained_variance
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.0f}%)", labelpad=AXIS_LABEL_PAD_XY, fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.0f}%)", labelpad=AXIS_LABEL_PAD_XY, fontsize=FONT_AXIS_LABEL)
    ax.set_zlabel(f"PC3 ({ev[2] * 100:.0f}%)", labelpad=AXIS_LABEL_PAD_Z, fontsize=FONT_AXIS_LABEL)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=DIMENSION_COLORS[dim],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=5,
            label=DIMENSION_LABELS[dim],
        )
        for dim in PANEL_ORDER
    ]
    if vector_deltas:
        handles.extend([
            mlines.Line2D(
                [],
                [],
                color=REFUSAL_MORE_COLOR,
                marker="o",
                linestyle="",
                markersize=7,
                alpha=0.55,
                label="More refusal",
            ),
            mlines.Line2D(
                [],
                [],
                color=REFUSAL_LESS_COLOR,
                marker="o",
                linestyle="",
                markersize=7,
                alpha=0.45,
                label="Less refusal",
            ),
            mlines.Line2D(
                [],
                [],
                color="0.35",
                marker="o",
                linestyle="",
                markersize=4,
                alpha=0.45,
                label="Small gap",
            ),
            mlines.Line2D(
                [],
                [],
                color="0.35",
                marker="o",
                linestyle="",
                markersize=10,
                alpha=0.45,
                label="Large gap",
            ),
        ])

    if show_title:
        fig.suptitle(
            "Cross-Dimension Attribute Geometry",
            fontsize=FONT_SUPTITLE,
            fontweight="bold",
            x=0.5,
            ha="center",
            y=0.85,
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, 0.10),
        bbox_transform=fig.transFigure,
        borderaxespad=0.0,
    )
    # Symmetric horizontal margins so the 3-D axes centroid sits at figure x≈0.5 (matching
    # ``suptitle`` / legend). Unequal left/right (e.g. right=0.90, left=0.02) shifts the axes
    # left while titles stay centered, so they appear offset to the right of the graphic.
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.17)

    out_base = output_dir / "cross_dimension_3d.png"
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[FIG] wrote {out_path}", flush=True)
    plt.close(fig)

    if interactive_html:
        _plot_interactive_html(
            coords=coords,
            embedding=embedding,
            refusal_overlay=refusal_overlay,
            output_dir=output_dir,
            position=position,
            axis_limit=axis_limit,
            max_abs_delta=max_abs_delta,
            show_title=show_title,
            show_vector_labels=show_vector_labels,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render P4 3D cross-dimensional geometry.")
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--vectors-dir", default="results/user_attr_vectors")
    parser.add_argument("--output-dir", default="figs/geometry/P4")
    parser.add_argument(
        "--position",
        default=DEFAULT_POSITION,
        choices=list(POSITIONS),
        help="Vector position to embed.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer consistency check for vector artifacts.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--refusal-metric",
        default="llm_refused",
        choices=["llm_refused", "keyword_refused", "none"],
        help="Optional endpoint cue for where refusal is higher/lower.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=DEFAULT_ELEV,
        help="3D camera elevation in degrees.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=DEFAULT_AZIM,
        help="3D camera azimuth in degrees.",
    )
    parser.add_argument(
        "--no-plot-title",
        action="store_true",
        help="Omit the figure title and panel titles.",
    )
    parser.add_argument(
        "--no-interactive-html",
        action="store_true",
        help="Do not write the freely rotatable Plotly HTML figure.",
    )
    parser.add_argument(
        "--no-vector-labels",
        action="store_true",
        help="Do not draw text labels at vector tips (matplotlib + HTML).",
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

    plot_cross_dimension_3d(
        model_id=model_id,
        vectors_dir=vectors_dir,
        output_dir=output_dir,
        formats=list(dict.fromkeys(args.formats)),
        position=args.position,
        layer=args.layer,
        refusal_metric=args.refusal_metric,
        elev=args.elev,
        azim=args.azim,
        interactive_html=not args.no_interactive_html,
        show_title=not args.no_plot_title,
        show_vector_labels=not args.no_vector_labels,
    )


if __name__ == "__main__":
    main()
