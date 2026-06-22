from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import yaml

from representation.math_utils import cos_sim
from representation.vector_specs import POSITIONS, VECTOR_SPECS

REPO_ROOT = Path(__file__).resolve().parents[1]


SUBCATEGORY_GROUPS: dict[str, list[str]] = {
    "knowledge": ["v_formal", "v_experiential", "v_autodidact"],
    "intent": ["v_curiosity", "v_educational", "v_professional", "v_institutional"],
    "emotion": ["v_valence", "v_arousal"],
    "belief": ["v_empirical", "v_normative", "v_conspiratorial"],
}

CROSS_DIMENSION_VECTORS = [
    "v_formal",
    "v_experiential",
    "v_autodidact",
    "v_curiosity",
    "v_educational",
    "v_professional",
    "v_institutional",
    "v_valence",
    "v_arousal",
    "v_empirical",
    "v_normative",
    "v_conspiratorial",
]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available")
    if device not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    return torch.device(device)


def model_vector_root(vectors_dir: str | Path, model_id: str) -> Path:
    return Path(vectors_dir) / model_id


def _spec_dimension_by_name() -> dict[str, str]:
    return {spec.name: spec.dimension for spec in VECTOR_SPECS}


def load_vector_artifacts(
    *,
    model_root: Path,
    layer: int | None = None,
    device: torch.device | str = "cpu",
    positions: tuple[str, ...] = POSITIONS,
) -> dict[str, dict[str, Any]]:
    """Load the configured user-attribute vector files into memory."""
    device = torch.device(device)
    spec_dims = _spec_dimension_by_name()
    vectors: dict[str, dict[str, Any]] = {}

    for spec in VECTOR_SPECS:
        path = model_root / spec.dimension / f"{spec.name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing vector artifact: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("vector_name") != spec.name:
            raise ValueError(f"{path} has vector_name={artifact.get('vector_name')!r}")
        if layer is not None and artifact.get("layer") != layer:
            raise ValueError(f"{path} has layer={artifact.get('layer')}, expected {layer}")

        artifact_positions = artifact.get("positions")
        if not isinstance(artifact_positions, dict):
            raise ValueError(f"{path} does not contain a positions dict")
        missing = [pos for pos in positions if pos not in artifact_positions]
        if missing:
            raise ValueError(f"{path} missing positions: {missing}")

        vectors[spec.name] = {
            "path": path,
            "dimension": artifact.get("dimension", spec_dims[spec.name]),
            "vector_type": artifact.get("vector_type", spec.vector_type),
            "layer": artifact.get("layer"),
            "positions": {
                pos: artifact_positions[pos].detach().float().to(device)
                for pos in positions
            },
            "metadata": {
                key: value
                for key, value in artifact.items()
                if key not in {"positions"}
            },
        }

    if len(vectors) != len(VECTOR_SPECS):
        raise ValueError(f"Loaded {len(vectors)} vectors, expected {len(VECTOR_SPECS)}")
    return vectors


def cosine_value(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(cos_sim(a, b).detach().cpu().item())


def _available_positions(vectors: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    if not vectors:
        return ()
    expected = set(next(iter(vectors.values()))["positions"])
    for name, artifact in vectors.items():
        current = set(artifact["positions"])
        if current != expected:
            raise ValueError(
                f"Inconsistent positions for {name}: {sorted(current)} != {sorted(expected)}"
            )
    return tuple(pos for pos in POSITIONS if pos in expected)


def cosine_matrix(
    vectors: dict[str, dict[str, Any]],
    names: list[str],
    position: str,
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row_name in names:
        row: list[float] = []
        for col_name in names:
            row.append(cosine_value(
                vectors[row_name]["positions"][position],
                vectors[col_name]["positions"][position],
            ))
        matrix.append(row)
    return matrix


def position_matrix_for_vector(
    artifact: dict[str, Any],
    positions: tuple[str, ...] = POSITIONS,
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row_pos in positions:
        row: list[float] = []
        for col_pos in positions:
            row.append(cosine_value(
                artifact["positions"][row_pos],
                artifact["positions"][col_pos],
            ))
        matrix.append(row)
    return matrix


def matrix_to_long_rows(
    matrix: list[list[float]],
    names: list[str],
    *,
    position: str,
    analysis: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            value = matrix[i][j]
            rows.append({
                "analysis": analysis,
                "position": position,
                "vector_a": left,
                "vector_b": right,
                "cosine": value,
                "abs_cosine": abs(value),
            })
    return rows


def compute_all_pairwise(vectors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = [spec.name for spec in VECTOR_SPECS]
    rows: list[dict[str, Any]] = []
    for position in _available_positions(vectors):
        for left in names:
            for right in names:
                value = cosine_value(
                    vectors[left]["positions"][position],
                    vectors[right]["positions"][position],
                )
                rows.append({
                    "position": position,
                    "vector_a": left,
                    "dimension_a": vectors[left]["dimension"],
                    "vector_type_a": vectors[left]["vector_type"],
                    "vector_b": right,
                    "dimension_b": vectors[right]["dimension"],
                    "vector_type_b": vectors[right]["vector_type"],
                    "cosine": value,
                    "abs_cosine": abs(value),
                })
    return rows


def compute_subcategory_collinearity(vectors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, names in SUBCATEGORY_GROUPS.items():
        for position in _available_positions(vectors):
            for left, right in combinations(names, 2):
                value = cosine_value(
                    vectors[left]["positions"][position],
                    vectors[right]["positions"][position],
                )
                rows.append({
                    "dimension": dimension,
                    "position": position,
                    "vector_a": left,
                    "vector_b": right,
                    "cosine": value,
                    "abs_cosine": abs(value),
                })
    return rows


def compute_subspace_intrinsic_dimension(vectors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimate the effective dimensionality spanned by subcategory directions.

    We use uncentered SVD on unit-normalized subcategory vectors. Centering three
    direction vectors would cap the rank at two, which is not the subspace
    question we want here.
    """
    rows: list[dict[str, Any]] = []
    for dimension, names in SUBCATEGORY_GROUPS.items():
        for position in _available_positions(vectors):
            matrix = torch.stack([
                vectors[name]["positions"][position].float()
                for name in names
            ])
            matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
            singular_values = torch.linalg.svdvals(matrix)
            eigenvalues = singular_values.square()
            ratios = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
            cumulative = ratios.cumsum(dim=0)
            participation_ratio = (
                eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)
            )
            entropy = -(ratios * ratios.clamp_min(1e-12).log()).sum()
            effective_rank = entropy.exp()

            component_ratios = [
                float(ratio.detach().cpu().item())
                for ratio in ratios
            ]
            while len(component_ratios) < 4:
                component_ratios.append(0.0)

            rows.append({
                "dimension": dimension,
                "position": position,
                "n_vectors": len(names),
                "participation_ratio": float(
                    participation_ratio.detach().cpu().item()
                ),
                "effective_rank_entropy": float(effective_rank.detach().cpu().item()),
                "pc1_var": component_ratios[0],
                "pc2_var": component_ratios[1],
                "pc3_var": component_ratios[2],
                "pc4_var": component_ratios[3],
                "top2_var": float(cumulative[min(1, len(cumulative) - 1)].detach().cpu().item()),
            })
    return rows


def compute_cross_dimension(
    vectors: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    rows: list[dict[str, Any]] = []
    matrices: dict[str, list[list[float]]] = {}
    for position in _available_positions(vectors):
        matrix = cosine_matrix(vectors, CROSS_DIMENSION_VECTORS, position)
        matrices[position] = matrix
        rows.extend(matrix_to_long_rows(
            matrix,
            CROSS_DIMENSION_VECTORS,
            position=position,
            analysis="cross_dimension",
        ))
    return rows, matrices


def compute_cross_position(
    vectors: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    rows: list[dict[str, Any]] = []
    matrices: dict[str, list[list[float]]] = {}
    positions = _available_positions(vectors)
    if len(positions) < 2:
        return rows, matrices
    input_positions = ("P1", "P2")
    integrated_positions = ("P3", "P4")
    if not all(position in positions for position in (*input_positions, *integrated_positions)):
        return rows, matrices

    for spec in VECTOR_SPECS:
        artifact = vectors[spec.name]
        matrix = position_matrix_for_vector(artifact, positions=positions)
        matrices[spec.name] = matrix
        input_to_integrated = [
            matrix[positions.index(left)][positions.index(right)]
            for left in input_positions
            for right in integrated_positions
        ]
        rows.append({
            "vector": spec.name,
            "dimension": spec.dimension,
            "vector_type": spec.vector_type,
            "p1_p2_cosine": matrix[0][1],
            "p3_p4_cosine": matrix[2][3],
            "input_to_integrated_mean_cosine": sum(input_to_integrated) / len(input_to_integrated),
            "input_to_integrated_min_cosine": min(input_to_integrated),
            "input_to_integrated_max_cosine": max(input_to_integrated),
        })
    return rows, matrices


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, names: list[str], matrix: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + row)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def make_summary(
    *,
    model_id: str,
    layer: int,
    device: torch.device,
    positions: tuple[str, ...],
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    sub_rows = rows["subcategory_collinearity"]
    intrinsic_rows = rows["subspace_intrinsic_dimension"]
    cross_pos_rows = rows["cross_position_consistency"]
    return {
        "model_id": model_id,
        "layer": layer,
        "device": str(device),
        "n_vectors": len(VECTOR_SPECS),
        "positions": list(positions),
        "row_counts": {key: len(value) for key, value in rows.items()},
        "subcategory_mean_cosine": {
            dimension: _mean([
                row["cosine"] for row in sub_rows
                if row["dimension"] == dimension
            ])
            for dimension in SUBCATEGORY_GROUPS
        },
        "subspace_participation_ratio": {
            dimension: {
                row["position"]: row["participation_ratio"]
                for row in intrinsic_rows
                if row["dimension"] == dimension
            }
            for dimension in SUBCATEGORY_GROUPS
        },
        "subspace_effective_rank_entropy": {
            dimension: {
                row["position"]: row["effective_rank_entropy"]
                for row in intrinsic_rows
                if row["dimension"] == dimension
            }
            for dimension in SUBCATEGORY_GROUPS
        },
        "cross_position_mean_input_to_integrated_cosine": _mean([
            row["input_to_integrated_mean_cosine"] for row in cross_pos_rows
        ]),
    }


def _safe_plot_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _plot_heatmap(
    *,
    matrix: list[list[float]],
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.8), max(4, len(labels) * 0.65)))
    sns.heatmap(
        matrix,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "cosine"},
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_bar(
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    hue_key: str | None,
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if hue_key:
        sns.barplot(data=df, x=x_key, y=y_key, hue=hue_key, ax=ax)
        ax.legend(title=hue_key.replace("_", " ").title(), fontsize=8)
    else:
        sns.barplot(data=df, x=x_key, y=y_key, ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_plots(
    *,
    figs_dir: Path,
    subcategory_rows: list[dict[str, Any]],
    intrinsic_dimension_rows: list[dict[str, Any]],
    cross_dimension_matrices: dict[str, list[list[float]]],
    cross_position_matrices: dict[str, list[list[float]]],
) -> None:
    figs_dir.mkdir(parents=True, exist_ok=True)

    for position, matrix in cross_dimension_matrices.items():
        _plot_heatmap(
            matrix=matrix,
            labels=CROSS_DIMENSION_VECTORS,
            title=f"Cross-Dimension Separation ({position})",
            path=figs_dir / f"cross_dimension_{position}.png",
        )

    for vector_name, matrix in cross_position_matrices.items():
        _plot_heatmap(
            matrix=matrix,
            labels=list(POSITIONS),
            title=f"Cross-Position Consistency: {vector_name}",
            path=figs_dir / "cross_position" / f"{_safe_plot_name(vector_name)}.png",
        )

    for dimension in SUBCATEGORY_GROUPS:
        rows = [
            row | {"pair": f"{row['vector_a']} vs {row['vector_b']}"}
            for row in subcategory_rows
            if row["dimension"] == dimension
        ]
        _plot_bar(
            rows=rows,
            x_key="pair",
            y_key="cosine",
            hue_key="position",
            title=f"{dimension.title()} Sub-Category Collinearity",
            path=figs_dir / f"{dimension}_subcategory_collinearity.png",
        )

    _plot_bar(
        rows=intrinsic_dimension_rows,
        x_key="position",
        y_key="participation_ratio",
        hue_key="dimension",
        title="Subspace Intrinsic Dimension (Participation Ratio)",
        path=figs_dir / "subspace_intrinsic_dimension.png",
    )


def run_geometry_analysis(
    *,
    model_id: str,
    vectors_dir: Path,
    layer: int,
    device: str = "auto",
    positions: tuple[str, ...] = POSITIONS,
    make_plots: bool = True,
    figures_dir: Path | None = None,
) -> Path:
    resolved_device = resolve_device(device)
    model_root = model_vector_root(vectors_dir, model_id)
    output_dir = model_root / "geometry"
    matrices_dir = output_dir / "matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    vectors = load_vector_artifacts(
        model_root=model_root,
        layer=layer,
        device=resolved_device,
        positions=positions,
    )

    all_pairwise = compute_all_pairwise(vectors)
    subcategory = compute_subcategory_collinearity(vectors)
    intrinsic_dimension = compute_subspace_intrinsic_dimension(vectors)
    cross_dimension, cross_dimension_matrices = compute_cross_dimension(vectors)
    cross_position, cross_position_matrices = compute_cross_position(vectors)

    rows_by_name = {
        "all_pairwise_cosines": all_pairwise,
        "subcategory_collinearity": subcategory,
        "subspace_intrinsic_dimension": intrinsic_dimension,
        "cross_dimension_separation": cross_dimension,
        "cross_position_consistency": cross_position,
    }

    for name, rows in rows_by_name.items():
        write_csv(output_dir / f"{name}.csv", rows)

    for position, matrix in cross_dimension_matrices.items():
        write_matrix_csv(
            matrices_dir / f"cross_dimension_{position}.csv",
            CROSS_DIMENSION_VECTORS,
            matrix,
        )
    for vector_name, matrix in cross_position_matrices.items():
        write_matrix_csv(
            matrices_dir / f"cross_position_{_safe_plot_name(vector_name)}.csv",
            list(positions),
            matrix,
        )

    summary = make_summary(
        model_id=model_id,
        layer=layer,
        device=resolved_device,
        positions=positions,
        rows=rows_by_name,
    )
    (output_dir / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if make_plots:
        resolved_figs = (
            figures_dir.resolve()
            if figures_dir is not None
            else (REPO_ROOT / "figs" / "geometry")
        )
        write_plots(
            figs_dir=resolved_figs,
            subcategory_rows=subcategory,
            intrinsic_dimension_rows=intrinsic_dimension,
            cross_dimension_matrices=cross_dimension_matrices,
            cross_position_matrices=cross_position_matrices,
        )
        print(f"Figures saved under {resolved_figs}", flush=True)

    print(f"Geometry analysis complete. Results in {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 6.3.1 geometry analysis.")
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--vectors-dir", default="results/user_attr_vectors")
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Directory for PNG plots (default: <repo>/figs/geometry).",
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--positions",
        nargs="+",
        choices=list(POSITIONS),
        default=list(POSITIONS),
        help="Vector positions to analyze. Use --positions P4 for P4-only geometry.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    run_geometry_analysis(
        model_id=model_cfg["model_id"],
        vectors_dir=Path(args.vectors_dir),
        layer=args.layer,
        device=args.device,
        positions=tuple(args.positions),
        make_plots=not args.no_plots,
        figures_dir=Path(args.figures_dir) if args.figures_dir else None,
    )


if __name__ == "__main__":
    main()
