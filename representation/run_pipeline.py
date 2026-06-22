from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from representation.build_vectors import build_vectors
from representation.data import DEFAULT_SPLIT_SEED, DEFAULT_TRAIN_FRACTION
from representation.extract_activations import extract_all_activations
from representation.projection import (
    DEFAULT_BUILD_POS,
    DEFAULT_EVAL_POS,
    DEFAULT_N_BINS,
    compute_projections,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract activations, build Section 6.2 vectors, "
                    "and compute Section 6.3.2 projection artifacts.",
    )
    parser.add_argument("--model-config", default="dimensions/configs/models/qwen3_4b.yaml")
    parser.add_argument("--output-dir", default="results/user_attr_vectors")
    parser.add_argument("--behavioral-dir", default="results/behavioral")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--build-pos", default=DEFAULT_BUILD_POS)
    parser.add_argument("--eval-pos", default=DEFAULT_EVAL_POS)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--skip-projection", action="store_true",
                        help="Skip the projection-analysis stage (Section 6.3.2).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    extract_all_activations(
        model_cfg=model_cfg,
        output_dir=Path(args.output_dir),
        layer=args.layer,
        batch_size=args.batch_size,
        limit=args.limit,
        conditions_filter=args.conditions,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        overwrite=args.overwrite,
    )
    build_vectors(
        model_id=model_cfg["model_id"],
        output_dir=Path(args.output_dir),
        layer=args.layer,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
    )
    if not args.skip_projection:
        compute_projections(
            model_id=model_cfg["model_id"],
            output_dir=Path(args.output_dir),
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

