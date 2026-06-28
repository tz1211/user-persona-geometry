# User Persona Subspaces Modulate Refusal Behavior in Language Models

This repository accompanies our study of how language models internally represent user personas and how those representations modulate refusal behavior. We show that Knowledge, Intent, Emotion, and Belief are encoded as structured low-dimensional subspaces in activation space, and that directions within these subspaces predict and causally shift refusal behavior and inferred user profiles. Our paper can be found [here](https://openreview.net/forum?id=F9oy1ouIef). 

## Setup

```bash
conda create -n user-persona-geometry python=3.12
conda activate user-persona-geometry
pip install -r requirements.txt
```

The default model is `Qwen/Qwen3-4B`; the default judge is
`Qwen/Qwen3-30B-A3B-Instruct-2507`. Edit
`dimensions/configs/models/qwen3_4b.yaml` or pass `--model-config` to use a
different compatible chat model.

## Data

Download and prepare the refusal benchmark:

```bash
bash scripts/00_prepare_refusal_data.sh
```

This writes JSONL files under `data/benchmarks/refusal/`. The script samples
1,450 prompts: 700 should-refuse prompts from HarmBench and OR-Bench toxic, and
750 should-not-refuse prompts from OR-Bench hard and XSTest safe.

## Reproduction Workflow

Run behavior generation:

```bash
bash scripts/01_generate_behavior.sh
```

Evaluate refusals with a judge:

```bash
bash scripts/02_evaluate_behavior.sh
```

Build activation caches, contrastive persona vectors, and P4-on-P4 projection artifacts:

```bash
bash scripts/03_build_representations.sh
```

Render refusal, geometry, and projection figures:

```bash
bash scripts/04_plot_refusal_and_geometry.sh
```

Run activation steering and forced-choice inferred-user probes:

```bash
bash scripts/05_run_steering.sh
bash scripts/06_run_choice_probe.sh
```

Optional appendix position sweeps:

```bash
bash scripts/07_projection_position_sweep.sh
```

All scripts accept environment-variable overrides such as `SEEDS`,
`MODEL_CONFIG`, `MODEL_ID`, `LIMIT`, `JUDGE_BACKEND`, `BATCH_SIZE`,
`VECTORS_DIR`, `RESULTS_DIR`, and `FIGS_DIR`.

## Figure Map

- Refusal behavior and refusal style: `viz.plot_refusal`
- Persona geometry and dimensionality: `representation.geometry`,
  `viz.plot_geometry`, `viz.plot_geometry_3d`
- Projection-binned refusal trends: `representation.projection`,
  `viz.plot_projection`
- Refusal steering and perplexity: `representation.steering`,
  `viz.plot_steering`
- Inferred-user forced-choice steering: `representation.trait_choice_probe`,
  `viz.plot_trait_choice_probe`

## Output Layout

Generated artifacts are written under:

```text
data/benchmarks/refusal/
results/behavioral/<model-id>/seed_<seed>/
results/user_attr_vectors/<model-id>/
results/steering/<model-id>/
results/trait_choice_probe/<model-id>/
figs/
```

## Citation 
```text
@article{zhou2026user,
  title = {User Persona Subspaces Modulate Refusal Behavior in Language Models},
  author = {Zhou, Yan and Zhang, Shichang and Xiong, Zidi and Lakkaraju, Himabindu},
  journal = {ICML Workshop on Mechanistic Interpretability},
  year = {2026},
}
```

## License 
This project is licensed under the Apache License 2.0 — see the [LICENSE](./LICENSE) file for details. 
