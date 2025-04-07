# Template
A template repository for developing deep learning pipelines.

## Repository structure

```
.
├── archive/        # old code kept for reference
├── checkpoints/    # model checkpoints (not tracked)
├── configs/        # Hydra config files
├── data/           # datasets
├── docs/           # notes, references etc.
├── notebooks/      # jupyter notebooks
├── outputs/        # run artifacts: resolved configs, logs (not tracked)
├── public/         # run artifacts / visualizations to share
├── scripts/        # executable entry points
├── src/            # core packages (trainer, model etc.)
└── third_party/    # external repos
```

## Setup

```bash
uv sync --group cpu   # or --group gpu (CUDA wheels, see [[tool.uv.index]] in pyproject.toml)
```

`dev` tools (ruff, ipykernel) are installed by default.

## Usage

```bash
python scripts/train.py                                 # full run with configs/train.yaml
python scripts/train.py debug=on                        # 10 steps on tiny splits, single-process loading
python scripts/train.py model.hidden_dim=256 seed=0     # override any value
python scripts/train.py -m sweep=example                # sweep defined in configs/sweep/example.yaml
python scripts/train.py -m model.hidden_dim=64,128      # one-off sweep from the CLI
```

## Starting a new project

1. Rename the package: `name` in `pyproject.toml`, `src/template/`, and every `template.` import and `_target_`.
2. Replace `RandomDataset` and `MLP` with your own dataset and model.
