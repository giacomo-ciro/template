# AGENTS.md

Deep learning pipeline on PyTorch Lightning, configured with Hydra, managed with uv.

## Layout

- `src/template/`: library code (`dataset.py`, `datamodule.py`, `model.py`, `progress_bar.py`, `scheduler.py`, `trainer.py`).
- `scripts/`: entry points only; they compose the library, no training logic.
- `configs/`: Hydra configs. `train.yaml` is the entry point; groups live in subfolders.
- Do not edit `third_party/` (external repos) or `archive/` (old code kept for reference).
- `outputs/` and `data/` are not tracked.

## Conventions

- Every tunable value lives in `configs/`, never hardcoded in `src/`. Directories come from `cfg.paths`.
- Components are built from the full config: `load_x(cfg)` resolves `cfg.x._target_` and calls the class with `cfg`; the class reads its parameters off `cfg.x`. Do not switch to `hydra.utils.instantiate` with keyword arguments.
- Training is step-based (`trainer.max_steps`, `trainer.val_every_n_steps`); do not add epoch-based settings.
- Dataset splits are fixed on disk. The datamodule only caps them (`datamodule.split`) and never reshuffles or re-splits.
- Log keys are prefixed `train/`, `val/`, `test/`.

## Commands

```bash
.venv/bin/python scripts/train.py debug=on   # end-to-end check, run it before finishing any change
.venv/bin/ruff check src scripts
```

Never run `uv sync` or `uv run` without `--group cpu` / `--group gpu`: without the group, uv replaces the installed torch build with the default PyPI one.
