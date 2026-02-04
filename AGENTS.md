# Repository Guidelines

## Project Structure & Module Organization

- `configs/`: YAML experiment configs (e.g. `configs/cxr_chexpert.yaml`).
- `data/`: dataset loaders and image transforms.
  - `data/dataset/`: local datasets + preprocessing scripts (`prepare_chexpert.py`, `prepare_nih.py`) and `processed/` JSONL.
- `models/`: model components (`vit.py`, `projector.py`, `vivid_model.py`).
- `training/`: losses and trainer (`losses.py`, `trainer.py`).
- `evaluation/`: UMS verifier + metrics (`verifier.py`, `metrics.py`).
- `scripts/`: runnable entry points (`train_cxr.py`, `test_pipeline.py`).
- `profle/`: design docs and schema notes (Markdown).

## Build, Test, and Development Commands

```bash
pip install -r requirements.txt
python data/dataset/prepare_chexpert.py
python data/dataset/prepare_nih.py
python scripts/test_pipeline.py
python scripts/train_cxr.py --debug
python scripts/train_cxr.py --config configs/cxr_chexpert.yaml
python scripts/train_cxr.py --resume outputs/cxr_chexpert/checkpoints/step_5000.pt
```

Notes:
- `scripts/test_pipeline.py` is a smoke test (data loading, model construction, forward pass, verifier). It may prompt to download the LLM; answering “y” can fetch multi‑GB weights.
- Training artifacts default under `outputs/` (see `training.output_dir` in the YAML).

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/variables, `CapWords` for classes.
- Prefer `pathlib.Path` for paths; keep dataset/config paths in YAML (`configs/`) rather than hardcoding.
- No formatter/linter is configured here—match surrounding style and keep changes minimal and readable.

## Testing Guidelines

- No unit-test framework is enforced. Use `python scripts/test_pipeline.py` for quick validation.
- If you add tests/scripts, prefer `test_*.py` naming and keep them runnable from the repo root.

## Commit & Pull Request Guidelines

- This workspace may not include Git metadata; if you initialize Git, use concise, imperative commit messages (e.g., “Add CheXpert preprocessing guardrails”).
- PRs: include a short summary, commands run, and any config/data assumptions; attach key metrics (tables/plots) when changing training/eval behavior.

## Security & Configuration Tips

- Datasets are medical images—treat `data/dataset/**` as sensitive and keep it local.
- Avoid committing generated artifacts such as `outputs/`, `vivid_env/`, and `__pycache__/`; add a `.gitignore` if/when you set up Git.
