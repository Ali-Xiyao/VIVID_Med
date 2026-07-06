# Repository Guidelines

## Project Structure & Module Organization

- `configs/`: YAML experiment configs (e.g. `configs/cxr_chexpert.yaml`).
- `data/`: dataset loaders and image transforms.
  - `data/dataset/`: local datasets + preprocessing scripts (`prepare_chexpert.py`, `prepare_nih.py`) and `processed/` JSONL.
- `models/`: model components (`vit.py`, `projector.py`, `vivid_model.py`).
- `training/`: losses and trainer (`losses.py`, `trainer.py`).
- `evaluation/`: UMS verifier + metrics (`verifier.py`, `metrics.py`).
- `scripts/`: runnable entry points (`train_cxr.py`, `test_pipeline.py`).
- `docs/`: active requirement ledgers, handoff indexes, and boundary notes (`docs/README.md` is the first stop).
- `profile/`: design docs and schema notes (Markdown).

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

## Agent Workflow & Skill Use

- Before starting any task or command sequence, check whether an available skill or superpower should be used; if one applies, load and follow it before executing file edits, experiments, or long-running commands.
- Prefer `superpowers:using-superpowers` as the pre-flight rule for skill selection: when there is even a small chance a skill applies, inspect it first, then proceed with the matching workflow.
- Use `subagent-plan-decomposer` (`C:\Users\Admin\.codex\skills\subagent-plan-decomposer\SKILL.md`) when converting an existing plan into a main-agent plus subagent execution plan. Keep final decisions, merge/conflict handling, and user-visible conclusions with the main agent.
- Use `superpowers:subagent-driven-development` (`C:\Users\Admin\.codex\superpowers\skills\subagent-driven-development\SKILL.md`) when executing an implementation plan with mostly independent tasks that benefit from fresh implementer/reviewer subagents. Do not use it for small, tightly coupled, or single-file edits unless the user explicitly asks.
- Use `planning-with-files` (`C:\Users\Admin\.codex\skills\planning-with-files\SKILL.md`) for complex multi-step work, research workflows, experiment queues, or work expected to span many tool calls. Keep `task_plan.md`, `findings.md`, and `progress.md` in the project root when this workflow is active.
- For current research handoff, read `docs/README.md`, `task_plan.md`, `findings.md`, and `progress.md` before interpreting old root-level proposal markdown. Treat generated files under `outputs/` as evidence, but keep them ignored unless the user explicitly asks to version artifacts.
- For simple one-step questions or narrow edits, briefly note that the skill check was considered and continue directly.
