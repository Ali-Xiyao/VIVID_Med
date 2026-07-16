# Repository Guidelines

## Active authority

`BiVES_CXR_MIA_TMI_ready_proposal.md` is the final research authority. The
active implementation is BiVES-CXR and uses multimodal Qwen3.5 only:

- Qwen3.5-2B for P0/debug;
- Qwen3.5-4B as the default main model;
- Qwen3.5-9B for the locked scale study;
- Qwen3.5-0.8B only for optional ultra-light smoke checks.

Do not introduce active Qwen2, Qwen2.5, Qwen3-VL, LLaDA, Gemma, InternVL,
Llama, BiomedCLIP, CEQ, CCSH, or AUCH model paths. Historical implementations
belong under `legacy/`.

## Project structure

- `bives_cxr/`: active evidence model, closed-form decoder, intervention logic,
  losses, metrics, Qwen3.5 adapter, and manifest dataset.
- `configs/bives_cxr/`: the only active experiment configurations.
- `scripts/train_bives_cxr.py`: server training entry.
- `scripts/smoke_bives_cxr.py`: synthetic CPU smoke.
- `scripts/smoke_qwen35_vision.py`: read-only real-weight vision-only smoke.
- `scripts/audit_bives_manifest.py`: manifest and split readiness audit.
- `scripts/{prepare,extract,audit}_vindr_cxr*.py`: current external-data tools.
- `tests/`: active BiVES contract tests.
- `docs/`: active handoff and schema documentation.
- `legacy/`: archived pre-BiVES code, configs, proposals, and tools.
- `data/`, `outputs/`, `pretrained/`: local/server assets; do not publish them.

## Build, test, and development commands

```bash
pip install -r requirements.txt
python scripts/smoke_bives_cxr.py
python scripts/smoke_qwen35_vision.py \
  --model-path /path/to/Qwen3.5-0.8B \
  --dtype fp32
python -m unittest discover -s tests -p "test_bives_*.py" -v
python scripts/audit_bives_manifest.py \
  --train data/bives_cxr/manifests/train_locked.jsonl \
  --val data/bives_cxr/manifests/val_locked.jsonl
python scripts/train_bives_cxr.py \
  --config configs/bives_cxr/qwen35_2b_p0.yaml \
  --debug
```

Local validation is synthetic and read-only. Formal model loading/training runs
on the server after the data readiness audit passes.

## Coding style

- Python uses 4-space indentation, `snake_case` functions/variables, and
  `CapWords` classes.
- Prefer `pathlib.Path`.
- Keep data and model paths in YAML.
- Keep the active decoder closed-form; do not add a trainable flat four-class
  head.

## Testing guidelines

- Run the BiVES smoke and `test_bives_*.py` suite after every active-code
  change.
- Assert the no-flat-four-class-head contract and the Qwen3.5-only path guard.
- Do not treat missing/corrupt images as `insufficient`; manifest IO is
  fail-fast.

## Commit and security rules

- Use concise, imperative commit messages.
- Do not commit medical images, patient data, generated outputs, checkpoints,
  model weights, environments, upload staging, or caches.
- Pull requests should include the commands run and data/model assumptions.

## Agent workflow

- Before starting a task, check whether an available skill applies.
- Use `planning-with-files` for complex work and keep `task_plan.md`,
  `findings.md`, and `progress.md` current.
- Read `BiVES_CXR_MIA_TMI_ready_proposal.md`, `docs/README.md`,
  `task_plan.md`, `findings.md`, and `progress.md` before interpreting anything
  under `legacy/`.
- Treat `outputs/` as ignored evidence unless the user explicitly asks to
  publish selected artifacts.
