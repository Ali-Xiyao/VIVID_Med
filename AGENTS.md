# Repository Guidelines

## Active authority

This branch activates the isolated strict VIVID/SPD clean extension under:

- `extensions/vivid_spd_clean/audit/VIVID_SPD_CLEAN_EXPERIMENT_PROTOCOL.md`;
- `extensions/vivid_spd_clean/audit/vivid_spd_clean_lock.json`;
- `extensions/vivid_spd_clean/AGENTS.md`.

The active question is the paired hard-UMS comparison between the historical
four-prefix projector and historical four-by-two SPD using frozen
Qwen3.5-2B. Qwen3.5-4B/9B remain locked behind the primary gate.

BiVES, RCSD, VSL, ARISE, VICER, and MORPH are frozen history. Do not merge
their targets, losses, selectors, posterior weighting, field anchors,
interventions, or claims into the strict VIVID/SPD route.

## Project structure

- `extensions/vivid_spd_clean/`: only active method, protocol, code, tests,
  launchers, and planning state on this branch.
- `extensions/rcsd_cxr/`: terminal RCSD audit history; not an active method.
- `bives_cxr/`: active evidence model, closed-form decoder, intervention logic,
  losses, metrics, Qwen3.5 adapter, and manifest dataset.
- `configs/bives_cxr/`: the only active experiment configurations.
- `scripts/train_bives_cxr.py`: local training entry.
- `scripts/smoke_bives_cxr.py`: synthetic CPU smoke.
- `scripts/smoke_qwen35_vision.py`: read-only real-weight vision-only smoke.
- `scripts/smoke_qwen35_bives_integration.py`: bounded local-CUDA official-vs-selective visual alignment plus two-step synthetic S/C/U/I gate.
- `scripts/audit_bives_manifest.py`: manifest and split readiness audit.
- `scripts/evaluate_bives_vindr_sc.py`: locked, selection-free B0-vs-B2 expert
  polarity evaluation on VinDr consensus test.
- `scripts/evaluate_bives_vindr_interventions.py`: positive-box target/control,
  evidence-only, and localization evaluation with full Qwen vision reruns.
- `scripts/{prepare,extract,audit}_vindr_cxr*.py`: current external-data tools.
- `scripts/prepare_bives_vindr_expert_sc.py`: fail-closed VinDr test-consensus
  S/C intake; it does not create U/I labels or authorize model evaluation.
- `tests/`: active BiVES contract tests.
- `docs/`: active handoff and schema documentation.
- `legacy/`: archived pre-BiVES code, configs, proposals, and tools.
- `data/`, `outputs/`, `pretrained/`: local assets; do not publish them.

## Build, test, and development commands

```bash
pip install -r requirements.txt
python scripts/smoke_bives_cxr.py
python scripts/smoke_qwen35_vision.py \
  --model-path /path/to/Qwen3.5-0.8B \
  --dtype fp32
python scripts/smoke_qwen35_bives_integration.py \
  --model-path /path/to/Qwen3.5-2B \
  --output-dir outputs/bives_cxr/integration_gate
python -m unittest discover -s tests -p "test_bives_*.py" -v
python scripts/audit_bives_manifest.py \
  --train data/bives_cxr/manifests/train_locked.jsonl \
  --val data/bives_cxr/manifests/val_locked.jsonl
# Formal P0 uses qwen35_2b_p0.yaml without --debug, only after its locks pass.

python -m unittest discover -s extensions/vivid_spd_clean/tests -v
python extensions/vivid_spd_clean/scripts/smoke_vivid_spd_clean.py
python extensions/vivid_spd_clean/scripts/audit_vivid_spd_lock.py
```

The user explicitly authorized server execution and cached models for the
strict extension. It may run through retained SUES allocation `3066` on
`gpu01` as a sequential queue after read-only ownership/capacity checks. Do not
stop or alter unrelated jobs. All older routes remain frozen and are not
authorized for server replay.

The nonclinical weak-label proxy P0 has completed and failed its held-out S/C
survival gate; do not scale it to 4B/9B without a justified data-side repair.
It may use frozen parser S/C/U candidates and provenance-preserving synthetic
insufficient images, but every row/result must remain marked unreviewed and
nonformal. Do not claim expert agreement or clinical U/I validity.

## Coding style

- Python uses 4-space indentation, `snake_case` functions/variables, and
  `CapWords` classes.
- Prefer `pathlib.Path`.
- Keep data and model paths in YAML.
- Keep the active decoder closed-form; do not add a trainable flat four-class
  head.
- Fixed exact-K is a K-budgeted evidence set; keep `lambda_min: 0` until an
  adaptive hard-concrete/L0 gate exists.

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
