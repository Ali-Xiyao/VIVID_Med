# Repository Guidelines

## Active authority

`audit/CXR_localization_causality_audit_proposal.md` is the only active
research authority. The project audits whether chest-X-ray localization
actually predicts causal model reliance under matched interventions.

The predecessor BiVES B2 route is frozen at tag
`bives-b2-terminal-8bb1a94`. Its archived proposal and terminal report are:

- `archive/BiVES_CXR_method_proposal_terminal.md`;
- `archive/BiVES_B2_terminal_negative_report.md`.

BiVES B2 is one audited model/explanation case, not a successful main method.
Do not create C6J, tune/replay C4/C5/C6I, repair B2, or scale that route to
Qwen3.5-4B/9B.

## Current execution boundary

The repository is in development-tooling phase. Model-free synthetic smokes,
unit tests, and precomputed-row audits may run locally. On 2026-07-19 the user
also explicitly opened local Qwen3.5-2B model/GPU development gates on both
workstation GPUs. This opening covers synthetic interface and determinism
gates only; it does not open CheXlocalize download/test, real-patient scoring,
or outcome-driven threshold selection. Future real-data execution still
requires a separately frozen data/model/explanation/operator lock and explicit
data/test opening.

All future validation and evaluation run on this workstation. Do not
synchronize active experiments to the server or submit SSH/Slurm jobs.

CheXlocalize validation was previously exposed in this repository. It may be
used only for protocol development, never as unbiased validation evidence.
CheXlocalize test is reserved for one-time locked evaluation.

## Project structure

- `audit/`: active proposal, split protocol, novelty matrix, and endpoints.
- `archive/`: frozen predecessor proposal and terminal negative report.
- `bives_cxr/`: retained BiVES implementation and reusable audited
  intervention/geometry utilities; frozen as a method route.
- `configs/bives_cxr/`: historical/frozen BiVES experiment identities; not an
  authorization to launch training or evaluation.
- `refine-logs/`: immutable C4/C5/C6I execution authorities and evidence.
- `scripts/audit_bives_b2_terminal.py`: read-only terminal-audit replay.
- `tests/`: BiVES and frozen-evidence contract tests.
- `docs/`: handoff, schema, and historical implementation documentation.
- `legacy/`: archived pre-BiVES code, configs, proposals, and tools.
- `data/`, `outputs/`, `pretrained/`, `local_runs/`: local assets/evidence; do
  not publish them.

## Audited model boundary

The new audit may eventually preregister multiple model families because model
family is an audit factor. That does not reactivate legacy model paths inside
BiVES. Exact models, weights, licenses, preprocessing, scores, and explanation
interfaces must be frozen during development before any locked test.

The frozen BiVES implementation remains multimodal Qwen3.5-only. Do not
introduce active Qwen2, Qwen2.5, Qwen3-VL, LLaDA, Gemma, InternVL, Llama,
BiomedCLIP, CEQ, CCSH, or AUCH paths into `bives_cxr/`.

## Validation commands

These commands validate the preserved code/evidence surface; they do not open a
new experiment:

```bash
pip install -r requirements.txt
python scripts/smoke_bives_cxr.py
python scripts/smoke_localization_causality_audit.py
python -m unittest discover -s tests -p "test_bives_*.py" -v
python scripts/audit_bives_b2_terminal.py --help
python scripts/audit_cxr_localization_causality.py --help
```

## Coding style

- Python uses 4-space indentation, `snake_case` functions/variables, and
  `CapWords` classes.
- Prefer `pathlib.Path` and keep data/model paths in YAML.
- Keep the frozen BiVES decoder closed-form; do not add a trainable flat
  four-class head.
- Missing/corrupt images are fail-fast and are never relabeled insufficient.
- New audit code must be deterministic, patient-aware, test-blind, and
  fail-closed on identity/geometry/strength mismatches.

## Testing guidelines

- Run the BiVES smoke and `test_bives_*.py` suite after changes touching the
  preserved implementation or evidence tooling.
- Preserve the no-flat-four-class-head and Qwen3.5-only BiVES contracts.
- New audit contracts must separately test patient identity, region geometry,
  matched controls, score orientation, deterministic explanations, operator
  strength diagnostics, and one-time test locks.

## Commit and security rules

- Use concise, imperative commit messages.
- Do not commit medical images, patient data, generated outputs, checkpoints,
  model weights, environments, upload staging, credentials, or caches.
- Pull requests should state commands run and data/model assumptions.
- Do not rewrite or regenerate frozen C4/C5/C6I evidence.

## Agent workflow

- Before starting, check whether an available skill applies.
- Use `planning-with-files` for complex work and keep `task_plan.md`,
  `findings.md`, and `progress.md` current.
- Read `audit/README.md`, `audit/CXR_localization_causality_audit_proposal.md`,
  `docs/README.md`, `task_plan.md`, `findings.md`, and `progress.md` before
  interpreting historical material.
- Treat `outputs/` and `local_runs/` as ignored evidence unless the user
  explicitly asks to publish selected, de-identified artifacts.
