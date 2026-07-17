# BiVES-CXR

BiVES-CXR learns a statement-conditioned bipolar visual evidence set for
chest-radiograph verification.

Given a chest X-ray and an atomic clinical statement, the active model learns:

- positive spatial evidence supporting the statement;
- negative spatial evidence contradicting the statement;
- evidence availability and decisiveness;
- a fixed-K budgeted evidence set that is sufficient when retained, necessary when
  removed, and stable to matched irrelevant-region deletion.

The four states are derived from the same evidence variables:

`support`, `contradict`, `uncertain`, and `insufficient`.

There is no flat four-class prediction head in the active model.

## Active mainline

- Proposal: [`BiVES_CXR_MIA_TMI_ready_proposal.md`](BiVES_CXR_MIA_TMI_ready_proposal.md)
- Current diagnostic authority: [`BiVES_next_direction_without_local_clinical_review_2026-07-17.md`](BiVES_next_direction_without_local_clinical_review_2026-07-17.md)
- Frozen Proxy-P0-A record: [`docs/bives_cxr_proxy_p0_a_freeze.md`](docs/bives_cxr_proxy_p0_a_freeze.md)
- Code: [`bives_cxr/`](bives_cxr/)
- Active configs: [`configs/bives_cxr/`](configs/bives_cxr/)
- Training entry: [`scripts/train_bives_cxr.py`](scripts/train_bives_cxr.py)
- CPU smoke: [`scripts/smoke_bives_cxr.py`](scripts/smoke_bives_cxr.py)
- Real-weight vision smoke: [`scripts/smoke_qwen35_vision.py`](scripts/smoke_qwen35_vision.py)
- Local CUDA integration gate: [`scripts/smoke_qwen35_bives_integration.py`](scripts/smoke_qwen35_bives_integration.py)
- Manifest audit: [`scripts/audit_bives_manifest.py`](scripts/audit_bives_manifest.py)
- Weak-label proxy-P0 builder: [`scripts/build_bives_proxy_p0.py`](scripts/build_bives_proxy_p0.py)
- Statement cache builder: [`scripts/build_bives_statement_embeddings.py`](scripts/build_bives_statement_embeddings.py)
- Locked-test release: [`scripts/evaluate_bives_final.py`](scripts/evaluate_bives_final.py)
- Tests: [`tests/test_bives_core.py`](tests/test_bives_core.py)
- Handoff index: [`docs/README.md`](docs/README.md)

## Model policy

All active BiVES-CXR experiments use the multimodal Qwen3.5 family:

| Role | Model |
| --- | --- |
| P0/debug | Qwen3.5-2B |
| Default main model | Qwen3.5-4B |
| Scale validation | Qwen3.5-9B |
| Optional ultra-light smoke | Qwen3.5-0.8B |

Qwen3-VL, Qwen2.5, CEQ, CCSH, AUCH, SAMEQ-CVCP, and earlier VIVID-Med routes
are not active model defaults. Their source/config/docs are preserved under
[`legacy/`](legacy/) for provenance and fair baselines.

## Quick validation

```bash
pip install -r requirements.txt
python scripts/smoke_bives_cxr.py
python -m unittest discover -s tests -p "test_bives_*.py" -v
python scripts/smoke_qwen35_vision.py \
  --model-path /path/to/Qwen3.5-0.8B \
  --dtype fp32
```

The first two checks use synthetic CPU tensors. The third is a read-only
real-weight vision-tower check and does not retain the language model or start
training.

Formal local runs use the audited package lock in
[`requirements-bives-local-lock.txt`](requirements-bives-local-lock.txt).
The bounded integration gate compares the selective vision-only loader against
the official full Qwen3.5 visual output, releases the full model, then runs two
synthetic S/C/U/I BiVES optimization steps.

## Local-only development and formal training

All active BiVES-CXR experiments run on this workstation. The active workflow
does not synchronize experiment code/assets to the server and does not submit
SSH or Slurm jobs. Changing the execution host does not relax any manifest,
dataset-lock, statement-cache, calibration, or locked-test gate.

The expanded `qwen35_2b_proxy_p0` cycle has completed as an explicitly
nonclinical weak-label engineering experiment. Its S/C/U labels come from the
frozen parser and its insufficient rows are reproducible synthetic
evidence-removal images. The 5,000-study parser-v3 run passes aggregate and
per-finding S/C ranking, but its uncalibrated four-state argmax collapses to
insufficient. A train-proxy decoder-parameter diagnostic improves held-out
accuracy only to `0.5417` and leaves contradict/uncertain weak, so 4B/9B
scaling remains stopped. See
[`docs/bives_cxr_proxy_p0_experiment_log.md`](docs/bives_cxr_proxy_p0_experiment_log.md).
It cannot be reported as expert-audited clinical ground truth or as a formal
locked-test result.

The bounded optimization-identifiability gate is now closed. A fixed
patient-group-disjoint logistic probe found usable frozen S/C signal (global
AUROC `0.7889`), but the 400-step state-only Qwen3.5-2B run reached only
`0.7917` train accuracy instead of the required `1.0`. It learned the correct
support/contradict polarity direction and lowest insufficient evidence, so the
failure does not justify a magnitude-polarity refactor. Run B was not started;
4B/9B scaling, decoder/loss changes, and further weak-label four-state runs
remain closed. See
[`docs/bives_cxr_optimization_identifiability_verdict.md`](docs/bives_cxr_optimization_identifiability_verdict.md).

The separate public-data intake is documented in
[`docs/bives_cxr_public_expert_sc_intake.md`](docs/bives_cxr_public_expert_sc_intake.md).
The bounded Qwen3.5-2B expert S/C and full-tower pixel-intervention run is now
closed. B2 improves expert AUROC for both findings but falls below B0 on
consolidation AUPRC. Its selected patches overlap expert boxes more than a
random mask, yet target deletion is not stronger than the equal-area disjoint
control; pleural-effusion TCIG is significantly negative. The declared stop
rule therefore keeps more seeds and 4B/9B closed. See
[`docs/bives_cxr_expert_polarity_intervention_verdict.md`](docs/bives_cxr_expert_polarity_intervention_verdict.md).
The aggregate post-stop failure taxonomy is in
[`docs/bives_cxr_post_stop_failure_taxonomy.md`](docs/bives_cxr_post_stop_failure_taxonomy.md):
the failed causal gate is associated with inconsistent selector localization
and disproportionate sensitivity to large arbitrary control deletions. This
VinDr-test diagnosis is descriptive only and cannot authorize tuning or a
same-test rerun.

For a bounded local engineering run, first copy the tracked template to the
ignored local-config area and point it at local manifest paths. This run is
explicitly marked `formal_result: false`; it selects at most two train steps,
uses `num_workers: 0`, reads local Qwen3.5-2B weights only, and never creates a
dataset/run lock or calibration/release artifact.

```bash
mkdir configs_local
copy configs\bives_cxr\qwen35_2b_local_debug.template.yaml configs_local\qwen35_2b_local_debug.yaml
python scripts/train_bives_cxr.py --config configs_local/qwen35_2b_local_debug.yaml
```

The default local debug config requests CUDA BF16 and fails before manifest or
model loading if the selected GPU is unavailable. Its metadata is written to
`local_runs/`, which is ignored by Git.

For the stronger local mechanism check, prepare an explicitly non-clinical
one-quartet overfit input and run up to 50 steps. The generated labels are
synthetic engineering probes, not dataset annotations or results.

```bash
python scripts/prepare_local_bives_overfit.py --source-image path/to/local.png --output-dir local_runs/bives_cxr/overfit_input
copy configs\bives_cxr\qwen35_2b_local_overfit.template.yaml configs_local\qwen35_2b_local_overfit.yaml
# Set the copied config's train/val manifests to the two files above, then run:
python scripts/train_bives_cxr.py --config configs_local/qwen35_2b_local_overfit.yaml
```

The checked-in `local_formal` configs are release-preparation templates, not a
shortcut around locks:

```bash
# Formal P0 uses qwen35_2b_p0.yaml without --debug, only after its locks pass.
```

Before formal training, build and audit the BiVES manifest and frozen statement
cache. Required JSONL
fields are documented in [`docs/bives_cxr_manifest_schema.md`](docs/bives_cxr_manifest_schema.md).

Primary validation and calibration use full sequential row coverage and assert
that every `sample_id` is evaluated exactly once. Same-statement grouped
evaluation is separate and reports pair/polarity mechanism diagnostics.
Temperature calibration is fitted only on the locked calibration split after
the validation-selected `best.pt` checkpoint is reloaded. The training entry
cannot evaluate the locked test; final release requires
`scripts/evaluate_bives_final.py --run-locked-test`.

Formal training creates `run_lock.json` before Qwen3.5 weight loading. Every
checkpoint carries that lock, its canonical SHA256, and the full statement
ontology. Calibration binds itself to the uncalibrated `best.pt`, and the final
evaluator rejects any checkpoint/calibration/cache/test/config/base-model/Git
combination that does not match the same lock. Evaluation controls use the
fixed cross-run seed `20260717`, independent of each model training seed.

## Repository layout

```text
bives_cxr/        Active evidence model, decoder, interventions, losses, data schema
configs/bives_cxr Active Qwen3.5-only experiment configs
scripts/          Active BiVES and reusable data-preparation entry points
tests/            CPU contract tests
docs/             Active handoff and implementation notes
legacy/vsl_cxr/   Archived VSL/CEQ/CCSH/AUCH/Qwen3-VL pilot line
legacy/vivid_med/ Archived pre-BiVES configs
legacy/delete_archive/ Archived small retired config/script cleanup bundle
data/             Local manifests/loaders; medical images remain ignored
outputs/          Generated experiment artifacts; ignored by Git
```

## Data and evidence boundaries

- Medical images and patient data remain local and are not committed.
- Generated outputs and checkpoints remain ignored.
- Historical outputs are preserved as pilot/provenance evidence; they are not
  BiVES-CXR results.
- Local debug outputs are engineering evidence only and are never formal
  results. `local_formal` remains gated by all four locked manifests, a
  revalidated dataset lock, frozen statement cache, clean source snapshot, and
  the final evaluator.
- Active keep/drop/control training is feature-space closure. Pixel-causal
  grounding claims remain locked until pixel keep/drop/equal-area controls are
  rerun through the complete vision tower.
- VinDr ZIP extraction and integrity checks remain active utilities; the old
  UMS/VSL manifest builder is archived and must not be used for BiVES labels.
