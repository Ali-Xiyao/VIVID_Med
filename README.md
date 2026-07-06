# VIVID-Med

VIVID-Med is a medical vision-language research repository centered on training deployable chest X-ray encoders from structured clinical supervision and instruction-style visual discrimination tasks.

## Current Main Method

**SAMEQ-CVCP**  
Same-Question Clinical Visual Curriculum Pretraining

Current active work treats `SAMEQ-CVCP` as the primary paper-ready method line. The core idea is to keep the clinical question fixed while changing the image, so the model must resolve the answer from image-specific evidence instead of text shortcuts.

Current active entry points:

- [`docs/README.md`](docs/README.md): handoff index for active research state, ledgers, and result artifacts.
- [`vivid_med_sameq_cvcp_next_experiment_plan_v4.md`](vivid_med_sameq_cvcp_next_experiment_plan_v4.md): current paper-ready experiment plan and final write-back.
- `outputs/final_tables/`: generated manifests, audits, and result tables for the active SAMEQ/CVCP/CCSH line.

## Legacy Method

**Frozen-LM JSON supervision**

The earlier project line used a frozen LLM as a structured semantic decoder / supervision space:

```text
Image -> ViT(train) -> Projector(train) -> Frozen LLM(forward) -> JSON token logits
```

In that setup, the LLM was used during training-time supervision and then discarded after training. This legacy route remains in the repository for historical baselines, ablations, and prior experiment traces, but it is no longer the top-level active method narrative.

## Repository Layout

```text
configs/      Experiment configs, including SAMEQ/CVCP/CCSH and legacy baselines
data/         Dataset loaders, local preprocessing, and processed manifests
models/       Vision, projector, and integrated model components
training/     Losses, trainers, and optimization utilities
evaluation/   Metrics and verification utilities
scripts/      Training, evaluation, summarization, and queue entry points
docs/         Active ledgers, handoff indexes, and boundary notes
profile/      Design notes, writing support, and historical method docs
outputs/      Generated experiment artifacts (kept out of Git by default)
```

## Common Entry Points

Install dependencies:

```bash
pip install -r requirements.txt
```

Current active docs-first workflow:

```bash
python scripts/train_qwen3vl_cvcp.py --config configs/qwen3vl_instruction/cvcp_ccsh/cvcp_v1_sameq_10k.yaml
python scripts/run_multiseed_manifest.py
python scripts/summarize_cvcp_ccsh_results.py
python scripts/summarize_multiseed_results.py
```

Legacy smoke test:

```bash
python scripts/test_pipeline.py
```

Note: many runnable experiment queues are also packaged as PowerShell helpers under `scripts/`.

## Documentation Policy

The top-level README is only a project overview. For the current research state, always start from [`docs/README.md`](docs/README.md) instead of older root-level proposal files.

## Data and Artifact Boundaries

- Medical datasets under `data/dataset/` should remain local.
- Generated instructions, checkpoints, and result packages under `outputs/` are evidence artifacts and are typically not committed.
- Historical method files are preserved for comparison and auditability, even when they are no longer the active paper line.
