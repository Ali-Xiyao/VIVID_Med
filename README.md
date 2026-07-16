# BiVES-CXR

BiVES-CXR learns a statement-conditioned bipolar visual evidence set for
chest-radiograph verification.

Given a chest X-ray and an atomic clinical statement, the active model learns:

- positive spatial evidence supporting the statement;
- negative spatial evidence contradicting the statement;
- evidence availability and decisiveness;
- a minimal evidence set that is sufficient when retained, necessary when
  removed, and stable to matched irrelevant-region deletion.

The four states are derived from the same evidence variables:

`support`, `contradict`, `uncertain`, and `insufficient`.

There is no flat four-class prediction head in the active model.

## Active mainline

- Proposal: [`BiVES_CXR_MIA_TMI_ready_proposal.md`](BiVES_CXR_MIA_TMI_ready_proposal.md)
- Code: [`bives_cxr/`](bives_cxr/)
- Active configs: [`configs/bives_cxr/`](configs/bives_cxr/)
- Training entry: [`scripts/train_bives_cxr.py`](scripts/train_bives_cxr.py)
- CPU smoke: [`scripts/smoke_bives_cxr.py`](scripts/smoke_bives_cxr.py)
- Manifest audit: [`scripts/audit_bives_manifest.py`](scripts/audit_bives_manifest.py)
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
```

These checks use synthetic CPU tensors and do not load model weights or run a
formal local experiment.

## Server training

The checked-in configs use the established server model cache:

```bash
python scripts/train_bives_cxr.py \
  --config configs/bives_cxr/qwen35_2b_p0.yaml \
  --debug
```

Before formal training, build and audit the BiVES manifest. Required JSONL
fields are documented in [`docs/bives_cxr_manifest_schema.md`](docs/bives_cxr_manifest_schema.md).

## Repository layout

```text
bives_cxr/        Active evidence model, decoder, interventions, losses, data schema
configs/bives_cxr Active Qwen3.5-only experiment configs
scripts/          Active BiVES and reusable data-preparation entry points
tests/            CPU contract tests
docs/             Active handoff and implementation notes
legacy/vsl_cxr/   Archived VSL/CEQ/CCSH/AUCH/Qwen3-VL pilot line
legacy/vivid_med/ Archived pre-BiVES configs
data/             Local manifests/loaders; medical images remain ignored
outputs/          Generated experiment artifacts; ignored by Git
```

## Data and evidence boundaries

- Medical images and patient data remain local/server-side and are not
  committed.
- Generated outputs and checkpoints remain ignored.
- Historical outputs are preserved as pilot/provenance evidence; they are not
  BiVES-CXR results.
- Formal local training is not part of this consolidation. The new code is
  prepared for server execution.
- VinDr ZIP extraction and integrity checks remain active utilities; the old
  UMS/VSL manifest builder is archived and must not be used for BiVES labels.
