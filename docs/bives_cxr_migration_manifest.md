# BiVES-CXR Migration Manifest

## Active

- `BiVES_CXR_MIA_TMI_ready_proposal.md`
- `bives_cxr/`
- `configs/bives_cxr/`
- `scripts/train_bives_cxr.py`
- `scripts/smoke_bives_cxr.py`
- `scripts/audit_bives_manifest.py`
- `scripts/extract_vindr_cxr.py`
- `scripts/audit_vindr_cxr_integrity.py`
- `tests/test_bives_core.py`
- `tests/test_bives_readiness.py`

## Archived

- `legacy/vsl_cxr/configs/`: Qwen3-VL/VSL/SAMEQ/CVCP configs.
- `legacy/vsl_cxr/scripts/`: VSL, CEQ, CCSH, AUCH, case-study, Qwen3-VL, and
  old VSL external-evaluation scripts.
- `legacy/vsl_cxr/models/`: CEQ, CCSH, and AUCH modules.
- `legacy/vsl_cxr/docs/`: VSL ledgers, story, plan, and prior proposal files.
- `legacy/vivid_med/configs/`: remaining pre-BiVES VIVID-Med and Qwen2.5-era
  configs.
- `legacy/vivid_med/scripts/`: pre-BiVES training, evaluation, analysis,
  plotting, queue, and legacy VinDr UMS builders.
- `legacy/vivid_med/models/`, `training/`, `evaluation/`, `data/`: pre-BiVES
  implementation packages and loaders.
- `legacy/vivid_med/profile/`, `prompts/`, `experiments/`, `docs/`: old design
  material, prompts, evaluator tools, and paper source snapshots.

## Preserved in place

- medical datasets and manifests under `data/`;
- ignored experiment evidence under `outputs/`;
- VinDr archive extraction and integrity-audit utilities;
- planning files recording provenance and server/storage boundaries.

## Claim boundary

Archived VSL/CEQ/CCSH/AUCH results are pilot evidence and baselines. They are
not BiVES-CXR results and must not be copied into a BiVES main table without a
matched rerun under the locked BiVES protocol.
