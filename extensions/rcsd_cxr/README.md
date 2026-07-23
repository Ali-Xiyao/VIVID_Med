# RCSD-CXR

Clean-room continuation of the stable VIVID-Med representation-learning line.
The project studies reliability-calibrated structured report supervision and
field-anchored distillation for a deployable chest-radiograph encoder.

## Scope

- Keep: frozen text teacher, UMS-compatible structured semantics, SPD 4x2
  baseline, and ViT-only deployment. Posterior fusion and field anchoring are
  retained only as terminal audited evidence, not active method components.
- Exclude: GFTM, FSA consistency, CXR/CT mixed training, VSL four-state
  clinical claims, CEQ/CCSH/AUCH, BiVES rescue, ARISE, VICER, and causal
  evidence claims.
- Raw images, reports, checkpoints, model weights, generated outputs, caches,
  credentials, and environments stay outside this repository.

The submitted proposal is preserved as an untrusted provenance input at
`provenance/RCSD_CXR_full_proposal_20260722.original.md`. The reviewed execution
authority is `docs/RCSD_CXR_active_protocol.md`.

## Data model

Datasets remain in their canonical local locations. Code reads a machine-local
YAML registry selected with `RCSD_DATA_REGISTRY`.

```powershell
$env:RCSD_DATA_REGISTRY = 'H:\Xiyao_Wang\02101\data_refs\datasets.local.yaml'
python scripts\validate_data_refs.py --roles track_a_train paper1_external
```

`data_refs/datasets.example.yaml` is portable and versionable.
`data_refs/datasets.local.yaml` contains the current workstation paths and is
ignored by Git.

## Status

The full RCSD combination is **NO-GO**. Multi-source fusion failed the G2
likelihood gate, and the surviving equal-budget field-anchor variant failed
the G3 NLL and macro-F1 gates. Full-data, external-test, multi-seed,
multi-institution, and Qwen3.5-size experiments are cancelled.

The current scientific authority is
`audit/RCSD_D0_D1_QWEN35_2B_TERMINAL_RESULT.md`. The final permitted
common-protocol D0-CP/D1 pilot completed, but D1 made validation token NLL
0.621% worse rather than at least 3% better. Selective agreement weighting is
therefore also terminal NO-GO. The machine lock is `TERMINAL`; training jobs
allowed: zero.

This closes the RCSD additions without declaring the stable historical
VIVID/SPD representation-learning line invalid. The only surviving paper-one
option is a strict VIVID/SPD extension with modern controlled validation, not
another reliability or field-anchor rescue.

## Validation

```powershell
python -m compileall rcsd_cxr scripts tests
python -m unittest discover -s tests -v
python scripts\audit_rcsd_component_status.py
python scripts\audit_rcsd_d0_d1_review.py
python scripts\validate_data_refs.py --roles track_a_train paper1_external
```

Capacity and remote feasibility are recorded in
`docs/data_and_server_capacity_audit_20260722.md`.
