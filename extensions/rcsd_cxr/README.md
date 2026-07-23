# RCSD-CXR

Clean-room continuation of the stable VIVID-Med representation-learning line.
The project studies reliability-calibrated structured report supervision and
field-anchored distillation for a deployable chest-radiograph encoder.

## Scope

- Keep: frozen text teacher, UMS-compatible structured semantics, SPD 4x2
  baseline, reliability-calibrated posteriors, and ViT-only deployment.
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

RCSD-CXR is terminal **NO-GO** as a new method. Multi-source fusion failed the
G2 likelihood gate, and the surviving equal-budget field-anchor variant failed
the G3 NLL and macro-F1 gates. Full-data, external-test, multi-seed,
multi-institution, and Qwen3.5-size experiments are cancelled.

The current scientific authority is
`docs/RCSD_CXR_terminal_gate_result_20260723.md`. The negative result applies
to the proposed RCSD additions; it does not reactivate or rewrite the frozen
VSL/BiVES/MORPH routes.

## Validation

```powershell
python -m compileall rcsd_cxr scripts tests
python -m unittest discover -s tests -v
python scripts\validate_data_refs.py --roles track_a_train paper1_external
```

Capacity and remote feasibility are recorded in
`docs/data_and_server_capacity_audit_20260722.md`.
