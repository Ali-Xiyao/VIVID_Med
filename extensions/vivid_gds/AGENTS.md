# VIVID-GDS Instructions

## Authority

`audit/VIVID_GDS_EXPERIMENT_PROTOCOL_20260724.md` and
`audit/vivid_gds_stage_a_lock.json` are the only active scientific and
execution authorities in this directory.

The method identity is fixed as VIVID-GDS:

- frozen Qwen3.5-2B;
- deterministic hard UMS;
- historical prefix4 projector;
- a training-only UMS schema head on the final ViT CLS representation;
- synchronized UMS field masks for generation and schema losses;
- `lambda_schema=0.5`, linearly ramped during the first 500 optimizer steps;
- Qwen, projector, and schema head removed for deployment.

## Boundaries

- Do not add SPD, a new projector, contrastive loss, graph modules, anatomy
  queries, consistency networks, causal selectors, or a second method name.
- Do not use Qwen3.5-4B/9B as a repair.
- Do not change the Stage-A thresholds, 20k patient split, hard targets,
  Qwen3.5-2B identity, ViT initialization, or checkpoint rule.
- Do not open CheXlocalize test or VinDr test.
- CheXpert expert-development is a development gate, not a blind test.
- Runtime outputs, medical images, checkpoints, models, caches, and patient
  data stay outside Git.

## Failure policy

Implementation failures preserve their logs and allow at most one
identity-preserving repair before the failed gate is restarted from zero.
Scientific failures stop scale-up. A bounded case study may use only the
controls preregistered in the protocol.

## Validation

```bash
python -m unittest discover -s extensions/vivid_gds/tests -v
python extensions/vivid_gds/scripts/smoke_vivid_gds.py
python extensions/vivid_gds/scripts/audit_vivid_gds_lock.py
```

Keep `task_plan.md`, `findings.md`, `progress.md`, and
`refine-logs/EXPERIMENT_TRACKER.md` current.
