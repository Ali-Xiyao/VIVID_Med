# Development-row schema

**Schema version:** `cxr_localization_causality_audit_v1`

This schema is a model-agnostic boundary between future frozen model adapters
and the audit/statistics layer. The current CLI accepts only precomputed
synthetic/development rows. It cannot open CheXlocalize test or load a model.

## Required identity fields

| Field | Meaning |
| --- | --- |
| `row_id` | Globally unique deterministic row identity |
| `patient_id` | Patient-level resampling key |
| `image_id` | Image identity within the local manifest |
| `pathology_id` | Frozen pathology/statement key |
| `model_id` | Frozen model/checkpoint identity alias |
| `explanation_id` | Frozen explanation method/config alias |
| `operator_id` | Frozen perturbation operator alias |
| `dataset_role` | `synthetic_development` or `development` only |

Identity values must be non-empty and cannot contain `|`, carriage returns, or
newlines because the summary uses a canonical pipe-delimited group key.

## Required status fields

```json
{
  "schema_version": "cxr_localization_causality_audit_v1",
  "formal_result": false,
  "test_opened": false,
  "score_direction": "higher_is_more_support"
}
```

Any test-like dataset role, `formal_result=true`, or `test_opened=true` fails
before summarization.

## Scores and contrasts

The builder receives five finite precomputed scores:

| Score | Intervention |
| --- | --- |
| `s0` | original image |
| `sX` | expert target `X` |
| `sCX` | expert-specific control `C_X` |
| `sE` | explanation target `E` |
| `sCE` | explanation-specific control `C_E` |

It derives `dX`, `dCX`, `dE`, `dCE`, `CS_X=dX-dCX`, and
`CS_E=dE-dCE`. One shared control is invalid when `X` and `E` differ in
geometry.

## Localization block

The builder records expert/explanation intersection, union, areas, IoU, Dice,
expert coverage, explanation precision, and optional point hit from a finite
continuous explanation map. It does not select an explanation threshold.

## Geometry and strength blocks

Both `X/C_X` and `E/C_E` must independently pass:

- shape-matched binary masks;
- non-empty in-content regions;
- exact target/control area;
- target/control disjointness;
- one connected control component;
- control disjointness from both relevant targets;
- finite, explicit perturbation-strength thresholds;
- masked L1/RMS, SSIM, and edge-change difference checks.

Rows that fail are rejected rather than repaired from their scores.

## Summary and lock outputs

`scripts/audit_cxr_localization_causality.py` writes:

- `audit_summary.json`: patient-cluster bootstrap intervals, group means,
  localization–`CS_E` Spearman descriptions, and cross-operator worst-case/sign
  agreement;
- `development_lock.json`: input file/row hashes, summary hash, source hashes,
  bootstrap identity, `formal_result=false`, and `test_opened=false`.

Generated development outputs belong under ignored `local_runs/` and are not
published as medical or locked-test evidence.
