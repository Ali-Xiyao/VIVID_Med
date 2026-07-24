# VIVID-GDS Experiment Plan — 2026-07-24

## Claim-to-experiment map

| Claim | Necessary comparison | Survival endpoint |
|---|---|---|
| UMS format adds value | A2 vs A1 | expert-development AUROC/AUPRC gate |
| Frozen Qwen adds value | A2 vs A0 | expert-development AUROC/AUPRC gate |
| UMS-SRB adds value | A3 vs A2 | expert-development AUROC/AUPRC gate |

## Execution blocks

1. Contract audit: hashes, patient split, fields/states, protected non-use.
2. 256-row overfit: A0, A1, A3.
3. 20k single-seed pilots: A0, A1, A3; reuse locked A2.
4. Identical CheXpert expert-development probes: A0–A3.
5. Frozen verdict and only then optional Stage-B multiseed.

Runtime outputs are written only under the server `local_runs/` root.
