# Experiment Plan 2026-07-23

This timestamped plan is identical in scientific content to
`EXPERIMENT_PLAN.md`. It records the initial frozen execution sequence:

1. audit identities, hashes, data overlap, model files, and server allocation;
2. run `ums_prefix4` and `ums_spd4x2` 256-row overfit sequentially;
3. if both pass, run their 20k-study pilots sequentially;
4. select each ViT checkpoint only by internal validation token NLL;
5. train identical CheXpert development linear probes;
6. apply the locked S3 thresholds and freeze the outcome;
7. promote, diagnose once, or terminate according to the protocol.

No protected test surface or larger teacher is part of this plan.
