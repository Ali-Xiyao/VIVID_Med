# Repository instructions

## Active authority

`audit/RCSD_D0_D1_QWEN35_2B_TERMINAL_RESULT.md` is the active scientific
authority. D1 selective agreement weighting is terminal NO-GO after the
frozen Qwen3.5-2B 20k gate; the machine lock authorizes zero training jobs.
`audit/RCSD_P0_NO_GO_VERDICT.md` is the earlier G2/G3 scientific authority.
`audit/RCSD_COMPONENT_ATTRIBUTION_PLAN.md` defines the frozen D0-D4 evidence
boundary. `audit/RCSD_D0_D1_REVIEW_PROTOCOL.md` and
`audit/rcsd_d0_d1_review_lock.json` preserve the completed final review.
`audit/RCSD_D0_D1_AUTO_DEV_PROTOCOL.md` records the user's explicit
2026-07-23 execution approval and the bounded automatic development loop.
The original proposal and historical execution protocol remain provenance and
may not override the audit.

## Scientific boundary

This repository is only for conservative VIVID-Med / RCSD-CXR representation
learning. Do not add BiVES, ARISE, VICER, VSL, CEQ, CCSH, AUCH, causal
localization claims, CT mixed pretraining, or test-driven method selection.
The historical SPD baseline is exactly four groups by two tokens.

## Data and test boundary

- Raw medical images, reports, patient data, model weights, checkpoints,
  outputs, environments, and credentials remain outside Git.
- Missing or corrupt samples fail closed; never replace them with black images
  or relabel them as negative/insufficient.
- Splits are patient-aware. Derived resources retain their parent lineage.
- CheXlocalize test is reserved for the separate paper-two protocol and must
  not be opened here.
- VinDr test and MS-CXR are previously inspected and cannot be presented as
  newly blinded evidence.

## Execution boundary

The user explicitly revoked the workstation-only boundary on 2026-07-23 and
authorized RCSD-CXR execution through retained Slurm allocation `3066` on
`gpu01`, with existing server datasets and caches referenced in place. The
primary Qwen3.5-2B D0-CP/D1 pilot completed and failed its first promotion
condition. No preregistered repair exists. D1 repair, expert-development
probing, paired-pilot replay, external-test evaluation, threshold relaxation,
result-driven label replacement, and unrelated frozen-method reactivation are
forbidden. Full-data, multi-seed, teacher-sensitivity, D2, D3, and D4 remain
locked.

## Coding and validation

- Python 3.10+, four-space indentation, type hints, `pathlib.Path`.
- Paths live in YAML registries, never in model code.
- Deterministic behavior and source/checkpoint manifests are mandatory.
- Run `python -m compileall rcsd_cxr scripts tests` and
  `python -m unittest discover -s tests -v` after implementation changes.
- Keep `task_plan.md`, `findings.md`, and `progress.md` current.
