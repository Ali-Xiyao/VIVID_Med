# Repository instructions

## Active authority

`audit/RCSD_P0_NO_GO_VERDICT.md` is the active scientific authority.
`audit/RCSD_COMPONENT_ATTRIBUTION_PLAN.md` defines the D0-D4 evidence boundary.
`audit/RCSD_D0_D1_REVIEW_PROTOCOL.md` and
`audit/rcsd_d0_d1_review_lock.json` define the only prepared next review.
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

Documentation, aggregate-evidence validation, and CPU unit tests are allowed.
No D0-D4 training is currently authorized. The D0/D1 review package has frozen
the proposed comparison but explicitly records incomplete implementation,
manifest, expert-development, launcher, and approval prerequisites. Do not
change `PREPARED_NOT_APPROVED`, fill artifact hashes without producing and
validating them, or start a job. External-test evaluation, data download,
server upload/replacement, and GPU or Slurm jobs remain unauthorized.

## Coding and validation

- Python 3.10+, four-space indentation, type hints, `pathlib.Path`.
- Paths live in YAML registries, never in model code.
- Deterministic behavior and source/checkpoint manifests are mandatory.
- Run `python -m compileall rcsd_cxr scripts tests` and
  `python -m unittest discover -s tests -v` after implementation changes.
- Keep `task_plan.md`, `findings.md`, and `progress.md` current.
