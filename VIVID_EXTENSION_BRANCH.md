# VIVID journal-extension branch

## Identity

- Repository: `Ali-Xiyao/VIVID_Med`
- Branch: `codex/vivid-extension-rcsd`
- Extension package: `extensions/rcsd_cxr/`
- Working name: RCSD-CXR
- Relationship to VIVID: clean-room journal-extension investigation of the
  stable structured-semantic representation-learning line

This branch is intentionally separate from the frozen VSL, BiVES, ARISE,
VICER, and MORPH method routes. Those historical routes are not reactivated by
this branch.

## Scientific status

RCSD-CXR is terminal **NO-GO** as a new method.

1. Multi-source posterior fusion failed G2 because NLL was 6.98% worse than
   the best single source, CheXbert.
2. The surviving field-anchor comparison failed G3. Relative NLL improvement
   was 0.046% against a 3% threshold, and macro-F1 improvement was 0.0648
   percentage points against a 0.5-point threshold.

The fail-closed protocol therefore cancelled:

- full-MIMIC training;
- external-test evaluation;
- multi-seed expansion;
- MIMIC + CheXpert-Plus scale-up;
- Qwen3.5 0.8B/4B/9B teacher-size sensitivity;
- post-hoc source, loss, query, teacher, or threshold rescue.

CheXlocalize test remained sealed.

## Authoritative files

- Terminal result:
  `extensions/rcsd_cxr/docs/RCSD_CXR_terminal_gate_result_20260723.md`
- Final task ledger: `extensions/rcsd_cxr/task_plan.md`
- Experiment tracker:
  `extensions/rcsd_cxr/refine-logs/EXPERIMENT_TRACKER_20260723T001428+0800.md`
- Reviewed protocol:
  `extensions/rcsd_cxr/docs/RCSD_CXR_active_protocol.md`
- Original proposal:
  `extensions/rcsd_cxr/provenance/RCSD_CXR_full_proposal_20260722.original.md`

## Publication boundary

The branch contains source code, tests, configuration, documentation, and
hash/provenance manifests only. It does not publish:

- medical images or reports;
- patient data;
- model weights or checkpoints;
- generated runtime outputs;
- local data-path registries;
- credentials, caches, or environments.

The packaged source surface contains 91 files and approximately 0.35 MiB.
The server-side experimental evidence remains outside Git.

## Validation

- Local RCSD unit tests: 45/45 passed.
- Server RCSD unit tests: 45/45 passed.
- Packaged source/server manifest entries: 90/90 hash-verified before branch
  publication.
