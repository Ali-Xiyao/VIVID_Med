# VIVID journal-extension branch

## Identity

- Repository: `Ali-Xiyao/VIVID_Med`
- Branch: `codex/rcsd-no-go-audit`
- Frozen implementation branch: `codex/vivid-extension-rcsd`
- Frozen implementation commit: `bc1105f880116e97e06da023110b7080debc28a4`
- Extension package: `extensions/rcsd_cxr/`
- Working name: RCSD-CXR
- Relationship to VIVID: clean-room journal-extension investigation of the
  stable structured-semantic representation-learning line

This branch is intentionally separate from the frozen VSL, BiVES, ARISE,
VICER, and MORPH method routes. Those historical routes are not reactivated by
this branch.

## Scientific status

The full RCSD combination is **NO-GO**, but the overall VIVID journal
extension is not declared failed.

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

The bounded attribution authority is
`extensions/rcsd_cxr/audit/RCSD_P0_NO_GO_VERDICT.md`. D2 posterior fusion and
the tested D3 field anchor are rejected; exact D0 and D1 have not both been
evaluated under the proposed common component-audit protocol. No new training
is authorized by that observation.

The next review artifact is
`extensions/rcsd_cxr/audit/RCSD_D0_D1_REVIEW_PROTOCOL.md`. It distinguishes
the immutable historical D0-H artifact from the common-protocol D0-CP
reconstruction and permits D1 to change only entropy-derived loss weights.
Its machine lock remains `PREPARED_NOT_APPROVED`, with zero training jobs
authorized.

## Authoritative files

- Terminal result:
  `extensions/rcsd_cxr/docs/RCSD_CXR_terminal_gate_result_20260723.md`
- Active component-attribution verdict:
  `extensions/rcsd_cxr/audit/RCSD_P0_NO_GO_VERDICT.md`
- D0-CP versus D1 review protocol:
  `extensions/rcsd_cxr/audit/RCSD_D0_D1_REVIEW_PROTOCOL.md`
- D0/D1 machine lock:
  `extensions/rcsd_cxr/audit/rcsd_d0_d1_review_lock.json`
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

- Current audit-branch unit tests: 59/59 passed.
- Component-attribution integrity: passed.
- D0/D1 review-lock and historical-source integrity: passed.
- Frozen base packaged source/server manifest entries: 90/90 hash-verified
  before the original extension-branch publication.
