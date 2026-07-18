# Research handoff index

## Active authority

| Purpose | File |
| --- | --- |
| Audit proposal | `../audit/CXR_localization_causality_audit_proposal.md` |
| CheXlocalize development/test protocol | `../audit/chexlocalize_validation_test_protocol.md` |
| Primary endpoints | `../audit/primary_endpoints.md` |
| Novelty matrix | `../audit/novelty_matrix.md` |
| Audit directory status | `../audit/README.md` |
| Repository overview | `../README.md` |
| Persistent task plan | `../task_plan.md` |
| Durable findings | `../findings.md` |
| Chronological progress | `../progress.md` |

The active state is `PROTOCOL_DESIGN_ONLY_NO_EXPERIMENT_AUTHORITY`. No new
dataset, model, GPU, score, test opening, server, or Slurm action is authorized.

## Frozen predecessor

| Purpose | File |
| --- | --- |
| Frozen tag | `bives-b2-terminal-8bb1a94` at commit `8bb1a94` |
| Archived method proposal | `../archive/BiVES_CXR_method_proposal_terminal.md` |
| Terminal negative report | `../archive/BiVES_B2_terminal_negative_report.md` |
| Terminal read-only audit | `../refine-logs/BIVES_B2_TERMINAL_READ_ONLY_AUDIT_RESULT_20260719.md` |
| C5 execution record | `../refine-logs/CONNECTED_CONTROL_C5_EXECUTION_LOG_20260718.md` |
| C6I authority | `../refine-logs/C6I_MS_CXR_ACTUAL_INPUT_RECOVERY_AUTHORITY_20260718.md` |
| C6I geometry log | `../refine-logs/C6I_MS_CXR_PREOPEN_GEOMETRY_EXECUTION_LOG_20260718.md` |
| C6I terminal result | `../refine-logs/C6I_MS_CXR_REPLACEMENT_EVALUATION_RESULT_20260718.md` |

BiVES B2 is preserved as an audited model/explanation case. It is not the
successful paper mainline, and the frozen evidence cannot be used for C6J,
same-test repair, rerun, operator tuning, or 4B/9B scale-up.

## Preserved code and contracts

| Surface | Path |
| --- | --- |
| BiVES package | `../bives_cxr/` |
| Frozen configs | `../configs/bives_cxr/` |
| Terminal audit module | `../bives_cxr/terminal_audit.py` |
| Terminal audit entrypoint | `../scripts/audit_bives_b2_terminal.py` |
| BiVES synthetic smoke | `../scripts/smoke_bives_cxr.py` |
| Active contract tests | `../tests/test_bives_*.py` |
| Implementation contract | `bives_cxr_implementation.md` |
| Manifest schema | `bives_cxr_manifest_schema.md` |
| Migration/archive manifest | `bives_cxr_migration_manifest.md` |

These surfaces remain testable and reusable for the audit, but they do not
authorize new BiVES training/evaluation.

## Historical BiVES records

The following root documents are historical design/diagnostic records, not
active authorities:

- `../BiVES_support_polarity_root_cause_and_repair.md`;
- `../BiVES_uncertain_selector_evidence_diagnostic_plan.md`;
- `../BiVES_next_direction_without_local_clinical_review_2026-07-17.md`;
- `../BiVES_995fb81_code_review_and_next_plan.md`;
- `../BiVES_C6G_MS_CXR_geometry_protocol_plan.md`.

Detailed implementation, proxy-P0, VinDr, MS-CXR, C6F/C6G/C6H/C6I, and
post-stop records remain in this directory and `../refine-logs/` for
provenance. They must be interpreted through the frozen terminal report.

## Data and execution boundary

- CheXlocalize validation is development-only due to prior exposure.
- CheXlocalize test remains unopened until a complete protocol and identity
  freeze plus separate explicit authorization.
- MS-CXR and VinDr evidence is frozen/supplemental and cannot be retuned.
- Future execution is local only; no server synchronization or SSH/Slurm job.
- Medical images, patient data, outputs, checkpoints, model weights, caches,
  and credentials remain local and uncommitted.
