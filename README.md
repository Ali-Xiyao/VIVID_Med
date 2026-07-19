# CXR Localization-Causality Audit

This repository now studies a narrower and more defensible question:

> When does a chest-X-ray explanation that localizes to an expert region also
> identify model evidence that is causally specific under matched intervention?

The active paper is an audit, not a claim that BiVES learned a successful
causal evidence set.

## Start here

- Active proposal:
  [`audit/CXR_localization_causality_audit_proposal.md`](audit/CXR_localization_causality_audit_proposal.md)
- CheXlocalize development/test protocol:
  [`audit/chexlocalize_validation_test_protocol.md`](audit/chexlocalize_validation_test_protocol.md)
- Primary endpoints:
  [`audit/primary_endpoints.md`](audit/primary_endpoints.md)
- Novelty matrix:
  [`audit/novelty_matrix.md`](audit/novelty_matrix.md)
- Development row schema:
  [`audit/development_row_schema.md`](audit/development_row_schema.md)
- Phase-C synthetic result:
  [`audit/phase_c_synthetic_development_result.md`](audit/phase_c_synthetic_development_result.md)
- Handoff index: [`docs/README.md`](docs/README.md)

## Frozen BiVES predecessor

BiVES B2 is frozen at tag `bives-b2-terminal-8bb1a94` (commit `8bb1a94`).
The former method proposal is preserved unchanged at
[`archive/BiVES_CXR_method_proposal_terminal.md`](archive/BiVES_CXR_method_proposal_terminal.md),
and the terminal decision/hashes are recorded in
[`archive/BiVES_B2_terminal_negative_report.md`](archive/BiVES_B2_terminal_negative_report.md).

The frozen evidence says:

- positive localization overlap did not guarantee matched target-vs-control
  causal specificity;
- the independent geometry-correct C6I result failed in a finding-dependent
  way;
- operator pixel-strength asymmetry explained only part of the result;
- there is no C6J, same-test repair, rerun, or BiVES 4B/9B scale-up.

BiVES remains in the repository as one audited model/explanation case and as a
source of deterministic intervention, geometry, and fail-closed tooling.

## New audit design

Every eligible row separates three regions:

1. expert region;
2. model explanation region;
3. disjoint perturbation-strength-matched control region.

Localization quality and causal specificity are separate endpoint families.
The audit reports model × explanation × pathology × operator interactions,
operator sign agreement/worst-case effects, and geometry/pixel-strength
diagnostics. A composite score cannot hide failure in either family.

## Current status

`LOCAL_QWEN35_SYNTHETIC_DEVELOPMENT_COMPLETE`

Model-free synthetic tooling and a real Qwen3.5-2B synthetic-image interface
gate have run locally. The model gate was repeated on both RTX 3090 GPUs and
produced byte-equivalent normalized rows/explanations. It is nonformal and
contains no patient data.

No CheXlocalize package or real-patient experiment has been opened by this
pivot.
CheXlocalize validation is development-only because this repository has prior
exposure to it. The official test split remains reserved for one-time local
evaluation after the complete protocol and identity package is frozen and the
user separately authorizes execution.

All future experiment execution is local to this workstation. Do not sync
active experiments to the server or submit SSH/Slurm jobs.

## Preserved validation

The existing commands check that the frozen BiVES/evidence surface remains
healthy; they do not authorize new research runs:

```powershell
python scripts/smoke_bives_cxr.py
python scripts/smoke_localization_causality_audit.py
python -m unittest discover -s tests -p "test_bives_*.py" -v
python scripts/audit_bives_b2_terminal.py --help
python scripts/audit_cxr_localization_causality.py --help
```

## Repository layout

```text
audit/              Active audit proposal, protocol, endpoints, novelty matrix
archive/            Frozen BiVES proposal and terminal negative-result report
bives_cxr/          Preserved BiVES and reusable audit/evidence utilities
configs/bives_cxr/  Frozen BiVES experiment identities
refine-logs/        Immutable C4/C5/C6I authorities and execution evidence
scripts/            Preserved tools and local audit development entrypoints
tests/              Contract tests
docs/               Handoff, schemas, and historical implementation records
legacy/             Pre-BiVES archived code and proposals
data/               Local-only data/manifests; ignored where required
outputs/             Generated evidence; ignored
local_runs/          Local runtime artifacts; ignored
```

Medical images, patient data, model weights, checkpoints, credentials, and
generated runtime outputs must not be committed.
