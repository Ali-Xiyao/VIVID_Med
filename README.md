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
- CheXlocalize Qwen3.5 development result:
  [`audit/phase_h_chexlocalize_qwen35_development_result.md`](audit/phase_h_chexlocalize_qwen35_development_result.md)
- ARISE-CXR method-development result:
  [`audit/arise_cxr_method_development_result.md`](audit/arise_cxr_method_development_result.md)
- VICER-CXR V0 intervention-validity result:
  [`audit/vicer_v0_intervention_validity_result.md`](audit/vicer_v0_intervention_validity_result.md)
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

`LOCAL_CHEXLOCALIZE_QWEN35_DEVELOPMENT_COMPLETE`

Model-free synthetic tooling and a real Qwen3.5-2B synthetic-image interface
gate have run locally. The model gate was repeated on both RTX 3090 GPUs and
produced byte-equivalent normalized rows/explanations. It is nonformal and
contains no patient data.

The approved CheXlocalize validation-only release has now been downloaded and
MD5-verified, and a frozen Qwen3.5-2B development matrix completed over 100
image-finding pairs from 70 patients. It remains nonformal because this
repository has prior exposure to the validation split. The result shows modest
expert overlap and little positive association between overlap and causal
specificity, despite positive mean explanation-region specificity under both
frozen operators. The official test split was not downloaded or opened and
remains reserved for a separately frozen one-time local evaluation.

A subsequent Qwen3.5-2B-only ARISE-CXR development ladder tested trained dense
and patch-MIL verifiers, VinDr-train box supervision, full visual re-encoding,
and result-blind statistics-matched controls. These changes materially improve
the mechanism, but the final oracle still fails the pleural-effusion blur and
minimum three-finding gates. Selector/four-state training, larger models, and
test execution therefore remain closed.

ARISE-v1 status:

- oracle intervention development closed;
- 3/4 finding/operator cells passed;
- pleural-effusion blur inconclusive;
- selector/U/I/scaling not authorized;
- CheXlocalize test unopened;
- successor method not authorized in this terminal snapshot.

VICER-CXR V0 subsequently completed on new VinDr-train development identities.
Independent heads passed, but no tested intervention family was valid across
all four findings; only 4/12 finding-family cells passed. V1 coverage analysis,
V2 coalition selection, test execution, and scaling therefore remain closed.

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
