# BiVES-CXR Candidate Rescue Planning Manifest

**Status:** DRAFT_REVIEW_REQUIRED
**Created:** 2026-07-18
**Base commit:** `7c3e7ed`
**Parent authority:** `../BiVES_CXR_MIA_TMI_ready_proposal.md`

This manifest registers a planning-only candidate authority. It does not
authorize model loading, training, data mutation, VinDr-test reuse,
Qwen3.5-4B/9B scaling, server execution, or a new clinical claim.

| Artifact | Versioned file | Fixed alias | SHA-256 | Verification |
| --- | --- | --- | --- | --- |
| Candidate experiment plan | `EXPERIMENT_PLAN_20260718.md` | `EXPERIMENT_PLAN.md` | `1c0edf56544fc1a6291deda1cb54a4130023a196a21188454e2ad9615c375017` | versioned and fixed files are byte-identical |
| Candidate experiment tracker | `EXPERIMENT_TRACKER_20260718.md` | `EXPERIMENT_TRACKER.md` | `fd9e2325dc9362174367b4f38333012676d5edbeebe448cbbc67b7bbb02531e7` | versioned and fixed files are byte-identical |

## Review gate

- Every tracker row is blocked pending review or an earlier dependency.
- VinDr test remains frozen as a diagnostic-only surface after E10.
- VinDr train may support image-disjoint development only; the public DICOMs
  expose no patient, study, or series identifier.
- An independent patient-grouped final dataset remains unavailable.
- Execution can begin only after explicit user acceptance of this candidate
  authority and then only in the preregistered row order.
