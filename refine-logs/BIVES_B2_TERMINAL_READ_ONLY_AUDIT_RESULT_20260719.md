# BiVES B2 terminal read-only audit

**Date:** 2026-07-19

**Status:** `COMPLETE_TERMINAL_NEGATIVE_NO_MODEL_ACCESS`

**Route decision:** C6I permanently ends the B2 rescue route. No C6J, rerun,
operator/control tuning, Qwen3.5-4B/9B scale-up, or model access is authorized.

## Scope and frozen inputs

The audit reads the frozen C5 confirmation rows/geometry/metrics, frozen C6I
evaluation rows/metrics, frozen C6I actual-input geometry/masks, and the bound
MS-CXR manifest/images. It does not load Qwen, compute a score, train, tune, or
create a new experiment opening. The two image operators remain exactly the
frozen local-ring mean and masked Gaussian blur implementations.

Generated ignored artifacts:

- `local_runs/bives_cxr/b2_terminal_read_only_audit/audit_summary.json`
- `local_runs/bives_cxr/b2_terminal_read_only_audit/effect_rows.jsonl`
- `local_runs/bives_cxr/b2_terminal_read_only_audit/image_space_rows.jsonl`
- `local_runs/bives_cxr/b2_terminal_read_only_audit/paired_effect_scatter.png`

| Artifact | SHA-256 |
| --- | --- |
| audit summary | `3f070db65a0e403ca34b871cc6502a30ee6885522b52d7153e5d365ec9339e86` |
| normalized effect rows | `09c78953288d2b233981ac4c9b3555b11d2335c5c462ef8d9619b6c66d3b7b25` |
| image-space rows | `a6171903611c2945c07cbc702bf024bf4ca41f7f4903b728d176814751becf64` |
| paired scatter | `6bc5a3f6a08b4d4c6df31b5731d2a5a43969d79e34581592ad97180d3682d45b` |

## Stage-level synthesis

| Stage | Data role | Mechanism result | Other gate | Decision |
| --- | --- | --- | --- | --- |
| C4 | VinDr-train protocol-design positives | pass | confirmation not opened | advance once to C5 |
| C5 | image-disjoint VinDr-train confirmation | pass | polarity fail | `FAIL_FINAL_STOP` |
| C6I | independent MS-CXR publisher test, positive-only | fail | no classification metric | `FAIL_FINAL_STOP` |

C4 therefore established only an internal protocol-design effect. C5 showed
that the mechanism effect could reproduce while the full B2 route still failed
its polarity condition. C6I then showed that a geometry-correct independent
positive-only intervention did not preserve the mechanism across findings.

## Target versus control decomposition

The audit uses a fixed, threshold-free four-way taxonomy: target sign reversal;
positive target dominance; positive control dominance/target inertness; or both
nonpositive/tied. The label is descriptive and is not used to select a model or
change an intervention.

| Source/operator/finding | n | mean target effect | mean control effect | mean TCIG | target dominant | target reversal | control dominant/inert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C5 local mean / Consolidation | 59 | 0.048009 | 0.009286 | 0.038723 | 54 | 0 | 5 |
| C5 local mean / Pleural Effusion | 318 | 0.035687 | 0.008293 | 0.027395 | 223 | 59 | 36 |
| C5 blur / Consolidation | 59 | 0.031616 | -0.000198 | 0.031815 | 45 | 10 | 4 |
| C5 blur / Pleural Effusion | 318 | 0.010272 | 0.001572 | 0.008700 | 201 | 85 | 32 |
| C6I local mean / Consolidation | 15 | 0.019766 | 0.015286 | 0.004480 | 7 | 4 | 4 |
| C6I local mean / Pleural Effusion | 14 | 0.003387 | 0.012486 | -0.009099 | 4 | 6 | 4 |
| C6I blur / Consolidation | 15 | 0.004897 | 0.020421 | -0.015523 | 4 | 7 | 4 |
| C6I blur / Pleural Effusion | 14 | 0.027186 | 0.001122 | 0.026064 | 10 | 4 | 0 |

The independent failure is finding/operator dependent. Blur produces the
intended target-dominant pattern in 10/14 Pleural Effusion cases, but only 4/15
Consolidation cases; 7/15 Consolidation cases reverse the target-effect sign.
Local mean is weaker and more control-sensitive in both findings.

## Image-space perturbation audit

The same frozen C6I images, masks, ring width, sigma, and truncate values were
replayed without model access. L1 is normalized to `[0,1]`; SSIM is a fixed
Gaussian-window luminance SSIM over the valid image content.

| Operator/finding | target masked L1 | control masked L1 | target-control | target SSIM | control SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| local mean / Consolidation | 0.139455 | 0.205362 | -0.065906 | 0.940376 | 0.936471 |
| local mean / Pleural Effusion | 0.128315 | 0.166724 | -0.038409 | 0.974631 | 0.972172 |
| blur / Consolidation | 0.031550 | 0.026277 | 0.005273 | 0.965053 | 0.978128 |
| blur / Pleural Effusion | 0.024956 | 0.017992 | 0.006964 | 0.990212 | 0.992434 |

Local mean changes control pixels more strongly than target pixels, so operator
asymmetry plausibly contributes to its weak independent TCIG. That explanation
does not rescue blur: target blur is the stronger pixel perturbation for both
findings, yet Consolidation has negative mean TCIG while Pleural Effusion is
positive. The terminal failure is therefore not reducible to a single global
perturbation-strength mismatch.

## Descriptive associations

No C6I localization feature shows a stable relationship with TCIG across all
four operator/finding cells. In particular, Spearman correlations between
top-K localization gain and TCIG range only from `-0.020` to `0.196` at
`n=14-15`. Some control-geometry associations reach moderate magnitudes in one
cell, but they change sign across cells and are too small for inferential or
causal claims. These are post-stop descriptive observations, not a selection
rule.

## Final interpretation

The correct main claim is no longer that the B2 evidence set is causally valid.
The defensible result is a localization-causality audit: positive localization
gain can coexist with failure of matched target-versus-control necessity, and
the failure can be strongly finding dependent even after exact geometry and
operator identity are fixed. Future CheXlocalize work requires a separately
frozen validation-then-test protocol; it is not opened by this audit.

## Reproducibility and validation

- A complete second audit replay reproduced all four ignored artifact SHA-256
  values byte-for-byte (`DETERMINISTIC_REPLAY=True`).
- New terminal-audit contracts: `4/4` pass.
- Complete active `test_bives_*.py` suite: `149/149` pass.
- Synthetic CPU smoke: normalized probabilities, finite gradients, and no flat
  state head.
- `py_compile` and `git diff --check`: pass.
