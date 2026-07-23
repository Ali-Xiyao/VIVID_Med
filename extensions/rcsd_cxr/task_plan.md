# RCSD-CXR component-attribution audit plan

## Goal

Freeze the exact G2/G3 NO-GO evidence, determine which D0-D4 arms genuinely
exist, and produce a deidentified decision package. Do not launch a new method
or diagnostic run in this phase.

## Authority

- Active verdict: `audit/RCSD_P0_NO_GO_VERDICT.md`.
- Arm inventory: `audit/RCSD_COMPONENT_ATTRIBUTION_PLAN.md`.
- Machine-readable status: `audit/rcsd_component_status.json`.
- Frozen historical result:
  `docs/RCSD_CXR_terminal_gate_result_20260723.md`.

## Phases

| Phase | Status | Output |
| --- | --- | --- |
| A. Freeze base implementation | completed | commit `bc1105f880116e97e06da023110b7080debc28a4` |
| B. Verify aggregate G2/G3 evidence | completed | exact metrics, gates, and hashes |
| C. Classify D0-D4 evidence | completed | exact D0 missing; D1 untested; D2/D3 NO-GO; D4 prohibited |
| D. Build deidentified verdict | completed | Markdown, JSON, and per-finding CSV |
| E. Validate audit package | completed | integrity pass, 48/48 tests, clean Git scope audit |
| F. Publish audit branch | completed | pushed `codex/rcsd-no-go-audit` |

## Binding decisions

- Full RCSD remains NO-GO.
- D2 posterior fusion is rejected by G2.
- The tested D3 field anchor is rejected by G3.
- The current unanchored SPD comparator is D0-like, not an exact original
  VIVID D0 reconstruction.
- D1 selective agreement weighting is untested.
- No new training is authorized by this audit.
- D4, external tests, full-data scaling, multi-seed expansion, institution
  mixing, and teacher scaling remain locked.

## Review gate for any future D1

Before a D1 run could be authorized, a separate protocol must freeze:

- exact original-SPD D0 comparator;
- scalar agreement-weight definition;
- 20k patient-locked dataset and seed;
- equal budget, optimizer, augmentation, and checkpoint rule;
- expert-development AUROC/AUPRC endpoints;
- reliability coverage/quartile table;
- promotion and stop thresholds.

Until that review exists, the correct action is evidence audit only.

## Errors encountered

| Error | Resolution |
| --- | --- |
| Historical G2 JSON uses top-level `pass: true` although `gate.g2_pass` is false. | Preserve the artifact; normalize audit completion and scientific gate outcome in the new JSON without rerunning G2. |
| Attached critique assumed gate metrics were unavailable from Git. | Retrieve only read-only aggregate server artifacts and publish deidentified metrics/hashes, not raw outputs. |
| A combined multi-file patch could not match one historical mojibake context line. | Split the change into narrow UTF-8-safe patches and apply each file independently. |
