# RCSD-CXR terminal execution plan

## Outcome

RCSD-CXR is terminal NO-GO. The formal fusion contribution failed G2 and the
simplified field-anchor contribution failed G3. No downstream method
experiment is active.

## Authority

- Terminal decision: `docs/RCSD_CXR_terminal_gate_result_20260723.md`.
- Binding G2 amendment: `docs/RCSD_CXR_protocol_amendment_20260723_G2.md`.
- Original reviewed protocol: `docs/RCSD_CXR_active_protocol.md`.
- Background proposal:
  `provenance/RCSD_CXR_full_proposal_20260722.original.md`.

## Completed gates

| Phase | Final status | Decision |
| --- | --- | --- |
| E0 runnable-surface audit | completed | clean project and server execution surface verified |
| G0 data identity | pass | 215,098 canonical MIMIC studies; no forbidden overlap/test rows |
| G1 historical audit | pass | accepted SPD checkpoint confirmed as 4x2 |
| G2 source preparation | pass | source manifests and independent report gold qualified |
| G2 posterior validity | NO-GO | fusion NLL 6.98% worse than CheXbert |
| G1 simplified trainability | pass | both equal-budget variants overfit 256 rows |
| G3 paired 20k pilot | NO-GO | NLL/F1 gains below frozen thresholds |
| G4 full MIMIC | cancelled | G3 failed |
| G5 multi-seed/downstream | cancelled | G3 failed |
| G6 scale/teacher sensitivity | cancelled | G2/G3 failed |

## Final closeout checklist

- [x] Preserve G2 and G3 artifacts and hashes.
- [x] Write terminal scientific decision.
- [x] Cancel every locked downstream experiment.
- [x] Correct the G3 artifact top-level pass field and rerun the audit only.
- [x] Run local and remote unit tests: 45/45 passed on each side.
- [x] Refresh the server upload manifest and verify exact remote hashes.
- [x] Confirm no RCSD screen/process remains on allocation 3066.

## Boundaries

- Do not tune or rerun the G2 mapping, sources, folds, temperatures, or
  thresholds.
- Do not tune or rerun the G3 data surface, loss, field definitions, teacher,
  checkpoint rule, or budget.
- Do not open external tests to rescue the method.
- Do not run Qwen3.5 size variants.
- Do not reactivate VSL, BiVES, ARISE, VICER, or MORPH.
- Allocation 3066 is retained infrastructure and must not be cancelled as part
  of this closeout.
