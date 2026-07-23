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
| G. Reconstruct exact D0 contract | completed | source identities and D0-H/D0-CP distinction frozen |
| H. Freeze D0-vs-D1 review protocol | completed | Markdown protocol, machine lock, pure CPU contracts |
| I. Validate and publish review package | completed | 59/59 tests, scope audit, and Git push completed |
| J. Record automatic execution authority | completed | workstation restriction revoked; server allocation 3066 authorized |
| K. Resolve R0-R2 prerequisites | completed | parity, real-weight smoke, manifests, 67/67 tests |
| L. Run paired 256-row overfit | completed | D0 98.20% at step 400; D1 98.05% at step 500 |
| M. Requalify Qwen3.5-2B teacher | completed | parity, fast-path real-weight smoke, paired overfit passed on allocation 3066 |
| N. Run paired 20k pilot | completed | both arms completed 3,000 steps; D1 failed the first frozen promotion condition |
| O. Case study and single-factor repair | completed | case study completed; no preregistered repair exists |
| P. Freeze terminal decision | completed | terminal NO-GO frozen; integrity audit PASS; 68/68 tests PASS |
| Q. Publish terminal closure | in progress | stage, commit, push, and verify the current audit branch |
| R. Continue method experiments | blocked | current terminal lock authorizes zero jobs; requires a new reviewed strict VIVID/SPD protocol |

## Binding decisions

- Full RCSD remains NO-GO.
- D2 posterior fusion is rejected by G2.
- The tested D3 field anchor is rejected by G3.
- The current unanchored SPD comparator is D0-like, not an exact original
  VIVID D0 reconstruction.
- D1 selective agreement weighting is untested.
- The user explicitly authorized the bounded D0/D1 automatic development loop
  on 2026-07-23. Training remains machine-blocked until R0-R2 pass.
- D4, external tests, full-data scaling, multi-seed expansion, institution
  mixing, and teacher scaling remain locked.
- The user's request to continue automatically does not reopen the failed D1
  identity. Further training must be a separately reviewed strict VIVID/SPD
  extension, not a post-hoc RCSD repair.

## Review gate for any future D1

Before a D1 run could be authorized, a separate protocol must freeze:

- exact original-SPD D0 comparator;
- scalar agreement-weight definition;
- 20k patient-locked dataset and seed;
- equal budget, optimizer, augmentation, and checkpoint rule;
- expert-development AUROC/AUPRC endpoints;
- reliability coverage/quartile table;
- promotion and stop thresholds.

The review now exists. The next action is to produce its missing prerequisites;
permission alone does not permit skipping them.

## Current continuation boundary

The later 2026-07-23 instruction explicitly authorizes automatic execution.
`audit/RCSD_D0_D1_AUTO_DEV_PROTOCOL.md` controls the state machine. The user's
2026-07-23 amendment authorizes allocation `3066`; all scientific/test locks
remain unchanged.

The subsequent teacher amendment replaces Qwen2.5-1.5B with Qwen3.5-2B as the
frozen primary teacher. Because this changes tokenization and projector output
dimension, the earlier Qwen2.5 overfit evidence does not authorize a 20k run.
Qwen3.5 parity, real-weight smoke, and paired overfit must pass first. Other
Qwen3.5 sizes remain conditional sensitivity experiments after the primary
D0/D1 gate, not competing methods selected from development results.

## Errors encountered

| Error | Resolution |
| --- | --- |
| Historical G2 JSON uses top-level `pass: true` although `gate.g2_pass` is false. | Preserve the artifact; normalize audit completion and scientific gate outcome in the new JSON without rerunning G2. |
| Attached critique assumed gate metrics were unavailable from Git. | Retrieve only read-only aggregate server artifacts and publish deidentified metrics/hashes, not raw outputs. |
| A combined multi-file patch could not match one historical mojibake context line. | Split the change into narrow UTF-8-safe patches and apply each file independently. |
| Step-level `squeue` rejected `%T` as an unsupported format token during a read-only monitor check. | The job and experiment were unaffected; future step checks use supported fields and treat the advancing D1 log plus queue state as the primary liveness evidence. |
| A PowerShell monitor command failed locally while parsing nested escaped quotes for remote `grep`; no SSH command was sent. | Replaced the fragile filtered command with a single-quoted literal remote command that reads the authoritative JSON and log directly. |
| A nonessential remote `find -printf` file-list suffix was split by SSH argument parsing. | The authoritative queue and D1 log were read successfully and showed a healthy run; omit formatted file listing until the final artifact freeze. |
| A combined PowerShell diagnostic command contained an empty pipe element, so it failed locally before SSH execution. | Split remote aggregation and local arithmetic into separate commands; the remote read-only diagnostic then completed successfully. |
| `docs/README.md` was absent during index inspection. | Use the extension root `README.md` and `audit/README.md` as the maintained entrypoints; no historical document was overwritten to fabricate an index. |
| A broad multi-file terminal-freeze patch missed the exact AGENTS.md line context and was rejected before applying. | Split the freeze into narrow per-file patches and preserved all existing changes. |
| `python -m unittest tests.test_rcsd_d0_d1_review` failed because `tests/` is not a Python package. | Ran the supported discovery command; the targeted file's six tests and the full 68-test suite all passed. |
