# RCSD-CXR component-attribution audit

This directory is the active authority for the bounded post-NO-GO audit.

Read in order:

1. `RCSD_D0_D1_QWEN35_2B_TERMINAL_RESULT.md`: terminal result for the final
   permitted D0-CP versus D1 gate.
2. `rcsd_d0_d1_qwen35_2b_terminal_result.json`: machine-readable metrics,
   case-study aggregates, hashes, and consequences.
3. `rcsd_d0_d1_review_lock.json`: terminal machine lock; zero training jobs
   are authorized.
4. `RCSD_P0_NO_GO_VERDICT.md`: earlier deidentified G2/G3 gate verdict.
5. `RCSD_COMPONENT_ATTRIBUTION_PLAN.md`: D0-D3 evidence inventory and the
   now-completed final decision boundary.
6. `rcsd_component_status.json`: machine-readable historical metrics, hashes,
   checks, and arm status.
7. `tables/rcsd_g3_per_finding_macro_f1.csv`: aggregate per-finding G3
   comparison.
8. `RCSD_D0_D1_REVIEW_PROTOCOL.md`: the frozen single-factor D0-CP versus D1
   protocol.
9. `RCSD_D0_D1_AUTO_DEV_PROTOCOL.md`: execution and stop authority.
10. `tables/rcsd_d0_source_contract.csv`: Git and normalized-byte identities
   for the retained historical D0 source contract.

The audit authorizes zero training jobs. D1, posterior fusion, and field
anchoring are terminal NO-GO; all frozen thresholds remain unchanged and
external tests remain sealed.
