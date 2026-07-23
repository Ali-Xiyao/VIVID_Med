# RCSD-CXR component-attribution audit

This directory is the active authority for the bounded post-NO-GO audit.

Read in order:

1. `RCSD_P0_NO_GO_VERDICT.md`: deidentified aggregate gate verdict.
2. `RCSD_COMPONENT_ATTRIBUTION_PLAN.md`: D0-D3 evidence inventory and the
   only permitted next decision.
3. `rcsd_component_status.json`: machine-readable metrics, hashes, checks, and
   arm status.
4. `tables/rcsd_g3_per_finding_macro_f1.csv`: aggregate per-finding G3
   comparison.
5. `RCSD_D0_D1_REVIEW_PROTOCOL.md`: the separately review-gated single-factor
   D0-CP versus D1 protocol.
6. `rcsd_d0_d1_review_lock.json`: machine lock that keeps execution disabled
   until every prerequisite is frozen and explicitly approved.
7. `tables/rcsd_d0_source_contract.csv`: Git and normalized-byte identities
   for the retained historical D0 source contract.

The audit does not authorize training. Full RCSD remains NO-GO, all frozen
thresholds remain unchanged, and external tests remain sealed.
