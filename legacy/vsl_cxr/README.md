# Legacy VSL-CXR Archive

This directory preserves the pre-BiVES research line for provenance, pilot
evidence, and fair baselines. It is not the active training surface.

Archived here:

- Qwen3-VL/VSL/SAMEQ/CVCP experiment configs;
- VSL, CEQ, CCSH, AUCH, case-study, and Qwen3-VL scripts;
- CEQ/CCSH/AUCH model modules;
- prior VSL-CXR plans, ledgers, story documents, and earlier root proposals.

Scientific boundary:

- SAMEQ and HNMB are data/sampling strategies or baselines.
- CEQ can be used only as a localization baseline or initialization study.
- CCSH and AUCH are baseline readouts.
- None of these modules are imported by the active `bives_cxr` package.
- Historical outputs remain in ignored `outputs/` paths and were not moved,
  deleted, or relabelled as BiVES results.

The active method is defined by
[`BiVES_CXR_MIA_TMI_ready_proposal.md`](../../BiVES_CXR_MIA_TMI_ready_proposal.md)
and implemented in `bives_cxr/`.
