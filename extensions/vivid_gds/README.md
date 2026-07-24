# VIVID-GDS

VIVID-GDS is the clean VIVID-Med journal-extension route after the terminal
strict-SPD result. It tests whether a training-only UMS schema readout can
align the frozen-Qwen generation objective with the representation that is
actually deployed and probed downstream.

Stage A is terminal `NO-GO`: A3 improved over A2, but the frozen A2-A1 and
A2-A0 prerequisite comparisons failed. The current result authority is
`audit/VIVID_GDS_STAGE_A_TERMINAL_NO_GO_20260724.md`. No external or protected
test surface was opened.

Start with:

1. `audit/VIVID_GDS_STAGE_A_TERMINAL_NO_GO_20260724.md`;
2. `audit/VIVID_GDS_CASE_STUDY_AND_NEXT_DIRECTION_20260724.md`;
3. `audit/VIVID_GDS_EXPERIMENT_PROTOCOL_20260724.md`;
4. `audit/vivid_gds_stage_a_lock.json`;
5. `refine-logs/EXPERIMENT_TRACKER.md`;
6. `progress.md`.

The first decision is deliberately small: paired 20k MIMIC Stage-A arms under
one data lock, followed by one CheXpert expert-development probe protocol.
No external or protected test set is used for method selection.
