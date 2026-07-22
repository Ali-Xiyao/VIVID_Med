# VICER-CXR V0 Experiment Tracker

| ID | Status | Artifact / evidence |
| --- | --- | --- |
| V000 | complete | Source commit `3aa8adb`; 3/3 contracts, compileall, CLI surfaces, hashes, and proposal frozen. |
| V001 | in_progress_v2_geometry | Data lock complete. Opening v1 was consumed by a score-free exact-translation infeasibility before model access. Opening `audit/local_vicer_v0_opening_v2_20260722.json` freezes the tested score-blind fallback and unchanged thresholds; rebuild 32/32 geometry records next. |
| V002 | locked | Wait for V001 data/geometry locks. |
| V003 | locked | Wait for independent head calibration pass. |
| V100 | locked | Requires V003 pass. |
| V200 | locked | Requires a separately frozen V1 pass. |
