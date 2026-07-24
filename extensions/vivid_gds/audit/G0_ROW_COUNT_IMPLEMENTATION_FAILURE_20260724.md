# G0 row-count implementation failure

The first server prelaunch audit was intentionally fail-closed.

## Observed

- hard-UMS SHA-256 matched the frozen authority;
- 19,533 train rows;
- 1,679 validation rows;
- 21,212 total rows;
- train/validation patients were disjoint;
- all targets, images, teacher weights, ViT weights, probe inputs, and reused
  A2 identity checks passed.

## Cause

The proposal and initial readiness assertion treated “20k pilot” as exactly
20,000 train rows. The immutable manifest produced after filtering actually
contains 19,533 train rows. This was a documentation/assertion error, not a
data or model failure.

## Single repair

The lock, protocol, proposal, plan, and readiness assertion now state the exact
19,533/1,679/21,212 counts. No row, split, target, threshold, teacher,
checkpoint, or protected surface changed. The original remote
`readiness_prelaunch.json` remains preserved.
