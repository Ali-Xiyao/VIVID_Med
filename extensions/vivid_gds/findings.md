# Findings

## 2026-07-24 intake

- The strict SPD route is terminal NO-GO; prefix4 remains the supported
  generation baseline and is reusable as A2.
- The frozen hard-UMS manifest contains `row_id`, patient/study identity,
  split, image path, and a deterministic JSON target. Its target contains only
  selected answerable findings, so the same JSON can be the single authority
  for A0/A1/A2/A3 field masks.
- The canonical 20k source CSV also retains `report_path`, but raw reports are
  not needed for the proposal's historical free-text ablation. The historical
  VIVID free-text arm rendered finding states through sentence templates.
- A1 will therefore deterministically render the exact selected UMS fields
  into historical-style free text. This changes only representation format,
  not data identity or field coverage.
- A0 has no generation objective, so its pretraining checkpoint must be chosen
  by validation schema NLL. A1/A2/A3 remain selected by generation token NLL.
- Allocation 3066 is running on gpu01 and, at the latest audit, only its batch
  step is present. A new exclusive step may be queued without stopping or
  competing with unrelated work.
