# Localization-causality audit authority

This directory is the active research surface. The frozen BiVES B2 method route
is retained under `../archive/` and in its original tracked execution records.

## Read order

1. [`CXR_localization_causality_audit_proposal.md`](CXR_localization_causality_audit_proposal.md)
2. [`chexlocalize_validation_test_protocol.md`](chexlocalize_validation_test_protocol.md)
3. [`primary_endpoints.md`](primary_endpoints.md)
4. [`novelty_matrix.md`](novelty_matrix.md)
5. [`development_row_schema.md`](development_row_schema.md)
6. [`phase_c_synthetic_development_result.md`](phase_c_synthetic_development_result.md)
7. [`local_model_development_opening_20260719.json`](local_model_development_opening_20260719.json)
8. [`MANIFEST.md`](MANIFEST.md)
9. [`../archive/BiVES_B2_terminal_negative_report.md`](../archive/BiVES_B2_terminal_negative_report.md)

## Current execution status

`LOCAL_QWEN35_SYNTHETIC_DEVELOPMENT_COMPLETE`

Local model-free synthetic tooling/tests and the separately opened
Qwen3.5-2B synthetic-image GPU gate are complete. The same frozen gate produced
identical normalized rows/explanations on both workstation RTX 3090 GPUs. No
CheXlocalize package, patient data, real-data score, or test split was opened.
Future real-data execution remains local and requires a separately frozen
protocol, data identity, model/explanation/operator manifest, and explicit
data/test opening lock.
