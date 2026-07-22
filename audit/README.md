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
7. [`phase_e_frozen_existing_data_retrospective_result.md`](phase_e_frozen_existing_data_retrospective_result.md)
8. [`phase_f_vindr_qwen35_development_result.md`](phase_f_vindr_qwen35_development_result.md)
9. [`phase_h_chexlocalize_qwen35_development_result.md`](phase_h_chexlocalize_qwen35_development_result.md)
10. [`arise_cxr_method_development_result.md`](arise_cxr_method_development_result.md)
11. [`vicer_v0_intervention_validity_result.md`](vicer_v0_intervention_validity_result.md)
12. [`../docs/SCIENTIFIC_STATUS.md`](../docs/SCIENTIFIC_STATUS.md)
12. [`local_model_development_opening_20260719.json`](local_model_development_opening_20260719.json)
13. [`local_vindr_qwen35_development_opening_20260719.json`](local_vindr_qwen35_development_opening_20260719.json)

The VICER successor-method entrypoint is
[`../vicer_cxr/VICER_CXR_method_proposal.md`](../vicer_cxr/VICER_CXR_method_proposal.md).
V0 is now terminal negative: no intervention family passed all four findings,
so V1 and V2 remain locked.
14. [`local_chexlocalize_validation_acquisition_opening_20260719.json`](local_chexlocalize_validation_acquisition_opening_20260719.json)
15. [`local_chexlocalize_qwen35_development_opening_20260719.json`](local_chexlocalize_qwen35_development_opening_20260719.json)
16. [`MANIFEST.md`](MANIFEST.md)
17. [`../archive/BiVES_B2_terminal_negative_report.md`](../archive/BiVES_B2_terminal_negative_report.md)
18. [`local_h_xiyao_asset_inventory.md`](local_h_xiyao_asset_inventory.md)

## Current execution status

`LOCAL_CHEXLOCALIZE_QWEN35_DEVELOPMENT_COMPLETE`

Local model-free synthetic tooling/tests and the separately opened
Qwen3.5-2B synthetic-image GPU gate are complete. The same frozen gate produced
identical normalized rows/explanations on both workstation RTX 3090 GPUs. A
subsequent read-only Phase-E retrospective aggregated immutable prior VinDr C5
and MS-CXR C6I scores without loading a model or creating a new score. Phase F
then completed a locked four-image VinDr-train Qwen3.5-2B development gate.
After the user's approved Redivis access, Phase H downloaded and MD5-verified
all 2,343 allowlisted CheXlocalize validation files, kept every test file
sealed, and completed the frozen 100-pair/70-patient Qwen3.5-2B development
matrix. Its nonformal result separates modest localization from positive but
weakly associated explanation-region causal specificity. Future test execution
still requires a separately frozen one-time authority.

The subsequent local ARISE-CXR candidate-method ladder is also complete and
fails closed before selector training. Box supervision and result-blind
statistics-matched controls materially improve the expert-region causal
effects, but pleural-effusion blur remains statistically unstable and only two
findings are available. See `arise_cxr_method_development_result.md`. Test,
4B/9B scaling, U/I, and server execution remain closed.

The independent VICER-CXR V0 ladder is also complete. All calibration heads
passed, but only 4/12 finding-family validity cells passed and no complete
operator family survived. The aggregate case study identifies removal
validity, followed by pneumothorax preservation, as the dominant failure;
valid-row target-control gaps remain positive. V1/V2 and test remain closed.
