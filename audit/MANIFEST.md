# Audit development manifest

**Branch:** `codex/localization-causality-audit`

**Frozen predecessor tag:** `bives-b2-terminal-8bb1a94`

**Status:** Phase-C synthetic development complete; CheXlocalize data/test closed

| Artifact | SHA-256 |
| --- | --- |
| `CXR_localization_causality_audit_proposal.md` | `89f7bcbf9d938895229d73125161c4178b196daa2c29500d198f6e8989af3efb` |
| `chexlocalize_validation_test_protocol.md` | `2e36e82fabb6ee790f41af2853b21a1f01cf4e90823bb112d30084198ee04dbc` |
| `primary_endpoints.md` | `c110a20f4b10b0a1db4f55e269c13e2ffa6987342fe0491e012a850225bc9b5e` |
| `novelty_matrix.md` | `1a56bf5e838b7a94ed91082806e8c91840c8cb63087884e927120a1c6c5c35bf` |
| `development_row_schema.md` | `5cd4e954ad4d4e08d0e717c95334155929509131dc4f79e7bbd68216f8b2569b` |
| `local_model_development_opening_20260719.json` | `8b1ad92b87fcb12fe13003e1b9fd6636f268d1c1c73242601a60b8290055f6ef` |
| `phase_c_synthetic_development_result.md` | `2c02ba69f2ca8d71c4874a57b00f25b0b152ba27f3cf283a4615fb6ba85d49b2` |
| `../bives_cxr/localization_causality.py` | `1765f11426f3ffc6128da9e5d927666594fd1f2a15f786e5e2f92fc2a7a7905c` |
| `../bives_cxr/qwen35_localization_audit.py` | `cd29de2183e06a84d08f69a42669ba3def9770fb6621cfba204846f136f7006e` |
| `../scripts/audit_cxr_localization_causality.py` | `f52df3bbd32c129a53dcc40b8bbed104d20ca01f01c8d69058b56ebe902ca842` |
| `../scripts/smoke_localization_causality_audit.py` | `748a3f321dbeb09bf27dbd9bf58c0573544f9043ff7ca439856db244d94c72cc` |
| `../scripts/smoke_qwen35_localization_causality.py` | `2d9ee42cd482f89823a3cef70bb985e501ba2220514783770d286ad880d82925` |
| `../tests/test_bives_localization_causality_audit.py` | `de18d2956d8dc44f376e866c9cd728cb6b5da3518311d5424fa214e36fcb31a6` |
| `../tests/test_bives_qwen35_localization_audit.py` | `485966792c390507566fec1e7be8492afd71cc153cb5518c8455c5649e406475` |
| `../archive/BiVES_B2_terminal_negative_report.md` | `61e745b69e6a1485b903c91a13b58ecd029c77dec913fdd87fb0709b5db57c2f` |

The source hashes bind the complete Phase-C synthetic development package.
The ignored runtime evidence is separately identified by:

- model-free CLI development-lock canonical SHA-256
  `159015e8c0dedd3c6f1cbbd6b6116f8638f0f6161eaaeb6f735b238fc33c7deb`;
- cross-GPU normalized Qwen3.5 synthetic-gate SHA-256
  `4c7931088e96a5a6fa1fb619d7366bc30130063960da58d475e0754a767c1cf4`;
- local Qwen3.5-2B snapshot SHA-256
  `6b57c58ceb3c97d4199d4e8e32b54f6ff11d0cf3a8bde3a4e940d5775ae12120`.

These identities do not open a real dataset or the one-time test. A future
real-data package must create a separately versioned data/model/explanation/
operator lock and must not silently replace these values.
