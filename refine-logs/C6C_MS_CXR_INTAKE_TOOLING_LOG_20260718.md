# C6C MS-CXR metadata-intake tooling log

Date: 2026-07-18
Scope: local metadata-only tooling; no download, login, terms acceptance, image
decode/render, annotation visualization, model load, score, GPU work,
experiment, server action, or C5 reopening.

## Authority and source contract

- Active research authority remains `BiVES_CXR_MIA_TMI_ready_proposal.md`.
- C5 remains `FAIL_FINAL_STOP`.
- Official source: `https://physionet.org/content/ms-cxr/1.1.0/`.
- Required release file: `MS_CXR_Local_Alignment_v1.1.0.json`.
- Frozen scope: publisher `test` annotations for Consolidation and Pleural
  Effusion only.
- Official test integrity counts: 15 Consolidation image-annotation pairs from
  15 subjects and 14 Pleural Effusion pairs from 14 subjects.
- User-side access remains mandatory: credentialed PhysioNet access, required
  CITI training, and signed DUA. The tooling does not perform those actions.

## Implemented surfaces

| Surface | SHA-256 | Purpose |
| --- | --- | --- |
| `bives_cxr/c6_ms_cxr.py` | `ac5b98f4d4f8d24c6eba1e9125620860d58588e12fa0749fb9c09604251baa51` | Fail-closed registry, license, COCO schema, test-count, box, MIMIC binding, overlap, and no-authority checks. |
| `scripts/audit_bives_c6_ms_cxr.py` | `fe02be5fd79cc193b6593550459205d82b86eca4f34387182bf9550920fa12a8` | Local CLI for building the frozen prior registry or auditing an acquired release. |
| `tests/test_bives_c6_ms_cxr.py` | `0b8cb99cb57ef4c9415bc597a3bd5fddd61e6db2ea9033937441dc22951dfc2f` | Ten synthetic fail-closed contract tests. |

## Frozen prior-use registry

The ignored artifact is
`local_runs/bives_cxr/c6_ms_cxr_intake/prior_mimic_access_registry.json`.

- file SHA-256:
  `6cdc625f28f955f886492a790f7875f4d94c2ed77d529291746e2268dae7f5bb`
- canonical artifact SHA-256:
  `d2b28786c69984538064b03546244b479ed8bb7a7b646cfa08d166be52deca70`
- patients: 1,414
- studies: 5,008
- patient-set SHA-256:
  `106e13b9500ff5ad9c7e67a168861c04a0f2486a9786ebc8850bf5000e207950`
- study-set SHA-256:
  `76e8ae65bc0d740908d064fff5748ddec390eb121c456a8f75f42020c472cd86`
- serialized raw patient-ID matches: 0
- serialized raw study-ID matches: 0
- Git boundary: ignored by `.gitignore: local_runs/`.

The registry was reconstructed from the frozen P0 5k, weak-S/C, and proxy-S/C
JSONL surfaces. Its counts and set hashes exactly reproduce the C6A authority.

## Validation

```text
python -m py_compile \
  bives_cxr/c6_ms_cxr.py \
  scripts/audit_bives_c6_ms_cxr.py \
  tests/test_bives_c6_ms_cxr.py

python -m unittest discover -s tests -p "test_bives_c6_ms_cxr.py" -v
10/10 passed

python -m unittest discover -s tests -p "test_bives_*.py" -v
126/126 passed

python scripts/smoke_bives_cxr.py
decoder_kind=monotone_bipolar_conditional
has_flat_state_head=false
finite_gradients=true

git diff --check
passed
```

## Current verdict

`TOOLING_COMPLETE_WAITING_USER_AUTHORIZED_PACKAGE`

Neither `H:\Xiyao_Wang\000_Public Dataset\MS-CXR` nor its lowercase variant
exists. Therefore no real MS-CXR intake artifact or data lock has been emitted.
Even a future metadata-intake pass will keep
`model_evaluation_authorized=false`; a separate reviewed post-C5 research
authority would still be required before any model evaluation.
