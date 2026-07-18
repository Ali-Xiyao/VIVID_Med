# BiVES-CXR C6B Metadata Intake Tooling Log

**Status:** `TOOLING_COMPLETE_WAITING_DATA`

**Date:** 2026-07-18

**Parent authority:** `C6A_OFFICIAL_DATA_ACQUISITION_PLAN_20260718.md`

## Scope

C6B implements only a local, fail-closed metadata intake surface for an
already-downloaded CheXlocalize release. It does not download data, accept
terms, store an Azure SAS URL, decode or render an image, visualize an expert
annotation, load a model, compute a score, run an experiment, reopen C5, or
authorize Qwen3.5-4B/9B.

## Implemented surfaces

- `bives_cxr/c6_intake.py`
  - parses official CheXpert/CheXlocalize patient-study-view identities;
  - builds a validation-access registry containing hashed identifiers only;
  - validates a user-authored license/provenance record without account data;
  - accepts only publisher test paths and rejects path traversal;
  - requires the official 668-image/500-patient test identity;
  - validates Consolidation and Pleural Effusion expert contour structure;
  - hashes metadata and test image bytes without decoding images;
  - fails on prior patient/study/image overlap;
  - emits no raw identifiers and no model-evaluation authority.
- `scripts/audit_bives_c6_chexlocalize.py`
  - `build-prior-registry` creates the ignored local hashed registry;
  - `audit-test-release` creates the ignored metadata-only intake record after
    the user supplies the release and license record.
- `tests/test_bives_c6_intake.py`
  - covers valid intake, raw-ID non-emission, official key parsing, validation
    split rejection, path traversal rejection, missing-target rejection, and
    prior-patient-overlap rejection.

## Source identities

| Artifact | SHA-256 |
| --- | --- |
| `bives_cxr/c6_intake.py` | `6f46100f32854778d3c059b1757ae70df1a93ba70831d55b868fdfc35dff2d3c` |
| `scripts/audit_bives_c6_chexlocalize.py` | `fa407013ca795d4f5bd2223890ba4b1bfb595a6928b76b8d65a39e2b9e1c0216` |
| `tests/test_bives_c6_intake.py` | `4bc2510bc0c1e0aae4e0ae5f409940a22084c0fafc3311eb1664216fba840f78` |
| local CheXpert `valid.csv` | `aa43403f7bb183a35ae3bb9152896c1fe9a5b297635e0b541e363b776830ec4a` |
| ignored prior-access registry file | `7f2d78255c932b9ec97c78b1820b0e090012a45cea4d76b3b01864c5faa9a95a` |
| prior-access registry canonical payload | `1703aaf6c548e3eea57db66ba41a3f0f516729afdb6839cba9202b35ebb736cd` |

The ignored registry contains 234 validation images, 200 patients, and 200
studies. A raw-identifier pattern audit found zero patient/study/view values in
the serialized registry. Individual identifier hashes remain ignored local
data and are not committed.

## Verification

```text
python -m py_compile bives_cxr/c6_intake.py \
  scripts/audit_bives_c6_chexlocalize.py tests/test_bives_c6_intake.py
python -m unittest discover -s tests -p "test_bives_c6_intake.py" -v
python -m unittest discover -s tests -p "test_bives_*.py"
python scripts/smoke_bives_cxr.py
```

Results:

- C6B narrow contracts: `7/7` passed.
- Full active BiVES suite: `116/116` passed.
- CPU smoke: finite gradients, normalized four-state probabilities, and
  `has_flat_state_head=false`.

## Current gate

No `H:\Xiyao_Wang\000_Public Dataset\CheXlocalize` directory exists. The
tooling is complete, but real metadata intake remains blocked until the user
downloads the official package and supplies a non-secret license/provenance
record. C5 remains `FAIL_FINAL_STOP` throughout.
