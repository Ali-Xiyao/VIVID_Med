# C6D MS-CXR real-package structure preflight log

Date: 2026-07-18
Scope: local, read-only release/schema/image-binding/independence preflight.
No image decode/render, annotation visualization, model load, score, GPU work,
experiment, server action, or C5 reopening occurred.

## Package placement and integrity

- Source ZIP was moved from `H:\2018b` to the public-data boundary at
  `H:\Xiyao_Wang\000_Public Dataset\MS-CXR` and extracted there.
- ZIP SHA-256:
  `62c829d307eb99a07fba82a3ee8346fd32dfcc5a226cfc00129049f684781bd9`.
- Publisher `SHA256SUMS.txt` entries for LICENSE, CSV, JSON, and converter all
  match their extracted files.
- No release file or medical image entered Git.

## Schema correction

The publisher JSON stores one COCO annotation row per box, not per unique
image-text pair. Therefore the official 15 Consolidation and 14 Pleural
Effusion figures are enforced on unique `(image_id, category_id, label_text)`
pairs, while every component box still undergoes LTWH bounds validation.

| Finding | Unique pairs | Boxes | Patients | Studies |
| --- | ---: | ---: | ---: | ---: |
| Consolidation | 15 | 25 | 15 | 15 |
| Pleural Effusion | 14 | 20 | 14 | 14 |

## Real structure preflight

Command:

```powershell
python scripts\audit_bives_c6_ms_cxr.py preflight-test-release
```

Result:

- status: `structure_preflight_passed_license_attestation_pending`
- target images bound to MIMIC metadata: 29/29
- local JPG files present and hashed: 29/29
- release-path mismatches: 0
- prior patient overlap: 0
- prior study overlap: 0
- raw identifiers emitted: false
- license gate passed: false
- model evaluation authorized: false
- canonical artifact SHA-256:
  `89d2b1c17541dfc6da9cf2567e428a24f11128125cec70a08504442dfbe98e50`
- ignored artifact file SHA-256:
  `0571be475f5e06a2153d8487aeee31d541a3d36b10f8efad05df63c6788e8801`

The preflight intentionally does not infer credentialed access, CITI training,
or a signed DUA from package possession. A user-authored local license record
is still required for the strict metadata intake. Even a strict intake pass
would not reopen C5; model evaluation requires a separately reviewed post-C5
research authority.

## Implementation hashes

| Surface | SHA-256 |
| --- | --- |
| `bives_cxr/c6_ms_cxr.py` | `551e0df6159db59aa9fb4117bd7afd7b93ba5aa243e52acaacc74703cd39024f` |
| `scripts/audit_bives_c6_ms_cxr.py` | `faa3954cbd34ce7595995325f4091da0c3a673f60e7189af63612b604dc3c231` |
| `tests/test_bives_c6_ms_cxr.py` | `d70babc5b246fdace74004f0c0f7c3b483465b224d303957c83036ef097b52be` |
| `docs/bives_cxr_c6_ms_cxr_intake.md` | `6fcb3d4008591c790ba4f13a59ff77393255c6423b79fd8099ec1bd9ad6bfb05` |

## Validation

```text
python -m py_compile bives_cxr/c6_ms_cxr.py scripts/audit_bives_c6_ms_cxr.py tests/test_bives_c6_ms_cxr.py
passed

python -m unittest discover -s tests -p "test_bives_c6_ms_cxr.py" -v
13/13 passed

python -m unittest discover -s tests -p "test_bives_*.py" -v
129/129 passed

python scripts/smoke_bives_cxr.py
decoder_kind=monotone_bipolar_conditional
has_flat_state_head=false
finite_gradients=true

git diff --check
passed
```

## Verdict

`STRUCTURE_PREFLIGHT_PASS_LICENSE_ATTESTATION_PENDING_NO_MODEL_AUTHORITY`
