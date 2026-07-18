# BiVES C6C MS-CXR local metadata intake

This is a local, metadata-only, fail-closed intake. It does not download data,
accept terms, decode or render images, load a model, access scores, or reopen
the C5 final stop.

## Local package placement

The acquired v1.1.0 release stays outside the repository:

```text
H:\Xiyao_Wang\000_Public Dataset\MS-CXR\
  MS_CXR_Local_Alignment_v1.1.0.json
  MS_CXR_Local_Alignment_v1.1.0.csv
  ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-1.1.0.zip
```

The ZIP SHA-256 is
`62c829d307eb99a07fba82a3ee8346fd32dfcc5a226cfc00129049f684781bd9`.
All four files named by the publisher `SHA256SUMS.txt` have been verified.

## Structure-only preflight

The structure preflight does not assert credentialed access, CITI training, or
DUA status and therefore cannot authorize model evaluation:

```powershell
python scripts\audit_bives_c6_ms_cxr.py preflight-test-release
```

The real release passes this preflight: 25/20 boxes collapse to the official
15/14 unique `(image, category, label_text)` pairs, all 29 target images bind
to local MIMIC metadata/JPG files, and patient/study overlap with the frozen
prior-use registry is zero. The ignored output explicitly records
`license_gate_passed=false` and `model_evaluation_authorized=false`.

## User-side access attestation

The user must personally confirm the PhysioNet requirements for MS-CXR v1.1.0
(credentialed access, required CITI training, and signed DUA). The tool never
infers those facts merely from possession of the package.

Do not paste credentials, cookies, tokens, or private download URLs into chat,
the repository, or the license record.

## License record

Create a local JSON file outside Git, for example
`H:\Xiyao_Wang\000_Public Dataset\MS-CXR\license_record.json`:

```json
{
  "dataset_name": "MS-CXR",
  "release_version": "1.1.0",
  "source_url": "https://physionet.org/content/ms-cxr/1.1.0/",
  "terms_url": "https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/",
  "retrieved_at": "YYYY-MM-DD",
  "package_sha256": "SHA256_OF_THE_RECEIVED_PACKAGE_OR_ARCHIVE",
  "credentialed_access_confirmed": true,
  "citi_training_confirmed": true,
  "dua_signed_by_user": true,
  "access_secret_not_persisted": true
}
```

## Commands

From the repository root:

```powershell
python scripts\audit_bives_c6_ms_cxr.py build-prior-registry

python scripts\audit_bives_c6_ms_cxr.py audit-test-release `
  --license-record "H:\Xiyao_Wang\000_Public Dataset\MS-CXR\license_record.json"
```

The strict audit uses the local MIMIC metadata and image tree by default. It
accepts only official `test` annotations, requires exactly 15 Consolidation
and 14 Pleural Effusion unique image-text pairs/subjects, validates all 25/20
LTWH boxes and image-file hashes, and rejects any patient/study overlap with
the frozen prior-use registry.

The 2026-07-18 strict local intake passed after explicit user access
confirmation. Its ignored artifact reports `license_gate_passed=true`, zero
prior patient/study overlap, and `model_evaluation_authorized=false`.

Outputs stay under ignored `local_runs/bives_cxr/c6_ms_cxr_intake/`. They must
not be committed. A successful metadata audit still records
`model_evaluation_authorized=false`; a new reviewed post-C5 authority would be
required before any model evaluation.
