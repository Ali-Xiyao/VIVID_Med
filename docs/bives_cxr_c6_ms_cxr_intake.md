# BiVES C6C MS-CXR local metadata intake

This is a local, metadata-only, fail-closed intake. It does not download data,
accept terms, decode or render images, load a model, access scores, or reopen
the C5 final stop.

## User-side prerequisites

The user must personally satisfy the PhysioNet requirements for MS-CXR v1.1.0
(credentialed access, required CITI training, and signed DUA) and place the
lawfully obtained release outside the repository:

```text
H:\Xiyao_Wang\000_Public Dataset\MS-CXR\
  MS_CXR_Local_Alignment_v1.1.0.json
```

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

The audit uses the local MIMIC metadata and image tree by default. It accepts
only official `test` annotations, requires exactly 15 Consolidation and 14
Pleural Effusion pairs/subjects, validates LTWH boxes and image-file hashes,
and rejects any patient/study overlap with the frozen prior-use registry.

Outputs stay under ignored `local_runs/bives_cxr/c6_ms_cxr_intake/`. They must
not be committed. A successful metadata audit still records
`model_evaluation_authorized=false`; a new reviewed post-C5 authority would be
required before any model evaluation.
