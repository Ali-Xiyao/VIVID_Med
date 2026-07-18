# C6E MS-CXR strict local intake log

Date: 2026-07-18
Scope: strict local metadata intake after explicit user confirmation of
credentialed PhysioNet access, required CITI training, and signed DUA.

## Access and package binding

- The user explicitly confirmed the three required access conditions.
- A local attestation was written outside Git at
  `H:\Xiyao_Wang\000_Public Dataset\MS-CXR\license_record.json`.
- The attestation contains no username, credential, token, cookie, SAS URL, or
  private download URL.
- Attestation SHA-256:
  `037bfea3c0ae112ccd188cb715f405bf696f820347f6a991348f47f25dee9ac7`.
- The strict audit independently re-hashed the package and matched it to the
  attested ZIP SHA-256:
  `62c829d307eb99a07fba82a3ee8346fd32dfcc5a226cfc00129049f684781bd9`.

## Strict intake command

```powershell
python scripts\audit_bives_c6_ms_cxr.py audit-test-release `
  --license-record "H:\Xiyao_Wang\000_Public Dataset\MS-CXR\license_record.json"
```

## Result

- status: `metadata_intake_ready_no_model_authority`
- license gate passed: true
- model evaluation authorized: false
- formal result: false
- publisher test only: true
- Consolidation: 15 pairs / 25 boxes / 15 patients / 15 studies
- Pleural Effusion: 14 pairs / 20 boxes / 14 patients / 14 studies
- target images present and hashed: 29/29
- prior patient overlap: 0
- prior study overlap: 0
- raw identifier regex matches in serialized artifact: 0
- canonical artifact SHA-256:
  `0027358c2998773e73dbd19da02a37dac27c060150bf42e59469d218fb24b4ed`
- ignored artifact file SHA-256:
  `33363f1dee5af29e8e9e2768f64de4a85c224f2505e8e0fcaf8fbcac34ccfb02`

The ignored artifact is
`local_runs/bives_cxr/c6_ms_cxr_intake/ms_cxr_test_intake.json`; it is excluded
by `.gitignore: local_runs/` and was not published.

## Boundary

This closes the access, package-integrity, schema, MIMIC binding, image
availability, and prior-overlap intake gates. It is not a clinical review and
does not retrospectively change C5. It also does not authorize a model score,
annotation visualization, Qwen3.5 evaluation, new seed, or 4B/9B scaling.
A separate reviewed post-C5 research authority must explicitly define and
approve any new MS-CXR model evaluation before execution.

## Verdict

`STRICT_INTAKE_PASS_NO_MODEL_AUTHORITY`
