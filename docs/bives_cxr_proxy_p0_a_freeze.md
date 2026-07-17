# BiVES-CXR Proxy-P0-A Freeze

This record freezes the bounded 50-step weak-label engineering run at Git
commit `c113b2abfc767b423e99e3ac61e20f660162fe57`. It is a diagnostic result,
not a clinical result and not a locked-test result.

## Frozen artifacts

| Artifact | SHA256 |
| --- | --- |
| Tracked template `configs/bives_cxr/qwen35_2b_proxy_p0.template.yaml` | `213d0a470311174e0fba0d159737a4c4de42338ab8b22928d698488d7ccf895b` |
| Ignored train proxy manifest | `5886197466eb5539b93e2cb775fcbd01080244b923fe046cb262b4b5417983c7` |
| Ignored validation proxy manifest | `be34d8c00da7ccc77e93bdc8410c9804ace3fbc0de90855d096fd2eb2f103976` |
| Ignored resolved config | `47d4f78b604e765a2ade26ea33047bfd60bb48b891e68222060ad871e6a39c9d` |
| Ignored `metrics_final.json` | `3c49435489f48f45fca7e135a4c14b264b5abdb9b8f61e2fefd4d1528c0d49e4` |
| Ignored final checkpoint | `e61922c7ed15ba71687902590c52033aff201d83d3245f65a7d3049bfe5b5936` |

The ignored runtime directory is
`local_runs/bives_cxr/qwen35_2b_proxy_p0_v4_5k/`. Runtime files remain local
and are not published.

## Frozen result

- Qwen3.5-2B vision backbone frozen; 48 train and 48 patient-disjoint
  validation rows; 50 optimization steps; seed 17; exact K=16.
- Final train-proxy accuracy `0.25`, macro-F1 `0.10`, NLL `1.36755`.
- Final validation accuracy `0.25`, macro-F1 `0.10`, NLL `1.36924`.
- All train and validation argmax predictions are `insufficient`.
- Train-proxy S/C AUROC `0.88194`; validation S/C AUROC `0.80556`; validation
  U/I AUROC `1.0`.

The positive S/C ranking signal is real engineering evidence, but the run did
not fit even its 48-row training proxy. Therefore its C-polarity pattern is a
failure representation, not proof of theoretical non-identifiability. It must
not justify 4B/9B scaling, a loss-weight sweep, decoder changes, or broader
parser labels.

## Next locked diagnostic

The next authority is
`BiVES_next_direction_without_local_clinical_review_2026-07-17.md`. It permits
only a frozen-feature patient-disjoint logistic probe and two matched bounded
Qwen3.5-2B optimization diagnostics: state-only first and the full objective
only if state-only fits the fixed 48-row training proxy. Both use 400 steps,
the same seed/model/K/data, and final-step selection rather than validation
selection.
