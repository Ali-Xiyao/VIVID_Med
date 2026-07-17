# BiVES-CXR Handoff Index

BiVES-CXR is the only active paper and code mainline.

## Start here

| Purpose | File |
| --- | --- |
| Final research proposal | `../BiVES_CXR_MIA_TMI_ready_proposal.md` |
| Repository overview | `../README.md` |
| Implementation contract | `bives_cxr_implementation.md` |
| Manifest schema | `bives_cxr_manifest_schema.md` |
| Migration/archive manifest | `bives_cxr_migration_manifest.md` |
| Persistent task plan | `../task_plan.md` |
| Durable findings | `../findings.md` |
| Chronological progress | `../progress.md` |

## Active code

| Surface | Path |
| --- | --- |
| Core package | `../bives_cxr/` |
| Qwen3.5 configs | `../configs/bives_cxr/` |
| Training entry | `../scripts/train_bives_cxr.py` |
| CPU smoke | `../scripts/smoke_bives_cxr.py` |
| Real-weight Qwen3.5 vision smoke | `../scripts/smoke_qwen35_vision.py` |
| Qwen3.5-to-BiVES server integration gate | `../scripts/smoke_qwen35_bives_integration.py` |
| Manifest audit | `../scripts/audit_bives_manifest.py` |
| Joint four-split dataset lock | `../scripts/lock_bives_dataset.py` |
| Source-only deployment manifest | `../scripts/write_bives_source_manifest.py` |
| Statement cache builder | `../scripts/build_bives_statement_embeddings.py` |
| Explicit locked-test evaluator | `../scripts/evaluate_bives_final.py` |
| CPU tests | `../tests/test_bives_core.py`, `../tests/test_bives_readiness.py` |
| VinDr archive/integrity utilities | `../scripts/extract_vindr_cxr.py`, `../scripts/audit_vindr_cxr_integrity.py` |

## Active model boundary

The active family is Qwen3.5 multimodal only:

- 2B for P0/debug;
- 4B for the main model;
- 9B for scale validation;
- 0.8B for optional ultra-light smoke.

No active config may reference Qwen3-VL, Qwen2.5, InternVL, LLaDA, Gemma, or
other prior model families.

## Legacy boundary

`../legacy/vsl_cxr/` preserves the prior VSL-CXR, CEQ, CCSH, AUCH, SAMEQ,
CVCP, case-study, Qwen3-VL, and old external-evaluation implementation.

`../legacy/vivid_med/` preserves pre-BiVES VIVID-Med scripts, models, training,
evaluation, loaders, configs, profiles, prompts, and external tools.

`../legacy/delete_archive/` preserves the small retired root-level cleanup
bundle that previously remained under `delete/`, including old Qwen2.5-era
configs and utility scripts.

Historical result files under ignored `outputs/` remain untouched. They are
pilot/provenance evidence only and cannot close a BiVES paper gate.

## Current execution status

The repository has an executable BiVES core, strict same-statement group
training, mask-before-contextual-block exact-K interventional closure, full-row
primary evaluation, a separate grouped mechanism evaluator, best-checkpoint
selection, decoder-temperature calibration, a vision-only Qwen3.5 loader,
provenance-complete statement caches, deterministic intervention controls,
an isolated locked-test release entry, synthetic CPU smoke, and proposal-level
unit tests.

## Local-first execution

The active YAMLs now default to this workstation: `data_root: data` and local
Qwen3.5 paths under `H:/Xiyao_Wang/001_models/`. The intended local checks are:

```powershell
python scripts/smoke_bives_cxr.py
python -m unittest discover -s tests -p "test_bives_*.py" -v
python scripts/evaluate_bives_final.py --help
```

`--validate-release-chain-only` on the final-evaluator CLI validates the full
checkpoint/calibration/cache/four-split lock chain without loading a model or
emitting locked-test metrics. The normal evaluator still requires
`--run-locked-test`.

Formal local training has not been started. Before local P0:

1. create locked train/validation/calibration/test manifests with provenance,
   hashes, complete S/C/U/I groups, and pass the strict mandatory audit;
2. create and retain the matching joint `dataset_lock.json` with
   `scripts/lock_bives_dataset.py`; bind it in the active YAML before training;
3. create a clean Git source manifest with
   `scripts/write_bives_source_manifest.py` before source-only server sync;
4. perform the uncertain-vs-insufficient expert pilot;
5. verify same-statement cross-state coverage;
6. run Qwen3.5-2B P0 locally;
7. unlock Qwen3.5-4B only after the proposal go/no-go gates pass.
