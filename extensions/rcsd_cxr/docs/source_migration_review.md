# Source migration review

## Verdict

Do not copy `legacy/vivid_med` wholesale. The broad text/code/config surface is
only about 2.16 MiB, but it mixes stable VIVID components with discarded
methods, hard-coded execution paths, obsolete launchers, and protocol defects.
The whole legacy directory is 525.31 MiB because it also contains figures and
a bundled model weight surface that must not migrate.

The clean project reimplements the small stable contracts and records every
source file considered in `provenance/source_migration_manifest.csv`.

## Defects corrected at migration

1. Historical `SPDProjector` defaulted to three groups although the result
   checkpoint identity was four groups by two tokens. The clean class rejects
   anything other than 4x2 for the controlled baseline.
2. The historical CheXpert loader replaced missing/corrupt images with a black
   image. The clean manifest loader fails immediately.
3. The historical training entrypoint could construct a random row-level
   validation subset when no validation manifest was given. That is not
   patient-safe and is not migrated.
4. Old configurations embedded machine/server paths. The clean code uses a
   separate YAML dataset registry.
5. The old monolithic VIVID model combined ViT, projector, tokenizer, and LLM
   loading. It is not copied until teacher/token-loss contracts are separately
   specified and unit tested.
6. Historical VSL/BiVES/ARISE/VICER code is scientifically out of scope and is
   absent from the executable project.

## Implemented clean contracts

- Dataset registry with parent lineage and paper-one/test-role locks.
- Fail-closed manifest image loader.
- Missing-aware log-opinion-pool posterior.
- SPD 4x2 projector.
- Field-anchored 4x2-token projector with an equal token budget.
- Read-only path and manifest-size validators.

## Deliberately pending behind Gate 0

- MIMIC canonical-frontal patient/study manifest builder.
- Source-specific CheXpert/NegBio/CheXbert/RadGraph adapters.
- Gold-set reliability estimator and confusion-matrix shrinkage.
- Frozen text-teacher adapter and token-alignment loss.
- Complete training/checkpoint engine.
- Linear-probe, fine-tuning, bootstrap, calibration, and subgroup runners.

These are not filled with placeholders that appear runnable. They remain
explicit gate deliverables because dataset identity, report gold data, and
checkpoint-selection rules must be frozen first.
