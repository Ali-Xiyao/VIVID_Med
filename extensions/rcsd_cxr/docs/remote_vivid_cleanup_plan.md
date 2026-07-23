# Remote VIVID cleanup plan

**Status:** batch 1 completed; later batches remain locked.

Batch 1 removed the out-of-scope CT/abdominal data and interrupted VinDr
staging listed in `remote_cleanup_batch1_result_20260722.md`. MIMIC, CheXpert,
VinDr qualified surfaces, outputs, and pretrained assets remain intact.

## Preserve before any cleanup

1. Source commit/tag/config identity for the stable A+UMS and A+UMS+SPD 4x2
   runs.
2. The exact selected checkpoints required for historical reconstruction.
3. Final metrics, per-seed tables, logs needed to prove checkpoint selection,
   and patient/split manifests.
4. Frozen BiVES/ARISE/VICER terminal reports and their small code/evidence
   manifests, without retaining bulk caches or every checkpoint.
5. SHA-256 and byte counts for every retained artifact.

## Candidate classes

| Class | Approximate size | Proposed action |
| --- | ---: | --- |
| AMOS/KiTS/LIDC/OrganAMNIST under old VIVID | 95.84 GiB | Remove from remote after canonical local copies and hashes pass |
| `.incoming_vindr` | 5.23 GiB | Remove if incomplete and fully superseded |
| `mimic-cxr` versus `mimic-cxr_less` | 19.29 vs 20.37 GiB | Audit coverage, keep one qualified Track-A surface |
| VinDr raw/PNG/incoming | about 14.53 GiB total | Keep one complete derived surface; remove partial duplicates |
| Historical `outputs/` | 70.03 GiB | Archive evidence subset, then remove disposable checkpoints/caches |
| `pretrained/` | 0.96 GiB | Keep only weights referenced by the new locked configs |

## Required deletion gate

For every candidate path:

- resolve its absolute path under the remote VIVID root;
- record current bytes, file count, modification range, and checksum manifest;
- prove an accessible local or archive copy for anything scientifically unique;
- show that no running job reads/writes the path;
- obtain an explicit user-approved deletion list;
- delete with one remote shell and exact literal paths;
- rerun quota and retained-evidence checks.

Do not delete the whole remote VIVID directory in one operation. The safest
replacement is selective pruning, reuse of qualified MIMIC/CheXpert data, and
installation of the clean code surface only after its Gate-0/Gate-1 package is
runnable.
