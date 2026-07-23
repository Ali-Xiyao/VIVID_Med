# Data and server capacity audit

**Observed:** 2026-07-22  
**Mode:** local metadata enumeration plus read-only SSH  
**No data copied, no remote file deleted, and no job submitted.**

> Post-audit update: cleanup batch 1 was subsequently authorized and completed.
> See `remote_cleanup_batch1_result_20260722.md`. It reclaimed 100.99 GiB and
> raised quota headroom to 239.38 GiB. No compression or image re-encoding was
> performed.

## Local code footprint

| Surface | Files | Bytes | MiB |
| --- | ---: | ---: | ---: |
| Entire tracked old VIVID repository | 1,151 | 21,285,811 | 20.30 |
| Broad legacy VIVID text/code/config surface | 292 | 2,268,829 | 2.16 |
| Minimum historical candidate files | 18 | 168,554 | 0.16 |
| Whole `legacy/vivid_med` including figures/bundled weights | 450 | 550,823,435 | 525.31 |

The clean project should transfer only versioned code/docs/config/tests. Its
size is negligible relative to datasets and checkpoints. The 525 MiB legacy
tree must not be uploaded wholesale.

## Local canonical resource footprints

Directory totals were enumerated without following Junctions. They describe
the current canonical packages, not independent patient cohorts.

| Resource | Files | Bytes | GiB |
| --- | ---: | ---: | ---: |
| CheXpert | 223,651 | 11,471,429,175 | 10.68 |
| NIH ChestX-ray14 | 112,128 | 45,077,186,807 | 41.98 |
| MIMIC-CXR family | 866,097 | 30,142,050,579 | 28.07 |
| VinDr-CXR | 18,008 | 205,970,857,822 | 191.82 |
| IU/OpenI | 11,433 | 2,752,604,064 | 2.56 |
| AMOS22 | 1,201 | 24,256,974,949 | 22.59 |
| KiTS21 | 36,950 | 70,644,123,345 | 65.79 |
| LIDC-IDRI slices | 77,740 | 153,064,106 | 0.14 |
| OrganAMNIST | 1 | 1,803,859,544 | 1.68 |
| CAMELYON16 partial | 69 | 107,036,987,371 | 99.69 |
| CheXlocalize validation | 2,345 | 3,849,728,336 | 3.59 |
| MS-CXR | 7 | 4,062,782 | <0.01 |
| Chest ImaGenome ZIP | 1 | 1,553,519,249 | 1.45 |
| CheXpert-Plus | 6 | 507,355,947 | 0.47 |
| CheXTemporal gold | 8 | 165,890 | <0.01 |

Total across these 15 canonical local package paths is
**505,223,969,966 bytes (470.53 GiB)**. This number must not be interpreted as
required paper-one storage: it includes CT, pathology, paper-two resources,
and derived annotation families.

## Paper-one transfer scenarios

| Scenario | Included local surfaces | Data footprint | Verdict |
| --- | --- | ---: | --- |
| Track A minimum | MIMIC plus Chest ImaGenome archive | 29.52 GiB | Fits the current quota if no duplicate is created |
| Full paper-one raw/canonical surface | MIMIC, CheXpert/Plus, NIH, raw VinDr package, IU, Chest ImaGenome, MS-CXR | 277.05 GiB | Does not fit current free quota |
| All 15 local resources | All rows above | 470.53 GiB | Out of scope and does not fit |

The raw VinDr DICOM package dominates the full-paper number. Paper one does
not need all CT/pathology data, CheXlocalize validation, or raw VinDr on the
training server. A canonical-frontal/derived VinDr representation may be used
only after its identity and pixel-transform contract are audited.

## MIMIC-CXR packaging decision

The local `MIMIC-CXR` root is not a full DICOM release. The inspected project
image surface consists of 224x224 grayscale JPEG files; one metadata probe was
6,562 bytes. The 30.14 GB root also contains reports, derivative tables,
archives, and an alternative `mimic-cxr_less` surface. The old remote VIVID
project already contains a `mimic-cxr` surface of approximately 19,749 MiB
(19.29 GiB), matching the expected order for the project-preprocessed pixels.
Therefore the default plan is:

1. Audit the remote file/study/patient coverage against the local official
   metadata and checksums.
2. Generate the patient-aware canonical-frontal manifest locally.
3. Reuse remote pixels by relative path when coverage is complete.
4. Upload only a missing delta, never a second full MIMIC copy.
5. Do not resize the existing 224 JPEGs again. Local preprocessing should only
   select the canonical frontal image, validate identity, and optionally pack
   files for transfer efficiency without changing pixels.
6. Treat this as a project-derived 224-pixel surface, not automatically as the
   official full-resolution MIMIC-CXR-JPG release. A future 384-pixel
   sensitivity experiment would require a separately qualified higher-
   resolution source and is not available from this derivative.

## Live server quota

Remote account: `dqxy11` on host `mu01`, GPFS filesystem `ipfs`.

`mmlsquota` reported:

| Item | KiB | GiB |
| --- | ---: | ---: |
| Used | 928,624,352 | 885.61 |
| Hard quota | 1,073,741,824 | 1,024.00 |
| Remaining | 145,117,472 | 138.39 |

The global `df` view shows about 1.2 PB available, but it is irrelevant to this
account once the 1 TiB user quota is reached.

## Existing remote VIVID observations

The measured top-level components sum to approximately **235.55 GiB**. Small
top-level files do not materially change this estimate. The project is thus a
major contributor to the account's 885.61 GiB usage.

| Surface | Observed allocated size | Initial classification |
| --- | ---: | --- |
| `outputs/` | 71,709 MiB (about 70.03 GiB) | High-value cleanup candidate after evidence manifesting |
| `data/` total | 163,039 MiB (about 159.22 GiB) | Mixed reusable and out-of-scope datasets |
| `data/dataset/mimic-cxr` | 19,749 MiB (about 19.29 GiB) | Reuse candidate for RCSD Track A |
| `data/dataset/mimic-cxr_less` | 20,861 MiB (about 20.37 GiB) | Possible duplicate/derivative; verify coverage then choose one |
| `data/dataset/AMOS22` | 25,688 MiB (about 25.09 GiB) | Out of paper-one scope; cleanup candidate after local-copy verification |
| `data/dataset/KITS21` | 70,200 MiB (about 68.55 GiB) | Out of paper-one scope; cleanup candidate after local-copy verification |
| `data/dataset/CheXpert-v1.0-small` | 14,071 MiB (about 13.74 GiB) | Potential Track B/external reuse; retain until lineage audit |
| VinDr raw/derived/incoming surfaces | 5,258 + 4,265 + 5,353 MiB | Redundant/partial candidates; retain one qualified representation |
| `.incoming_vindr/` alone | 5,353 MiB (about 5.23 GiB) | Interrupted staging candidate after hash/coverage check |
| `pretrained/` | 982 MiB | Small; retain only referenced weights |
| executable code/docs | under 0.2 GiB | Negligible |

The final remote top-level size audit is still required before deletion. In
particular, `.incoming_*`, `.recovery_chunks`, duplicate VinDr surfaces,
checkpoints inside `outputs`, and `mimic-cxr_less` must be classified.

### Largest output groups

The largest single output family is the terminal historical
`qwen3vl_case_study_multiseed` tree at 18,635 MiB. The controlled VIVID
`A+UMS+SPD` and `A+UMS` output identities occupy about 2,262 MiB and
1,023 MiB respectively. `outputs/final_tables` is only 148 MiB. This makes a
selective evidence archive much more efficient than retaining the entire
70.03 GiB output tree.

### Staged reclaim estimate

Subject to hash and local-copy verification, the following is realistic:

- CT/other-paper data (AMOS, KiTS, LIDC, OrganAMNIST): about 95.84 GiB;
- interrupted `.incoming_vindr`: about 5.23 GiB;
- one redundant MIMIC derivative: up to 20.37 GiB;
- redundant VinDr representations: roughly 5--10 GiB;
- old outputs after retaining final tables, manifests, logs needed for audit,
  and the original A+UMS/A+UMS+SPD checkpoint identities: roughly 60--66 GiB.

Total reclaim potential is approximately **181--198 GiB**, but this is not a
deletion authorization. The selected evidence must be copied to a manifest-led
archive and verified first.

## Live compute availability

The account association allows `gres/gpu=2`. Both account GPUs are currently
allocated:

- job 4161, `036_IndexMemory`, one GPU;
- job 3066, `036_IndexMemory_git`, one GPU.

The cluster has two eight-GPU `tesla` nodes, but the account-level two-GPU
allocation is the relevant limit. Therefore RCSD cannot start immediately
without waiting for or deliberately releasing an existing allocation. This
audit did not stop or enter either job.

## Current verdict

- **Can the clean code replace the executable old VIVID code?** Yes, after its
  pending training/evaluation adapters pass the proposal gates. Code storage
  is not a constraint.
- **Can all current local paper-one data be uploaded as another copy?** No.
- **Can Track A run on the server?** Storage-wise, likely yes by reusing the
  existing 19.29 GiB MIMIC surface and uploading only manifests/posteriors.
- **Can the full experiment matrix run today?** Not yet. Storage headroom must
  include checkpoints, optimizer states, logs, cached features, and external
  datasets, and both account GPUs are already allocated. A minimum 80--120 GiB
  working reserve is prudent; the current 138.39 GiB free is too tight without
  cleanup for the complete matrix.
- **May the old project be deleted immediately?** No. First preserve hashes and
  final evidence, verify local canonical data copies, then delete only the
  approved candidates. This audit has not deleted anything.

## Post-upload verified state (2026-07-23)

After the separately authorized cleanup and uncompressed staging:

- clean `02101` project: about 1.1 MiB;
- new `02101_data` surface: about 46 GiB;
- NIH: 112,128 files, 45,077,186,807 bytes, zero partial files;
- NIH local/remote path-size digest:
  `74ea0031732606cb72665180db3d2c0af1dd74ff4f4d78dc4143bcdd7e4bb2da`;
- CheXpert-Plus, MS-CXR, and Chest ImaGenome: per-file SHA-256 verified;
- MIMIC, CheXpert, and derived VinDr: referenced from retained remote VIVID
  paths rather than duplicated;
- GPFS quota: 838.50 GiB used, 185.50 GiB remaining.

The remaining quota exceeds the planned 80--120 GiB working reserve. This is
a storage conclusion only; training remains behind the scientific/data gates
and GPU-allocation checks.
