# Remote cleanup batch 1 result

**Completed:** 2026-07-22  
**Deletion method:** guarded Python deletion with exact literal targets after
`realpath` containment checks  
**Compression/re-encoding:** none

## Removed

- `data/dataset/AMOS22`
- `data/dataset/KITS21`
- `data/dataset/LIDC-IDRI-slices`
- `data/dataset/organamnist_224.npz`
- `.incoming_vindr`

All five targets were confirmed absent after deletion.

## Retained and confirmed present

- `data/dataset/mimic-cxr`
- `data/dataset/mimic-cxr_less`
- `data/dataset/CheXpert-v1.0-small`
- `data/dataset/vinbigdata_xhlulu_512png`
- `data/dataset/vindr-cxr-an-open-dataset-of-chest-x-rays-with-radiologists-annotations-1.0.0`
- `outputs`
- `pretrained`

## Quota result

| Item | Before | After | Change |
| --- | ---: | ---: | ---: |
| Used | 885.61 GiB | 784.62 GiB | -100.99 GiB |
| Available under 1 TiB quota | 138.39 GiB | 239.38 GiB | +100.99 GiB |

GPFS also reported about 0.38 GiB `in_doubt` immediately after deletion, so the
settled free value may move slightly.

## Recovery status

The four removed public datasets remain locally at their canonical paths. The
interrupted VinDr tar was not a qualified dataset and is superseded by retained
VinDr dataset/PNG surfaces. The deletion is not recoverable from a remote
trash, but the dataset content is recoverable from the verified local sources.

## Next decision

The current 239.38 GiB headroom is sufficient for the expected uncompressed
increment if existing MIMIC, CheXpert, and VinDr surfaces are reused. Do not
delete another remote project yet. First audit local-versus-remote MIMIC
coverage and enumerate exactly which new files are absent.
