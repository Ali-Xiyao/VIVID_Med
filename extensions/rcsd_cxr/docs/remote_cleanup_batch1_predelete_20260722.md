# Remote cleanup batch 1 pre-delete manifest

**Authorized by user:** 2026-07-22  
**Remote root:** `/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID`  
**Purpose:** free storage for the separate RCSD-CXR paper-one project without
compressing or re-encoding datasets.

## Safety evidence

- `realpath -e` resolved every target below the exact remote VIVID root.
- The only two active user jobs are under `036_IndexMemory` and
  `036_IndexMemory_git`; neither command/work directory references VIVID.
- Canonical local copies exist under `H:\Xiyao_Wang\000_Public Dataset`.
- These CT/abdominal resources are explicitly excluded from RCSD-CXR paper one.
- `.incoming_vindr` contains one incomplete staging tar plus directory
  metadata; qualified VinDr data remains under `data/dataset/`.

## Exact targets

| Remote literal target | Remote MiB | Remote files | Local recovery source | Local status |
| --- | ---: | ---: | --- | --- |
| `data/dataset/AMOS22` | 25,688 | 41,446 | `H:\Xiyao_Wang\000_Public Dataset\AMOS22` | exists; 1,201 package files, 24,256,974,949 bytes |
| `data/dataset/KITS21` | 70,200 | 151,736 | `H:\Xiyao_Wang\000_Public Dataset\KITS21` | exists; 36,950 package files, 70,644,123,345 bytes |
| `data/dataset/LIDC-IDRI-slices` | 526 | 77,740 | `H:\Xiyao_Wang\000_Public Dataset\LIDC-IDRI-slices` | exists; same 77,740-file count |
| `data/dataset/organamnist_224.npz` | 1,721 | 1 | `H:\Xiyao_Wang\000_Public Dataset\organamnist_224.npz` | SHA-256 exact match `a19bae532ac0cf979f7474aba7eb923dc9bbf67c1bcba3cb941dce51a59951e9` |
| `.incoming_vindr` | 5,353 | 1 payload | qualified VinDr package and PNG derivative remain remotely and locally | interrupted `vindr_test_chunk_0001.tar`, 5,612,087,808 bytes |

Expected allocated-space reclaim: **103,488 MiB (about 101.06 GiB)**.

## Explicitly retained

- `data/dataset/mimic-cxr`
- `data/dataset/mimic-cxr_less` pending equivalence decision
- `data/dataset/CheXpert-v1.0-small`
- both VinDr dataset/PNG surfaces pending qualification
- `outputs/`, including A+UMS/A+UMS+SPD evidence
- `pretrained/`, code, docs, configs, and terminal audit evidence
