# BiVES-CXR C6A Official Data Acquisition Plan

**Status:** `PLAN_COMPLETE_WAITING_USER_ACCESS`

**Date:** 2026-07-18

**Parent authority:** `../BiVES_CXR_MIA_TMI_ready_proposal.md`
**Predecessor verdict:** C5 is `FAIL_FINAL_STOP`; C6 is
`BLOCKED_DATA_NO_ELIGIBLE_LOCAL_CANDIDATE`.

## 1. Scope and non-authorization

This document defines only a lawful, provenance-preserving route for obtaining
an independent patient-identified expert-region dataset. It does **not**
authorize a download, account login, license acceptance, model load, score
access, training, evaluation, additional seed, Qwen3.5-4B/9B scale-up, server
work, or reuse of VinDr test.

Clinical review is not required by this acquisition plan. Released expert
regions are used only under their publisher-provided provenance. No local row
may be relabeled as expert-reviewed, and no pseudo-region may be substituted.

Even if an eligible dataset is acquired and locked, it does not automatically
reopen the stopped C5 route. A new reviewed research authority must separately
authorize any model evaluation and freeze the claim, endpoint, comparator,
thresholds, and one-time opening rule before annotations or scores are opened.

## 2. Candidate ranking

### Candidate A: CheXlocalize test-only — preferred practical intake

Official Stanford AIMI documentation describes board-certified-radiologist
pixel segmentations and points for CheXpert validation and test images,
including Consolidation and Pleural Effusion. It reports 234 validation images
from 200 patients and 668 test images from 500 patients. The official repository
states that the release contains CheXpert validation/test images and labels,
ground-truth annotations/segmentations, and a separate test human benchmark.

Official sources:

- <https://aimi.stanford.edu/datasets/chexlocalize>
- <https://github.com/rajpurkarlab/cheXlocalize>
- canonical Redivis dataset: <https://stanford.redivis.com/datasets/efx9-5nspnbb4b>

**Frozen boundary:** CheXlocalize validation is permanently ineligible. This
repository has historical CheXpert-validation access and evaluation use, so the
validation split cannot support a new project-wide final claim. Only the
publisher-defined CheXlocalize test split may enter metadata-only intake, and
only if its annotations have not previously been opened for BiVES selection.

CheXlocalize test remains a candidate, not an authorized final set, until all
of the following are recorded locally: the exact downloaded release, dataset
terms/license, patient/study/view keys, target counts, checksums, absence from
all prior BiVES manifests/results, and a patient-level lock made before any
model score is computed.

### Candidate B: MS-CXR official test split — conditional secondary intake

PhysioNet MS-CXR v1.1.0 reports 1,162 radiologist-verified image-sentence
bounding-box pairs over 851 subjects. The release includes Consolidation (117
pairs from 109 subjects) and Pleural Effusion (96 pairs from 95 subjects), with
a recommended patient-level 70:15:15 split. The official test split is small:
15 Consolidation pairs/subjects and 14 Pleural Effusion pairs/subjects. Images
must be joined separately from MIMIC-CXR/JPG by released identifiers.

Official source: <https://physionet.org/content/ms-cxr/1.1.0/>

MS-CXR requires a credentialed PhysioNet user, completion of the required CITI
training, acceptance of the PhysioNet Credentialed Health Data License 1.5.0,
and a signed project DUA. Those are user/account decisions and cannot be
accepted or represented by the code agent.

Because MS-CXR is derived from MIMIC-CXR, its subject IDs must be compared
against the frozen prior-use registry below. Only publisher-test subjects with
zero overlap may survive. If either target lacks enough nonoverlapping patients
for the new authority's prespecified endpoint, the route fails closed; official
test rows may not be supplemented with train/validation rows after inspection.

## 3. Frozen prior-use registry

Patient/study identifiers stay local and are not committed. The acquisition
audit binds only counts and SHA-256 hashes of sorted identifier sets.

| Prior surface | Rows | Patients | Studies | Patient-set SHA-256 | Study-set SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| `p0_intake_5k` | 8,220 | 1,414 | 5,000 | `106e13b9500ff5ad9c7e67a168861c04a0f2486a9786ebc8850bf5000e207950` | `b43b2ce1df50537776933bb7cc19792b457c40a9d2a3dfae1db3b5e1ffde168a` |
| `weak_sc_v1` | 1,090 | 715 | 1,022 | `1e0221dd9da27b98bb9a74756bc515edb63dc058acb721ee3566251f429064f3` | `2dff36898d915a5a7cde3ca07410084ec4c6ac9db9512f72e84261c1b7d78c6b` |
| `proxy_sc_v3` | 32 | 30 | 32 | `ce57c053c1a4419140afc7d3da5ea590ecda29198d8b237a38ec7a530b334687` | `b07fb972ba803a962ffc926696aa88f6196250da9af17f04835012381dd03c2b` |
| Strict prior union | — | 1,414 | 5,008 | `106e13b9500ff5ad9c7e67a168861c04a0f2486a9786ebc8850bf5000e207950` | `76e8ae65bc0d740908d064fff5748ddec390eb121c456a8f75f42020c472cd86` |

The union is intentionally strict: any subject/study previously admitted to a
BiVES MIMIC intake, proxy, weak-label, training, validation, selection, cache,
or audit surface is treated as prior use.

## 4. User-side acquisition boundary

The user must personally obtain any restricted package and accept its terms.
Raw packages must be placed outside the repository:

```text
H:\Xiyao_Wang\000_Public Dataset\CheXlocalize\
H:\Xiyao_Wang\000_Public Dataset\MS-CXR\
```

Do not place medical images, patient identifiers, licenses containing account
details, or raw annotation packages under `021_260129VIVID` or Git.

## 5. Mandatory metadata-only intake gates

The first local action after user-side acquisition is metadata-only. No image
tensor, model, checkpoint, annotation visualization, metric, or score may be
opened until these gates pass in order:

1. **Release identity:** record publisher, version, canonical URL, retrieval
   date, terms/license, file inventory, and SHA-256 for every received package.
2. **Schema identity:** identify immutable patient, study, image/view, finding,
   region, and publisher-split fields; fail if patient grouping is absent.
3. **Split freeze:** keep only CheXlocalize test or MS-CXR official test. Never
   merge validation/train after target counts are observed.
4. **Prior-access audit:** prove the candidate's patient/image keys do not occur
   in any prior BiVES manifest, cache, checkpoint selection, score table, or
   result artifact. MS-CXR additionally requires exact subject/study overlap
   comparison against the frozen registry above.
5. **Target coverage:** independently count patients and images with released
   expert regions for Consolidation and Pleural Effusion. No label conversion,
   synonym expansion, pseudo-box, or locally inferred region is permitted.
6. **Image binding:** verify every annotation resolves to exactly one image and
   hash the resolved pixels/files. Missing, duplicate, or ambiguous joins fail
   closed.
7. **Patient lock:** write a patient-level manifest and lock whose hash binds
   release files, license/provenance record, schema, inclusion/exclusion counts,
   prior-overlap result, image hashes, and code hashes. Raw IDs remain ignored.
8. **Blind release gate:** before any annotation visualization or model score,
   obtain a new reviewed research authority. C5's stopped authority cannot be
   silently reused or amended after observing the new final data.

## 6. Current decision

`CheXlocalize test-only` is the preferred practical acquisition candidate.
`MS-CXR official test` is a conditional secondary candidate whose small target
counts and MIMIC overlap risk may make it fail closed. Neither package is
currently present or authorized for automated acquisition.

Current status is therefore:

```text
PLAN_COMPLETE_WAITING_USER_ACCESS
NO_DOWNLOAD
NO_EXPERIMENT
C5_REMAINS_FINAL_STOP
```
