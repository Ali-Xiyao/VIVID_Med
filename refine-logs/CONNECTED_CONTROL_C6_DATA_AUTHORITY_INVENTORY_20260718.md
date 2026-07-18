# BiVES-CXR C6 Independent Final-Data Authority Inventory

**Date:** 2026-07-18

**Mode:** local, read-only metadata inventory

**Result:** `BLOCKED_DATA_NO_ELIGIBLE_LOCAL_CANDIDATE`

**Parent stop:** C5 is `COMPLETE_FAIL_FINAL_STOP`

## Scope and non-authorization

This inventory checks whether an already-local chest-X-ray dataset can satisfy
the frozen C6 requirement for a new patient-identified, expert-region final
evaluation surface. It did not decode images, load a model, use a GPU, download
data, mutate any dataset, open VinDr test, or run an experiment.

C6 availability would not erase the C5 final stop. Because C5 failed its
conjunctive consolidation polarity gate, a future dataset lock alone cannot
reopen the frozen model route; model evaluation would require a separately
reviewed research authority.

## Frozen eligibility gates

A C6 candidate must satisfy all of the following before any score is read:

1. independent of every VinDr development, confirmation, and test surface;
2. stable patient or study identity suitable for a patient-disjoint lock;
3. expert region annotations tied to image identity;
4. coverage of both frozen findings, `consolidation` and `pleural_effusion`;
5. reviewable provenance and permitted local research use;
6. an immutable manifest/annotation/image hash chain approved before model use.

Missing one gate is a failure, not an invitation to impute regions or relabel
the data.

## Bounded local inventory

| Local source | Patient/study identity | Region evidence | Frozen finding coverage | C6 verdict |
| --- | --- | --- | --- | --- |
| NIH Chest X-rays | `Data_Entry_2017.csv` contains 112,120 images and 30,805 patient IDs. All 984 box rows link to metadata, covering 726 patients. | `BBox_List_2017.csv` contains 984 boxes over 880 images. The CSV itself does not encode annotator identity or review provenance. | Effusion has 153 boxed images from 142 patients. Consolidation is absent from the eight box labels. | **Fail.** Useful only as a possible separately reviewed single-finding audit; it cannot support the frozen two-finding C6 final. |
| MIMIC-CXR metadata and labeled test | Metadata/split tables contain 377,110 DICOM IDs, 227,835 studies, and 65,379 subjects. Of 687 labeled test studies, 685 link to 296 subjects. | No local box, mask, phrase-region, MS-CXR, or equivalent expert-region table was found in the bounded MIMIC trees. The supplement contains RadGraph metric JSON and a landmark-observation adjacency directory, not image regions. | The labeled test table contains both Consolidation and Pleural Effusion columns. | **Fail.** Patient identity and findings are available, but expert image regions are absent. |
| CheXpert-v1.0-small | Paths expose 64,540 train patients and 200 validation patients. | The local package contains images plus train/validation label CSVs; no region annotation package is present. | Both Consolidation and Pleural Effusion label columns are present. | **Fail.** Patient identity and findings are available, but expert image regions are absent. |
| IU X-ray/OpenI | A local image/report tree is present. | No bounded filename evidence of a box, mask, or expert-region package was found. | Not established as a locked two-finding regional surface. | **Fail.** No eligible regional final surface is locally established. |
| VinDr-CXR | Previously used development and one-time confirmation surfaces; public DICOMs lack patient/study/series identifiers in this release. | Expert boxes exist, but the dataset is not new and its test surface remains prohibited. | Both frozen findings exist. | **Excluded by authority.** It cannot serve as C6 or be reopened for selection. |

No top-level `CheXlocalize` or `MS-CXR` package was present in the local public
dataset root. The bounded raw archive directory contains only AMOS22,
CheXpert-small, and NIH archives.

## Exact metadata evidence

### NIH boxed subset

- Box rows: `984`; unique boxed images: `880`; linked patient IDs: `726`;
  unmatched box rows: `0`.
- Per-label patients: Atelectasis `167`, Cardiomegaly `128`, Effusion `142`,
  Infiltrate `115`, Mass `81`, Nodule `77`, Pneumonia `116`, Pneumothorax `82`.
- There is no Consolidation row in the box table.

### MIMIC labeled test linkage

- Labeled studies: `687`; matched through the official split table: `685`;
  matched patients: `296`.
- Consolidation values: positive `56`, negative `24`, uncertain `26`, empty
  `581`.
- Pleural Effusion values: positive `274`, negative `83`, uncertain `23`,
  empty `307`.
- These are study labels, not expert image regions.

## Source SHA-256

| File | SHA-256 |
| --- | --- |
| `NIH Chest X-rays/BBox_List_2017.csv` | `0bbfea9d4c4e9771481b3023b1bc9f0df9dea924453b12986beb29b0c4d0c95b` |
| `NIH Chest X-rays/Data_Entry_2017.csv` | `88f75094e25ccc0c6f1f9cdfd4b2f94f9379a0ae07d5ff4dcf94242707b07462` |
| `mimic_cxr_other/mimic-cxr-2.0.0-metadata.csv.gz` | `6a3748ce77724c0dfe7d2def8f47643e989e3bbf0795bc13b89c1578e1649d6b` |
| `mimic_cxr_other/mimic-cxr-2.0.0-split.csv.gz` | `515997bd6649045d7443d60c59a4ce9f6cca6c478871b8f2fb13454462bedb2f` |
| `mimic_cxr_other/mimic-cxr-2.1.0-test-set-labeled.csv` | `9bf491397a2cb2bff79803e7d77be8b820328b82684638db236be86a8bafa061` |
| `CheXpert-v1.0-small/train.csv` | `b4bcd3d7942de23349bbe922a2389185d639cb06a24e700efb4980caae80b8e8` |
| `CheXpert-v1.0-small/valid.csv` | `aa43403f7bb183a35ae3bb9152896c1fe9a5b297635e0b541e363b776830ec4a` |

## Decision

`C007/C6` remains `BLOCKED_DATA`. No local candidate passes the conjunctive
patient identity, expert-region, two-finding coverage, provenance, and
independence requirements. No C6 manifest, dataset lock, cache, score, or run
may be created from the currently available files.

The data gate can be reconsidered only after a separately obtained and
authorized package such as CheXlocalize, MS-CXR, or an equivalent dataset is
placed under the public-dataset boundary with images, region annotations,
patient/study identifiers, and license/provenance documentation. That review
must occur before any model access, and it still would not override the C5
final stop without a new research authority.
