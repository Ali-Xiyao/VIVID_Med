# BiVES-CXR Candidate Rescue Planning Manifest

**Status:** STOPPED_R002_GEOMETRY_FAIL
**Created:** 2026-07-18
**Base commit:** `7c3e7ed`
**Parent authority:** `../BiVES_CXR_MIA_TMI_ready_proposal.md`

This manifest registers the accepted candidate authority and its CPU-only
R001/R002 execution. R002 failed its geometry survival gate, so it does not
authorize model loading, training, VinDr-test reuse, Qwen3.5-4B/9B scaling,
server execution, or a new clinical claim.

| Artifact | Versioned file | Fixed alias | SHA-256 | Verification |
| --- | --- | --- | --- | --- |
| Experiment plan and verdict | `EXPERIMENT_PLAN_20260718.md` | `EXPERIMENT_PLAN.md` | `b7202dd1ab08d6b4109d779e452b935b19646cfbb0f3b6479cbadb766fdb6a0a` | versioned and fixed files are byte-identical |
| Experiment tracker and stop state | `EXPERIMENT_TRACKER_20260718.md` | `EXPERIMENT_TRACKER.md` | `a4685ec4ed9c7fa716dbd9b318395fe678003c1c343b2c630eb434fa413e12e8` | versioned and fixed files are byte-identical |
| R001/R002 execution log | `R001_R002_EXECUTION_LOG_20260718.md` | none | `8f9c8dcee8418f332ad7094e9dfed5a56c08b2544c809f91310277f64b799922` | R001 pass; R002 hard-stop fail; final provenance replay recorded |

## Review gate

- R001 is complete-pass; R002 is complete-fail-hard-stop; R003 onward did not
  run or remain dependency/data blocked.
- VinDr test remains frozen as a diagnostic-only surface after E10.
- VinDr train may support image-disjoint development only; the public DICOMs
  expose no patient, study, or series identifier.
- An independent patient-grouped final dataset remains unavailable.
- A new continuation requires a separately reviewed control-family authority;
  it cannot lower the 90% threshold or reuse VinDr test for selection.

## Accepted connected-control candidate

The stopped R001/R002 package above remains immutable. A separate candidate
authority now defines an exact-area, target-disjoint, 4-connected control from
the same coarse content-coordinate zone. It explicitly weakens exact
target-shape matching and does not call the coordinate bins true anatomy.

| Artifact | Versioned file | Fixed alias | SHA-256 | Status |
| --- | --- | --- | --- | --- |
| Connected-control plan | `CONNECTED_CONTROL_RESCUE_PLAN_20260718.md` | `CONNECTED_CONTROL_RESCUE_PLAN.md` | `c5100a32d8c5b9f9858d37c73538695849413f80fa37ee45c471703c49ee844e` | byte-identical; stopped at C5 confirmation polarity gate |
| Connected-control tracker | `CONNECTED_CONTROL_RESCUE_TRACKER_20260718.md` | `CONNECTED_CONTROL_RESCUE_TRACKER.md` | `66688af2c3f009d3078412efdb44f834205e8edda04c9e9a852ef3d4ec3435b5` | byte-identical; C001-C005 pass; C006 final-stop fail; C007 blocked with no eligible local candidate |
| C1/C2 execution log | `CONNECTED_CONTROL_C1_C2_EXECUTION_LOG_20260718.md` | none | `7e1e9c317d1560a5d369fefa034253cd0732f78eb4f82c0f111ff32bc47096c8` | 98/98 tests; 375/377 geometry pass; full rows replay identical |
| C3 execution log | `CONNECTED_CONTROL_C3_EXECUTION_LOG_20260718.md` | none | `ffc59f9872b65e4345dbf05e073b317047c86cd646f69a079ce680b19170bbca` | 101/101 tests; zero replay error; C4 estimate 0.2461 h |
| C4 execution log | `CONNECTED_CONTROL_C4_EXECUTION_LOG_20260718.md` | none | `4a4dfc4ebf2861ae179ac481ffb578412a61d8185aecdf8934cb62280592d67f` | 106/106 tests; both co-primary operators pass every C4 gate |
| C5 execution log | `CONNECTED_CONTROL_C5_EXECUTION_LOG_20260718.md` | none | `b1f8d9ca51c822dd674dc6c66bdbb9142962161847650a70873f905c252e6804` | mechanism replicates; consolidation AUPRC below B0; final stop |
| C6 data-authority inventory | `CONNECTED_CONTROL_C6_DATA_AUTHORITY_INVENTORY_20260718.md` | none | `2453216d654290f6bb32cfc3b32edacc15c78040aa55bf0376295a654768651f` | read-only metadata audit; no eligible local patient-identified two-finding expert-region candidate; no run authorized |
| C6A official acquisition plan | `C6A_OFFICIAL_DATA_ACQUISITION_PLAN_20260718.md` | none | `4fd9234f2ba68a6535a5ef410790c2ec6204e1a48f8439b6b845a0cdb414bff3` | CheXlocalize test-only preferred; MS-CXR official test conditional; waiting on user-side access; no download/run authorized |
| C6B metadata intake tooling log | `C6B_METADATA_INTAKE_TOOLING_LOG_20260718.md` | none | `07c2fdaffd7874d4043ea204f633650b7b4feae61a0a41d098d8dacad9ffeffc` | fail-closed test-only intake ready; hashed validation registry ready; 116/116 tests; real package absent |
| C6C MS-CXR intake tooling log | `C6C_MS_CXR_INTAKE_TOOLING_LOG_20260718.md` | none | `12cdc09dd9517ffbb4207b42f8e3eda2fcb23e9792d40f74a8bd469761182d6a` | fail-closed official-test COCO intake ready; strict hashed MIMIC registry reproduced; 126/126 tests; real package absent |
| C6D MS-CXR real-package preflight log | `C6D_MS_CXR_REAL_PACKAGE_PREFLIGHT_LOG_20260718.md` | none | `f252ab3d9f46d6d88d5828fb4995c9b3b250a5472a2111906ce4fad684f2fb5d` | real package checksums pass; 15/14 pairs and 25/20 boxes; 29/29 images bound; zero prior overlap; license/model authority remain false |
| C6E MS-CXR strict intake log | `C6E_MS_CXR_STRICT_INTAKE_LOG_20260718.md` | none | `0cf2813122b2621908ec193c9b651a8619fcef11c72a4334775f7851afe5d360` | explicit user access confirmation; exact package binding; strict intake pass; model authority remains false |
| C6F MS-CXR post-C5 authority | `C6F_MS_CXR_POST_C5_EVALUATION_AUTHORITY_20260718.md` | none | `6599212f1c9b4177379196435a65deffd278440e94f4574f3057d4b107bc207c` | separate user authorization; positive-only 2B mechanism protocol; frozen C6E intake remains unchanged |
| C6F frozen evaluation config | `C6F_MS_CXR_QWEN35_2B_EVAL_CONFIG_20260718.yaml` | none | `5bfd243c7c3a42a113e5e5b50b171d988cecb4a1d6a1426b4c94ad7a8e1ffb5a` | Qwen3.5-2B only; unchanged C4/C5 operators and gates; classification metrics and 4B/9B forbidden |
| C6F pre-open geometry log | `C6F_MS_CXR_PREOPEN_GEOMETRY_EXECUTION_LOG_20260718.md` | none | `acbe78ac76d08b7ef9d4acd62d1c4d861299ab094631afc77a17985552bb82cd` | 29-patient manifest passes; connected-control geometry 28/29; 133/133 tests; model/GPU/score never opened |
| C6G geometry-only authority | `C6G_MS_CXR_GEOMETRY_ONLY_AUTHORITY_20260718.md` | none | `17305e6887491b179a672d6f67ea1186f06acb8236feef5e0a4f3f6cf62355e0` | new score-free control family; model/GPU/image-decode/score authorization false |
| C6G frozen geometry thresholds | `C6G_MS_CXR_GEOMETRY_THRESHOLDS_20260718.json` | none | `69a3bedeb43b65065eab41d28fdefe4870214babebbfa7872f5c5e8146ecb5ab` | maxima from 752 accepted frozen C4/C5 geometry rows only |
| C6G geometry execution log | `C6G_MS_CXR_GEOMETRY_EXECUTION_LOG_20260718.md` | none | `da2b9026575aacda75fed137ce82fdde48a67beeda0d2458623ffb60ad80b6e1` | 29/29 geometry pass; three byte-identical geometry replays; final lock bound to `db3c033`; 137/137 tests; no model/GPU/score |

The user accepted this candidate by replying `继续` on 2026-07-18. C001-C005
are complete-pass. C006/C5 opened the image-disjoint VinDr-train
`rescue_confirm` surface exactly once and failed its frozen per-finding polarity
gate because consolidation B2 AUPRC fell below B0. This route is final-stopped;
VinDr test, training, result-driven tuning/reruns, scale-up, and later rows
remain blocked.

C007/C6 was subsequently inspected as a read-only local data-authority gate.
NIH supplies patient-linked Effusion boxes but no Consolidation boxes;
MIMIC-CXR and CheXpert supply patient keys and both finding labels but no local
expert-region annotations. VinDr remains excluded. C007 therefore stays
`BLOCKED_DATA_NO_ELIGIBLE_LOCAL_CANDIDATE`, and this bookkeeping does not
reopen the C5 final stop.

C6A then freezes a lawful acquisition route without changing C007. It ranks
CheXlocalize test-only as the preferred practical intake, permanently excludes
CheXlocalize validation because of prior project access, and keeps MS-CXR
official test conditional on PhysioNet authorization plus zero overlap with
the local prior-use MIMIC registry. User-side access and a later reviewed
research authority remain mandatory; no download or experiment is authorized.

C6B implements that plan's local metadata-only intake boundary. It builds an
ignored hashed registry for the already-accessed CheXpert validation split and
can audit an already-downloaded CheXlocalize test release without decoding or
rendering images. The package is not present, so the tool has not emitted a
real intake lock and C007/C5 remain unchanged.

C6C adds the conditional MS-CXR route without changing that verdict. It binds
publisher-test COCO annotations to the local MIMIC metadata/JPG release,
requires the official 15/14 target counts, rejects any patient/study overlap
with the frozen prior-use registry, and hashes image bytes without decoding.
The ignored registry reproduces 1,414 patients and 5,008 studies with the
frozen C6A set hashes and no raw identifiers. The restricted package is absent,
so no real intake result, model authority, or C5 reopening exists.

C6D records the subsequently supplied real v1.1.0 package without rewriting
the historical C6C tooling log. Publisher hashes pass; the repaired audit
distinguishes 29 unique image-text pairs from 45 component boxes, binds all 29
images, and finds zero prior patient/study overlap. The structure preflight is
followed by the strict C6E licensed intake.

C6F is a new, independent post-C5 protocol authorized by the user; it does not
rewrite C5 or C6E. The positive-only MS-CXR question is limited to the frozen
expert-box mechanism gate. Its score-free connected-control audit fails one of
29 rows, so the dataset lock remains closed and the evaluator correctly stops
before any model, GPU, JPG decode, or score. No result-driven row exclusion,
control change, rerun, or Qwen3.5-4B/9B scale-up is authorized.

not a license attestation or research authority and cannot authorize a score,
annotation visualization, model evaluation, or C5 reopening.

C6E records the strict intake after explicit user confirmation of credentialed
access, required CITI training, and signed DUA. The local attestation and
ignored intake artifact remain outside Git. Package hash, official pair/box
counts, MIMIC binding, image hashes, and zero prior overlap pass. The result is
still nonformal and explicitly carries no model-evaluation authority; C5 is
unchanged pending a new reviewed research authority.

C6G is a separate geometry-only authority rather than a repair or rerun of
C6F. It replaces the categorical same-zone gate with frozen continuous
centroid/perimeter limits derived only from accepted C4/C5 geometry and expands
the deterministic connected candidate family uniformly for all 29 rows. The
final score-free build passes 29/29 and freezes masks plus a geometry lock, but
keeps model, GPU, image-decode, and score authorization false. C6H remains a
separate, not-yet-authorized decision.
