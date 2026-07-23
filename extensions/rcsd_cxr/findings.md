# Findings

## Initial facts

- The requested target `H:\Xiyao_Wang\02101` did not exist before this task.
- The source repository is on branch `codex/morph-cxr-separability` and has
  pre-existing user changes in the public-dataset inventory and planning/index
  files. They are preserved and will not be reverted or folded into unrelated
  source commits.
- The verified inventory records 15 public dataset/annotation families, not 15
  independent cohorts. MIMIC-derived and CheXpert-derived add-ons must retain
  lineage labels.
- The supplied conversation contains the proposed RCSD-CXR paper plan, but is
  background material. The new repository must independently validate code,
  data availability, novelty, and gates before execution.

## Decisions

- Build a new clean project rather than mutate or copy the full historical
  repository.
- Copy no raw/local runtime data. Use a machine-local path manifest and a
  portable template instead.
- Preserve old frozen methods only as provenance summaries or links, not
  executable packages in the new project.

## Verified implementation findings

- The old tracked repository is 20.30 MiB; broad legacy VIVID text/code/config
  is 2.16 MiB; `legacy/vivid_med` becomes 525.31 MiB only because it also
  contains figures and a bundled weight surface.
- Historical SPD source defaults to three query groups, while the accepted
  experiment identity is four groups by two tokens. The clean baseline rejects
  any non-4x2 configuration.
- Historical image loading could replace missing/corrupt images with a black
  image. The clean manifest dataset fails closed.
- Historical training could create a random row-level validation subset when
  no validation manifest was supplied. It was not migrated because it is not
  patient-safe.
- The local MIMIC project pixels inspected are already 224x224 grayscale JPEG,
  so preprocessing should select canonical frontal studies and validate
  identity rather than resize them again.

## Capacity findings

- Fifteen canonical local resource paths occupy 470.53 GiB, but the minimum
  Track-A data surface is only 29.52 GiB and the raw/full paper-one set is
  277.05 GiB.
- The server user quota is 1,024 GiB, with 885.61 GiB used and 138.39 GiB
  remaining. Global filesystem free space is not the limiting value.
- The old remote VIVID project occupies approximately 235.55 GiB: about
  159.22 GiB data, 70.03 GiB outputs, 5.23 GiB interrupted VinDr staging, and
  about 1 GiB other large surfaces.
- The server already has about 19.29 GiB of MIMIC pixels and 13.74 GiB of
  CheXpert, so the new project should reuse qualified remote data instead of
  uploading duplicate copies.
- Approximately 181--198 GiB is potentially reclaimable after evidence hashes,
  local-copy checks, and an explicit deletion approval.
- Both account GPUs are currently allocated to two IndexMemory jobs; no RCSD
  GPU job can start without waiting or intentionally releasing an allocation.

## SSH transfer recovery

- The earlier hypothesis that local SSH process count exhausted a connection
  allowance was falsified: the same reset persisted after every local
  SSH/SFTP/SCP process was stopped.
- TCP, host-key exchange, and public-key authentication all succeed. Non-PTY
  exec and SFTP channels then open with a zero remote receive window and stall
  or reset.
- Forced-TTY sessions succeed on `mu01`; account process and file-descriptor
  limits are not close to exhaustion, no stale `~/.ssh/rc` exists, and no old
  failed SSH children remained under the account.
- Standard SFTP with a forced PTY is unusable because terminal processing
  corrupts the binary protocol. A raw/no-echo PTY followed by explicit
  `/usr/libexec/openssh/sftp-server` startup succeeds.
- A real 277-byte upload through the raw-PTY SFTP path was read back with an
  identical SHA-256 and then removed. This establishes a verified workaround
  without compression or test-file residue.
- CheXpert-Plus, MS-CXR, and Chest ImaGenome were subsequently transferred
  through the workaround and verified by per-file SHA-256. An initial MS-CXR
  aggregate digest differed only because Windows and Linux sorted mixed-case
  file names differently; all seven authoritative per-file hashes match.
- NIH completed with exact local/remote agreement on file count (112,128),
  bytes (45,077,186,807), and deterministic relative-path/size digest
  (`74ea0031732606cb72665180db3d2c0af1dd74ff4f4d78dc4143bcdd7e4bb2da`).
  No partial upload files remain.
- The final new server surfaces occupy about 46 GiB for data and 1.1 MiB for
  code. GPFS quota reports 838.50 GiB used and 185.50 GiB remaining, preserving
  the planned 80--120 GiB working reserve.

## Remaining gate evidence

- Exact report-gold dataset and source-specific labeler contracts.
- MIMIC local/remote file, study, patient, and hash coverage equivalence.
- A deletion-ready remote evidence manifest and explicit approved target list.
- License/access qualification and patient/split hashes before training.
- Training, checkpoint-selection, and evaluation adapters behind Gate 0/1.

## 2026-07-23 execution preflight

- The user explicitly authorized RCSD-CXR server execution on allocation
  `4161`; this does not authorize any old VIVID/BiVES/MORPH experiment.
- Live Slurm state confirms job `4161` (`tpami`) is RUNNING on `gpu01` with one
  A800 80GB GPU, four CPUs, and 64GB RAM.
- The allocation is shared in practice: PID `455229` is an unrelated
  IndexMemory Qwen3.5-4B process using about 10.2 GiB VRAM and active GPU
  compute. It will not be stopped or signalled.
- The new repository still lacks a complete trainer, report-source adapters,
  label-model fitter, checkpoint selector, and downstream evaluators. The
  present code can validate paths and core tensor/posterior contracts but
  cannot truthfully execute the full proposal yet.
- Qwen3.5 model sizes must be inventoried live. They belong to a controlled
  teacher-sensitivity experiment after the primary teacher and method have
  survived earlier gates, not to an unconditional all-size launch.
- The server has Qwen3.5-0.8B (1.7 GiB), 2B (4.3 GiB), 4B (18 GiB), and 9B
  (25 GiB) directories under the shared model root. All advertise
  `Qwen3_5ForConditionalGeneration` with a vision configuration.
- The server environment is currently suitable at the package level: PyTorch
  2.9.0+cu128, Transformers 5.12.1, timm 1.0.27, pandas 2.3.3, scikit-learn
  1.7.2, PyYAML 6.0.3, and Pillow 12.2.0. Model loadability remains to be
  tested without disturbing the active GPU process.
- Qwen3.5-2B is frozen as the primary teacher candidate before visual results;
  0.8B/4B/9B are conditional sensitivity variants. Using all four as competing
  main methods would confound teacher selection with test-driven scaling.
- Queue attempt 1 stopped correctly at `G0_registry`: the controller itself ran
  in `vivid_med310`, but its task shell resolved bare `python` to a system
  interpreter without NumPy. This is an environment-binding defect, not a data
  or scientific-gate failure. Attempt 2 uses the absolute verified interpreter
  path in every enabled task and a new state directory.

## 2026-07-23 gate results

- G0 passed for the controlled MIMIC Track A surface: 215,098 canonical frontal
  studies from 63,656 patients (213,365 train and 1,733 validate), with zero
  test rows, patient overlap, missing images, missing reports, or empty files.
  The canonical manifest SHA-256 is
  `00fde375c608017d5e5700f946a15f32097d44ceecec885ebae41dfc58578133`.
- The lineage/license audit qualifies MIMIC, the official MIMIC metadata,
  Chest ImaGenome, and MS-CXR for their declared Track-A roles. CheXpert,
  CheXpert-Plus, NIH, and VinDr remain conditional or locked for later roles;
  CheXlocalize validation/test remain excluded from paper-one execution.
- The static historical audit confirmed the accepted SPD checkpoint contains
  exactly four query groups of shape `1 x 2 x 768`. Its artifacts, configs,
  trainer source, global steps, validation losses, and hashes are frozen.
- The official CheXpert/NegBio source manifest contains all 215,098 canonical
  rows. Both source tables omit the same eight studies; these are preserved as
  all-fields-missing rather than dropped or converted to negatives. Across all
  concepts, 51,204 rows have no observed label in either source.
- The two official sources disagree nontrivially, including 5,449 cardiomegaly,
  2,351 lung-opacity, and 2,194 edema rows. This establishes the practical need
  to measure source reliability rather than silently choose one parser.
- Chest ImaGenome gold has 21,594 attribute-relation rows covering 500 patients
  and 500 studies. Ten mapped findings contain both positive and negative
  examples, but this surface provides no uncertain gold and cannot unlock the
  three-state G2 posterior gate.
- LUNGUAGE is the required independent three-state report-gold source. The
  logged-in PhysioNet account does not currently have access and requires the
  user to personally sign its data-use agreement. This is the active external
  authorization blocker.
- Allocation 4161 remains RUNNING. The latest job-local GPU snapshot was idle
  (10 MiB used, about 81.1 GiB free, 0% utilization), but no GPU experiment was
  launched because the posterior-validity gate has not passed.
- The user confirmed the LUNGUAGE DUA was signed. The protected PhysioNet file
  URLs still could not be acquired automatically: the authenticated Chrome
  content page repeatedly timed out through the control channel, and direct
  file navigation was blocked by the browser client. No credentials, cookies,
  or alternate access-control bypass were attempted.
- The official LUNGUAGE release contains `Lunguage.csv` and
  `Lunguage_vocab.csv`. A local qualification script now checks their published
  1,473-report/230-patient/17,949-entity/3,827-vocabulary-row contract, hashes,
  three-state fields, and zero overlap with the Track-A training studies.
- The manually acquired protected files match between local and server:
  `Lunguage.csv` is 17,296,674 bytes with SHA-256
  `175fa761648f0ade00c0046537fa5a89924f170fb5bc1752fff12ddf464ac2cd`;
  `Lunguage_vocab.csv` is 357,148 bytes with SHA-256
  `f00ab49fb0a06d708d0098047b8bc0e25284cbdf151a65137dc7d2fe4f6008ca`.
- The release contains the exact published 230 patients and 1,473 studies, but
  the downloaded files contain 17,946 entity rows and 3,868 vocabulary rows,
  versus 17,949 and 3,827 on the release page. The qualification record keeps
  this as a publisher-count warning; exact file hashes and observed counts are
  authoritative.
- LUNGUAGE has zero study overlap with the controlled Track-A train/validation
  surface. Its report-gold mapping yields 5,730 study-finding labels across
  1,455 studies and 11 conservatively mapped findings; Enlarged
  Cardiomediastinum remains intentionally unmapped.
- The fixed five-fold two-source diagnostic did not support posterior fusion.
  NegBio was the best single source (macro-F1 0.7811, NLL 0.4746, ECE 0.0488).
  CheXpert+NegBio fusion reached macro-F1 0.7637, NLL 0.4943, and ECE 0.0437:
  -1.74 pp F1 and +4.15% NLL relative to NegBio. This result is preserved and
  the gold mapping will not be altered after seeing it.
- LUNGUAGE `report` is section-specific. CheXbert input is therefore frozen as
  unique `section_report` text from findings followed by impression, excluding
  history. There are 1,472 studies with at least one findings/impression
  section, with no conflicting section text.
- The server CheXbert checkpoint contains the complete BERT and classifier
  state, so only the small BERT config/tokenizer vocabulary is staged; the
  534-MiB generic BERT file is not duplicated. The loader requires zero missing
  and zero unexpected checkpoint keys.
- Formal five-fold evaluation used disjoint likelihood-training, calibration,
  and test patient folds. CheXbert was the best single source (macro-F1 0.8005,
  NLL 0.3712, ECE 0.0059). Three-source fusion improved macro-F1 to 0.8133 but
  worsened NLL to 0.3971 (+6.98%) and ECE to 0.0153. G2 is therefore NO-GO for
  posterior fusion; CheXbert is frozen and no fourth-source rescue is allowed.
- Allocation 3066 (`bash-gpu01-64g4c`) is RUNNING on gpu01 with four CPUs,
  64 GiB RAM, and one job-visible A800 80GB. Its job-visible GPU was at 10 MiB
  and 0% utilization when accepted as the new RCSD execution allocation.
- The deterministic simplified development surface contains 20,000 training
  studies plus all 1,733 canonical validation studies. CheXbert source labels
  were produced for all 21,733 reports. A 256-row patient-aware overfit subset
  contains all three report states across the aggregate target surface.
- The uploaded ImageNet ViT-B initialization is 346,284,714 bytes with SHA-256
  `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`;
  local and remote hashes match.
- Equal-budget overfit passed for both models. SPD reached 98.55% observed
  target accuracy with 95.66% loss reduction at step 200. Field anchoring
  reached 98.13% with 95.22% loss reduction at step 100. Both have 85,798,656
  backbone, 13,788,672 projector, and 77,860 state-head trainable parameters.
- Qwen3.5-2B produced 96 frozen field templates at hidden size 2,048:
  observation `[12,2048]`, assertion `[12,3,2048]`, anatomy `[12,2048]`, and
  global `[12,3,2048]`. Teacher-size selection is not reopened.

## 2026-07-23 terminal RCSD finding

- The paired 20k development run completed after fail-closed exclusion of
  all-missing targets. The final surface contained 19,533 train and 1,679
  validation studies; no missing target was converted to absent.
- Unanchored SPD reached NLL 0.478478, macro-F1 0.701950, accuracy 0.819177,
  and ECE 0.010646 at step 1,000.
- Equal-budget field anchoring reached NLL 0.478258, macro-F1 0.702598,
  accuracy 0.819046, and ECE 0.007876 at step 1,000.
- Relative NLL improvement was only 0.046% versus the frozen 3% requirement.
  Macro-F1 improvement was only 0.0648 pp versus the 0.5 pp requirement.
  ECE and per-finding safety checks passed, but the two primary mechanism
  checks failed.
- G3 is terminal NO-GO. The effect is too small to support field anchoring as
  a new method contribution. G4-G6, external evaluation, multi-seed scaling,
  multi-institution training, and Qwen3.5 size sensitivity are cancelled.
- A schema defect made the initial gate artifact's top-level `pass` field mean
  “audit completed” while `comparison.g3_pass` correctly recorded false. The
  schema is corrected so top-level `pass` equals the scientific gate result;
  metrics and thresholds are unchanged.

## 2026-07-23 component-attribution audit

- The correct project-level statement is not “the whole VIVID extension
  failed.” Full RCSD is NO-GO; the VIVID extension remains open only for a
  bounded attribution decision.
- G2 directly rejects D2 posterior fusion. CheXbert was the best single source
  at macro-F1 0.800504, NLL 0.371156, and ECE 0.005906. Fusion reached
  macro-F1 0.813270 but worsened NLL by 6.981% and ECE to 0.015316.
- G3 rejects the tested D3 field anchor under the structured-state pilot.
  NLL improved only 0.046% and macro-F1 only 0.0648 pp.
- The existing unanchored SPD arm is D0-like rather than exact D0: it uses
  CheXbert report-state targets and Qwen field prototypes, not a verified
  reconstruction of the original VIVID objective.
- D1 selective agreement weighting has never been trained.
- D2 did not proceed to visual training because its label-layer prerequisite
  failed.
- Expert-development macro AUROC/AUPRC and a visual reliability-quartile table
  are unavailable. They must remain missing, not inferred from macro-F1.
- The only possible future diagnostic is a separately reviewed exact D0 versus
  D1 pair. The current audit does not authorize it.

## 2026-07-23 D0 reconstruction review

- The existing 20k unanchored-SPD implementation is not an exact historical
  D0. Its targets are CheXbert three-state report labels and its auxiliary
  semantic objects are Qwen3.5 field prototypes.
- Exact D0 must be reconstructed from the retained historical VIVID code,
  accepted 4x2 checkpoint/config metadata, UMS serialization and masking,
  token-prediction loss, SPD orthogonality term, optimizer, image selection,
  training surface, and strictly-lower-validation-loss checkpoint rule.
- The clean RCSD migration deliberately did not copy the old monolithic
  VIVID model or token-loss contract. Therefore the new review protocol cannot
  call the current state trainer “original SPD” without an explicit provenance
  mapping and parity checks.
- D1 must preserve every frozen D0 element and add only one scalar per-sample
  or per-field reliability multiplier. It may not add posterior fusion,
  field-anchored queries, new teacher prototypes, new labels, or new losses.
- The accepted historical files are now identified as
  `legacy/vivid_med/configs/ablation_A_ums_spd_12label.yaml`,
  `legacy/vivid_med/models/spd.py`, `legacy/vivid_med/training/losses.py`, and
  `legacy/vivid_med/training/trainer.py`, paired with the audited historical
  `outputs/ablation_A_ums_spd_12label/checkpoints/best.pt`.
- The accepted SPD config freezes ViT-B/16 at 224 pixels, Qwen2.5-1.5B
  Instruct, 12 labels, JSON UMS, 4 groups x 2 tokens, orthogonality weight
  0.02, seed 42, 10,000 steps, effective batch 32, ViT LR 2e-5, projector LR
  1e-4, weight decay 0.01, warmup 500, BF16, validation every 500 steps, and
  best-checkpoint selection by strictly lower validation loss.
- The historical primary loss is next-token cross-entropy over serialized JSON
  tokens. SPD adds its attention-pattern orthogonality penalty. This is not
  equivalent to the current three-state classification loss.
- Historical validation checkpointing uses token-generation validation loss
  only; the SPD orthogonality penalty is added during training but not to the
  validation loss. A faithful D0 comparator must preserve this asymmetry.
- Historical training applies the JSON token loss to all target tokens unless
  an explicitly enabled weighting/mask config is present. The accepted SPD
  config enables neither token weighting nor answerability masking.
- Historical loading uses separate CheXpert UMS train/validation JSONL files,
  a fixed 1,000-row validation cap, shuffled training, and deterministic seed
  42. The legacy image fallback and non-patient-safe fallback split are known
  defects; the controlled reconstruction must preserve scientific semantics
  while retaining the clean project's fail-closed image and patient-lock
  safety contracts.
- The supplied diagnostic specifies D1 as entropy-based agreement weighting,
  not a learned posterior: average the available source one-hot distributions,
  compute normalized entropy, and use
  `w_ic = m_ic * (1 - H(q_bar_ic) / log(3))`. The hard target itself must
  remain the frozen best single source.
- To keep the comparison single-factor, D0 and D1 must use identical hard UMS
  strings. D1 may alter only the token-loss weights associated with a finding;
  it may not replace states, add a learned confidence model, or change the SPD
  projector.
- “Exact D0” needs two explicit identities: D0-H is the immutable historical
  CheXpert artifact; D0-CP is the controlled-protocol reconstruction of its
  method contract on the locked 20k MIMIC surface. D1 must be paired against
  D0-CP, not compared numerically against the historical artifact.
- The earlier source-migration manifest is not sufficient as the D0 authority:
  several byte hashes differ from the current Windows checkout and it omitted
  the full D0 contract. The review package must record current Git blob
  identity plus LF-normalized SHA-256 and must fail closed if those identities
  drift before implementation review.
- The current `track_a_pilot.yaml` remains a placeholder-era RCSD config and
  cannot authorize D0/D1. A separate machine-readable review lock is required;
  all dataset, reliability, expert-development, and implementation artifact
  hashes that do not yet exist must stay explicit `null` prerequisites.
- Historical narrative surfaces disagree: the May revision log reports a
  single SPD LP AUC of 0.8208 below no-SPD UMS at 0.8439, while the older
  mentor-summary surface reports a later/compiled three-seed SPD mean of
  0.8588 above UMS at 0.8431. The latter document also contains a stale 3x2
  method description beside a 4-group table. These values cannot be pooled or
  treated as a single protocol.
- No deidentified G1 checkpoint-audit JSON is present in the local clean
  worktree. The checkpoint and audit hashes therefore remain unresolved
  prerequisites rather than being inferred from narrative documents.
- The retained UMS v0.2 schema is broader than the training target. The actual
  historical dataset renderer deliberately emits the simplified target with
  only `modality`, selected `findings` objects (`state`, normally `score`), and
  `study_view`. D0-CP must reproduce this renderer, not the full schema.

## 2026-07-23 automatic D0/D1 continuation

- The user explicitly approved continuous experiments, failure case studies,
  and bounded improvements. The approval is recorded without changing any
  scientific threshold or opening a test surface.
- Higher-priority repository authority keeps all active work on the
  workstation. Server and Slurm permission will not be exercised.
- Live preflight found two RTX 3090 GPUs. GPU 1 was idle; GPU 0 had only about
  0.8 GiB baseline allocation. GPU ownership will be rechecked before launch.
- The frozen Qwen2.5-1.5B-Instruct teacher is already present in the local
  Hugging Face cache with a 3,087,467,144-byte `model.safetensors`.
- Local PyTorch 2.5.1+cu121 sees both GPUs; Transformers 5.5.3, timm 1.0.24,
  and torchvision 0.20.1+cu121 are installed.
- The historical local SPD output directory exists but its checkpoint
  directory is empty. D0-H hashes cannot be fabricated from the directory and
  remain an unresolved provenance prerequisite unless an existing validated
  audit artifact supplies them.
- Local MIMIC pixels/reports and official CheXpert/NegBio label tables are
  available. Existing local runtime output contains only the Gate-0
  qualification artifact, so the 20k CheXbert, hard-UMS, reliability, and
  expert-development manifests must be produced locally and hashed.
- R0 parity now passes. Clean and retained SPD implementations each have
  13,000,704 parameters and produce 205 visual tokens (8 query plus 197 ViT)
  in the 1,536-dimensional Qwen space. D0 and D1 token IDs and hard labels are
  identical.
- A real-weight one-sample forward/backward smoke passed with finite ViT/SPD
  gradients, frozen teacher gradients, and 4.36 GB peak allocated GPU memory.
- The frozen visual surface contains 19,533 train and 1,679 validation rows
  after fail-closed removal of all-missing CheXbert targets.
- D1 changes only 5,176 of 97,298 observed finding weights (5.32%); 74 are
  effectively zero. The only rounded weights are 0, 0.369070246429,
  0.420619835714, and 1. The percentile cuts are all 1, so reliability
  quartiles are explicitly degenerate rather than silently presented as four
  distinct confidence strata.
- CheXpert probe manifests are frozen at 191,027 frontal training rows and 202
  frontal expert-development rows with zero patient overlap.
- R0-R2 passed, all 67 tests passed, and the machine lock now authorizes only
  two sequential 256-row overfit arms on workstation GPU 1.
- The strict deterministic paired overfit gate passed. D0 reached token
  accuracy 0.981992 and NLL 0.050750 at step 400 (97.07% NLL reduction);
  D1 reached 0.980542 and 0.071878 at step 500 (95.86% reduction).
- Both arms had finite, nonzero backbone and projector gradients. The 256-row
  D1 surface had 46/1,176 nonunit finding weights (3.91%) and no exact zero
  weight. This proves learnability only, not D1 scientific benefit.

## 2026-07-23 remote execution facts

- The active user amendment permits only the bounded RCSD D0/D1 paper-one
  experiment on SUES allocation `3066`; it does not reopen BiVES, ARISE,
  VICER, MORPH, CheXlocalize test, VinDr test, or any post-hoc rescue route.
- Allocation `3066` is live on `gpu01` with one GPU, four CPUs, and 64 GiB RAM.
  The remote project, data, MIMIC pixels, and Python environment are present.
- The exact Qwen2.5-1.5B-Instruct snapshot is absent remotely. The local
  authority contains a 3,087,467,144-byte `model.safetensors` with SHA-256
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`;
  upload and remote hash verification are prerequisites to launch.
- A fresh server run must begin at D0 step zero and execute D0 then D1
  sequentially under the same allocation. The stopped workstation D0 step-30
  attempt is retained only as an invalid pre-metric operational record.

## 2026-07-23 Qwen3.5 inventory and design consequence

- Complete remote models exist at
  `/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model/Qwen3.5-{0.8B,2B,4B,9B}`.
  Their current directory sizes are approximately 1.77 GB, 4.57 GB,
  18.69 GB, and 25.92 GB.
- The Qwen3.5-2B config and weight authority matches the local model:
  config SHA-256 `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`,
  index SHA-256 `aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9`,
  and weight-shard SHA-256
  `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
- Qwen3.5-2B has a 2,048-dimensional text hidden state, so the SPD projector
  remains structurally 4x2 but changes its output dimension from 1,536 to
  2,048. This is a teacher-family amendment shared by D0 and D1, not a D1-only
  architectural delta.

## 2026-07-23 Qwen3.5 acceleration qualification

- The initial Qwen3.5-2B overfit step was stopped at step 10 before any gate
  metric because Transformers was using its slow fallback path. The run is
  explicitly invalid and will not be resumed or compared.
- `flash-linear-attention==0.5.2` and
  `causal-conv1d==1.6.2.post1` are now installed in the approved remote
  environment. The latter was built from the PyPI source distribution for
  `sm_80`; its locally built wheel reported SHA-256
  `679e464f024cdbb486c2b47416b2dacb43aabbcb4a290d953f4b6ea3d5436fe4`.
- The first source build failed before installation because the CUDA C++
  compiler could not find `cuda_runtime_api.h`. The header existed under the
  Conda target prefix; explicitly adding that include path fixed the
  environment without changing code, data, weights, or thresholds.
- A job-local CUDA kernel test passed forward and backward with finite values.
  Transformers reports the Qwen3.5 fast path available, both FLA gated-delta
  kernels are callable, and `pip check` reports no broken requirements.
- The real-weight RCSD smoke passed twice with exact repeat equality:
  token loss `3.524507999420166`, orthogonality `0.9854222536087036`, peak
  allocation `5,406,058,496` bytes, finite ViT/projector gradients, frozen
  teacher, and 205 visual tokens. The warm repeat took 17.41 seconds.
- The replacement paired overfit must start from zero in the new `s2` output
  root. Launchers now fail closed if the Qwen3.5 acceleration imports or fast
  path are unavailable.
- The replacement `s2` paired overfit passed both arms at step 350. D0 reached
  token accuracy `0.9876102800`, token NLL `0.0342915525`, and `98.15%` NLL
  reduction. D1 reached token accuracy `0.9854622171`, token NLL
  `0.0401251204`, and `97.83%` NLL reduction. Both gradient audits passed.
- The frozen D0, D1, and queue-state SHA-256 values are respectively
  `9e027a7337003744e10ae6a3612d0ae13c19adabf1320324d7b126a10937697d`,
  `328c90a838b56ab4d819c2c795d9c17bd73111598d222ce7a3c3f90354b5becf`,
  and `bb70b6a02d6d605b909079e44673ce7a84b4b9e9721ee8bf8fe456e3a460e9af`.
- This is a learnability qualification only. It authorizes the paired 20k
  D0/D1 pilot but is not evidence that D1 improves representation quality.
- The authorized 20k run identity is
  `d0_d1_token_pilot_qwen35_2b_20260723_s2`. It started in Slurm step
  `3066.19012` with D0 first; D1 remains mechanically blocked until D0 exits
  successfully. The first observed D0 training record was step 30 with finite
  loss and no acceleration fallback.
- D0 completed its first frozen 20k validation at step 500 with token NLL
  `0.0868966725` and token accuracy `0.9660930965` over 169,523 observed
  tokens. This is a checkpoint-selection trajectory point, not a promotion
  decision; training continues to the unchanged 3,000-step budget.
- At step 1,000, D0 validation token NLL improved to `0.0820955950` and token
  accuracy to `0.9678922624` on the same 169,523-token surface. The strictly
  lower-NLL checkpoint rule therefore selects step 1,000 over step 500; no
  endpoint or gate has been applied early.
- At step 1,500, D0 validation token NLL improved again to `0.0805892760` and
  token accuracy reached `0.9684113660`. Step 1,500 is therefore the current
  strictly lower-NLL checkpoint; the full 3,000-step budget remains active.
- At step 2,000, D0 validation token NLL improved to `0.0785858087` and token
  accuracy to `0.9694849666`. The current checkpoint advanced to step 2,000
  under the same rule; there were no non-finite values or resource failures.
- At step 2,500, D0 validation token NLL improved to `0.0774704031` and token
  accuracy to `0.9697209228`. Step 2,500 became the current best checkpoint;
  the final step-3,000 validation remains binding.
- D0 completed all 3,000 steps in 5,915.03 seconds. Its final and best
  validation was step 3,000 with token NLL `0.0768119148` and token accuracy
  `0.9698801932`; all scheduled validation NLL values improved monotonically.
  ViT/projector gradient audits were finite and nonzero.
- The frozen D0 summary SHA-256 is
  `6d39051eecb6fb00af58da2b14269fe2f7eb2f4262f85dd393f43b9e9c6b52dd`;
  the selected checkpoint SHA-256 is
  `37ab60d2782043d0cffc7e8925b385143264731f4283112e5666744384e14895`.
  D1 then started from zero under the same queue and budget.
- D1 step-500 validation produced token NLL `0.0883252861` and token accuracy
  `0.9653321378`. At the matched step, D1 NLL was 1.64% higher than D0
  (`0.0868966725`) and accuracy was 0.0761 percentage points lower. This is
  an intermediate trajectory comparison only; the frozen promotion gate uses
  the final selected checkpoints and remains unopened until D1 completes.
- D1 step-1,000 validation produced token NLL `0.0827696097` and token
  accuracy `0.9675147325` over the same 169,523-token surface. Against D0 at
  matched step 1,000 (`0.0820955950`, `0.9678922624`), D1 NLL is 0.82%
  higher and accuracy is 0.0378 percentage points lower. The gap narrowed
  relative to step 500, but this remains an intermediate trajectory point;
  no promotion endpoint has been applied.
- D1 step-1,500 validation produced token NLL `0.0813829365` and token
  accuracy `0.9679571504`. Against D0 at matched step 1,500
  (`0.0805892760`, `0.9684113660`), D1 NLL is 0.98% higher and accuracy is
  0.0454 percentage points lower. D1 remains stable and its own NLL continues
  to improve, but it has not crossed D0 at any matched validation point.
- D1 step-2,000 validation produced token NLL `0.0790323267` and token
  accuracy `0.9693433929`. Against D0 at matched step 2,000
  (`0.0785858087`, `0.9694849666`), D1 NLL is 0.57% higher and accuracy is
  0.0142 percentage points lower. This is D1's narrowest matched gap so far,
  but the direction remains below D0 and the final selected-checkpoint gate
  is still unopened.
- D1 step-2,500 validation produced token NLL `0.0780412085` and token
  accuracy `0.9694200787`. Against D0 at matched step 2,500
  (`0.0774704031`, `0.9697209228`), D1 NLL is 0.74% higher and accuracy is
  0.0301 percentage points lower. D1's own NLL improved again, but all five
  matched validation comparisons remain directionally below D0.
- D1 completed the full 3,000-step budget with final/best token NLL
  `0.0772889282` and token accuracy `0.9697445184`. D0's frozen final/best
  values were `0.0768119148` and `0.9698801932`. D1 therefore made token NLL
  approximately 0.62% worse rather than meeting the frozen requirement of at
  least 3% improvement (`relative change <= -0.03`). The first required
  promotion condition fails, so downstream expert-development probing is not
  unlocked.
- Both queue arms exited with return code 0 and the queue ended with
  `pass: true`; this is execution completion, not a scientific gate pass. The
  D1 summary SHA-256 is
  `200d5c3cc541ccee2634b2781f71eb9bf03f26cacab00c9b3291c243831178ab`
  and the final queue-state SHA-256 is
  `c7c8c26a8910497b3df237fc37d79f373b1dc0beaf2824fd19cdde167d1a1548`.
- A read-only hash command assumed the checkpoint filename
  `best_checkpoint.pt`, which does not exist. The already frozen D0 checkpoint
  hash remains valid, but the exact D1 checkpoint filename and hash must be
  resolved from a file inventory before terminal freeze.
- The actual checkpoint filename in both arms is `best.pt`. D0 retains SHA-256
  `37ab60d2782043d0cffc7e8925b385143264731f4283112e5666744384e14895`;
  D1's selected checkpoint SHA-256 is
  `f08357394e4c96ef1c4d2d42b92add9eb2e8313a45e5a0cee616fb3216b3023e`.
- A repository-wide preregistration search found no named in-identity repair
  for a valid 20k D1 scientific failure. The review protocol uniquely states
  that D1 failure closes RCSD reliability and field-anchor development, and
  the auto-development protocol permits case study only unless such a repair
  exists. Therefore no rerun, expert-development probe, teacher-size
  sensitivity, or new weighting rule is authorized.
- The development-only case study found that only 4,791/89,622 (5.35%) of
  observed training finding fields and 385/7,676 (5.02%) of validation fields
  were downweighted. Only 4.04% of validation target tokens were affected;
  mean target-token weight remained 0.9762. Pneumonia had the highest
  concentration, with 19.75% of its training fields downweighted.
- D1 was behind D0 at every frozen matched validation point, from step 500
  through step 3,000. This rules out a checkpoint-selection explanation.
  Together with the passed overfit and finite-gradient audits, the result is a
  valid scientific failure of the weakly supported scalar weighting mechanism,
  not an implementation or learnability failure.
