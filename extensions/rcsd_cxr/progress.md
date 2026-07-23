# Progress log

## 2026-07-22

- Loaded and read the `planning-with-files` skill.
- Ran planning session catch-up in the source repository.
- Confirmed `H:\Xiyao_Wang\02101` did not previously exist and created the
  directory explicitly at the user's request.
- Checked live source Git state and recorded pre-existing user changes.
- Began reading the required audit authority, handoff indexes, planning files,
  and verified public-dataset inventory.
- Created the new project's persistent planning files before migration work.
- Logged one read-only PowerShell parser error from a file-size inventory; the
  command made no filesystem change and the next attempt uses an explicit
  result array.
- Browser connection was established, but the first direct ChatGPT conversation
  navigation timed out before page state was returned. Logged the error; no
  external or local file was changed by that failed attempt.
- No data, model, checkpoint, output, experiment, GPU, server, test split, Git
  commit, or remote publication action has been opened.
- Located the user-supplied RCSD proposal in the source repository, recorded its
  42,251-byte source hash, and preserved a text-identical normalized copy plus
  task lineage under `provenance/`.
- Built the clean `02101` scaffold, active authority, portable/local data
  registries, source migration manifest, and fail-closed implementation core.
- Reimplemented the SPD 4x2 and field-anchored equal-budget projectors, a
  missing-aware posterior fusion primitive, a fail-closed manifest loader, and
  a patient-aware canonical-frontal MIMIC manifest builder.
- Validated five active local data references without opening reserved test
  data.
- Ran Python compilation and 12 unit tests; all passed.
- Scanned the clean tree: 35 non-cache files, 115,522 bytes (0.110 MiB), with
  no model weights, medical images, archives, checkpoints, or files over 5 MiB.
- Measured all 15 canonical local resource paths and documented Track-A,
  paper-one, and all-resource capacity scenarios.
- Performed read-only SSH checks of GPFS quota, old remote VIVID sizes, Slurm
  association, queue, and job identities. No remote file or job was changed.
- Wrote a staged remote cleanup plan before deletion; later batches still
  require fresh evidence and explicit literal targets.
- Received explicit authorization to begin cleanup without compression or
  pixel re-encoding.
- Froze batch-1 local/remote recovery evidence, verified that the two active
  Slurm jobs belong to IndexMemory rather than VIVID, and removed five exact
  remote targets with containment guards.
- Confirmed all five batch-1 targets absent and all protected MIMIC, CheXpert,
  VinDr, outputs, and pretrained surfaces present.
- Reclaimed 100.99 GiB: user-quota headroom increased from 138.39 GiB to
  239.38 GiB. No other project was touched.
- Confirmed that no additional compression or image re-encoding will be used.
- Added a server-side dataset registry that reuses existing MIMIC, CheXpert,
  and derived VinDr paths and marks NIH/Chest-ImaGenome/CheXpert-Plus absent.
- Built a file-level uncompressed server-upload manifest; local-only Windows
  paths and Python caches are excluded.
- Computed the local MIMIC tree summary: 377,110 images (2,329,089,036 bytes)
  and 227,835 reports (151,686,630 bytes). The remote path/size digest was
  stopped after prolonged GPFS traversal; it must resume when SSH capacity is
  available.
- Falsified the initial local-connection-count explanation by stopping all
  local SSH clients and reproducing the failure from a clean state.
- Isolated the SUES failure to non-PTY session initialization: TCP and public
  key authentication succeed, while non-PTY exec/SFTP stalls at a zero remote
  receive window; forced-TTY SSH succeeds.
- Audited the remote account through a forced TTY: process/file limits are
  healthy, no stale `~/.ssh/rc` exists, and the SFTP server is available at
  `/usr/libexec/openssh/sftp-server`.
- Prototyped and verified raw-PTY SFTP with a real upload/readback SHA-256
  match, removed the probe, added a dedicated `sues-hpc-tty` alias, and added
  `scripts/upload_rawpty_sftp.py` for resumable verified staging.
- Cleared the exact remote `02101`/`02101_data` staging targets, uploaded the
  rebuilt project through raw-PTY SFTP, and independently verified 40/40
  manifest hashes plus 12/12 remote unit tests.
- Uploaded and content-verified CheXpert-Plus (6 files, 507,355,947 bytes),
  MS-CXR (7 files, 4,062,782 bytes), and the publisher-provided Chest
  ImaGenome archive (1,553,519,249 bytes). All individual SHA-256 values match.
- Completed the resumable NIH upload: 112,128 files and 45,077,186,807 bytes.
  The local and remote relative-path/size digest is identically
  `74ea0031732606cb72665180db3d2c0af1dd74ff4f4d78dc4143bcdd7e4bb2da`,
  with zero `.rcsd-part` files and uploader exit code 0.
- Final staged project/data sizes are about 1.1 MiB and 46 GiB. Fresh GPFS
  quota after upload is 838.50 GiB used and 185.50 GiB remaining.
- Final server acceptance: 41/41 project-manifest hashes matched, 12/12 unit
  tests passed, and all seven paper-one server data references validated.

## 2026-07-23

- Loaded the `experiment-plan`, `run-experiment`, and `planning-with-files`
  instructions for the newly authorized server-execution task.
- Archived the completed clean-extraction phase boundary and replaced the
  active task plan with a gate-controlled RCSD experiment plan.
- Re-read the reviewed RCSD authority, supplied proposal, repository scope,
  current code, configs, server registry, and planning evidence.
- Freshly verified allocation `4161`: RUNNING as `tpami` on `gpu01`, one A800
  80GB, four CPUs, 64GB, job-local CUDA device 0.
- Detected and preserved the unrelated active IndexMemory Qwen3.5-4B process
  on the allocated GPU; no process or job was stopped.
- Confirmed that the current clean project is a scaffold rather than a complete
  training implementation. No GPU experiment has been falsely marked started.
- Audited remote Qwen3.5 assets and package versions. The 0.8B, 2B, 4B, and 9B
  model directories are present and referenced in place; no weights were
  copied.
- Created timestamped and fixed-name claim-driven experiment plan/tracker
  artifacts under `refine-logs/`, including the conditional all-size Qwen3.5
  sensitivity queue.
- Implemented a fail-closed MIMIC Gate-0 manifest audit and a sequential queue
  controller with prerequisite markers, per-task logs, command hashes, and a
  non-destructive GPU headroom guard. Local validation is 15/15 tests passed.
- Uploaded the official MIMIC metadata, split, CheXpert, NegBio, labelled-test
  table, and publisher SHA manifest (about 87 MiB total). Independent local and
  remote SHA-256 values match for all six files.
- Synchronized the 53-file RCSD execution surface to the server and verified
  53/53 hashes, 15/15 tests, and six selected server data references.
- Launched queue attempt 1 on allocation `4161`. It stopped at `G0_registry`
  before any manifest or GPU work because a bare task-shell `python` resolved
  to a system interpreter without NumPy. Preserved the failure logs and changed
  attempt 2 to use the absolute verified `vivid_med310` interpreter.
- Queue attempts 2/3 completed the controlled G0 registry, canonical MIMIC
  build/audit, data-lineage qualification, and static historical checkpoint
  audit. No GPU work ran.
- G0 froze 215,098 canonical frontal study rows from 63,656 patients, with no
  test rows or missing assets and a manifest SHA-256 of
  `00fde375c608017d5e5700f946a15f32097d44ceecec885ebae41dfc58578133`.
- The historical audit froze the UMS/SPD checkpoint identities and confirmed
  the real SPD artifact is 4 groups x 2 tokens, resolving the old 3x2/4x2
  description mismatch without replaying the experiment.
- Implemented and tested the official MIMIC source-label join and a direct ZIP
  audit of Chest ImaGenome gold. Local and remote suites now pass 21/21 tests.
- The first G2 source-manifest attempt correctly stopped because eight
  canonical studies were absent from both official source tables. Updated the
  contract to preserve those source rows as missing (never negative), added a
  regression test, synchronized it, and reran from the failed task.
- G2 preparation then completed: a 215,098-row/45 MiB source manifest was
  written, and Chest ImaGenome gold was audited over 21,594 rows/500 patients.
  Both tasks have pass markers in the sequential queue.
- Confirmed that Chest ImaGenome contains binary present/absent evidence but no
  uncertain gold. The queue is now intentionally stopped at the LUNGUAGE DUA
  boundary; visual training and every Qwen3.5 size remain locked.
- The user confirmed the LUNGUAGE DUA was signed. Automatic protected-file
  download remained blocked by the authenticated Chrome transfer path; no
  browser credentials or cookies were inspected or exported.
- Added `audit_lunguage_gold.py` and regression tests so the two protected CSVs
  can be qualified immediately after manual download. Local validation is now
  23/23 tests passed.
- Qualified the manually downloaded LUNGUAGE release, uploaded both protected
  CSVs through the verified raw-PTY SFTP path, and confirmed exact local/remote
  SHA-256 agreement. The gold studies are independent of Track-A train/val.
- Implemented a conservative fixed 12-finding LUNGUAGE entity mapper and
  regression tests. Unmentioned concepts remain missing, tentative assertions
  map only to report uncertainty, and ambiguous anatomy terms are excluded.
- Ran the frozen five-fold CheXpert/NegBio diagnostic. The fusion failed both
  discrimination and NLL thresholds relative to NegBio; no post-hoc mapping or
  threshold change was made.
- Implemented and tested standalone CheXbert inference that reconstructs the
  published architecture directly from the frozen checkpoint, excludes
  history text, and outputs deidentified study-level source labels. Local and
  remote suites pass 31/31 tests.
- Staged only CheXbert config/tokenizer assets, freshly checked the shared
  A800, and launched the low-memory source inference in detached screen
  `rcsd_g2_chexbert_20260723`. The unrelated IndexMemory process remains
  active and untouched.
- CheXbert inference completed over 1,472 LUNGUAGE findings/impression reports:
  227 checkpoint keys loaded with zero missing/unexpected keys and 1,472
  deidentified source rows were written.
- Switched future RCSD execution to retained allocation 3066 after a live
  job-local check showed one idle A800 80GB, four CPUs, and 64 GiB RAM.
- Ran the formal three-source G2 gate on allocation 3066. The fusion passed the
  F1 threshold but failed the NLL threshold versus CheXbert, so the fusion
  contribution and Qwen3.5 size lane were cancelled before visual training.
- Wrote a dated binding protocol amendment. The only surviving route is a
  paired, equal-budget unanchored-SPD versus field-anchored learnability test
  using frozen CheXbert targets.
- Generated the deterministic 20k-train plus 1,733-validation CheXbert source
  manifest and a frozen 256-row overfit subset on allocation 3066.
- Uploaded and hash-verified the cached 346-MiB ImageNet ViT-B weight without
  compression or duplicate model download.
- Implemented an equal-trainable-parameter visual state model, masked
  three-state loss, fail-closed image loading, and sequential overfit launcher.
  Local/remote tests passed before launch.
- Both overfit variants passed. The diagnostic emitted cuBLAS deterministic
  warnings because the workspace variable was not set before process start;
  no failure occurred and both variants shared the same condition. The formal
  20k launcher now sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Built and froze 96 Qwen3.5-2B field prototypes, then launched the sequential
  20k SPD/field-anchor pilot plus automatic gate audit on allocation 3066.
- Completed the paired 20k SPD/field-anchor pilot on allocation 3066.
- Preserved the first pre-training failure caused by all-missing target rows,
  added fail-closed exclusion at dataset initialization, and restarted both
  variants from zero on the same frozen surface.
- Applied the prospective G3 thresholds. Field anchoring improved NLL by only
  0.046% and macro-F1 by only 0.0648 pp, so both primary checks failed.
- Declared RCSD-CXR terminal NO-GO and cancelled full-MIMIC, external-test,
  multi-seed, multi-institution, and Qwen-size work.
- Wrote `docs/RCSD_CXR_terminal_gate_result_20260723.md` and updated the active
  protocol, tracker, plan, README, findings, and progress surfaces.
- Corrected the gate-artifact schema so top-level `pass` mirrors `g3_pass`;
  no experiment, metric, threshold, or model was rerun or changed.
- Reran only the gate audit, producing top-level `pass: false` with SHA-256
  `159768072505e163ccc3e224567452714f22b40cdac28f65be8b00ab77535c41`.
- Completed final local and remote validation: 45/45 unit tests passed on both
  sides.
- Created `codex/rcsd-no-go-audit` from frozen implementation commit
  `bc1105f880116e97e06da023110b7080debc28a4`.
- Read the supplied component-attribution critique and reconciled it against
  the existing aggregate G2/G3 server evidence.
- Added a deidentified NO-GO verdict, machine-readable component status,
  D0-D4 evidence inventory, and per-finding G3 macro-F1 table under `audit/`.
- Corrected the overall interpretation from “RCSD-CXR as a whole is closed” to
  “full RCSD and the tested D2/D3 contributions are NO-GO; the VIVID extension
  remains open only for a bounded, review-gated component audit.”
- Added an integrity validator and tests. No training, external test, data
  download, server mutation, or GPU/Slurm job was launched.
- Validated the audit package: the integrity checker passed, Python sources
  compiled, and all 48 unit tests passed.
- Published the bounded audit to `origin/codex/rcsd-no-go-audit`; the audit
  content commit was `cd45646b2bf586609a1bd0e18311a457b1d6c235`.
