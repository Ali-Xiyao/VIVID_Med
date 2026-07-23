# RCSD-CXR reconstruction validation report

## Outcome

The clean project scaffold is internally consistent and safe to hand off for
Gate-0 data/protocol work. It is not yet a full training package.

## Commands and results

| Check | Result |
| --- | --- |
| `python -m compileall rcsd_cxr scripts tests` | PASS |
| `python -m unittest discover -s tests -v` | PASS, 12/12 |
| Local registry roles `track_a_train paper1_external structured_supervision` | PASS, 5 paths |
| Non-cache project footprint | 34 files, 109,809 bytes |
| Binary/large-file scan | PASS, no medical images, model weights, archives, checkpoints, or file over 5 MiB |
| Frozen SPD identity | PASS, non-4x2 construction rejected |
| Missing report source semantics | PASS, all-missing posterior returns missing |
| Corrupt/missing image behavior | PASS, fail-closed |
| Patient split overlap contract | PASS, cross-split patient rejected |

## Boundaries not tested

- No medical image was used in a model forward pass.
- No report parser, gold label set, text teacher, or pretrained backbone was
  downloaded or loaded.
- No checkpoint, training loop, external evaluation, GPU, or Slurm job ran.
- No CheXlocalize test asset was downloaded or opened.
- No remote file was copied, replaced, or deleted.

## Handoff gate

The next authorized work should be a read-only equivalence audit between the
local and remote 224-pixel MIMIC surfaces, followed by a report-label source and
gold-set qualification document. Training code should be completed only after
those identities and the checkpoint-selection rule are frozen.
