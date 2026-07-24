# Launcher CRLF implementation failure

The first launcher process exited before creating a Slurm step or run root.

## Cause

The Windows checkout exported `launch_vivid_gds_stage_a_3066.sh` with CRLF
line endings. Remote Bash rejected `set -o pipefail` because the option name
contained a carriage return.

## Evidence and effect

- remote `vivid_gds_stage_a_3066.launch.log` captured the error;
- only allocation step `3066.batch` existed;
- no VIVID-GDS run root was created;
- no GPU training, data iteration, checkpoint, or metric occurred.

## Identity-preserving repair

`extensions/vivid_gds/.gitattributes` now enforces LF for all shell scripts.
The failed launch log is preserved before relaunch. No experiment identity,
threshold, data, model, target, or checkpoint rule changed.
