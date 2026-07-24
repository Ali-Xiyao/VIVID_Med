# VIVID-GDS Fresh Replication

This is a separately locked replication of the unchanged VIVID-GDS training
bridge. It does not reactivate or repair the terminal Stage-A experiment.

The data identity and survival gate are frozen in:

- `audit/VIVID_GDS_FRESH_REPLICATION_PROTOCOL_20260724.md`;
- `audit/vivid_gds_fresh_replication_lock.json`.

The server workflow creates a score-free patient split, audits readiness, then
runs A0/A2/A3 for seeds 0/1/2 sequentially on retained allocation 3066. No
fresh-development score had been opened when the lock was created.

Runtime artifacts stay under the isolated SUES project:

`/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_vivid_gds_fresh_replication`
