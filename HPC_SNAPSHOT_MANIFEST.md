# HPC QPSL NociCIM Snapshot

This branch preserves the HPC working-tree snapshot used for the QPSL
`nocicim_close_range` 5-seed, 100-episode evaluation table.

Source host path:

`/datapool/home/1001900174/home/DYJ/saferl_kit`

Remote archive:

`/tmp/saferl_hpc_snapshot_qpsl_20260702.tar.gz`

Archive SHA256:

`cb1c204033e988a31e2a4ec70d6e8a9b7ed79a48e96791b86b8fe4b3c76946b3`

Excluded from the snapshot archive:

- `.git`
- `.venv-saferlkit`
- `logs`
- `models`
- `__pycache__`
- local `.pytest_cache` after extraction

Primary result table:

`runs/tables/sweep_qpsl_nocicim_close_route_100ep_20260702/five_seed_100ep_summary.csv`

Primary result table SHA256:

`7a597ede011a2edeb2dd5acaa23f7bf3b3bd9b7b23558f19a2bd3a64a13a0dae`

Key source SHA256 values from the HPC snapshot:

- `eval_memristive_frontend.py`: `855ed3cdce826b49a90e93d69056a52cd7ccc866592de12fae15ea55326af4f0`
- `train_metadrive.py`: `a1727e7b5d3c7d2f108bd2eb36385af7affedffa2508d3ba7c50f148c73c76e8`
- `sweep_memristive_frontend.py`: `4b7be7287569f8cfd14c3cd3e34e3d845d1ab6bea3cd6b8b718f6422a5fa3f9a`

Important boundary:

This is the original HPC snapshot for the reported table. It is not the later
`codex/nocicim-risk-memory-upgrade` branch that adds trend-gated memory writes.
