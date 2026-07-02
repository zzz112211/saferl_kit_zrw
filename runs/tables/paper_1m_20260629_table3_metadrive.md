# Paper-Style MetaDrive Results

Step: 1,000,000. Values are mean ± normal 95% confidence over 5 seeds.

| Environment | Metric | Safety Layer | Recovery RL | Lagrangian | FAC | EPO |
|---|---|---|---|---|---|---|
| MetaDrive | SuccessRate | **0.85 ± 0.08** | 0.75 ± 0.14 | 0.75 ± 0.10 | 0.57 ± 0.06 | 0.77 ± 0.08 |
| MetaDrive | Ep-Cost | 16.35 ± 2.13 | 14.12 ± 5.16 | 4.36 ± 1.21 | **3.97 ± 2.13** | 5.35 ± 1.54 |
| MetaDrive | CostRate | 0.043 ± 0.006 | 0.055 ± 0.006 | 0.019 ± 0.003 | **0.013 ± 0.001** | 0.015 ± 0.002 |

Notes: Safety Layer corresponds to the repository's `--use_qpsl`; `SuccessRate` is logged as `EpRet` in `train_metadrive.py` for MetaDrive.
