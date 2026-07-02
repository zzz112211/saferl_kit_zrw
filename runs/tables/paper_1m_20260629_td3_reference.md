# TD3 Reference MetaDrive Results

Step: 1,000,000. Values are mean ± normal 95% confidence over 5 seeds.

| Environment | Metric | TD3 |
|---|---|---|
| MetaDrive | SuccessRate | **0.81 ± 0.07** |
| MetaDrive | Ep-Cost | **15.39 ± 2.16** |
| MetaDrive | CostRate | **0.059 ± 0.009** |

Notes: Safety Layer corresponds to the repository's `--use_qpsl`; `SuccessRate` is logged as `EpRet` in `train_metadrive.py` for MetaDrive.
