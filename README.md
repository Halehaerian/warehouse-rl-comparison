Python version: 3.10.19

## Training
- `python train.py --algo dqn` (or `ppo`, `sac`, `all`)

## Evaluation and plots (proposal metrics)
- **Table 1 & metrics:** `python scripts/evaluate.py` (reads `outputs/*_metrics.json`). Optionally `--out-table results/table1.csv`.
- **Learning curves:** `python scripts/plot_learning_curves.py` (saves reward, success rate, battery death figures to `outputs/`).

## Config
- `configs/config.py`: `ENV_CONFIG` (default), `ENV_CONFIG_5x5` for small grid (proposal Section 4).
