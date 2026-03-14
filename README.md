# Warehouse RL Comparison

Comparison of reinforcement learning algorithms (DQN, DDQN, PPO, SAC) on a robotic warehouse delivery task using the RWARE environment.

## Setup

1. **Python 3.10** recommended
2. Create and activate a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python -m venv venv` then `source venv/bin/activate`
3. Install PyTorch (choose one):
   - CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
   - CPU only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
   - See [PyTorch](https://pytorch.org/get-started/locally/) for other CUDA versions.
4. Install project dependencies:
   ```
   pip install -r requirements.txt
   ```
   If `rware` fails, install from source:
   ```
   pip install git+https://github.com/semitable/robotic-warehouse.git
   ```

## Training

```
python train.py --algo <algorithm> [options]
```

| Option | Values | Default |
|--------|--------|---------|
| `--algo` | `dqn`, `ddqn`, `ppo`, `sac`, `all` | `dqn` |
| `--episodes` | any integer | from config |
| `--env` | `default`, `small`, `10x10`, `16x16`, `22x22` | `default` |
| `--device` | `auto`, `cuda`, `cpu` | `auto` |
| `--seed` | any integer | none |
| `--resume` | path to checkpoint | none |

**Examples:**
```
python train.py --algo dqn --episodes 10000 --env 10x10
python train.py --algo ppo --episodes 5000 --env 16x16 --device cuda
python train.py --algo sac --episodes 10000 --env 22x22 --seed 42
python train.py --algo all --episodes 5000 --env 10x10
```

## Visualization

Run a trained agent with rendering:
```
python visualize.py --algo <algorithm> --model <path> --env <preset>
```

**Examples:**
```
python visualize.py --algo dqn --model models/dqn_ep10000.pt --env 10x10
python visualize.py --algo sac_original --model models/Archive/sac_final.pt --env 10x10
```

## Evaluation & Plots

- **Evaluate metrics** (reads `outputs/*_metrics.json`):
  ```
  python scripts/evaluate.py --outputs outputs
  ```
  Prints six tables covering all paper results:
  | Table | Content |
  |-------|---------|
  | 1 | Full 10,000-episode summary (success rate, battery deaths, episodes to 80%) |
  | 2 | Last 1,000 episodes (reward, steps, deliveries, charges, battery deaths) |
  | 3 | Battery management — deaths, death rate, avg charges, avg battery left |
  | 4 | Training stability ep 5,001–10,000 (mean SR, std dev, min, max, dips <90%) |
  | 5 | Learning milestones — rolling-window SR at ep 500/1k/2k/3k/5k/10k |
  | 6 | Convergence — episodes to reach 80%, 95%, 99% success rate |

- **Plot learning curves** (saves figures to `outputs/`):
  ```
  python scripts/plot_learning_curves.py
  ```
- **Statistical analysis:**
  ```
  python scripts/analysis.py
  ```
- **Analysis plots:**
  ```
  python scripts/plot_analysis.py
  ```
- **Poster figures:**
  ```
  python scripts/plot_poster.py
  ```

> **Note:** Pre-trained metrics are already in `outputs/*_metrics.json` and pre-trained model checkpoints are in `models/`, so you can run evaluation and visualization without retraining.

## Map Presets

| Preset | Shelf Columns | Shelf Rows | Max Steps | Max Deliveries |
|--------|---------------|------------|-----------|----------------|
| `default` | 3 | 4 | 500 | 5 |
| `small` | 3 | 4 | 500 | 5 |
| `10x10` | 3 | 4 | 500 | 5 |
| `16x16` | 5 | 7 | 800 | 5 |
| `22x22` | 7 | 10 | 1200 | 5 |

Custom map: `--shelf-columns N --shelf-rows N --max-steps M`

## Project Structure

```
train.py              # Main training script
visualize.py          # Run and render a trained agent
configs/config.py     # Environment presets and algorithm hyperparameters
agents/               # Agent implementations (DQN, PPO, SAC)
envs/warehouse.py     # Custom warehouse environment wrapper
scripts/evaluate.py   # Metrics evaluation — prints 6 tables (final perf, battery, stability, milestones, convergence)
scripts/analysis.py   # Statistical analysis
scripts/plot_learning_curves.py  # Learning curve plots
scripts/plot_analysis.py         # Analysis plots
scripts/plot_poster.py           # Poster figures
models/               # Saved model checkpoints (pre-trained, ready to use)
outputs/              # Training metrics (JSON, pre-computed) and plots
```
