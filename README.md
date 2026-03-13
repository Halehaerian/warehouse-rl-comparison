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
  python scripts/evaluate.py
  ```
- **Plot learning curves** (saves figures to `outputs/`):
  ```
  python scripts/plot_learning_curves.py
  ```

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
scripts/evaluate.py   # Metrics evaluation
scripts/plot_learning_curves.py  # Learning curve plots
models/               # Saved model checkpoints
outputs/              # Training metrics (JSON) and plots
```
