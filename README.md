Python version: 3.10.19

## Setup (CUDA or CPU)

**1. Create and activate a virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # macOS/Linux
```

**2. Install PyTorch — choose one:**

- **With CUDA (if you have an NVIDIA GPU):**  
  Pick the line that matches your [CUDA version](https://pytorch.org/get-started/locally/) (check with `nvidia-smi`).
```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

- **CPU only (no GPU or to force CPU):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**3. Install the rest of the project**
```bash
pip install -r requirements.txt
```
If `rware` fails (not on PyPI), install from GitHub:
```bash
pip install git+https://github.com/semitable/robotic-warehouse.git
```

**4. Check whether CUDA is available**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
```

Training uses **`--device auto`** by default: it uses CUDA if available, otherwise CPU. No code change needed.

## Training
- `python train.py --algo ppo` (or `dqn`, `sac`, `all`)
- `python train.py --algo ppo --episodes 1000`
- **Reproducibility:** `python train.py --algo ppo --episodes 500 --seed 42` (same seed ⇒ same results)
- **CUDA:** `python train.py --algo ppo --device cuda` or `--device auto` (use GPU if available). Requires PyTorch with CUDA.

## CUDA and scaling (bigger maps, more agents)
- **Default:** Training uses CPU. With small models and single-agent, this is often fine.
- **When CUDA helps:** (1) **Larger networks** (e.g. 512→1024 hidden, more layers) — more matrix ops. (2) **Longer/bigger rollouts** — PPO batches 2048 steps; larger batches benefit from GPU. (3) **Multiple agents** — if you later batch forward passes for N agents, GPU parallelizes. (4) **Bigger maps** — same obs size here, but if you add more features or train many envs in parallel, GPU helps.
- **When it doesn’t:** The **environment step** (RWARE) runs on CPU. So each transition is CPU-bound; GPU only speeds up the neural network forward/backward. For 1 agent and small nets, speedup may be modest (e.g. 1.2–2×). For larger models and batched multi-agent, expect larger gains.
- **Recommendation:** Use `--device auto` so GPU is used when present. For scaling to smarter models and multiple bots, CUDA will matter more for training time.

## Why does PPO performance vary between runs?
- Each run **starts from random weights** and uses a **stochastic policy**, so results naturally vary (e.g. 75% vs 87% success with the same setup).
- **`outputs/ppo_metrics.json` is overwritten** every time you train PPO, so `evaluate.py` always shows the **last** run, not a comparison across runs.
- Use **`--seed 42`** (or any int) to get **reproducible** runs: same config + same seed ⇒ same training and metrics.

## Warehouse map (bigger scale)
- **Presets:** `--env default | 5x5 | 7x7 | 10x10`
  - `default`: small (3×3 shelves), 200 steps
  - `5x5`: 5×5 shelves, 200 steps
  - `7x7`: 7×7 shelves, 400 steps
  - `10x10`: 10×10 shelves, 600 steps
- **Override map:** `--shelf-columns N --shelf-rows N --max-steps M`
- **Examples:**
  - `python train.py --algo ppo --env 5x5 --episodes 1000`
  - `python train.py --algo ppo --env 7x7 --episodes 2000`
  - `python train.py --algo ppo --env 10x10 --episodes 3000`
  - `python train.py --algo ppo --shelf-columns 8 --shelf-rows 8 --max-steps 500`

## Map parameters (configs/config.py)
| Parameter        | Meaning              | Example |
|-----------------|----------------------|---------|
| shelf_columns   | Number of shelf columns (width) | 3, 5, 7, 10 |
| shelf_rows      | Number of shelf rows (height)   | 3, 5, 7, 10 |
| column_height   | Shelves per column              | 1 |
| max_steps       | Max steps per episode           | 200–600 |
| max_deliveries  | Deliveries per episode          | 1 |
| n_agents        | Number of agents                | 1 |

## Evaluation and plots (proposal metrics)
- **Table 1 & metrics:** `python scripts/evaluate.py` (reads `outputs/*_metrics.json`). Optionally `--out-table results/table1.csv`.
- **Learning curves:** `python scripts/plot_learning_curves.py` (saves reward, success rate, battery death figures to `outputs/`).

## Config
- `configs/config.py`: env presets (`ENV_PRESETS`), algorithm configs, `ENV_CONFIG_5x5`, `ENV_CONFIG_7x7`, `ENV_CONFIG_10x10`.
