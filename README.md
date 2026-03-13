## Warehouse RL Comparison

### Environment Setup
1. **Python 3.10 recommended**
2. Create and activate a virtual environment:
  - Windows: `python -m venv venv && venv\Scripts\activate`
  - Mac/Linux: `python -m venv venv && source venv/bin/activate`
3. Install dependencies:
  - `pip install -r requirements.txt`
  - If `rware` fails: `pip install git+https://github.com/semitable/robotic-warehouse.git`
4. Install PyTorch (choose CUDA or CPU):
  - See [PyTorch site](https://pytorch.org/get-started/locally/) for your CUDA version.
  - Example: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (CUDA 12.1)
  - For CPU only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Training
- Run training for an algorithm:
  - `python train.py --algo ppo`
  - `python train.py --algo dqn`
  - `python train.py --algo sac`
  - `python train.py --algo ddqn`
- Set episodes: `--episodes N` (default: 1000)
- Set environment: `--env default | 5x5 | 7x7 | 10x10`
- Use GPU: `--device cuda` or auto-detect: `--device auto`
- Set random seed: `--seed 42` (for reproducibility)

### Map Presets
- `--env default`: 3x3 shelves, 200 steps
- `--env 5x5`: 5x5 shelves, 200 steps
- `--env 7x7`: 7x7 shelves, 400 steps
- `--env 10x10`: 10x10 shelves, 600 steps
- Custom: `--shelf-columns N --shelf-rows N --max-steps M`

### Evaluation & Plots
- Evaluate metrics: `python scripts/evaluate.py` (reads `outputs/*_metrics.json`)
- Plot learning curves: `python scripts/plot_learning_curves.py` (outputs to `outputs/`)

### Model Files
- Trained models are saved in `models/` and `models/Archive/`
- Metrics are saved in `outputs/`

### Configurations
- Edit environment and algorithm configs in `configs/config.py`

---
For any issues, check your Python version, dependencies, and CUDA setup. All scripts are run from the project root.
