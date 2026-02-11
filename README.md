# Warehouse RL Comparison

Compare DQN, PPO, and SAC on the RWARE (Robotic Warehouse) environment with battery management.

## Features

- **Three Algorithms**: DQN, PPO, SAC with a shared `BaseAgent` interface
- **Battery Management**: Optional battery drain/recharge (kept modular for future use)
- **Mission Flow**: SEEK shelf → PICKUP → DELIVER to goal → DROP
- **Metrics**: Per-episode logging saved to JSON
- **Visualization**: Pyglet-based renderer with battery bar, charger, and real-time stats

## Project Structure

```
warehouse-rl-comparison/
├── train.py                  # CLI entry point
├── visualize.py              # Graphical visualization
├── configs/
│   └── config.py             # All hyperparameters (env, battery, DQN, PPO, SAC)
├── agents/
│   ├── base.py               # Abstract BaseAgent
│   ├── dqn.py                # DQN (replay buffer + target network)
│   ├── ppo.py                # PPO (actor-critic + GAE)
│   └── sac.py                # SAC-Discrete (twin critics + entropy tuning)
├── envs/
│   └── warehouse.py          # RWARE wrapper with reward shaping + observations
├── training/
│   └── trainer.py            # Unified training loop for all algorithms
├── utils/
│   └── metrics.py            # MetricsCollector (JSON export)
├── models/                   # Saved model weights (.pt)
└── outputs/                  # Training metrics (.json)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install torch numpy gymnasium rware pyglet
```

## Usage

### Training

```bash
python train.py --algo dqn               # Train DQN (default)
python train.py --algo ppo               # Train PPO
python train.py --algo sac               # Train SAC
python train.py --algo all               # Train all three
python train.py --algo dqn --episodes 5000  # Override episode count
```

### Visualization

```bash
python visualize.py --algo dqn            # Visualize DQN agent
python visualize.py --algo ppo            # Visualize PPO agent
python visualize.py --algo sac --episodes 5 --delay 0.05
python visualize.py --model models/dqn_best.pt  # Specific model file
```

## Environment

### RWARE
- Grid-based warehouse with direction-based movement
- Actions: NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD
- Pick up requested shelves and deliver to goal zones

### Extended Observations (12 extra dims)
| Dims | Description |
|------|-------------|
| 2 | Agent position (normalized) |
| 2 | Target position (shelf or goal, dynamic) |
| 2 | Direction to target (dx, dy) |
| 4 | Facing direction one-hot (UP/DOWN/LEFT/RIGHT) |
| 1 | Carrying flag |
| 1 | Battery level |

### Reward Shaping
| Reward | Event |
|--------|-------|
| +500 * eff | Delivery (efficiency-scaled) |
| +300 * eff | Mission complete bonus |
| +150 | Pickup at correct shelf |
| +20 | Arrival at target location |
| +15 | Toggle at correct position |
| +8 / -5 | Distance shaping (closer / farther) |
| -1 growing | Step penalty (increases over time) |
| -10 | Hesitation (at target, not toggling) |
| -50 | Wrong drop |
| -100 | Battery death |

## Algorithms

| | DQN | PPO | SAC |
|-|-----|-----|-----|
| Type | Value-based | Policy gradient | Max-entropy |
| Buffer | Replay buffer | Rollout buffer | Replay buffer |
| Exploration | ε-greedy | Stochastic policy | Entropy bonus |
| Networks | Q-network + target | Actor-Critic (shared) | Actor + Twin Critics |

## License

MIT
