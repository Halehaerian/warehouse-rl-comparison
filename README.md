# Warehouse RL with Battery Management

Train a DQN agent to complete deliveries in the RWARE (Robotic Warehouse) environment while managing battery levels.

## Features

- **Battery Management**: Agent must monitor battery and visit charger when low
- **Mission Flow**: PICKUP shelf → DELIVER to goal → repeat
- **Visual Feedback**: Graphical visualization with battery bar and charger station
- **Prioritized Experience Replay**: Efficient learning from important experiences

## Project Structure

```
warehouse-rl-comparison/
├── train.py              # Main training script
├── visualize.py          # Graphical visualization
├── envs/
│   ├── battery_wrapper.py   # Battery wrapper for RWARE
│   └── __init__.py
├── agents/
│   ├── dqn_agent.py         # DQN with PER
│   └── __init__.py
├── utils/
│   ├── sum_tree.py          # SumTree for PER
│   └── __init__.py
├── models/                   # Saved model weights
└── outputs/                  # Training metrics
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch numpy gymnasium rware pyglet
```

## Usage

### Training

```bash
# Train with default settings
python train.py

# Train with custom settings
python train.py --episodes 5000 --max_steps 500 --lr 0.001

# Full options
python train.py \
    --env rware-tiny-1ag-v2 \
    --episodes 3000 \
    --max_steps 500 \
    --max_deliveries 1 \
    --battery_drain 0.05 \
    --charge_rate 50.0 \
    --lr 0.001 \
    --gamma 0.99 \
    --epsilon_decay 0.997
```

### Visualization

```bash
# Visualize with latest model
python visualize.py

# Visualize specific model
python visualize.py --model models/battery_dqn_best.pt

# Adjust speed and episodes
python visualize.py --episodes 5 --delay 0.05 --epsilon 0.0
```

## Environment Details

### RWARE (Robotic Warehouse)
- Grid-based warehouse environment
- Agent actions: FORWARD, LEFT, RIGHT, TOGGLE (pickup/drop), NOOP
- Goal: Pick up requested shelves and deliver to goal zones

### Battery Wrapper
Extends RWARE with:
- **Battery drain**: Loses charge each step
- **Charger station**: Yellow tile where agent recharges
- **Mission state machine**: Prevents exploit behaviors
- **Reward shaping**: Encourages efficient delivery and smart charging

### Observation Space
Extended with 8 additional dimensions:
- Agent position (x, y) normalized
- Target position (x, y) - dynamic based on mission state
- Charger position (x, y)
- Holding status (0 or 1)
- Battery level (0-100)

### Reward Function
```
+100  Delivery (drop at goal)
+10   Pickup (grab requested shelf)
+50   Mission complete bonus
+5    Smart charging (when battery low)
+2    Moving closer to target
-0.5  Step penalty
-2    Wasteful charging
-5    Dropped without delivering
-50   Battery death
```

## Training Tips

1. **Start simple**: Use 1 delivery target first
2. **Slow drain**: Start with low battery_drain (0.05)
3. **More exploration**: Keep epsilon_min around 0.15
4. **Patience**: RWARE is sparse-reward, may need 2000+ episodes

## Visualization Legend

- 🟡 **Yellow square** = Charger station
- 🔴 **Red circle** = Agent carrying shelf
- 🟠 **Orange circle** = Agent empty
- 🌊 **Teal** = Requested shelf
- 🟦 **Dark blue** = Regular shelf
- ⬛ **Dark gray** = Goal zone
- **Battery bar** = Green (>50%), Yellow (20-50%), Red (<20%)

## License

MIT
