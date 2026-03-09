"""
Configuration for warehouse RL comparison.

All algorithm-specific hyperparameters live here.
Battery config is kept separate for future expansion.

Square grid sizes available in RWARE:
  - 10x10: shelf_columns=3, column_height=1, shelf_rows=4 => 22 shelves
  - 16x16: shelf_columns=5, column_height=1, shelf_rows=7 => 68 shelves
  - 22x22: shelf_columns=7, column_height=1, shelf_rows=10 => 138 shelves
"""

# --- Environment ---

# Default: 10x10 square grid, 22 shelves (good for training & demo)
ENV_CONFIG = {
    "max_steps": 500,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 4,
    "n_agents": 1,
}

# Small: same as default (10x10 is the smallest square RWARE supports)
ENV_CONFIG_SMALL = {
    "max_steps": 500,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 4,
    "n_agents": 1,
}

# 10x10 square grid, 22 shelves
ENV_CONFIG_10x10 = {
    "max_steps": 500,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 4,
    "n_agents": 1,
}

# 16x16 square grid, 68 shelves (larger, needs more training)
ENV_CONFIG_16x16 = {
    "max_steps": 800,
    "max_deliveries": 5,
    "shelf_columns": 5,
    "column_height": 1,
    "shelf_rows": 7,
    "n_agents": 1,
}

# 22x22 square grid, 138 shelves (hardest)
ENV_CONFIG_22x22 = {
    "max_steps": 1200,
    "max_deliveries": 5,
    "shelf_columns": 7,
    "column_height": 1,
    "shelf_rows": 10,
    "n_agents": 1,
}

# Named env presets (used by train.py --env)
ENV_PRESETS = {
    "default": ENV_CONFIG,
    "small": ENV_CONFIG_SMALL,
    "10x10": ENV_CONFIG_10x10,
    "16x16": ENV_CONFIG_16x16,
    "22x22": ENV_CONFIG_22x22,
}

# --- Battery ---
# 10x10 grid: max manhattan distance ~18
# drain=1.0 -> dies at step 100 without charging
# threshold=30: agent seeks charger at step ~70
# charge_rate=25 per step -> fast recharge (3 steps from 5 to 80)
# resume=80 gives ~50 more steps before next charge trigger
# ~2 charge cycles per 5-delivery episode
# Charger at (0,0) top-left corner
BATTERY_CONFIG = {
    "max_battery": 100.0,
    "battery_drain": 1.0,
    "charge_rate": 25.0,
    "battery_threshold": 30.0,
    "battery_resume": 80.0,
    "charger_location": (0, 0),
}

# --- DQN (Double DQN with tuned exploration) ---
DQN_CONFIG = {
    "lr": 5e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9990,      
    "batch_size": 128,
    "memory_size": 100000,
    "hidden_size": 256,
    "tau": 0.005,               
    "double": True,             
    "warmup": 500,
}

# --- Vanilla DQN (standard DQN, no Double DQN) ---
VANILLA_DQN_CONFIG = {
    "lr": 5e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9990,      
    "batch_size": 128,
    "memory_size": 100000,
    "hidden_size": 256,
    "tau": 0.005,
    "double": False,            # Vanilla DQN
    "warmup": 500,
}

# --- PPO (tuned for 10x10 warehouse) ---
PPO_CONFIG = {
    "lr": 3e-4,
    "lr_min": 1e-5,
    "lr_decay": 0.9999,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_clip": 0,          # disabled: was causing critic lag in high-variance reward env
    "ppo_epochs": 8,          # reduced from 10: larger rollouts need fewer epochs to avoid overfitting
    "batch_size": 128,
    "rollout_len": 1024,      # increased from 512: ~2 episodes per update, lower gradient variance
    "hidden_size": 256,
    "n_layers": 2,
    "use_layer_norm": True,
    "vf_coef": 0.5,
    "ent_coef": 0.01,         # constant entropy bonus (standard PPO)
    "reward_scale": 0.02,     # raised from 0.01 -- stronger delivery signal
    "max_grad_norm": 0.5,
}

# --- SAC (discrete, auto-entropy tuning) ---
SAC_CONFIG = {
    "lr": 3e-4,
    "alpha": 3e-4,
    "layer1_size": 256,
    "layer2_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "memory_size": 1000000,
}

# --- Training ---
TRAINING_CONFIG = {
    "episodes": 10000,
    "eval_freq": 200,
    "save_freq": 1000,
}

# Helper to get algorithm config by name
ALGO_CONFIGS = {
    "dqn": DQN_CONFIG,
    "vanilla_dqn": VANILLA_DQN_CONFIG,
    "ppo": PPO_CONFIG,
    "sac": SAC_CONFIG,
}
