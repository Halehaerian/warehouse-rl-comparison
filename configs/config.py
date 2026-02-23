"""
Configuration for warehouse RL comparison.

All algorithm-specific hyperparameters live here.
Battery config is kept separate for future expansion.
"""

# --- Environment ---
ENV_CONFIG = {
    "max_steps": 200,
    "max_deliveries": 1,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 1,
    "n_agents": 1,
}

# Proposal: start with small grid (5×5) then scale (see Section 4).
ENV_CONFIG_5x5 = {
    "max_steps": 200,
    "max_deliveries": 1,
    "shelf_columns": 5,
    "column_height": 1,
    "shelf_rows": 5,
    "n_agents": 1,
}

# Larger warehouses (more shelf columns/rows = bigger map).
ENV_CONFIG_7x7 = {
    "max_steps": 400,
    "max_deliveries": 1,
    "shelf_columns": 7,
    "column_height": 1,
    "shelf_rows": 7,
    "n_agents": 1,
}

ENV_CONFIG_10x10 = {
    "max_steps": 600,
    "max_deliveries": 1,
    "shelf_columns": 10,
    "column_height": 1,
    "shelf_rows": 10,
    "n_agents": 1,
}

# Named env presets (used by train.py --env)
ENV_PRESETS = {
    "default": ENV_CONFIG,
    "5x5": ENV_CONFIG_5x5,
    "7x7": ENV_CONFIG_7x7,
    "10x10": ENV_CONFIG_10x10,
}

# --- Battery ---
# max_battery=100 + drain=0.3 → dies at step 333 without charging
# 1 efficient delivery (~40 steps) drains 12 → battery=88
# Agent has time to learn before battery becomes critical
BATTERY_CONFIG = {
    "max_battery": 100.0,
    "battery_drain": 0.3,
    "charge_rate": 25.0,
    "battery_threshold": 25.0,
    "charger_location": (0, 0),
}

# --- DQN ---
DQN_CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.999,      # reach ~0.05 by ep 3000
    "batch_size": 128,
    "memory_size": 100000,
    "hidden_size": 256,
    "target_update_freq": 100,
    "warmup": 500,
}

# --- PPO (tuned for warehouse: larger net, more exploration, LR decay) ---
PPO_CONFIG = {
    "lr": 3e-4,
    "lr_min": 1e-4,
    "lr_decay": 0.9998,
    "gamma": 0.99,
    "gae_lambda": 0.98,
    "clip_eps": 0.2,
    "value_clip": 0.2,
    "ppo_epochs": 10,
    "batch_size": 128,
    "rollout_len": 2048,
    "hidden_size": 512,
    "n_layers": 3,
    "use_layer_norm": True,
    "vf_coef": 0.25,
    "ent_coef": 0.025,
    "reward_scale": 0.01,
    "max_grad_norm": 0.5,
}

# --- SAC ---
SAC_CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "memory_size": 50000,
    "hidden_size": 128,
    "warmup": 256,
}

# --- Training ---
TRAINING_CONFIG = {
    "episodes": 15000,
    "eval_freq": 200,
    "save_freq": 1000,
}

# Helper to get algorithm config by name
ALGO_CONFIGS = {
    "dqn": DQN_CONFIG,
    "ppo": PPO_CONFIG,
    "sac": SAC_CONFIG,
}
