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

# --- PPO ---
PPO_CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "ppo_epochs": 4,
    "batch_size": 64,
    "rollout_len": 128,
    "hidden_size": 128,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
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
