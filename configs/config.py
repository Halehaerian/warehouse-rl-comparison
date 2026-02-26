"""
Configuration for warehouse RL comparison.

All algorithm-specific hyperparameters live here.
Battery config is kept separate for future expansion.
"""

# --- Environment ---
# 5 deliveries, optimal ~77 steps. max_steps=300 gives margin for charging.
ENV_CONFIG = {
    "max_steps": 300,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 1,
    "n_agents": 1,
}

# Proposal: start with small grid (5x5) then scale (see Section 4).
ENV_CONFIG_5x5 = {
    "max_steps": 300,
    "max_deliveries": 5,
    "shelf_columns": 5,
    "column_height": 1,
    "shelf_rows": 5,
    "n_agents": 1,
}

# --- Battery ---
# drain=2.5 -> dies at step 40 without charging
# Threshold=50 triggers CHARGING at step 20 -- EVERY episode must charge
# Even a fast 5-delivery run (~22 steps/delivery) must detour to charger
# charge_rate=25 per step -> recharge 50->85 in 2 steps at charger
# resume=85 gives ~14 more steps before next charge trigger
# => Charging is a first-class quest waypoint, not an optional detour
# Charger at (0,0)
BATTERY_CONFIG = {
    "max_battery": 100.0,
    "battery_drain": 2.5,
    "charge_rate": 25.0,
    "battery_threshold": 50.0,
    "battery_resume": 85.0,
    "charger_location": (0, 0),
}

# --- DQN ---
DQN_CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,          # higher floor for more exploration (helps discover charging)
    "epsilon_decay": 0.9997,      # slower decay: reaches ~0.05 around ep 11000
    "batch_size": 128,
    "memory_size": 200000,         # larger buffer for 5-delivery episodes (longer episodes)
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
