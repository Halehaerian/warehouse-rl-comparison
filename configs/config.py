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
# Gentler drain so the agent can complete deliveries and learn to charge (post-merge fix).
# drain=1.0 -> 100 steps to empty; CHARGING at step 50 (threshold 50) gives time to reach (0,0).
# For "hard" battery ablation use battery_drain=2.5 (dies at step 40 without charging).
# charge_rate=25 per step -> recharge 50->85 in ~2 steps at charger; resume=85.
# Charger at (0,0).
BATTERY_CONFIG = {
    "max_battery": 100.0,
    "battery_drain": 1.0,
    "charge_rate": 25.0,
    "battery_threshold": 50.0,
    "battery_resume": 85.0,
    "charger_location": (0, 0),
}

# --- DQN ---
# epsilon_decay 0.9985 -> ~0.22 at 1000 ep, ~0.05 by ~2000 ep (so DQN exploits in short runs).
# reward_scale matches PPO for stable Q-learning with large env rewards.
DQN_CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9985,
    "batch_size": 128,
    "memory_size": 200000,
    "hidden_size": 256,
    "target_update_freq": 100,
    "warmup": 500,
    "reward_scale": 0.01,
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
