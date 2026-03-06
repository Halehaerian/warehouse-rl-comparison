"""
Configuration for warehouse RL comparison.

All algorithm-specific hyperparameters live here.
Battery config is kept separate for future expansion.
"""

# --- Environment ---
# 7x7 shelf grid. 5 deliveries, max_steps=800 gives margin for navigation + charging.
ENV_CONFIG = {
    "max_steps": 800,
    "max_deliveries": 5,
    "shelf_columns": 7,
    "column_height": 1,
    "shelf_rows": 7,
    "n_agents": 1,
}

# Small env kept for reference / quick tests
ENV_CONFIG_SMALL = {
    "max_steps": 300,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 1,
    "n_agents": 1,
}

# Proposal: start with small grid (5x5) then scale (see Section 4).
ENV_CONFIG_5x5 = {
    "max_steps": 500,
    "max_deliveries": 5,
    "shelf_columns": 5,
    "column_height": 1,
    "shelf_rows": 5,
    "n_agents": 1,
}

# Larger warehouses (more shelf columns/rows = bigger map).
ENV_CONFIG_7x7 = {
    "max_steps": 800,
    "max_deliveries": 5,
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
    "small": ENV_CONFIG_SMALL,
    "5x5": ENV_CONFIG_5x5,
    "7x7": ENV_CONFIG_7x7,
    "10x10": ENV_CONFIG_10x10,
}

# --- Battery ---
# drain=2.5 -> dies at step 40 without charging
# Threshold=50 means battery hits 50 at step 20; with (battery < threshold) CHARGING starts on step 21
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
    "lr": 5e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9994,      # reaches 0.05 at ep ~5000; 15k exploitation episodes remain
    "batch_size": 256,
    "memory_size": 300000,
    "hidden_size": 512,
    "target_update_freq": 200,
    "warmup": 1000,
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
    "rollout_len": 4096,          # bigger rollouts for longer 7x7 episodes
    "hidden_size": 512,
    "n_layers": 3,
    "use_layer_norm": True,
    "vf_coef": 0.25,
    "ent_coef": 0.03,             # more entropy for larger exploration space
    "reward_scale": 0.01,
    "max_grad_norm": 0.5,
}

# --- SAC ---
SAC_CONFIG = {
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "memory_size": 200000,        # 4x increase — critical for SAC to learn on longer episodes
    "hidden_size": 256,           # larger network for bigger obs space
    "warmup": 1000,               # more warmup for bigger env
}

# --- Training ---
TRAINING_CONFIG = {
    "episodes": 20000,
    "eval_freq": 200,
    "save_freq": 1000,
}

# Helper to get algorithm config by name
ALGO_CONFIGS = {
    "dqn": DQN_CONFIG,
    "ppo": PPO_CONFIG,
    "sac": SAC_CONFIG,
}
