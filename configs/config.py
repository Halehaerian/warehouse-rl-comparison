ENV_CONFIG = {
    "max_steps": 500,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 4,
    "n_agents": 1,
}

ENV_CONFIG_SMALL = {
    "max_steps": 500,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 4,
    "n_agents": 1,
}

ENV_CONFIG_10x10 = {
    "max_steps": 500,
    "max_deliveries": 5,
    "shelf_columns": 3,
    "column_height": 1,
    "shelf_rows": 4,
    "n_agents": 1,
}

ENV_CONFIG_16x16 = {
    "max_steps": 800,
    "max_deliveries": 5,
    "shelf_columns": 5,
    "column_height": 1,
    "shelf_rows": 7,
    "n_agents": 1,
}

ENV_CONFIG_22x22 = {
    "max_steps": 1200,
    "max_deliveries": 5,
    "shelf_columns": 7,
    "column_height": 1,
    "shelf_rows": 10,
    "n_agents": 1,
}

ENV_PRESETS = {
    "default": ENV_CONFIG,
    "small": ENV_CONFIG_SMALL,
    "10x10": ENV_CONFIG_10x10,
    "16x16": ENV_CONFIG_16x16,
    "22x22": ENV_CONFIG_22x22,
}

BATTERY_CONFIG = {
    "max_battery": 100.0,
    "battery_drain": 1.0,
    "charge_rate": 25.0,
    "battery_threshold": 30.0,
    "battery_resume": 80.0,
    "charger_location": (0, 0),
}

DDQN_CONFIG = {
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
    "double": False,
    "warmup": 500,
}

PPO_CONFIG = {
    "lr": 3e-4,
    "lr_min": 1e-5,
    "lr_decay": 0.99998,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_clip": 0,
    "ppo_epochs": 6,         
    "batch_size": 128,
    "rollout_len": 2048,      
    "hidden_size": 256,
    "n_layers": 2,
    "use_layer_norm": True,
    "vf_coef": 0.5,
    "ent_coef": 0.05,        
    "ent_coef_min": 0.005,    
    "ent_coef_decay": 0.9995, 
    "reward_scale": 0.1,      
    "max_grad_norm": 0.5,
}

SAC_CONFIG = {
    "lr": 3e-4,
    "hidden_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "memory_size": 100000,
    "warmup": 500,
}

TRAINING_CONFIG = {
    "episodes": 10000,
    "eval_freq": 200,
    "save_freq": 1000,
}

ALGO_CONFIGS = {
    "ddqn": DDQN_CONFIG,
    "dqn": DQN_CONFIG,
    "ppo": PPO_CONFIG,
    "sac": SAC_CONFIG,
    "sac_original": SAC_CONFIG,
}
