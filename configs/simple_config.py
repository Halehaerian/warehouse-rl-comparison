"""
MINIMAL Simple Configuration for FASTEST Learning
Stripped-down for direct learning: go to pickup, grab, go to destination
"""

# Environment Settings (ABSOLUTE MINIMUM)
ENV_CONFIG = {
    'env_id': None,  # Use direct Warehouse creation
    'max_steps': 100,  # Need enough for turns (RWARE uses FORWARD/LEFT/RIGHT, not grid moves)
    'max_deliveries': 1,
}

# Battery Settings (SIMPLE)
BATTERY_CONFIG = {
    'max_battery': 100.0,
    'battery_drain': 0.1,            # Very low drain - focus on navigation
    'charge_rate': 50.0,             # Fast charging if needed
    'battery_threshold': 5.0,        # Rarely triggers - focus on task
    'charger_location': (0, 0),
}

# DQN Hyperparameters (SIMPLIFIED for SPEED)
DQN_CONFIG = {
    'lr': 1e-3,                      # HIGHER LR = faster learning
    'gamma': 0.99,                   # High gamma - value future delivery reward
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,             # VERY LOW - rely on learned policy
    'epsilon_decay': 0.9995,          # Slower decay = longer exploration over 10k episodes
    'batch_size': 64,                # Reasonable batch for stability
    'memory_size': 50000,            # Larger buffer = more diverse experience
    'hidden_size': 128,               # Bigger network for more obs features
    'train_freq': 1,                 # Train EVERY step (aggressive)
    'warmup_steps': 100,             # MINIMAL warmup
}

# Training Settings
TRAINING_CONFIG = {
    'episodes': 10000,  # Need more episodes to learn full pickup→deliver cycle
    'eval_freq': 200,
    'save_freq': 500,
}
