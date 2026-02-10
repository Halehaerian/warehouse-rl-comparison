"""
Train DQN Agent on RWARE - SIMPLIFIED
Simple config, simple agent, fast learning.

Usage:
    python train.py --simple
    python train.py --simple --episodes 1000
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on RWARE")
    
    parser.add_argument("--simple", action="store_true", default=True,
                        help="Use simple config (default: True)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Override episodes")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Always use simple config
    from configs.simple_config import ENV_CONFIG, BATTERY_CONFIG, DQN_CONFIG, TRAINING_CONFIG
    from training.train_dqn_simple import train_dqn_simple
    
    print(f"\n{'='*60}")
    print(f"SIMPLE DQN TRAINING - FAST LEARNING")
    print(f"{'='*60}\n")
    
    # Override episodes if provided
    if args.episodes != 500:
        TRAINING_CONFIG = dict(TRAINING_CONFIG)
        TRAINING_CONFIG['episodes'] = args.episodes
    
    train_dqn_simple(ENV_CONFIG, BATTERY_CONFIG, DQN_CONFIG, TRAINING_CONFIG)


if __name__ == "__main__":
    main()
