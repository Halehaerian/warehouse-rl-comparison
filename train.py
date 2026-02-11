"""
Train RL agents on the warehouse environment.

Usage:
    python train.py --algo dqn
    python train.py --algo ppo
    python train.py --algo sac
    python train.py --algo all
    python train.py --algo dqn --episodes 5000
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

from configs.config import ENV_CONFIG, BATTERY_CONFIG, TRAINING_CONFIG, ALGO_CONFIGS
from training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train warehouse RL agents")
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "ppo", "sac", "all"],
                        help="Algorithm to train (default: dqn)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    args = parser.parse_args()

    training_config = TRAINING_CONFIG.copy()
    if args.episodes:
        training_config["episodes"] = args.episodes

    algos = ["dqn", "ppo", "sac"] if args.algo == "all" else [args.algo]

    for algo in algos:
        config = ALGO_CONFIGS[algo]
        train(algo, ENV_CONFIG, BATTERY_CONFIG, config, training_config)


if __name__ == "__main__":
    main()
