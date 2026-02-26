"""
Train RL agents on the warehouse environment.

Usage:
    python train.py --algo ppo
    python train.py --algo ppo --env 5x5 --episodes 1000
    python train.py --algo ppo --env 7x7 --episodes 2000
    python train.py --algo ppo --env 10x10 --episodes 3000
    python train.py --algo ppo --shelf-columns 8 --shelf-rows 8 --max-steps 500
"""

import argparse
import os
import copy
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"

import torch

from configs.config import ENV_PRESETS, BATTERY_CONFIG, TRAINING_CONFIG, ALGO_CONFIGS
from training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train warehouse RL agents")
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "ppo", "sac", "all"],
                        help="Algorithm to train (default: dqn)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--env", type=str, default="default",
                        choices=list(ENV_PRESETS.keys()),
                        help="Warehouse map preset: default, 5x5, 7x7, 10x10")
    parser.add_argument("--shelf-columns", type=int, default=None,
                        help="Override shelf columns (map width)")
    parser.add_argument("--shelf-rows", type=int, default=None,
                        help="Override shelf rows (map height)")
    parser.add_argument("--column-height", type=int, default=None,
                        help="Override shelf column height")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps per episode")
    parser.add_argument("--n-agents", type=int, default=None,
                        help="Override number of agents")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (same seed => same run)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for robust reporting, e.g. --seeds 42 123 456")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device: auto (CUDA if available), cuda, or cpu")
    args = parser.parse_args()

    if args.seeds is not None and args.seed is not None:
        raise ValueError("Use either --seed or --seeds, not both.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env_config = copy.deepcopy(ENV_PRESETS[args.env])
    if args.shelf_columns is not None:
        env_config["shelf_columns"] = args.shelf_columns
    if args.shelf_rows is not None:
        env_config["shelf_rows"] = args.shelf_rows
    if args.column_height is not None:
        env_config["column_height"] = args.column_height
    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps
    if args.n_agents is not None:
        env_config["n_agents"] = args.n_agents

    training_config = TRAINING_CONFIG.copy()
    if args.episodes:
        training_config["episodes"] = args.episodes

    algos = ["dqn", "ppo", "sac"] if args.algo == "all" else [args.algo]

    seed_list = args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else [None])

    for algo in algos:
        config = ALGO_CONFIGS[algo]
        for seed in seed_list:
            if len(seed_list) > 1:
                print(f"\n>>> Run {seed_list.index(seed) + 1}/{len(seed_list)} (seed={seed})")
            train(algo, env_config, BATTERY_CONFIG, config, training_config, seed=seed, device=device)


if __name__ == "__main__":
    main()
