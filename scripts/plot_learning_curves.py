"""
Plot learning curves from training metrics: reward, success rate, battery death rate.

Reads outputs/*_metrics.json and saves figures to outputs/ or a given directory.

Usage:
    python scripts/plot_learning_curves.py
    python scripts/plot_learning_curves.py --outputs outputs --savedir outputs/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def rolling_mean(x: list, window: int) -> np.ndarray:
    """Rolling mean; first (window-1) values are nan or partial mean."""
    if window <= 0 or len(x) == 0:
        return np.array(x, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.mean(x[start : i + 1])
    return out


def plot_curves(metrics_dir: Path, savedir: Path, window: int = 200):
    metrics_dir = Path(metrics_dir)
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    files = sorted(metrics_dir.glob("*_metrics.json"))
    if not files:
        print(f"No *_metrics.json in {metrics_dir}")
        return

    algo_styles = {"dqn": "C0", "ppo": "C1", "sac": "C2"}
    algo_names = {"dqn": "DQN", "ppo": "PPO", "sac": "SAC"}

    # --- Reward ---
    plt.figure(figsize=(8, 5))
    for path in files:
        episodes = load_metrics(path)
        if not episodes:
            continue
        algo = path.stem.replace("_metrics", "")
        ep_num = [e["episode"] for e in episodes]
        rewards = [e.get("reward", 0) for e in episodes]
        smooth = rolling_mean(rewards, window)
        color = algo_styles.get(algo, None)
        plt.plot(ep_num, smooth, label=algo_names.get(algo, algo.upper()), color=color, lw=2)
        plt.plot(ep_num, rewards, alpha=0.15, color=color)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning curve (reward)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savedir / "learning_curve_reward.png", dpi=150)
    plt.close()
    print(f"Saved {savedir / 'learning_curve_reward.png'}")

    # --- Success rate (rolling) ---
    plt.figure(figsize=(8, 5))
    for path in files:
        episodes = load_metrics(path)
        if not episodes:
            continue
        algo = path.stem.replace("_metrics", "")
        ep_num = [e["episode"] for e in episodes]
        success = [1.0 if e.get("mission_complete", False) else 0.0 for e in episodes]
        smooth = rolling_mean(success, window) * 100
        color = algo_styles.get(algo, None)
        plt.plot(ep_num, smooth, label=algo_names.get(algo, algo.upper()), color=color, lw=2)
    plt.xlabel("Episode")
    plt.ylabel("Success rate (%)")
    plt.title("Success rate (rolling)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=80, color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(savedir / "learning_curve_success_rate.png", dpi=150)
    plt.close()
    print(f"Saved {savedir / 'learning_curve_success_rate.png'}")

    # --- Battery death rate (rolling) ---
    plt.figure(figsize=(8, 5))
    for path in files:
        episodes = load_metrics(path)
        if not episodes:
            continue
        algo = path.stem.replace("_metrics", "")
        ep_num = [e["episode"] for e in episodes]
        battery_dead = [1.0 if e.get("battery_dead", False) else 0.0 for e in episodes]
        smooth = rolling_mean(battery_dead, window) * 100
        color = algo_styles.get(algo, None)
        plt.plot(ep_num, smooth, label=algo_names.get(algo, algo.upper()), color=color, lw=2)
    plt.xlabel("Episode")
    plt.ylabel("Battery death rate (%)")
    plt.title("Battery death rate (rolling)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savedir / "learning_curve_battery_death.png", dpi=150)
    plt.close()
    print(f"Saved {savedir / 'learning_curve_battery_death.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from metrics JSON")
    parser.add_argument("--outputs", type=str, default="outputs",
                        help="Directory with *_metrics.json")
    parser.add_argument("--savedir", type=str, default="outputs",
                        help="Directory to save figures")
    parser.add_argument("--window", type=int, default=200,
                        help="Rolling window for smoothing")
    args = parser.parse_args()
    plot_curves(Path(args.outputs), Path(args.savedir), window=args.window)


if __name__ == "__main__":
    main()
