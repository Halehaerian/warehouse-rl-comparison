"""Metrics collection and persistence."""

import json
import time
from pathlib import Path
from collections import defaultdict


class MetricsCollector:
    """Collects per-episode metrics and saves to JSON."""

    def __init__(self):
        self.episodes = []
        self.start_time = time.time()

    def log_episode(self, episode, reward, steps, info=None, **extra):
        entry = {
            "episode": episode,
            "reward": round(reward, 2),
            "steps": steps,
            "elapsed": round(time.time() - self.start_time, 1),
        }
        if info:
            entry["deliveries"] = info.get("deliveries", 0)
            entry["mission_complete"] = info.get("mission_complete", False)
            entry["battery_dead"] = info.get("battery_dead", False)
        entry.update(extra)
        self.episodes.append(entry)

    def recent_avg(self, key="reward", n=200):
        vals = [e[key] for e in self.episodes[-n:] if key in e]
        return sum(vals) / max(len(vals), 1)

    def best(self, key="reward"):
        vals = [e[key] for e in self.episodes if key in e]
        return max(vals) if vals else float("-inf")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.episodes, f, indent=2)

    def summary(self, last_n=200):
        recent = self.episodes[-last_n:]
        if not recent:
            return {}
        rewards = [e["reward"] for e in recent]
        steps = [e["steps"] for e in recent]
        return {
            "avg_reward": round(sum(rewards) / len(rewards), 2),
            "best_reward": round(max(rewards), 2),
            "avg_steps": round(sum(steps) / len(steps), 1),
            "total_episodes": len(self.episodes),
        }
