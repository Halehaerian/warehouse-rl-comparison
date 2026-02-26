"""Compute proposal metrics from training outputs.

Metrics:
  - Success rate %          (mission_complete episodes)
  - Battery death %         (battery_dead episodes)
  - Avg deliveries / episode
  - Avg steps per delivery  (efficiency of routing)
  - Battery efficiency %    (avg battery remaining at end, higher = more efficient)
  - Episodes to 80% success
  - Table 1 summary

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --outputs outputs --window 200
    python scripts/evaluate.py --outputs outputs --out-table results/table1.csv
"""

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> list:
    """Load episode list from a metrics JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def success_rate(episodes: list) -> float:
    """Fraction of episodes where mission_complete is True."""
    if not episodes:
        return 0.0
    n = sum(1 for e in episodes if e.get("mission_complete", False))
    return 100.0 * n / len(episodes)


def battery_death_rate(episodes: list) -> float:
    """Fraction of episodes where battery_dead is True."""
    if not episodes:
        return 0.0
    n = sum(1 for e in episodes if e.get("battery_dead", False))
    return 100.0 * n / len(episodes)


def episodes_to_success_threshold(episodes: list, threshold: float = 0.80,
                                  window: int = 200) -> int | None:
    """
    First episode at which the rolling success rate (over last `window` episodes)
    reaches or exceeds `threshold`. Returns None if never reached.
    """
    if not episodes or window <= 0:
        return None
    for i in range(window - 1, len(episodes)):
        window_eps = episodes[i - window + 1 : i + 1]
        successes = sum(1 for e in window_eps if e.get("mission_complete", False))
        if successes / window >= threshold:
            return episodes[i]["episode"]
    return None


def mean_reward(episodes: list) -> float:
    if not episodes:
        return 0.0
    return sum(e.get("reward", 0) for e in episodes) / len(episodes)


def mean_steps(episodes: list) -> float:
    if not episodes:
        return 0.0
    return sum(e.get("steps", 0) for e in episodes) / len(episodes)


def _algo_base_name(path: Path) -> str:
    """e.g. ppo_metrics_seed42.json -> ppo; ppo_metrics.json -> ppo."""
    stem = path.stem  # e.g. ppo_metrics_seed42 or ppo_metrics
    stem = stem.replace("_metrics", "")
    if "_seed" in stem:
        stem = stem.split("_seed")[0]
    return stem
def avg_deliveries(episodes: list) -> float:
    """Average number of deliveries per episode."""
    if not episodes:
        return 0.0
    return sum(e.get("deliveries", 0) for e in episodes) / len(episodes)


def avg_steps_per_delivery(episodes: list) -> float:
    """Average steps per completed delivery across all episodes."""
    total_steps = 0
    total_deliveries = 0
    for e in episodes:
        segs = e.get("delivery_segment_steps", [])
        if segs:
            total_steps += sum(segs)
            total_deliveries += len(segs)
    if total_deliveries == 0:
        return 0.0
    return total_steps / total_deliveries


def battery_efficiency(episodes: list) -> float:
    """Average battery remaining (%) at end of completed episodes."""
    completed = [e for e in episodes if e.get("mission_complete", False)]
    if not completed:
        return 0.0
    return sum(e.get("battery_remaining", 0) for e in completed) / len(completed)


def avg_charging_events(episodes: list) -> float:
    """Average number of charging events per episode."""
    if not episodes:
        return 0.0
    return sum(e.get("charging_events", 0) for e in episodes) / len(episodes)

def evaluate_one(path: Path, window: int = 200) -> dict:
    """Compute all metrics for one algorithm's metrics file."""
    episodes = load_metrics(path)
    base = _algo_base_name(path)
    if not episodes:
        return {"algorithm": base, "error": "No episodes", "path": str(path)}
    ep80 = episodes_to_success_threshold(episodes, threshold=0.80, window=window)
    return {
        "algorithm": base,
        "success_rate_pct": round(success_rate(episodes), 1),
        "battery_death_pct": round(battery_death_rate(episodes), 1),
        "avg_deliveries": round(avg_deliveries(episodes), 2),
        "avg_steps_per_delivery": round(avg_steps_per_delivery(episodes), 1),
        "battery_efficiency_pct": round(battery_efficiency(episodes), 1),
        "avg_charging": round(avg_charging_events(episodes), 2),
        "episodes_to_80pct": ep80 if ep80 is not None else "N/A",
        "mean_reward": round(mean_reward(episodes), 2),
        "mean_steps": round(mean_steps(episodes), 1),
        "total_episodes": len(episodes),
    }


def _mean_std(vals, fmt=".1f"):
    if not vals:
        return "—", "—"
    import statistics
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:{fmt}}", f"{s:{fmt}}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate training metrics for proposal Table 1")
    parser.add_argument("--outputs", type=str, default="outputs",
                        help="Directory containing *_metrics.json files")
    parser.add_argument("--window", type=int, default=200,
                        help="Rolling window for 'episodes to 80%% success'")
    parser.add_argument("--out-table", type=str, default=None,
                        help="Optional path to save Table 1 as CSV")
    parser.add_argument("--aggregate-seeds", action="store_true",
                        help="Group *_metrics_seed*.json by algorithm and report mean ± std (robust)")
    args = parser.parse_args()

    out_dir = Path(args.outputs)
    if not out_dir.exists():
        print(f"Outputs directory not found: {out_dir}")
        return

    metrics_files = sorted(out_dir.glob("*_metrics*.json"))
    if not metrics_files:
        print(f"No *_metrics*.json files in {out_dir}")
        return

    per_file = [evaluate_one(p, window=args.window) for p in metrics_files]

    if args.aggregate_seeds:
        # Group by algorithm base (ppo, dqn, sac)
        by_algo = defaultdict(list)
        for r in per_file:
            if "error" in r:
                continue
            by_algo[r["algorithm"]].append(r)
        results = []
        for algo in sorted(by_algo.keys()):
            runs = by_algo[algo]
            if len(runs) == 1:
                r0 = runs[0]
                results.append({
                    "algorithm": algo,
                    "success_rate_pct": f"{r0['success_rate_pct']}",
                    "battery_death_pct": f"{r0['battery_death_pct']}",
                    "mean_reward": f"{r0['mean_reward']}",
                    "episodes_to_80pct": str(r0["episodes_to_80pct"]),
                    "n_runs": 1,
                })
                continue
            sr_vals = [x["success_rate_pct"] for x in runs]
            bd_vals = [x["battery_death_pct"] for x in runs]
            rew_vals = [x["mean_reward"] for x in runs]
            e80_vals = [x["episodes_to_80pct"] for x in runs if isinstance(x["episodes_to_80pct"], int)]
            sr_m, sr_s = _mean_std(sr_vals)
            bd_m, bd_s = _mean_std(bd_vals)
            rew_m, rew_s = _mean_std(rew_vals, ".2f")
            e80_m = f"{sum(e80_vals)/len(e80_vals):.0f}" if e80_vals else "N/A"
            e80_s = f"±{((sum((x - sum(e80_vals)/len(e80_vals))**2 for x in e80_vals)/len(e80_vals))**0.5):.0f}" if len(e80_vals) > 1 else ""
            results.append({
                "algorithm": algo,
                "success_rate_pct": f"{sr_m} ± {sr_s}",
                "battery_death_pct": f"{bd_m} ± {bd_s}",
                "mean_reward": f"{rew_m} ± {rew_s}",
                "episodes_to_80pct": f"{e80_m} {e80_s}".strip() or "N/A",
                "n_runs": len(runs),
            })
        print("\n--- Robust summary (mean ± std over runs) ---")
        for r in results:
            print(f"\n{r['algorithm'].upper()} (n={r.get('n_runs', 1)} runs)")
            print(f"  Success rate:        {r['success_rate_pct']}%")
            print(f"  Battery death rate:  {r['battery_death_pct']}%")
            print(f"  Mean reward:         {r['mean_reward']}")
            print(f"  Episodes to 80%:     {r['episodes_to_80pct']}")
        print("\n" + "=" * 60)
        print("Table 1 (robust: mean ± std)")
        print("=" * 60)
        row_fmt = "{:12} {:>22} {:>22} {:>18}"
        print(row_fmt.format("Algorithm", "Success Rate", "Battery Deaths", "Episodes to 80%"))
        print("-" * 60)
        for r in results:
            sr = r["success_rate_pct"] if "%" in str(r["success_rate_pct"]) else f"{r['success_rate_pct']}%"
            bd = r["battery_death_pct"] if "%" in str(r["battery_death_pct"]) else f"{r['battery_death_pct']}%"
            print(row_fmt.format(r["algorithm"].upper(), sr, bd, r["episodes_to_80pct"]))
    else:
        results = per_file
        for r in results:
            if "error" in r:
                print(f"{r['algorithm']}: {r['error']}")
                continue
            print(f"\n{r['algorithm'].upper()}")
            print(f"  Success rate:        {r['success_rate_pct']}%")
            print(f"  Battery death rate:  {r['battery_death_pct']}%")
            print(f"  Episodes to 80%:     {r['episodes_to_80pct']}")
            print(f"  Mean reward:         {r['mean_reward']}")
            print(f"  Mean steps:          {r['mean_steps']}")
            print(f"  Total episodes:      {r['total_episodes']}")

        print("\n" + "=" * 60)
        print("Table 1 (proposal-style summary)")
        print("=" * 60)
        headers = ["Algorithm", "Success Rate", "Battery Deaths", "Episodes to 80%"]
        row_fmt = "{:12} {:>14} {:>16} {:>18}"
        print(row_fmt.format(*headers))
        print("-" * 60)
        seen = set()
        for r in results:
            if "error" in r:
                continue
            key = r["algorithm"]
            if key in seen:
                continue
            seen.add(key)
            sr = f"{r['success_rate_pct']}%"
            bd = f"{r['battery_death_pct']}%"
            e80 = str(r["episodes_to_80pct"])
            print(row_fmt.format(key.upper(), sr, bd, e80))

    if args.out_table:
        out_path = Path(args.out_table)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            if args.aggregate_seeds:
                f.write("algorithm,success_rate_pct,battery_death_pct,episodes_to_80pct,mean_reward,n_runs\n")
                for r in results:
                    f.write(f"{r['algorithm']},{r.get('success_rate_pct','')},{r.get('battery_death_pct','')},{r.get('episodes_to_80pct','')},{r.get('mean_reward','')},{r.get('n_runs','')}\n")
            else:
                f.write("algorithm,success_rate_pct,battery_death_pct,episodes_to_80pct,mean_reward,mean_steps,total_episodes\n")
                for r in results:
                    if "error" in r:
                        continue
                    e80 = r["episodes_to_80pct"] if isinstance(r["episodes_to_80pct"], int) else ""
                    f.write(f"{r['algorithm']},{r['success_rate_pct']},{r['battery_death_pct']},{e80},{r['mean_reward']},{r['mean_steps']},{r['total_episodes']}\n")
        print(f"\nTable saved to {out_path}")


if __name__ == "__main__":
    main()
