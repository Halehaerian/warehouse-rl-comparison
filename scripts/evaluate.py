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
    if not episodes:
        return {"algorithm": path.stem.replace("_metrics", ""), "error": "No episodes"}
    ep80 = episodes_to_success_threshold(episodes, threshold=0.80, window=window)
    return {
        "algorithm": path.stem.replace("_metrics", ""),
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate training metrics for proposal Table 1")
    parser.add_argument("--outputs", type=str, default="outputs",
                        help="Directory containing *_metrics.json files")
    parser.add_argument("--window", type=int, default=200,
                        help="Rolling window for 'episodes to 80%% success'")
    parser.add_argument("--out-table", type=str, default=None,
                        help="Optional path to save Table 1 as CSV")
    args = parser.parse_args()

    out_dir = Path(args.outputs)
    if not out_dir.exists():
        print(f"Outputs directory not found: {out_dir}")
        return

    metrics_files = sorted(out_dir.glob("*_metrics.json"))
    if not metrics_files:
        print(f"No *_metrics.json files in {out_dir}")
        return

    results = []
    for path in metrics_files:
        r = evaluate_one(path, window=args.window)
        results.append(r)
        if "error" in r:
            print(f"{r['algorithm']}: {r['error']}")
            continue
        print(f"\n{r['algorithm'].upper()}")
        print(f"  Success rate:        {r['success_rate_pct']}%")
        print(f"  Battery death rate:  {r['battery_death_pct']}%")
        print(f"  Avg deliveries/ep:   {r['avg_deliveries']}")
        print(f"  Avg steps/delivery:  {r['avg_steps_per_delivery']}")
        print(f"  Battery efficiency:  {r['battery_efficiency_pct']}%")
        print(f"  Avg charging events: {r['avg_charging']}")
        print(f"  Episodes to 80%:     {r['episodes_to_80pct']}")
        print(f"  Mean reward:         {r['mean_reward']}")
        print(f"  Mean steps:          {r['mean_steps']}")
        print(f"  Total episodes:      {r['total_episodes']}")

    # Table 1 style summary
    print("\n" + "=" * 80)
    print("Table 1 (proposal-style summary)")
    print("=" * 80)
    headers = ["Algorithm", "Success Rate", "Battery Deaths", "Avg Del/Ep",
               "Steps/Del", "Battery Eff", "Charges", "Ep to 80%"]
    row_fmt = "{:12} {:>14} {:>16} {:>12} {:>10} {:>12} {:>9} {:>10}"
    print(row_fmt.format(*headers))
    print("-" * 90)
    for r in results:
        if "error" in r:
            continue
        sr = f"{r['success_rate_pct']}%"
        bd = f"{r['battery_death_pct']}%"
        ad = f"{r['avg_deliveries']}"
        sd = f"{r['avg_steps_per_delivery']}"
        be = f"{r['battery_efficiency_pct']}%"
        ch = f"{r['avg_charging']}"
        e80 = str(r["episodes_to_80pct"])
        print(row_fmt.format(r["algorithm"].upper(), sr, bd, ad, sd, be, ch, e80))

    if args.out_table:
        out_path = Path(args.out_table)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("algorithm,success_rate_pct,battery_death_pct,avg_deliveries,avg_steps_per_delivery,battery_efficiency_pct,avg_charging,episodes_to_80pct,mean_reward,mean_steps,total_episodes\n")
            for r in results:
                if "error" in r:
                    continue
                e80 = r["episodes_to_80pct"] if isinstance(r["episodes_to_80pct"], int) else ""
                f.write(f"{r['algorithm']},{r['success_rate_pct']},{r['battery_death_pct']},{r['avg_deliveries']},{r['avg_steps_per_delivery']},{r['battery_efficiency_pct']},{r['avg_charging']},{e80},{r['mean_reward']},{r['mean_steps']},{r['total_episodes']}\n")
        print(f"\nTable saved to {out_path}")


if __name__ == "__main__":
    main()
