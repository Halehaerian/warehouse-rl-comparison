import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_metrics(path: Path) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def success_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    n = sum(1 for e in episodes if e.get("mission_complete", False))
    return 100.0 * n / len(episodes)


def battery_death_rate(episodes: list) -> float:
    if not episodes:
        return 0.0
    n = sum(1 for e in episodes if e.get("battery_dead", False))
    return 100.0 * n / len(episodes)


def episodes_to_success_threshold(episodes: list, threshold: float = 0.80,
                                  window: int = 200) -> int | None:
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
    stem = path.stem  
    stem = stem.replace("_metrics", "")
    if "_seed" in stem:
        stem = stem.split("_seed")[0]
    return stem
def avg_deliveries(episodes: list) -> float:
    if not episodes:
        return 0.0
    return sum(e.get("deliveries", 0) for e in episodes) / len(episodes)


def avg_steps_per_delivery(episodes: list) -> float:
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
    completed = [e for e in episodes if e.get("mission_complete", False)]
    if not completed:
        return 0.0
    return sum(e.get("battery_remaining", 0) for e in completed) / len(completed)


def avg_charging_events(episodes: list) -> float:
    if not episodes:
        return 0.0
    return sum(e.get("charging_events", 0) for e in episodes) / len(episodes)

def stability_stats(episodes: list, window: int = 200, start_ep: int = 5001) -> dict:
    second_half = episodes[start_ep - 1:]
    sr_wins = []
    for i in range(window - 1, len(second_half)):
        w = second_half[i - window + 1 : i + 1]
        sr_wins.append(100 * sum(1 for e in w if e.get("mission_complete", False)) / window)
    if not sr_wins:
        return {}
    import statistics as _st
    return {
        "mean_sr": round(_st.mean(sr_wins), 2),
        "std_sr":  round(_st.stdev(sr_wins), 2),
        "min_sr":  round(min(sr_wins), 1),
        "max_sr":  round(max(sr_wins), 1),
        "dips_below_90": sum(1 for x in sr_wins if x < 90),
    }


def convergence_thresholds(episodes: list, thresholds: list, window: int = 200) -> dict:
    result = {}
    for threshold in thresholds:
        found = None
        for i in range(window - 1, len(episodes)):
            w = episodes[i - window + 1 : i + 1]
            sr = sum(1 for e in w if e.get("mission_complete", False)) / window
            if sr >= threshold:
                found = episodes[i]["episode"]
                break
        result[threshold] = found
    return result


def milestone_sr(episodes: list, milestones: list, window: int = 200) -> dict:
    result = {}
    for m in milestones:
        idx = m - 1
        if idx >= len(episodes):
            result[m] = None
            continue
        w = episodes[max(0, idx - window + 1) : idx + 1]
        result[m] = round(100 * sum(1 for e in w if e.get("mission_complete", False)) / len(w), 1)
    return result


def evaluate_one(path: Path, window: int = 200) -> dict:
    episodes = load_metrics(path)
    base = _algo_base_name(path)
    if not episodes:
        return {"algorithm": base, "error": "No episodes", "path": str(path)}
    ep80 = episodes_to_success_threshold(episodes, threshold=0.80, window=window)

    # Last 1000 episodes metrics
    last_1000 = episodes[-1000:] if len(episodes) >= 1000 else episodes
    def safe(val): return round(val, 2) if isinstance(val, float) else val
    return {
        "algorithm": base,
        "success_rate_pct": round(success_rate(episodes), 1),
        "battery_death_pct": round(battery_death_rate(episodes), 1),
        "avg_deliveries": round(avg_deliveries(episodes), 2),
        "avg_steps_per_delivery": round(avg_steps_per_delivery(episodes), 1),
        "battery_efficiency_pct": round(battery_efficiency(episodes), 1),
        "avg_battery_left_all": round(sum(e.get("battery_remaining", 0) for e in episodes) / len(episodes), 1),
        "battery_deaths_count": sum(1 for e in episodes if e.get("battery_dead", False)),
        "avg_charging": round(avg_charging_events(episodes), 2),
        "episodes_to_80pct": ep80 if ep80 is not None else "N/A",
        "mean_reward": round(mean_reward(episodes), 2),
        "mean_steps": round(mean_steps(episodes), 1),
        "total_episodes": len(episodes),
        # Last 1000
        "last_1000_success_rate_pct": round(success_rate(last_1000), 1),
        "last_1000_battery_death_pct": round(battery_death_rate(last_1000), 1),
        "last_1000_avg_deliveries": round(avg_deliveries(last_1000), 2),
        "last_1000_avg_steps_per_delivery": round(avg_steps_per_delivery(last_1000), 1),
        "last_1000_battery_efficiency_pct": round(battery_efficiency(last_1000), 1),
        "last_1000_avg_charging": round(avg_charging_events(last_1000), 2),
        "last_1000_mean_reward": round(mean_reward(last_1000), 2),
        "last_1000_mean_steps": round(mean_steps(last_1000), 1),
        # Stability & milestones
        "stability": stability_stats(episodes, window=window),
        "milestones": milestone_sr(episodes, [500, 1000, 2000, 3000, 5000, 10000], window=window),
        "convergence": convergence_thresholds(episodes, [0.80, 0.95, 0.99], window=window),
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

        print("\n" + "=" * 80)
        print("Table 2 (Last 1000 Episodes)")
        print("=" * 80)
        h2 = "{:10} {:>12} {:>14} {:>12} {:>10} {:>14} {:>12} {:>12}".format(
            "Algorithm", "Success%", "Batt Death%", "Avg Deliv", "Avg Steps",
            "Steps/Deliv", "Avg Charge", "Avg Reward")
        print(h2)
        print("-" * 80)
        seen2 = set()
        for r in results:
            if "error" in r:
                continue
            key = r["algorithm"]
            if key in seen2:
                continue
            seen2.add(key)
            print("{:10} {:>12} {:>14} {:>12} {:>10} {:>14} {:>12} {:>12}".format(
                key.upper(),
                f"{r['last_1000_success_rate_pct']}%",
                f"{r['last_1000_battery_death_pct']}%",
                f"{r['last_1000_avg_deliveries']}",
                f"{r['last_1000_mean_steps']}",
                f"{r['last_1000_avg_steps_per_delivery']}",
                f"{r['last_1000_avg_charging']}",
                f"{r['last_1000_mean_reward']}",
            ))

        # --- Battery Management Table ---
        print("\n" + "=" * 80)
        print("Table 3 (Battery Management: full 10,000 episodes)")
        print("=" * 80)
        print("{:22} {:>10} {:>10} {:>10} {:>10}".format("Metric", "SAC", "DDQN", "DQN", "PPO"))
        print("-" * 80)
        batt_results = {r["algorithm"]: r for r in results if "error" not in r}
        order = ["sac", "ddqn", "dqn", "ppo"]
        def bv(key, algo): return batt_results.get(algo, {}).get(key, "N/A")
        print("{:22} {:>10} {:>10} {:>10} {:>10}".format(
            "Battery Deaths",
            *[str(bv("battery_deaths_count", a)) for a in order]))
        print("{:22} {:>10} {:>10} {:>10} {:>10}".format(
            "Death Rate",
            *[str(bv("battery_death_pct", a))+"%" for a in order]))
        print("{:22} {:>10} {:>10} {:>10} {:>10}".format(
            "Avg Charges/Ep",
            *[str(bv("avg_charging", a)) for a in order]))
        print("{:22} {:>10} {:>10} {:>10} {:>10}".format(
            "Avg Battery Left",
            *[str(bv("avg_battery_left_all", a)) for a in order]))

        # --- Stability Table ---
        print("\n" + "=" * 80)
        print("Table 4 (Stability: episodes 5,001-10,000, rolling window=200)")
        print("=" * 80)
        print("{:10} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
            "Algorithm", "Mean SR", "Std Dev", "Min Window", "Max Window", "Dips <90%"))
        print("-" * 80)
        seen3 = set()
        for r in results:
            if "error" in r or r["algorithm"] in seen3:
                continue
            seen3.add(r["algorithm"])
            s = r.get("stability", {})
            if not s:
                continue
            print("{:10} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
                r["algorithm"].upper(),
                f"{s['mean_sr']}%",
                f"{s['std_sr']}%",
                f"{s['min_sr']}%",
                f"{s['max_sr']}%",
                str(s["dips_below_90"]),
            ))

        # --- Milestones Table ---
        milestone_cols = [500, 1000, 2000, 3000, 5000, 10000]
        print("\n" + "=" * 80)
        print("Table 5 (Milestones: rolling window=200 SR at episode)")
        print("=" * 80)
        hdr = "{:10}".format("Algorithm") + "".join(f" {'Ep'+str(m):>10}" for m in milestone_cols)
        print(hdr)
        print("-" * 80)
        seen4 = set()
        for r in results:
            if "error" in r or r["algorithm"] in seen4:
                continue
            seen4.add(r["algorithm"])
            m_data = r.get("milestones", {})
            row = "{:10}".format(r["algorithm"].upper())
            for m in milestone_cols:
                val = m_data.get(m)
                row += f" {(str(val)+'%') if val is not None else 'N/A':>10}"
            print(row)

        # --- Convergence Table ---
        conv_thresholds = [0.80, 0.95, 0.99]
        print("\n" + "=" * 80)
        print("Table 6 (Convergence: episodes to reach SR threshold, rolling window=200)")
        print("=" * 80)
        print("{:10} {:>12} {:>12} {:>12}".format("Algorithm", "80%", "95%", "99%"))
        print("-" * 80)
        seen5 = set()
        for r in results:
            if "error" in r or r["algorithm"] in seen5:
                continue
            seen5.add(r["algorithm"])
            c = r.get("convergence", {})
            row = "{:10}".format(r["algorithm"].upper())
            for t in conv_thresholds:
                val = c.get(t)
                row += " {:>12}".format(str(val) if val else "Never")
            print(row)

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
