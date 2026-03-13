import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


ALGOS = ['sac', 'ddqn', 'dqn', 'ppo']
ALGO_NAMES = {'sac': 'SAC', 'ddqn': 'DDQN', 'dqn': 'DQN', 'ppo': 'PPO'}
ALGO_COLORS = {'sac': '#2ecc71', 'ddqn': '#3498db', 'dqn': '#e74c3c', 'ppo': '#f39c12'}
WINDOW = 200


def load_metrics(algo, metrics_dir):
    path = Path(metrics_dir) / f'{algo}_metrics.json'
    with open(path) as f:
        return json.load(f)


def rolling_mean(x, window):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.mean(x[start:i + 1])
    return out


def plot_success_rate(all_data, savedir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGOS:
        data = all_data[algo]
        eps = [e['episode'] for e in data]
        success = [1.0 if e.get('mission_complete', False) else 0.0 for e in data]
        smooth = rolling_mean(success, WINDOW) * 100
        ax.plot(eps, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)

    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.6, label='80% threshold')
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.4, label='95% threshold')
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Success Rate (%)', fontsize=13)
    ax.set_title('Success Rate Over Training', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    fig.tight_layout()
    fig.savefig(savedir / 'analysis_success_rate.png', dpi=200)
    plt.close(fig)
    print(f'Saved {savedir / "analysis_success_rate.png"}')


def plot_reward(all_data, savedir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGOS:
        data = all_data[algo]
        eps = [e['episode'] for e in data]
        rewards = [e['reward'] for e in data]
        smooth = rolling_mean(rewards, WINDOW)
        ax.plot(eps, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)
        ax.fill_between(eps, smooth - 20, smooth + 20, color=ALGO_COLORS[algo], alpha=0.08)

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Average Reward', fontsize=13)
    ax.set_title('Reward Progression', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savedir / 'analysis_reward.png', dpi=200)
    plt.close(fig)
    print(f'Saved {savedir / "analysis_reward.png"}')


def plot_battery_death(all_data, savedir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGOS:
        data = all_data[algo]
        eps = [e['episode'] for e in data]
        bd = [1.0 if e.get('battery_dead', False) else 0.0 for e in data]
        smooth = rolling_mean(bd, WINDOW) * 100
        ax.plot(eps, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Battery Death Rate (%)', fontsize=13)
    ax.set_title('Battery Death Rate Over Training', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savedir / 'analysis_battery_death.png', dpi=200)
    plt.close(fig)
    print(f'Saved {savedir / "analysis_battery_death.png"}')


def plot_convergence_bars(all_data, savedir):
    thresholds = [80, 95, 99]
    results = {}
    for algo in ALGOS:
        data = all_data[algo]
        successes = [1 if e.get('mission_complete', False) else 0 for e in data]
        n = len(successes)
        algo_res = {}
        for t in thresholds:
            found = None
            for i in range(WINDOW, n + 1):
                rate = sum(successes[i - WINDOW:i]) / WINDOW * 100
                if rate >= t:
                    found = i
                    break
            algo_res[t] = found if found else n + 500  
        results[algo] = algo_res

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALGOS))
    width = 0.25
    for idx, t in enumerate(thresholds):
        values = [results[a][t] for a in ALGOS]
        labels_text = []
        for a in ALGOS:
            v = results[a][t]
            labels_text.append(str(v) if v <= len(all_data[a]) else 'N/A')
        bars = ax.bar(x + idx * width, values, width, label=f'{t}% success',
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, txt in zip(bars, labels_text):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                    txt, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Algorithm', fontsize=13)
    ax.set_ylabel('Episodes to Reach Threshold', fontsize=13)
    ax.set_title('Convergence Speed Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([ALGO_NAMES[a] for a in ALGOS], fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(savedir / 'analysis_convergence.png', dpi=200)
    plt.close(fig)
    print(f'Saved {savedir / "analysis_convergence.png"}')


def plot_final_performance(all_data, savedir):
    metrics = {}
    for algo in ALGOS:
        last = all_data[algo][-1000:]
        sr = sum(1 for e in last if e.get('mission_complete', False)) / len(last) * 100
        avg_r = np.mean([e['reward'] for e in last])
        avg_s = np.mean([e['steps'] for e in last])
        bd = sum(1 for e in last if e.get('battery_dead', False)) / len(last) * 100
        metrics[algo] = {'success': sr, 'reward': avg_r, 'steps': avg_s, 'battery_death': bd}

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    names = [ALGO_NAMES[a] for a in ALGOS]
    colors = [ALGO_COLORS[a] for a in ALGOS]

    vals = [metrics[a]['success'] for a in ALGOS]
    axes[0].bar(names, vals, color=colors, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_title('Success Rate', fontsize=12, fontweight='bold')
    axes[0].set_ylim(85, 102)
    axes[0].set_ylabel('%')
    axes[0].grid(True, alpha=0.3, axis='y')

    vals = [metrics[a]['reward'] for a in ALGOS]
    axes[1].bar(names, vals, color=colors, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 2, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_title('Avg Reward', fontsize=12, fontweight='bold')
    axes[1].set_ylim(min(vals) - 30, max(vals) + 30)
    axes[1].grid(True, alpha=0.3, axis='y')

    vals = [metrics[a]['steps'] for a in ALGOS]
    axes[2].bar(names, vals, color=colors, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(vals):
        axes[2].text(i, v + 2, f'{v:.0f}', ha='center', fontsize=10, fontweight='bold')
    axes[2].set_title('Avg Steps (lower=better)', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, max(vals) + 30)
    axes[2].grid(True, alpha=0.3, axis='y')

    vals = [metrics[a]['battery_death'] for a in ALGOS]
    axes[3].bar(names, vals, color=colors, edgecolor='white', linewidth=0.5)
    for i, v in enumerate(vals):
        axes[3].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[3].set_title('Battery Death (lower=better)', fontsize=12, fontweight='bold')
    axes[3].set_ylim(0, max(vals) + 5)
    axes[3].set_ylabel('%')
    axes[3].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Final Performance (Last 1000 Episodes)', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(savedir / 'analysis_final_performance.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {savedir / "analysis_final_performance.png"}')


def plot_steps_efficiency(all_data, savedir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGOS:
        data = all_data[algo]
        eps = []
        steps = []
        for e in data:
            if e.get('mission_complete', False):
                eps.append(e['episode'])
                steps.append(e['steps'])
        if len(eps) > WINDOW:
            smooth = rolling_mean(steps, WINDOW)
            ax.plot(eps, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Steps per Successful Episode', fontsize=13)
    ax.set_title('Execution Efficiency (Successful Episodes Only)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savedir / 'analysis_steps_efficiency.png', dpi=200)
    plt.close(fig)
    print(f'Saved {savedir / "analysis_steps_efficiency.png"}')


def main():
    parser = argparse.ArgumentParser(description='Generate analysis plots')
    parser.add_argument('--outputs', type=str, default='outputs', help='Metrics directory')
    parser.add_argument('--savedir', type=str, default='outputs/figures', help='Save directory')
    args = parser.parse_args()

    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for algo in ALGOS:
        all_data[algo] = load_metrics(algo, args.outputs)
        print(f'Loaded {algo}: {len(all_data[algo])} episodes')

    plot_success_rate(all_data, savedir)
    plot_reward(all_data, savedir)
    plot_battery_death(all_data, savedir)
    plot_convergence_bars(all_data, savedir)
    plot_final_performance(all_data, savedir)
    plot_steps_efficiency(all_data, savedir)

    print(f'\nAll 6 plots saved to {savedir}/')


if __name__ == '__main__':
    main()
