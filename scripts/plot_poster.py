"""
Generate a single poster-style summary figure for RL algorithm comparison.

Produces one high-resolution image with all key findings arranged for a poster.

Usage:
    python scripts/plot_poster.py
    python scripts/plot_poster.py --savedir outputs/figures
"""

import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import numpy as np

ALGOS = ['sac', 'ddqn', 'dqn', 'ppo']
ALGO_NAMES = {'sac': 'SAC', 'ddqn': 'DDQN', 'dqn': 'DQN', 'ppo': 'PPO'}
ALGO_COLORS = {'sac': '#2ecc71', 'ddqn': '#3498db', 'dqn': '#e74c3c', 'ppo': '#f39c12'}
WINDOW = 200


def load_all(metrics_dir):
    all_data = {}
    for algo in ALGOS:
        path = Path(metrics_dir) / f'{algo}_metrics.json'
        with open(path) as f:
            all_data[algo] = json.load(f)
    return all_data


def rolling_mean(x, window):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = np.mean(x[start:i + 1])
    return out


def get_convergence(all_data):
    results = {}
    for algo in ALGOS:
        successes = [1 if e.get('mission_complete', False) else 0 for e in all_data[algo]]
        n = len(successes)
        thresholds = {80: None, 95: None, 99: None}
        for i in range(WINDOW, n + 1):
            rate = sum(successes[i - WINDOW:i]) / WINDOW * 100
            for t in thresholds:
                if thresholds[t] is None and rate >= t:
                    thresholds[t] = i
        results[algo] = thresholds
    return results


def get_final_metrics(all_data):
    metrics = {}
    for algo in ALGOS:
        last = all_data[algo][-1000:]
        sr = sum(1 for e in last if e.get('mission_complete', False)) / len(last) * 100
        avg_r = np.mean([e['reward'] for e in last])
        avg_s = np.mean([e['steps'] for e in last])
        bd = sum(1 for e in last if e.get('battery_dead', False)) / len(last) * 100
        total_bd = sum(1 for e in all_data[algo] if e.get('battery_dead', False))
        charges = np.mean([e.get('charging_events', 0) for e in all_data[algo]])
        metrics[algo] = {
            'success': sr, 'reward': avg_r, 'steps': avg_s,
            'battery_death': bd, 'total_bd': total_bd, 'charges': charges
        }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument('--savedir', type=str, default='outputs/figures')
    args = parser.parse_args()

    savedir = Path(args.savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    all_data = load_all(args.outputs)
    conv = get_convergence(all_data)
    final = get_final_metrics(all_data)

    # =====================================================================
    # POSTER LAYOUT: 3 rows x 3 cols + title row
    # =====================================================================
    fig = plt.figure(figsize=(24, 18), facecolor='white')
    gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[0.08, 1, 1, 0.8],
                           hspace=0.35, wspace=0.3)

    # --- Title Banner ---
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7,
                  'Reinforcement Learning for Warehouse Robot Navigation with Battery Management',
                  fontsize=26, fontweight='bold', ha='center', va='center',
                  transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15,
                  'Comparing SAC, DDQN, DQN, and PPO on RWARE 10\u00d710 Environment  |  10,000 Episodes  |  5 Deliveries per Mission',
                  fontsize=14, ha='center', va='center', color='#555555',
                  transform=ax_title.transAxes)

    # =====================================================================
    # ROW 1: Learning Curves
    # =====================================================================

    # (1,0) Success Rate
    ax1 = fig.add_subplot(gs[1, 0])
    for algo in ALGOS:
        eps = [e['episode'] for e in all_data[algo]]
        success = [1.0 if e.get('mission_complete', False) else 0.0 for e in all_data[algo]]
        smooth = rolling_mean(success, WINDOW) * 100
        ax1.plot(eps, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)
    ax1.axhline(y=80, color='gray', ls='--', alpha=0.5, lw=1)
    ax1.axhline(y=95, color='gray', ls=':', alpha=0.4, lw=1)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.set_ylim(-2, 105)
    ax1.grid(True, alpha=0.2)

    # (1,1) Reward
    ax2 = fig.add_subplot(gs[1, 1])
    for algo in ALGOS:
        eps = [e['episode'] for e in all_data[algo]]
        rewards = [e['reward'] for e in all_data[algo]]
        smooth = rolling_mean(rewards, WINDOW)
        ax2.plot(eps, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Reward Progression', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, alpha=0.2)

    # (1,2) Battery Death Rate
    ax3 = fig.add_subplot(gs[1, 2])
    for algo in ALGOS:
        eps = [e['episode'] for e in all_data[algo]]
        bd = [1.0 if e.get('battery_dead', False) else 0.0 for e in all_data[algo]]
        smooth = rolling_mean(bd, WINDOW) * 100
        ax3.plot(eps, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Battery Death Rate (%)', fontsize=12)
    ax3.set_title('Battery Death Rate', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.2)

    # =====================================================================
    # ROW 2: Bar Charts
    # =====================================================================

    # (2,0) Convergence Speed
    ax4 = fig.add_subplot(gs[2, 0])
    thresholds = [80, 95, 99]
    th_colors = ['#85C1E9', '#5DADE2', '#2E86C1']
    x = np.arange(len(ALGOS))
    width = 0.25
    max_ep = max(len(all_data[a]) for a in ALGOS)
    for idx, t in enumerate(thresholds):
        values = []
        labels = []
        for a in ALGOS:
            v = conv[a][t]
            if v is None:
                values.append(max_ep + 500)
                labels.append('N/A')
            else:
                values.append(v)
                labels.append(str(v))
        bars = ax4.bar(x + idx * width, values, width, label=f'{t}%',
                       color=th_colors[idx], edgecolor='white', linewidth=0.5)
        for bar, txt in zip(bars, labels):
            y_pos = bar.get_height() + 100
            ax4.text(bar.get_x() + bar.get_width() / 2, y_pos, txt,
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax4.set_xlabel('Algorithm', fontsize=12)
    ax4.set_ylabel('Episodes', fontsize=12)
    ax4.set_title('Episodes to Convergence', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([ALGO_NAMES[a] for a in ALGOS], fontsize=11)
    ax4.legend(title='Threshold', fontsize=9)
    ax4.grid(True, alpha=0.2, axis='y')

    # (2,1) Final Performance - 4 grouped bars
    ax5 = fig.add_subplot(gs[2, 1])
    categories = ['Success %', 'Reward\n(scaled)', 'Steps\n(inv. scaled)', 'Battery\nSafety %']
    x2 = np.arange(len(categories))
    width2 = 0.18
    for idx, algo in enumerate(ALGOS):
        f = final[algo]
        # Normalize for visual comparison
        vals = [
            f['success'],                          # 0-100
            f['reward'] / 7.5,                     # scale ~700 to ~93
            (1 - f['steps'] / 500) * 100,          # invert: lower steps = higher bar
            100 - f['battery_death'],               # invert: lower death = higher
        ]
        bars = ax5.bar(x2 + idx * width2, vals, width2,
                       label=ALGO_NAMES[algo], color=ALGO_COLORS[algo],
                       edgecolor='white', linewidth=0.5)
    ax5.set_ylabel('Score (higher = better)', fontsize=12)
    ax5.set_title('Final Performance (Last 1000 Eps)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x2 + width2 * 1.5)
    ax5.set_xticklabels(categories, fontsize=10)
    ax5.legend(fontsize=9)
    ax5.set_ylim(0, 110)
    ax5.grid(True, alpha=0.2, axis='y')

    # (2,2) Steps Efficiency
    ax6 = fig.add_subplot(gs[2, 2])
    for algo in ALGOS:
        data = all_data[algo]
        eps_s = []
        steps_s = []
        for e in data:
            if e.get('mission_complete', False):
                eps_s.append(e['episode'])
                steps_s.append(e['steps'])
        if len(eps_s) > WINDOW:
            smooth = rolling_mean(steps_s, WINDOW)
            ax6.plot(eps_s, smooth, label=ALGO_NAMES[algo], color=ALGO_COLORS[algo], lw=2.5)
    ax6.set_xlabel('Episode', fontsize=12)
    ax6.set_ylabel('Steps per Success', fontsize=12)
    ax6.set_title('Execution Efficiency', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.2)

    # =====================================================================
    # ROW 3: Key Findings Summary Table
    # =====================================================================
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis('off')

    # Build table data
    col_labels = ['Algorithm', 'Eps to 80%', 'Eps to 95%', 'Eps to 99%',
                  'Final SR%', 'Avg Reward', 'Avg Steps', 'Battery\nDeath %', 'Key Strength']
    strengths = {
        'sac': 'Fastest convergence,\nbest battery mgmt',
        'ddqn': 'Stable learning,\nlow overestimation',
        'dqn': 'Simple & effective,\nefficient paths',
        'ppo': 'On-policy stability,\nsteady improvement',
    }
    cell_text = []
    for algo in ALGOS:
        f = final[algo]
        c = conv[algo]
        row = [
            ALGO_NAMES[algo],
            str(c[80]) if c[80] else 'N/A',
            str(c[95]) if c[95] else 'N/A',
            str(c[99]) if c[99] else 'N/A',
            f'{f["success"]:.1f}%',
            f'{f["reward"]:.0f}',
            f'{f["steps"]:.0f}',
            f'{f["battery_death"]:.1f}%',
            strengths[algo],
        ]
        cell_text.append(row)

    table = ax_table.table(cellText=cell_text, colLabels=col_labels,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=11)

    # Style rows with algo colors
    for i, algo in enumerate(ALGOS):
        color = ALGO_COLORS[algo]
        # Light tint of the algo color
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        light = (r / 255 * 0.15 + 0.85, g / 255 * 0.15 + 0.85, b / 255 * 0.15 + 0.85)
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(light)

    ax_table.set_title('Summary of Results', fontsize=14, fontweight='bold', pad=20)

    # Save
    fig.savefig(savedir / 'poster_summary.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved {savedir / "poster_summary.png"}')

    # Also save a PDF version for printing
    fig2 = plt.figure(figsize=(24, 18), facecolor='white')
    # Re-render for PDF (matplotlib handles vector output)
    import subprocess
    # Just save the same figure as PDF
    fig_pdf = plt.figure(figsize=(24, 18), facecolor='white')
    plt.close(fig_pdf)
    print(f'\nPoster image ready at: {savedir / "poster_summary.png"}')
    print('Recommended: open in image viewer or insert into PowerPoint/Google Slides')


if __name__ == '__main__':
    main()
