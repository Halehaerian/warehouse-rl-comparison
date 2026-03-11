"""Comprehensive RL algorithm comparison analysis."""
import json
import numpy as np

algos = ['sac', 'ddqn', 'dqn', 'ppo']
window = 200

print('=' * 80)
print('COMPREHENSIVE RL ALGORITHM COMPARISON ANALYSIS')
print('=' * 80)

all_data = {}
for algo in algos:
    path = f'outputs/{algo}_metrics.json'
    try:
        with open(path) as f:
            data = json.load(f)
        all_data[algo] = data
    except FileNotFoundError:
        print(f'{algo}: FILE NOT FOUND')

# --- 1. Convergence Speed ---
print('\n' + '=' * 80)
print('1. CONVERGENCE SPEED (rolling window=200)')
print('=' * 80)
print(f'{"Algo":<8} {"Eps to 80%":<15} {"Eps to 95%":<15} {"Eps to 99%":<15}')
print('-' * 53)

for algo in algos:
    data = all_data[algo]
    successes = [1 if ep.get('mission_complete', False) else 0 for ep in data]
    n = len(successes)
    thresholds = {80: None, 95: None, 99: None}
    for i in range(window, n + 1):
        rate = sum(successes[i - window:i]) / window * 100
        for t in thresholds:
            if thresholds[t] is None and rate >= t:
                thresholds[t] = i
    row = f'{algo.upper():<8}'
    for t in [80, 95, 99]:
        val = str(thresholds[t]) if thresholds[t] else 'Never'
        row += f' {val:<15}'
    print(row)

# --- 2. Final Performance (last 1000 episodes) ---
print('\n' + '=' * 80)
print('2. FINAL PERFORMANCE (last 1000 episodes)')
print('=' * 80)
print(f'{"Algo":<8} {"Success%":<12} {"Avg Reward":<14} {"Avg Steps":<12} {"Avg Deliv.":<12}')
print('-' * 58)

for algo in algos:
    data = all_data[algo]
    last = data[-1000:]
    sr = sum(1 for ep in last if ep.get('mission_complete', False)) / len(last) * 100
    avg_r = np.mean([ep['reward'] for ep in last])
    avg_s = np.mean([ep['steps'] for ep in last])
    avg_d = np.mean([ep.get('deliveries', 0) for ep in last])
    print(f'{algo.upper():<8} {sr:>6.1f}%     {avg_r:>8.1f}      {avg_s:>6.1f}      {avg_d:>6.2f}')

# --- 3. Stability (second half of training) ---
print('\n' + '=' * 80)
print('3. STABILITY (success rate variance, second half of training)')
print('=' * 80)
print(f'{"Algo":<8} {"Mean SR%":<12} {"Std Dev":<10} {"Min Window":<12} {"Max Window":<12} {"Dips <90%":<10}')
print('-' * 64)

for algo in algos:
    data = all_data[algo]
    successes = [1 if ep.get('mission_complete', False) else 0 for ep in data]
    n = len(successes)
    rates = []
    for i in range(window, n + 1):
        rate = sum(successes[i - window:i]) / window * 100
        rates.append(rate)
    half = rates[len(rates) // 2:]
    mean_sr = np.mean(half)
    std_sr = np.std(half)
    min_sr = np.min(half)
    max_sr = np.max(half)
    dips = sum(1 for r in half if r < 90)
    print(f'{algo.upper():<8} {mean_sr:>6.1f}%     {std_sr:>5.2f}%   {min_sr:>6.1f}%     {max_sr:>6.1f}%     {dips}')

# --- 4. Sample Efficiency (steps per successful episode) ---
print('\n' + '=' * 80)
print('4. SAMPLE EFFICIENCY (steps in successful episodes)')
print('=' * 80)
print(f'{"Algo":<8} {"Total Success":<15} {"Avg Steps":<12} {"Min Steps":<12} {"Max Steps":<12}')
print('-' * 59)

for algo in algos:
    data = all_data[algo]
    successful = [ep for ep in data if ep.get('mission_complete', False)]
    if successful:
        steps = [ep['steps'] for ep in successful]
        print(f'{algo.upper():<8} {len(successful):<15} {np.mean(steps):>7.1f}     {np.min(steps):>6}      {np.max(steps):>6}')
    else:
        print(f'{algo.upper():<8} 0')

# --- 5. Battery Management ---
print('\n' + '=' * 80)
print('5. BATTERY MANAGEMENT')
print('=' * 80)
print(f'{"Algo":<8} {"Battery Deaths":<16} {"Death Rate%":<14} {"Avg Charges":<14} {"Avg Batt Left":<14}')
print('-' * 66)

for algo in algos:
    data = all_data[algo]
    bd = sum(1 for ep in data if ep.get('battery_dead', False))
    bd_rate = bd / len(data) * 100
    charges = np.mean([ep.get('charging_events', 0) for ep in data])
    batt_left = np.mean([ep.get('battery_remaining', 0) for ep in data])
    print(f'{algo.upper():<8} {bd:<16} {bd_rate:>6.1f}%       {charges:>7.2f}       {batt_left:>7.1f}')

# --- 6. Learning Phases ---
print('\n' + '=' * 80)
print('6. LEARNING PHASES (success rate at milestones)')
print('=' * 80)
milestones = [500, 1000, 2000, 3000, 5000, 8000, 10000]
header = f'{"Algo":<8}'
for m in milestones:
    header += f' {"Ep" + str(m):<10}'
print(header)
print('-' * (8 + 10 * len(milestones)))

for algo in algos:
    data = all_data[algo]
    successes = [1 if ep.get('mission_complete', False) else 0 for ep in data]
    n = len(successes)
    row = f'{algo.upper():<8}'
    for m in milestones:
        if m <= n and m >= window:
            rate = sum(successes[m - window:m]) / window * 100
            row += f' {rate:>5.1f}%    '
        else:
            row += f' {"N/A":<10}'
    print(row)

# --- 7. Average Reward Comparison ---
print('\n' + '=' * 80)
print('7. REWARD PROGRESSION (avg reward per 1000-ep block)')
print('=' * 80)
blocks = [(1, 1000), (1001, 2000), (2001, 3000), (3001, 5000), (5001, 8000), (8001, 10000)]
header = f'{"Algo":<8}'
for s, e in blocks:
    header += f' {"Ep" + str(s) + "-" + str(e):<14}'
print(header)
print('-' * (8 + 14 * len(blocks)))

for algo in algos:
    data = all_data[algo]
    row = f'{algo.upper():<8}'
    for s, e in blocks:
        chunk = [ep['reward'] for ep in data[s - 1:e]]
        if chunk:
            row += f' {np.mean(chunk):>8.1f}      '
        else:
            row += f' {"N/A":<14}'
    print(row)

print('\n' + '=' * 80)
print('ANALYSIS COMPLETE')
print('=' * 80)
