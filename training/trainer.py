"""
Unified training loop for DQN, PPO, and SAC agents.
"""

import warnings
import numpy as np
import torch
from pathlib import Path

from envs.warehouse import make_env
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.sac import SACAgent
from utils.metrics import MetricsCollector

warnings.filterwarnings("ignore")

AGENT_CLASSES = {
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
}


def _obs_to_array(obs):
    """Convert possibly-tuple observation to numpy array (first agent)."""
    if isinstance(obs, tuple):
        return np.asarray(obs[0], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


def _wrap_action(env, action):
    """Wrap single-agent action into multi-agent tuple if needed."""
    if hasattr(env.action_space, "spaces"):
        n = len(env.action_space.spaces)
        return tuple([action] + [0] * (n - 1))  # others NOOP (Action.NOOP=0)
    return action


def _scalar_reward(reward):
    """Convert reward to float scalar."""
    if isinstance(reward, (list, tuple)):
        return float(reward[0])
    if isinstance(reward, np.ndarray):
        return float(reward.item())
    return float(reward)


def train(algo, env_config, battery_config, algo_config, training_config,
          verbose=True):
    """
    Train an agent.

    Args:
        algo: "dqn", "ppo", or "sac"
        env_config: environment settings dict
        battery_config: battery settings dict
        algo_config: algorithm hyperparameters dict
        training_config: episodes, eval_freq, save_freq
        verbose: print progress

    Returns:
        MetricsCollector with all episode data
    """
    device = torch.device("cpu")
    env = make_env(env_config, battery_config)

    # Determine obs/action sizes from env
    obs, _ = env.reset()
    state = _obs_to_array(obs)
    obs_size = state.shape[0]
    n_actions = env.action_space.spaces[0].n if hasattr(env.action_space, "spaces") else env.action_space.n

    # Create agent
    AgentClass = AGENT_CLASSES[algo]
    agent = AgentClass(obs_size, n_actions, device, algo_config)

    episodes = training_config["episodes"]
    eval_freq = training_config.get("eval_freq", 200)
    save_freq = training_config.get("save_freq", 500)
    metrics = MetricsCollector()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    best_reward = float("-inf")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {algo.upper()} | {episodes} episodes")
        print(f"Obs: {obs_size} | Actions: {n_actions}")
        print(f"{'='*60}\n")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state = _obs_to_array(obs)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_obs, reward, terminated, truncated, info = env.step(_wrap_action(env, action))
            done = terminated or truncated
            reward = _scalar_reward(reward)
            next_state = _obs_to_array(next_obs)

            # Algorithm-specific update
            if algo == "ppo":
                agent.store_transition(state, action, reward, done)
                if agent.ready_to_update():
                    agent.update(next_state=next_state)
            else:
                # DQN and SAC: store + train in update()
                agent.update(state, action, reward, next_state, done)

            ep_reward += reward
            state = next_state

        agent.end_episode()

        # PPO: flush remaining rollout at episode end
        if algo == "ppo" and len(agent.buf_states) > 0:
            agent.update(next_state=state)

        metrics.log_episode(ep, ep_reward, env.total_steps, info,
                            epsilon=getattr(agent, "epsilon", None))

        # Logging
        if ep % eval_freq == 0 and verbose:
            avg = metrics.recent_avg("reward", eval_freq)
            extra = f" | eps={agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
            print(f"Ep {ep}/{episodes} | Avg Reward: {avg:.1f}{extra}")

            if avg > best_reward:
                best_reward = avg
                agent.save(f"models/{algo}_best.pt")
                print(f"  -> New best! Saved models/{algo}_best.pt")

        if ep % save_freq == 0:
            agent.save(f"models/{algo}_ep{ep}.pt")

    # Final save
    agent.save(f"models/{algo}_final.pt")
    metrics.save(f"outputs/{algo}_metrics.json")

    if verbose:
        s = metrics.summary()
        print(f"\nDone! Avg: {s.get('avg_reward', 0):.1f} | Best: {s.get('best_reward', 0):.1f}")

    return metrics
