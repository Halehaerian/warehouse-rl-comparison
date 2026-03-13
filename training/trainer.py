import warnings
import numpy as np
import torch
from pathlib import Path

from envs.warehouse import make_env
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.sac.sac_original import SACAgent
from utils.metrics import MetricsCollector

warnings.filterwarnings("ignore")

AGENT_CLASSES = {
    "ddqn": DQNAgent,
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
}


def _obs_to_array(obs):
    if isinstance(obs, tuple):
        return np.asarray(obs[0], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


def _wrap_action(env, action):
    if hasattr(env.action_space, "spaces"):
        n = len(env.action_space.spaces)
        return tuple([action] + [0] * (n - 1))  # others NOOP (Action.NOOP=0)
    return action


def _scalar_reward(reward):
    if isinstance(reward, (list, tuple)):
        return float(reward[0])
    if isinstance(reward, np.ndarray):
        return float(reward.item())
    return float(reward)


def _save_checkpoint(agent, path, episode):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = agent.state_dict()
    data["episode"] = episode
    torch.save(data, path)


def train(algo, env_config, battery_config, algo_config, training_config,
          verbose=True, seed=None, device=None, resume_path=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device is not None and device.type == "cuda":
            torch.cuda.manual_seed(seed)

    if device is None:
        device = torch.device("cpu")
    env = make_env(env_config, battery_config)

    reset_kw = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kw)
    state = _obs_to_array(obs)
    obs_size = state.shape[0]
    n_actions = env.action_space.spaces[0].n if hasattr(env.action_space, "spaces") else env.action_space.n

    AgentClass = AGENT_CLASSES[algo]
    agent = AgentClass(obs_size, n_actions, device, algo_config)

    episodes = training_config["episodes"]
    eval_freq = training_config.get("eval_freq", 200)
    save_freq = training_config.get("save_freq", 500)
    metrics = MetricsCollector()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    best_reward = float("-inf")
    model_suffix = f"_seed{seed}" if seed is not None else ""

    start_ep = 1
    if resume_path is not None:
        p = Path(resume_path)
        if p.exists():
            checkpoint = torch.load(str(p), map_location=device)
            agent.load_state_dict(checkpoint)
            if "episode" in checkpoint:
                start_ep = checkpoint["episode"] + 1
            elif "_ep" in p.stem:
                try:
                    start_ep = int(p.stem.split("_ep")[-1]) + 1
                except ValueError:
                    start_ep = 1
            if verbose:
                print(f"Resumed from {resume_path} (starting at episode {start_ep})")
        else:
            print(f"Warning: resume checkpoint {resume_path} not found, training from scratch")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {algo.upper()} | episodes {start_ep}-{episodes}")
        print(f"Obs: {obs_size} | Actions: {n_actions}" + (f" | Seed: {seed}" if seed is not None else "") + f" | Device: {device}")
        print(f"{'='*60}\n")

    for ep in range(start_ep, episodes + 1):
        reset_kw = {"seed": seed + ep} if seed is not None else {}
        obs, info = env.reset(**reset_kw)
        state = _obs_to_array(obs)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_obs, reward, terminated, truncated, info = env.step(_wrap_action(env, action))
            done = terminated or truncated
            reward = _scalar_reward(reward)
            next_state = _obs_to_array(next_obs)

            if algo == "ppo":
                agent.store_transition(state, action, reward, terminated)
            else:
                agent.update(state, action, reward, next_state, terminated)

            ep_reward += reward
            state = next_state

        agent.end_episode()

        if algo == "ppo" and agent.ready_to_update():
            agent.update(next_state=state)

        metrics.log_episode(ep, ep_reward, env.total_steps, info,
                            epsilon=getattr(agent, "epsilon", None))

        if ep % eval_freq == 0 and verbose:
            avg = metrics.recent_avg("reward", eval_freq)
            extra = f" | eps={agent.epsilon:.3f}" if hasattr(agent, "epsilon") else ""
            print(f"Ep {ep}/{episodes} | Avg Reward: {avg:.1f}{extra}")

            if avg > best_reward:
                best_reward = avg
                _save_checkpoint(agent, f"models/{algo}_best.pt", ep)
                print(f"  -> New best! Saved models/{algo}_best.pt")

        if ep % save_freq == 0:
            _save_checkpoint(agent, f"models/{algo}_ep{ep}.pt", ep)

    _save_checkpoint(agent, f"models/{algo}_final.pt", episodes)
    metrics.save(f"outputs/{algo}_metrics.json")

    if verbose:
        s = metrics.summary()
        print(f"\nDone! Avg: {s.get('avg_reward', 0):.1f} | Best: {s.get('best_reward', 0):.1f}")

    return metrics
