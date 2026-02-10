"""
Train Simple DQN on RWARE - FAST LEARNING
Minimal code, maximum speed.
"""

import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path

from envs.battery_wrapper import make_battery_warehouse
from agents.simple_dqn_agent import SimpleDQNAgent


def train_dqn_simple(env_config, battery_config, dqn_config, training_config):
    """Train simple DQN agent."""
    
    import warnings
    warnings.filterwarnings('ignore')  # Suppress gym warnings
    
    device = torch.device("cpu")  # Force CPU to avoid CUDA issues
    
    # Create environment
    env = make_battery_warehouse(
        env_id=env_config['env_id'],
        max_steps=env_config['max_steps'],
        max_battery=battery_config['max_battery'],
        battery_drain=battery_config['battery_drain'],
        charge_rate=battery_config['charge_rate'],
        battery_threshold=battery_config['battery_threshold'],
        charger_location=battery_config['charger_location'],
        max_deliveries=env_config['max_deliveries'],
    )
    
    # Create agent
    agent = SimpleDQNAgent(
        env=env,
        device=device,
        lr=dqn_config['lr'],
        gamma=dqn_config['gamma'],
        epsilon=dqn_config['epsilon_start'],
        epsilon_min=dqn_config['epsilon_min'],
        epsilon_decay=dqn_config['epsilon_decay'],
        batch_size=dqn_config['batch_size'],
        memory_size=dqn_config['memory_size'],
        hidden_size=dqn_config['hidden_size'],
        train_freq=dqn_config['train_freq'],
    )
    
    episodes = training_config['episodes']
    eval_freq = training_config['eval_freq']
    save_freq = training_config['save_freq']
    
    # Training loop
    rewards_log = []
    best_reward = -999999
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Select and execute action
            # Convert tuple obs to array if needed
            obs_for_agent = obs[0] if isinstance(obs, tuple) else obs
            action = agent.select_action(obs_for_agent, training=True)
            
            # Handle multi-agent actions
            if hasattr(env.action_space, 'spaces'):
                n_agents = len(env.action_space.spaces)
                actions = tuple([action] + [4 for _ in range(1, n_agents)])
            else:
                actions = action
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Convert reward to scalar (handle list/array rewards from multi-agent env)
            if isinstance(reward, (list, tuple)):
                reward = float(reward[0])
            elif isinstance(reward, np.ndarray):
                reward = float(reward.item())
            else:
                reward = float(reward)
            
            # Convert observations to arrays
            obs_array = obs[0] if isinstance(obs, tuple) else obs
            next_obs_array = next_obs[0] if isinstance(next_obs, tuple) else next_obs
            
            # Store experience
            agent.remember(obs_array, action, reward, next_obs_array, done)
            
            # Train if we're collecting enough data
            if len(agent.memory) > dqn_config['batch_size']:
                loss = agent.train()
            
            episode_reward += reward
            obs = next_obs
            steps += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        rewards_log.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': steps,
            'epsilon': agent.epsilon,
        })
        
        # Evaluation and logging
        if (episode + 1) % eval_freq == 0:
            avg_reward = np.mean([r['reward'] for r in rewards_log[-eval_freq:]])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.1f} | Epsilon: {agent.epsilon:.3f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                model_path = f"models/dqn_battery_best.pt"
                agent.save(model_path)
                print(f"  [OK] Saved best model: {model_path}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/dqn_battery_{timestamp}.pt"
            agent.save(model_path)
    
    # Save final metrics
    with open("outputs/dqn_metrics.json", "w") as f:
        json.dump(rewards_log, f, indent=2)
    
    print(f"\nTraining complete! Final model saved.")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Metrics saved to outputs/dqn_metrics.json")


if __name__ == "__main__":
    from configs.simple_config import ENV_CONFIG, BATTERY_CONFIG, DQN_CONFIG, TRAINING_CONFIG
    train_dqn_simple(ENV_CONFIG, BATTERY_CONFIG, DQN_CONFIG, TRAINING_CONFIG)
