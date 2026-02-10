"""
SIMPLE DQN Agent - Stripped down for FAST learning
NO dueling, NO PER, NO complex features - just basic DQN that works
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gymnasium.spaces import Tuple as TupleSpace


class SimpleQNetwork(nn.Module):
    """Minimal Q-network: 2 small layers for speed."""
    
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)


class SimpleDQNAgent:
    """
    Basic DQN: Simple replay buffer, no PER, no dueling.
    Fast inference, fast training, easy to debug.
    """
    
    def __init__(self, env, device=None, lr=1e-3, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99,
                 batch_size=32, memory_size=10000, hidden_size=64,
                 train_freq=1):
        
        self.env = env
        self.device = device or torch.device("cpu")  # Force CPU
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_freq = train_freq
        
        # Get sizes
        self.obs_size = self._get_obs_size()
        self.n_actions = self._get_action_size()
        
        # Simple networks
        self.q_network = SimpleQNetwork(self.obs_size, self.n_actions, hidden_size).to(self.device)
        self.target_network = SimpleQNetwork(self.obs_size, self.n_actions, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Simple replay buffer
        self.memory = deque(maxlen=memory_size)
        self.step_count = 0
        
        print(f"\n{'='*60}")
        print("SIMPLE DQN INITIALIZED")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Obs Size: {self.obs_size}, Actions: {self.n_actions}")
        print(f"LR: {lr}, Gamma: {gamma}, Epsilon: {epsilon}")
        print(f"Batch: {batch_size}, Memory: {memory_size}")
        print(f"{'='*60}\n")
    
    def _get_obs_size(self):
        obs_space = self.env.observation_space
        if isinstance(obs_space, TupleSpace):
            return obs_space.spaces[0].shape[0]
        return obs_space.shape[0]
    
    def _get_action_size(self):
        action_space = self.env.action_space
        if isinstance(action_space, TupleSpace):
            return action_space.spaces[0].n
        return action_space.n
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train on batch."""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Current Q values: get Q for the actions taken
        q_all = self.q_network(states)  # (batch_size, n_actions)
        # Simple indexing instead of gather
        q_values = q_all[torch.arange(self.batch_size), actions]  # (batch_size,)
        
        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]  # (batch_size,)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)  # (batch_size,)
        
        # Simple MSE loss
        loss = nn.MSELoss()(q_values, target_q)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step_count += 1
        
        # Hard update target network every 100 steps
        if self.step_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(1).item()
    
    def decay_epsilon(self):
        """Reduce exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        import os
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        try:
            torch.save(self.q_network.state_dict(), filepath)
        except Exception as e:
            print(f"ERROR saving model to {filepath}: {e}")
    
    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
