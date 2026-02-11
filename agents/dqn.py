"""DQN Agent with target network and replay buffer."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from agents.base import BaseAgent


class QNetwork(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(self, obs_size, n_actions, device, config):
        super().__init__(obs_size, n_actions, device, config)

        hidden = config.get("hidden_size", 128)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.9995)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 100)

        self.q = QNetwork(obs_size, n_actions, hidden).to(device)
        self.q_target = QNetwork(obs_size, n_actions, hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=config.get("lr", 1e-3))
        self.memory = deque(maxlen=config.get("memory_size", 50000))
        self.steps = 0

    # ----- BaseAgent interface -----

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q(t).argmax(1).item()

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        warmup = self.config.get("warmup", 100)
        if len(self.memory) < max(self.batch_size, warmup):
            return {}

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(np.array(s2)).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        q_vals = self.q(s)[torch.arange(len(a)), a]
        with torch.no_grad():
            # Double DQN: online net selects, target net evaluates
            best_actions = self.q(s2).argmax(1)
            q_next = self.q_target(s2)[torch.arange(len(best_actions)), best_actions]
            target = r + self.gamma * q_next * (1 - d)

        loss = nn.SmoothL1Loss()(q_vals, target)  # Huber loss for stability
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": loss.item()}

    def end_episode(self):
        """Called at episode end to decay epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def state_dict(self):
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, data):
        self.q.load_state_dict(data["q"])
        self.q_target.load_state_dict(data["q_target"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
