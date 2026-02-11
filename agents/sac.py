"""SAC Agent (Soft Actor-Critic) adapted for discrete action spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

from agents.base import BaseAgent


class DiscreteActor(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_action(self, x):
        probs = self(x)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = torch.log(probs + 1e-8)
        return action, probs, log_prob


class TwinCritic(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.q1(x), self.q2(x)


class SACAgent(BaseAgent):
    def __init__(self, obs_size, n_actions, device, config):
        super().__init__(obs_size, n_actions, device, config)

        hidden = config.get("hidden_size", 128)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 64)
        self.memory_size = config.get("memory_size", 50000)
        self.warmup = config.get("warmup", 100)

        # Networks
        self.actor = DiscreteActor(obs_size, n_actions, hidden).to(device)
        self.critic = TwinCritic(obs_size, n_actions, hidden).to(device)
        self.target_critic = TwinCritic(obs_size, n_actions, hidden).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.get("lr", 3e-4))
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.get("lr", 3e-4))

        # Auto-tune entropy
        self.target_entropy = -np.log(1.0 / n_actions) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=config.get("lr", 3e-4))

        self.memory = deque(maxlen=self.memory_size)
        self.steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ----- BaseAgent interface -----

    def select_action(self, state, training=True):
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if training:
                action, _, _ = self.actor.get_action(t)
                return action.item()
            else:
                probs = self.actor(t)
                return probs.argmax(1).item()

    def update(self, state, action, reward, next_state, done, **kwargs):
        self.memory.append((state, action, reward, next_state, done))
        self.steps += 1

        if len(self.memory) < self.warmup:
            return {}

        batch = random.sample(self.memory, self.batch_size)
        s = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        a = torch.LongTensor([b[1] for b in batch]).to(self.device)
        r = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        ns = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        d = torch.FloatTensor([b[4] for b in batch]).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_probs = self.actor(ns)
            next_log_probs = torch.log(next_probs + 1e-8)
            tq1, tq2 = self.target_critic(ns)
            next_q = torch.min(tq1, tq2)
            next_v = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(1)
            target_q = r + self.gamma * (1 - d) * next_v

        q1, q2 = self.critic(s)
        q1_a = q1.gather(1, a.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, a.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update ---
        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_pi, q2_pi = self.critic(s)
            min_q = torch.min(q1_pi, q2_pi)

        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Alpha (entropy) update ---
        entropy = -(probs.detach() * log_probs.detach()).sum(1)
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Soft update target critic ---
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def end_episode(self):
        pass  # SAC has no epsilon

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, data):
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.target_critic.load_state_dict(data["target_critic"])
        self.log_alpha.data.copy_(data["log_alpha"].to(self.device))
