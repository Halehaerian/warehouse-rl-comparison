"""PPO Agent (Proximal Policy Optimization) for discrete actions."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.base import BaseAgent


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, x):
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        logits, values = self(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values.squeeze(-1)


class PPOAgent(BaseAgent):
    def __init__(self, obs_size, n_actions, device, config):
        super().__init__(obs_size, n_actions, device, config)

        hidden = config.get("hidden_size", 128)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.epochs = config.get("ppo_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.rollout_len = config.get("rollout_len", 128)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.ent_coef = config.get("ent_coef", 0.01)

        self.net = ActorCritic(obs_size, n_actions, hidden).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.get("lr", 3e-4))

        # Rollout buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.buf_states = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_rewards = []
        self.buf_dones = []
        self.buf_values = []

    # ----- BaseAgent interface -----

    def select_action(self, state, training=True):
        t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if training:
                action, log_prob, value = self.net.get_action(t)
                self._last_log_prob = log_prob.item()
                self._last_value = value.item()
                return action.item()
            else:
                logits, _ = self.net(t)
                return logits.argmax(1).item()

    def store_transition(self, state, action, reward, done):
        """Store one transition in rollout buffer."""
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_log_probs.append(self._last_log_prob)
        self.buf_rewards.append(reward)
        self.buf_dones.append(done)
        self.buf_values.append(self._last_value)

    def ready_to_update(self):
        return len(self.buf_states) >= self.rollout_len

    def update(self, next_state=None, **kwargs):
        """Run PPO update on collected rollout."""
        if not self.ready_to_update():
            return {}

        # Bootstrap value for last state
        with torch.no_grad():
            if next_state is not None:
                t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, last_val = self.net(t)
                last_val = last_val.item()
            else:
                last_val = 0.0

        # Compute GAE advantages
        advantages = []
        gae = 0.0
        values = self.buf_values + [last_val]
        for t in reversed(range(len(self.buf_rewards))):
            delta = self.buf_rewards[t] + self.gamma * values[t + 1] * (1 - self.buf_dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.buf_dones[t]) * gae
            advantages.insert(0, gae)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buf_states)).to(self.device)
        actions = torch.LongTensor(self.buf_actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buf_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.buf_values).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_loss = 0.0
        n = len(states)
        for _ in range(self.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                b = idx[start:end]

                new_log_probs, entropy, new_values = self.net.evaluate(states[b], actions[b])
                ratio = (new_log_probs - old_log_probs[b]).exp()

                # Clipped surrogate
                surr1 = ratio * advantages[b]
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[b]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(new_values, returns[b])
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self._reset_buffer()
        return {"loss": total_loss}

    def end_episode(self):
        pass  # PPO doesn't decay epsilon

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, data):
        self.net.load_state_dict(data["net"])
