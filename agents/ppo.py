"""PPO Agent (Proximal Policy Optimization) for discrete actions.

Improved: deeper/wider network, value clipping, orthogonal init, reward scaling,
and configurable rollout/entropy for better exploration and stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.base import BaseAgent


def orthogonal_init(m, gain=1.0):
    """Orthogonal initialization for stability; smaller gain for last layer."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ActorCritic(nn.Module):
    """Deeper actor-critic with optional layer norm and proper init."""

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256,
                 n_layers: int = 2, use_layer_norm: bool = True):
        super().__init__()
        layers = []
        in_dim = obs_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m, gain=np.sqrt(2))
        orthogonal_init(self.actor, gain=0.01)  # near-uniform policy at start
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m, gain=1.0)

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

        hidden = config.get("hidden_size", 256)
        n_layers = config.get("n_layers", 2)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.value_clip = config.get("value_clip", 0.2)  # PPO2-style value clipping
        self.epochs = config.get("ppo_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.rollout_len = config.get("rollout_len", 2048)
        self.vf_coef = config.get("vf_coef", 0.25)
        self.ent_coef = config.get("ent_coef", 0.02)
        self.reward_scale = config.get("reward_scale", 0.01)  # scale down large env rewards
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.lr = config.get("lr", 3e-4)
        self.lr_min = config.get("lr_min", 1e-4)
        self.lr_decay = config.get("lr_decay", 1.0)  # per update; 1.0 = no decay

        self.net = ActorCritic(
            obs_size, n_actions, hidden,
            n_layers=n_layers,
            use_layer_norm=config.get("use_layer_norm", True),
        ).to(device)
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            eps=1e-5,
        )
        self._reset_buffer()

    def _reset_buffer(self):
        self.buf_states = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_rewards = []
        self.buf_dones = []
        self.buf_values = []

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
        """Store one transition; reward is scaled for value stability."""
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_log_probs.append(self._last_log_prob)
        self.buf_rewards.append(reward * self.reward_scale)
        self.buf_dones.append(done)
        self.buf_values.append(self._last_value)

    def ready_to_update(self):
        return len(self.buf_states) >= self.rollout_len

    def update(self, next_state=None, **kwargs):
        """Run PPO update with value clipping and normalized advantages."""
        if not self.ready_to_update():
            return {}

        with torch.no_grad():
            if next_state is not None:
                t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, last_val = self.net(t)
                last_val = last_val.item()
            else:
                last_val = 0.0

        # GAE
        advantages = []
        gae = 0.0
        values = self.buf_values + [last_val]
        for t in reversed(range(len(self.buf_rewards))):
            delta = self.buf_rewards[t] + self.gamma * values[t + 1] * (1 - self.buf_dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.buf_dones[t]) * gae
            advantages.insert(0, gae)

        states = torch.FloatTensor(np.array(self.buf_states)).to(self.device)
        actions = torch.LongTensor(self.buf_actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buf_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.buf_values).to(self.device)
        old_values = torch.FloatTensor(self.buf_values).to(self.device)

        # Normalize advantages (critical for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n = len(states)
        for _ in range(self.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                b = idx[start:end]

                new_log_probs, entropy, new_values = self.net.evaluate(states[b], actions[b])
                ratio = (new_log_probs - old_log_probs[b]).exp()

                # Clipped policy loss
                surr1 = ratio * advantages[b]
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[b]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (PPO2): clip new value to old value ± clip
                if self.value_clip > 0:
                    value_clipped = old_values[b] + (new_values - old_values[b]).clamp(
                        -self.value_clip, self.value_clip
                    )
                    vf_loss1 = (new_values - returns[b]).pow(2)
                    vf_loss2 = (value_clipped - returns[b]).pow(2)
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * (new_values - returns[b]).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        # Learning rate decay (helps fine-tune in longer runs)
        if self.lr_decay < 1.0:
            for g in self.optimizer.param_groups:
                g["lr"] = max(self.lr_min, g["lr"] * self.lr_decay)

        self._reset_buffer()
        return {"loss": total_loss}

    def end_episode(self):
        pass

    def state_dict(self):
        return {"net": self.net.state_dict()}

    def load_state_dict(self, data):
        self.net.load_state_dict(data["net"])
