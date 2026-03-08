#Code adapted form Phil Tabor's Youtube video https://www.youtube.com/watch?v=ioidsRlf79o

import os
import torch as T
import torch.nn.functional as F
import numpy as np

from agents.sac.buffer import ReplayBuffer
from agents.sac.networks import ActorNetwork, CriticNetwork, ValueNetwork
from agents.base import BaseAgent

class SACAgent(BaseAgent):
    def __init__(self, obs_size, n_actions, device, config):
        super().__init__(obs_size, n_actions, device, config)
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.memory = ReplayBuffer(config['memory_size'], obs_size, n_actions)
        self.batch_size = config['batch_size']
        self.n_actions = n_actions
        self.device = device

        self.actor = ActorNetwork(config['alpha'], obs_size, n_actions=n_actions).to(self.device)
        self.critic_1 = CriticNetwork(config['lr'], obs_size, n_actions=n_actions).to(self.device)
        self.critic_2 = CriticNetwork(config['lr'], obs_size, n_actions=n_actions).to(self.device)
        self.value = ValueNetwork(config['lr'], obs_size).to(self.device)
        self.target_value = ValueNetwork(config['lr'], obs_size).to(self.device)

        self.scale = config['reward_scale']
        self.update_network_parameters(tau=1)

        self.steps = 0

    def select_action(self, observation, training=True):
        #state = T.Tensor([observation]).to(self.device)
        state = T.as_tensor([observation], dtype=T.float32, device=self.device)
        with T.no_grad():
           action,_ = self.actor.sample(state)
           if not training:
              action = np.argmax(action.numpy())
        return action

    def update_network_parameters(self, tau=None):
        if tau is None:
           tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau* value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)

    def update(self, state, action, reward, next_state, done, **kwargs):
        self.memory.store_transition(state, action, reward, next_state, done)

        self.steps += 1

        if self.memory.mem_cntr < self.batch_size:
           return

        state, new_state, action, reward, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        # Training the Value network
        actions,log_probs = self.actor.sample(state)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs.squeeze()
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Training the Actor network
        actions, log_probs = self.actor.sample(state)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.actor.optimizer.zero_grad()
        actor_loss = log_probs.squeeze() - critic_value
        actor_loss = T.mean(actor_loss)
        actor_loss.backward(retain_graph = True)
        self.actor.optimizer.step()

        # Training Critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def end_episode(self):
        pass

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "value": self.value.state_dict(),
            "target_value" : self.target_value.state_dict(),
         }

    def load_state_dict(self, data):
        self.actor.load_state_dict(data["actor"])
        self.critic_1.load_state_dict(data["critic_1"])
        self.critic_2.load_state_dict(data["critic_2"])
        self.value.load_state_dict(data["value"])
        self.target_value.load_state_dict(data["target_value"])
