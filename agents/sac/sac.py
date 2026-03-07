#Code derived form Phil Tabor's Youtube video https://www.youtube.com/watch?v=ioidsRlf79o

import os
import torch as T
import torch.nn.functional as F
import numpy as np

from agents.sac.buffer import ReplayBuffer
from agents.sac.networks import ActorNetwork, CriticNetwork, ValueNetwork


class SACAgent():
    def __init__(self, n_actions, config, input_dims):
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.memory = ReplayBuffer(config['memory_size'], input_dims, n_actions)
        self.batch_size = config['batch_size']
        self.n_actions = n_actions

        self.actor = ActorNetwork(config['alpha'], input_dims, n_actions=n_actions, name='actor')
        self.critic_1 = CriticNetwork(config['lr'], input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(config['lr'], input_dims, n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(config['lr'], input_dims, name='value')
        self.target_value = ValueNetwork(config['lr'], input_dims, name='target_value')

        self.scale = config['reward_scale']
        self.update_network_parameters(tau=1)

    def select_action(self, observation, training=True):
        if training:
           state = T.Tensor([observation]).to(self.actor.device)
           action,_ = self.actor.sample(state)
           return action 

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

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

    def save(self, location): #We ignore location here.
        print('....saving models....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('....loading models....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def update(self):
        if self.memory.mem_cntr < self.batch_size:
           return

        state, new_state, action, reward, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
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

    def end_episode(self):
        pass
