#Code adapted form Phil Tabor's Youtube video https://www.youtube.com/watch?v=ioidsRlf79o

import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch .distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        # This neural network has input > fully connected layer 1 > ReLu Activation > fully connected layer 2 > ReLu Activation > output

        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.criticNN = nn.Sequential(nn.Linear(self.input_dims+self.n_actions, self.fc1_dims), nn.ReLU(),
                                      nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(),
                                      nn.Linear(self.fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr = beta)

    def forward(self, state, action):
        return self.criticNN(T.cat([state,action], dim=1))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256):
        # This neural network has input > fully connected layer 1 > ReLu Activation > fully connected layer 2 > ReLu Activation > output

        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.valueNN = nn.Sequential(nn.Linear(self.input_dims, self.fc1_dims), nn.ReLU(),
                                     nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(),
                                     nn.Linear(self.fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr = beta)

    def forward(self, state):
        return self.valueNN(state)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        # This neural network has input > fully connected layer 1 > ReLu Activation > fully connected layer 2 > ReLu Activation > output > Softmax

        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.actorNN = nn.Sequential(nn.Linear(self.input_dims, self.fc1_dims), nn.ReLU(),
                                     nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(),
                                     nn.Linear(self.fc2_dims, self.n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)

    def forward(self, state):
        return self.actorNN(state)

    def sample(self, state):
        actor_out = self(state)
        action = F.gumbel_softmax(actor_out, tau=1.0, hard=True, dim=-1)
        log_probs = F.log_softmax(actor_out, dim=1)
        log_probs_sel = action.detach() * log_probs
        return action, log_probs_sel.sum(dim=-1, keepdim=True)

