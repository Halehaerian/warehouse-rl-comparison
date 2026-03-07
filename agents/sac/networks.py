#Code derived form Phil Tabor's Youtube video https://www.youtube.com/watch?v=ioidsRlf79o

import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch .distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='../../models/sac'):
        # This neural network has input > fully connected layer 1 > ReLu Activation > fully connected layer 2 > ReLu Activation > output

        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.criticNN = nn.Sequential(nn.Linear(self.input_dims[0]+self.n_actions, self.fc1_dims), nn.ReLU(),
                                      nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(),
                                      nn.Linear(self.fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        return self.criticNN(T.cat([state,action], dim=1))

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='../../models/sac'):
        # This neural network has input > fully connected layer 1 > ReLu Activation > fully connected layer 2 > ReLu Activation > output

        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.valueNN = nn.Sequential(nn.Linear(*self.input_dims, self.fc1_dims), nn.ReLU(),
                                     nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(),
                                     nn.Linear(self.fc2_dims, 1))

        self.optimizer = optim.Adam(self.parameters(), lr = beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        return self.valueNN(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='actor', checkpt_dir='../../models/sac'):
        # This neural network has input > fully connected layer 1 > ReLu Activation > fully connected layer 2 > ReLu Activation > output > Softmax

        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.actorNN = nn.Sequential(nn.Linear(*self.input_dims, self.fc1_dims), nn.ReLU(),
                                     nn.Linear(self.fc1_dims, self.fc2_dims), nn.ReLU(),
                                     nn.Linear(self.fc2_dims, self.n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        return self.actorNN(state)

    def sample(self, state):
        actor_out = self.forward(state)
        action = F.gumbel_softmax(actor_out, tau=1.0, hard=True, dim=-1)
        log_probs = F.log_softmax(actor_out, dim=1)
        log_probs_sel = action.detach() * log_probs
        return action, log_probs_sel.sum(dim=-1, keepdim=True)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


