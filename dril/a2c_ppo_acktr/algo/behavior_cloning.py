# prerequisites
import copy
import glob
import sys
import os
import time
from collections import deque

import gym

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class BehaviorCloning:
    def __init__(self, policy,  device, batch_size=None, lr=None, expert_dataset=None,
         num_batches=np.float('inf'), training_data_split=None, envs=None, ensemble_size=None):
        super(BehaviorCloning, self).__init__()

        self.actor_critic = policy

        self.optimizer  = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size

        datasets = expert_dataset.load_demo_data(training_data_split, batch_size, ensemble_size)
        self.trdata = datasets['trdata']
        self.tedata = datasets['tedata']

        self.num_batches = num_batches
        self.action_space = envs.action_space

    def update(self, update=True, data_loader_type=None):
        if data_loader_type == 'train':
            data_loader = self.trdata
        elif data_loader_type == 'test':
            data_loader = self.tedata
        else:
            raise Exception("Unknown Data loader specified")

        total_loss = 0
        for batch_idx, batch in enumerate(data_loader, 1):
            self.optimizer.zero_grad()
            (states, actions) = batch
            expert_states  = states.float().to(self.device)
            expert_actions = actions.float().to(self.device)

            dynamic_batch_size = expert_states.shape[0]
            try:
                # Regular Behavior Cloning
                pred_actions = self.actor_critic.get_action(expert_states).view(dynamic_batch_size, -1)
            except AttributeError:
                # Ensemble Behavior Cloning
                pred_actions = self.actor_critic(expert_states).view(dynamic_batch_size, -1)

            if isinstance(self.action_space, gym.spaces.Box):
                pred_actions = torch.clamp(pred_actions, self.action_space.low[0],self.action_space.high[0])
                expert_actions = torch.clamp(expert_actions.float(), self.action_space.low[0],self.action_space.high[0])
                loss = F.mse_loss(pred_actions, expert_actions)
            elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
                loss = F.cross_entropy(pred_actions, expert_actions.flatten().long())
            elif self.action_space.__class__.__name__ == "MultiBinary":
                loss = torch.binary_cross_entropy_with_logits(pred_actions, expert_actions).mean()

            if update:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            if batch_idx >= self.num_batches:
                break

        return (total_loss / batch_idx)

    def reset(self):
        self.optimizer  = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

