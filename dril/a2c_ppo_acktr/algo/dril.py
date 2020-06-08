import os
import numpy as np
import torch
import gym
import pandas as pd

import dril.a2c_ppo_acktr.ensemble_models as ensemble_models
from baselines.common.running_mean_std import RunningMeanStd
from collections import defaultdict

from torch.utils.data import DataLoader, TensorDataset

# This file creates the reward function used by dril. Both reinforcement algorithms
# ppo (line: 102) and a2c (line: 92), have dril bc udpates.

class DRIL:
    def __init__(self, device=None, envs=None, ensemble_policy=None, env_name=None,
        expert_dataset=None, ensemble_size=None, ensemble_quantile_threshold=None,
        dril_bc_model=None, dril_cost_clip=None, num_dril_bc_train_epoch=None,\
        training_data_split=None):

        self.ensemble_quantile_threshold = ensemble_quantile_threshold
        self.dril_cost_clip = dril_cost_clip
        self.device = device
        self.num_dril_bc_train_epoch = num_dril_bc_train_epoch
        self.env_name = env_name
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.observation_space = envs.observation_space

        if envs.action_space.__class__.__name__ == "Discrete":
            self.num_actions = envs.action_space.n
        elif envs.action_space.__class__.__name__ == "Box":
            self.num_actions = envs.action_space.shape[0]
        elif envs.action_space.__class__.__name__ == "MultiBinary":
            self.num_actions = envs.action_space.shape[0]

        self.ensemble_size = ensemble_size
        # use full data since we don't use a validation set
        self.trdata = expert_dataset.load_demo_data(1.0, 1, self.ensemble_size)['trdata']

        self.ensemble = ensemble_policy
        self.bc = dril_bc_model
        self.bc.num_batches = num_dril_bc_train_epoch
        self.clip_variance = self.policy_variance(envs=envs)

    def policy_variance(self, q=0.98, envs=None):
        q   = self.ensemble_quantile_threshold
        obs = None
        acs = None

        variance = defaultdict(lambda:[])
        for batch_idx, batch in enumerate(self.trdata):
            (state, action) = batch
            action = action.float().to(self.device)

            # Image observation
            if len(self.observation_space.shape) == 3:
                state = state.repeat(self.ensemble_size, 1,1,1).float().to(self.device)
            # Feature observations
            else:
                state = state.repeat(self.ensemble_size, 1).float().to(self.device)

            if isinstance(envs.action_space, gym.spaces.discrete.Discrete):
                # Note: this is just a place holder
                action_idx = int(action.item())
                one_hot_action = torch.FloatTensor(np.eye(self.num_actions)[int(action.item())])
                action = one_hot_action
            elif envs.action_space.__class__.__name__ == "MultiBinary":
                # create unique id for each combination
                action_idx = int("".join(str(int(x)) for x in action[0].tolist()), 2)
            else:
                action_idx = 0

            with torch.no_grad():
                ensemble_action = self.ensemble(state).squeeze()
            if isinstance(envs.action_space, gym.spaces.Box):
                action = torch.clamp(action, envs.action_space.low[0], envs.action_space.high[0])

                ensemble_action = torch.clamp(ensemble_action, envs.action_space.low[0],\
                                            envs. action_space.high[0])

            cov = np.cov(ensemble_action.T.cpu().numpy())
            action = action.cpu().numpy()

            # If the env has only one action then we need to reshape cov
            if envs.action_space.__class__.__name__ == "Box":
                if envs.action_space.shape[0] == 1:
                    cov = cov.reshape(-1,1)

            #variance.append(np.matmul(np.matmul(action, cov), action.T).item())
            if isinstance(envs.action_space, gym.spaces.discrete.Discrete):
                for action_idx in range(envs.action_space.n):
                    one_hot_action = torch.FloatTensor(np.eye(self.num_actions)[action_idx])
                    variance[action_idx].append(np.matmul(np.matmul(one_hot_action, cov), one_hot_action.T).item())
            else:
                variance[action_idx].append(np.matmul(np.matmul(action, cov), action.T).item())


        quantiles = {key: np.quantile(np.array(variance[key]), q) for key in list(variance.keys())}
        if self.dril_cost_clip == '-1_to_1':
            return {key: lambda x: -1 if x > quantiles[key] else 1 for key in list(variance.keys())}
        elif self.dril_cost_clip == 'no_clipping':
            return {key: lambda x: x for i in list(variance.keys())}
        elif self.dril_cost_clip == '-1_to_0':
            return {key: lambda x: -1 if x > quantiles[key] else 0 for key in list(variance.keys())}

    def predict_reward(self, actions, states, envs):
        rewards = []
        for idx in range(actions.shape[0]):

            # Image observation
            if len(self.observation_space.shape) == 3:
                state = states[[idx]].repeat(self.ensemble_size, 1,1,1).float().to(self.device)
            # Feature observations
            else:
                state = states[[idx]].repeat(self.ensemble_size, 1).float().to(self.device)

            if isinstance(envs.action_space, gym.spaces.discrete.Discrete):
                one_hot_action = torch.FloatTensor(np.eye(self.num_actions)[int(actions[idx].item())])
                action = one_hot_action
                action_idx = int(actions[idx].item())
            elif isinstance(envs.action_space, gym.spaces.Box):
                action = actions[[idx]]
                action_idx = 0
            elif isinstance(envs.action_space, gym.spaces.MultiBinary):
                raise Exception('Envrionment shouldnt be MultiBinary')
            else:
                raise Exception("Unknown Action Space")

            with torch.no_grad():
                ensemble_action = self.ensemble(state).squeeze().detach()

            if isinstance(envs.action_space, gym.spaces.Box):
                action = torch.clamp(action, envs.action_space.low[0], envs.action_space.high[0])
                ensemble_action = torch.clamp(ensemble_action, envs.action_space.low[0],\
                                            envs. action_space.high[0])

            cov = np.cov(ensemble_action.T.cpu().numpy())
            action = action.cpu().numpy()

            # If the env has only one action then we need to reshape cov
            if envs.action_space.__class__.__name__ == "Box":
                if envs.action_space.shape[0] == 1:
                    cov = cov.reshape(-1,1)

            ensemble_variance = (np.matmul(np.matmul(action, cov), action.T).item())

            if action_idx in self.clip_variance:
                reward = self.clip_variance[action_idx](ensemble_variance)
            else:
                reward = -1
            rewards.append(reward)
        return torch.FloatTensor(np.array(rewards)[np.newaxis].T)

    def normalize_reward(self, state, action, gamma, masks, reward, update_rms=True):
        if self.returns is None:
            self.returns = reward.clone()

        if update_rms:
            self.returns = self.returns * masks * gamma + reward
            self.ret_rms.update(self.returns.cpu().numpy())

        return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

    def bc_update(self):
        for dril_epoch in range(self.num_dril_bc_train_epoch):
            dril_train_loss = self.bc.update(update=True, data_loader_type='train')
