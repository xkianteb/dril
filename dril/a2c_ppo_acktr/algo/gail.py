import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
import gym

from torch.utils.data import DataLoader, TensorDataset

from baselines.common.running_mean_std import RunningMeanStd
from dril.a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, gail_reward_type=None,
                 clip_gail_action=None, envs=None, disc_lr=None):
        super(Discriminator, self).__init__()

        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=disc_lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.reward_type = gail_reward_type
        self.clip_gail_action = clip_gail_action
        self.action_space = envs.action_space

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        if self.clip_gail_action and isinstance(self.action_space, gym.spaces.Box):
            expert_action = torch.clamp(expert_action, self.action_space.low[0], self.action_space.high[0])
            policy_action = torch.clamp(policy_action, self.action_space.low[0], self.action_space.high[0])

        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self,expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            if self.reward_type == 'unbias':
                reward = s.log() - (1 - s).log()
            elif self.reward_type == 'favor_zero_reward':
                reward = reward = s.log()
            elif self.reward_type == 'favor_non_zero_reward':
                reward = - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class DiscriminatorCNN(nn.Module):
    def __init__(self, obs_shape, hidden_dim, num_actions, device, disc_lr,\
                 gail_reward_type=None, envs=None):
        super(DiscriminatorCNN, self).__init__()

        self.device = device

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.num_actions = num_actions
        self.action_emb = nn.Embedding(num_actions, num_actions).cuda()
        num_inputs = obs_shape.shape[0] + num_actions


        self.cnn = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_dim)), nn.ReLU()).to(device)


        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.cnn.train()
        self.trunk.train()

        self.optimizer = torch.optim.Adam(list(self.trunk.parameters()) + list(self.cnn.parameters()), lr=disc_lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

        self.reward_type = gail_reward_type

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)

        '''
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)
        '''

        expert_data = self.combine_states_actions(expert_state, expert_action, detach=True)
        policy_data = self.combine_states_actions(policy_state, policy_action, detach=True)

        alpha = alpha.view(-1, 1, 1, 1).expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(self.cnn(mixup_data))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def combine_states_actions(self, states, actions, detach=False):
        batch_size, height, width = states.shape[0], states.shape[2], states.shape[3]
        action_emb = self.action_emb(actions).squeeze()
        action_emb = action_emb.view(batch_size, self.num_actions, 1, 1).expand(batch_size, self.num_actions, height, width)
        if detach:
            action_emb = action_emb.detach()
        state_actions = torch.cat((states / 255.0, action_emb), dim=1)
        return state_actions

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_data = self.combine_states_actions(policy_state, policy_action)
            policy_d = self.trunk(self.cnn(policy_data))

            expert_state, expert_action = expert_batch

            if obsfilt is not None:
                expert_state = obsfilt(expert_state.numpy(), update=False)
                expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_state = expert_state.to(self.device)

            expert_data = self.combine_states_actions(expert_state, expert_action)

            expert_d = self.trunk(self.cnn(expert_data))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1
            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            policy_data = self.combine_states_actions(state, action)
            d = self.trunk(self.cnn(policy_data))
            s = torch.sigmoid(d)

            if self.reward_type == 'unbias':
                reward = s.log() - (1 - s).log()
            elif self.reward_type == 'favor_zero_reward':
                reward = reward = s.log()
            elif self.reward_type == 'favor_non_zero_reward':
                reward = - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

