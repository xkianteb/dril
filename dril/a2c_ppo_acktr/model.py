#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

# Convert weights from tensorflow to pytorch
def copy_mlp_weights(baselines_model):
    model_params = baselines_model.get_parameters()

    params = {
        'base.actor.0.weight':model_params['model/pi_fc0/w:0'].T,
        'base.actor.0.bias':model_params['model/pi_fc0/b:0'].squeeze(),
        'base.actor.2.weight':model_params['model/pi_fc1/w:0'].T,
        'base.actor.2.bias':model_params['model/pi_fc1/b:0'].squeeze(),
        'base.critic.0.weight':model_params['model/vf_fc0/w:0'].T,
        'base.critic.0.bias':model_params['model/vf_fc0/b:0'].squeeze(),
        'base.critic.2.weight':model_params['model/vf_fc1/w:0'].T,
        'base.critic.2.bias':model_params['model/vf_fc1/b:0'].squeeze(),
        'base.critic_linear.weight':model_params['model/vf/w:0'].T,
        'base.critic_linear.bias':model_params['model/vf/b:0'],
        'dist.fc_mean.weight':model_params['model/pi/w:0'].T,
        'dist.fc_mean.bias':model_params['model/pi/b:0'],
        'dist.logstd._bias':model_params['model/pi/logstd:0'].T
    }

    for key in params.keys():
        params[key] = torch.tensor(params[key])
    return params

def copy_cnn_weights(baselines_model):
    model_params = baselines_model.get_parameters()

      # Convert images to torch format
    def conv_to_torch(obs):
        obs = np.transpose(obs, (3, 2, 0, 1))
        return obs

    params = {
        'base.conv1.weight':conv_to_torch(model_params['model/c1/w:0']),
        'base.conv1.bias':conv_to_torch(model_params['model/c1/b:0']).squeeze(),
        'base.conv2.weight':conv_to_torch(model_params['model/c2/w:0']),
        'base.conv2.bias':conv_to_torch(model_params['model/c2/b:0']).squeeze(),
        'base.conv3.weight':conv_to_torch(model_params['model/c3/w:0']),
        'base.conv3.bias':conv_to_torch(model_params['model/c3/b:0']).squeeze(),
        'base.fc1.weight': model_params['model/fc1/w:0'].T,
        'base.fc1.bias': model_params['model/fc1/b:0'].squeeze(),
        'base.critic_linear.weight': model_params['model/vf/w:0'].T,
        'base.critic_linear.bias': model_params['model/vf/b:0'],
        'dist.linear.weight': model_params['model/pi/w:0'].T,
        'dist.linear.bias': model_params['model/pi/b:0'].squeeze()
    }

    for key in params.keys():
        params[key] = torch.tensor(params[key])
    return params


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, load_expert=None,
                 env_name=None, rl_baseline_zoo_dir=None, expert_algo=None, normalize=True):
        super(Policy, self).__init__()

        #TODO: Pass these parameters in
        self.epsilon = 0.1
        self.dril = True

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if env_name in  ['duckietown']:
                base = DuckieTownCNN
            elif len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], normalize=normalize, **base_kwargs)
        self.action_space = None
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
            self.action_space = "Discrete"
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
            self.action_space = "Box"
        elif action_space.__class__.__name__ == "MultiBinary":
            raise Exception('Error')
        else:
            raise NotImplementedError

        if load_expert == True and env_name not in ['duckietown', 'highway-v0']:
            print('[Loading Expert --- Base]')
            model_path = os.path.join(rl_baseline_zoo_dir, 'trained_agents', f'{expert_algo}')
            try:
                import mpi4py
                from stable_baselines import TRPO
            except ImportError:
                mpi4py = None
                DDPG, TRPO = None, None

            from stable_baselines import PPO2

            model_path = f'{model_path}/{env_name}.pkl'
            if env_name in ['AntBulletEnv-v0']:
                baselines_model = TRPO.load(model_path)
            else:
                baselines_model = PPO2.load(model_path)
            for key, value in baselines_model.get_parameters().items():
                print(key, value.shape)

            if base.__name__ == 'CNNBase':
                print(['Loading CNNBase expert model'])
                params = copy_cnn_weights(baselines_model)
            elif load_expert == True and base.__name__ == 'MLPBase':
                print(['Loading MLPBase expert model'])
                params = copy_mlp_weights(baselines_model)

            #TODO: I am not sure what this is doing
            try:
                self.load_state_dict(params)
                self.obs_shape = obs_shape[0]
            except:
                self.base = base(obs_shape[0]+ 1, **base_kwargs)
                self.load_state_dict(params)
                self.obs_shape = obs_shape[0] +1


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_action(self, inputs, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, None, None)#, rnn_hxs, masks)
        if self.action_space == "Discrete":
            return self.dist.get_logits(actor_features)
        elif self.action_space == "MultiBinary":
            return self.dist.get_logits(actor_features)
        elif self.action_space == "Box":
            return self.dist.get_mean(actor_features)



    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if (self.dril and random.random() <= self.epsilon) or deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, normalize=True):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = (nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = (nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = (nn.Conv2d(64, 64, 3, stride=1))
        self.fc1  = (nn.Linear(32*7*7*2, hidden_size))
        self.relu = nn.ReLU()
        self.flatten = Flatten()
        self.critic_linear = (nn.Linear(hidden_size, 1))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = (nn.Linear(hidden_size, 1))
        self.normalize = normalize

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        if self.normalize:
            x = (inputs/ 255.0)
        else:
            x = inputs
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.flatten(x)
        x = self.relu(self.fc1(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, normalize=None):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

# https://github.com/duckietown/gym-duckietown/blob/master/learning/imitation/iil-dagger/model/squeezenet.py
class DuckieTownCNN(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(DuckieTownCNN, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, hidden_size)


        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = (inputs/255.0)
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(x), x, rnn_hxs

