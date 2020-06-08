import os

import gym
from gym.wrappers import TimeLimit
import numpy as np
import torch
from gym.spaces.box import Box
import pickle

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame, ClipRewardEnv, FrameStack, ScaledFloatFrame
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
from baselines.common.retro_wrappers import make_retro #, wrap_deepmind_retro

from dril.a2c_ppo_acktr.stable_baselines.base_vec_env import VecEnvWrapper
from dril.a2c_ppo_acktr.stable_baselines.running_mean_std import RunningMeanStd

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

env_hyperparam = ['BipedalWalkerHardcore-v2', 'BipedalWalker-v2',\
                  'HalfCheetahBulletEnv-v0', 'HopperBulletEnv-v0',\
                  'HumanoidBulletEnv-v0', 'MinitaurBulletEnv-v0',\
                  'MinitaurBulletDuckEnv-v0', 'Walker2DBulletEnv-v0',\
                  'AntBulletEnv-v0', 'LunarLanderContinuous-v2',
                  'CartPole-v1','Acrobot-v1', 'Pendulum-v0', 'MountainCarContinuous-v0',
                  'CartPoleContinuousBulletEnv-v0','ReacherBulletEnv-v0']

retro_envs = ['SuperMarioKart-Snes', 'StreetFighterIISpecialChampionEdition-Genesis',\
              'AyrtonSennasSuperMonacoGPII-Genesis']

def make_env(env_id, seed, rank, log_dir, allow_early_resets, time=False, max_steps=None):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id in ['duckietown']:
            from a2c_ppo_acktr.duckietown.env import launch_env
            from a2c_ppo_acktr.duckietown.wrappers import NormalizeWrapper, ImgWrapper,\
                 DtRewardWrapper, ActionWrapper, ResizeWrapper
            from a2c_ppo_acktr.duckietown.teacher import PurePursuitExpert
            env = launch_env()
            env = ResizeWrapper(env)
            env = NormalizeWrapper(env)
            env = ImgWrapper(env)
            env = ActionWrapper(env)
            env = DtRewardWrapper(env)
        elif env_id in retro_envs:
            env = make_retro(game=env_id)
            #env = SuperMarioKartDiscretizer(env)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id, max_episode_steps=max_steps)

        env.seed(seed + rank)

        #TODO: Figure out what todo here
        if is_atari:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif env_id in retro_envs:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind_retro(env, frame_stack=0)
        elif len(env.observation_space.shape) == 3:
            if env_id not in ['duckietown'] and env_id not in retro_envs:
                raise NotImplementedError(
                    "CNN models work only for atari,\n"
                    "please use a custom wrapper for a custom pixel input env.\n"
                    "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        if env_id not in ['duckietown']:
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

        if time:
            env = TimeFeatureWrapper(env)

        return env

    return _thunk

def wrap_deepmind_retro(env, scale=True, frame_stack=0):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in wrap_deepmind
    """
    env = WarpFrame(env, grayscale=False)
    env = ClipRewardEnv(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if scale:
        env = ScaledFloatFrame(env)
    return env

class SuperMarioKartDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SuperMarioKartDiscretizer, self).__init__(env)
        buttons = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
        actions = [['B'], ['B', 'LEFT', 'R'], ['LEFT'], ['B', 'LEFT'], ['B', 'RIGHT'], ['B', 'DOWN', 'LEFT'], ['DOWN', 'RIGHT'], ['RIGHT'], ['DOWN', 'LEFT'], ['RIGHT', 'A'], ['A'], [], ['B', 'R'], ['LEFT', 'A'], ['B', 'UP', 'RIGHT'], ['B', 'RIGHT', 'R'], ['B', 'DOWN'], ['B', 'DOWN', 'RIGHT'], ['B', 'UP', 'LEFT'], ['DOWN', 'RIGHT', 'A'], ['B', 'UP', 'LEFT', 'R'], ['B', 'RIGHT', 'A'], ['B', 'LEFT', 'A'], ['DOWN', 'LEFT', 'A'], ['B', 'A'], ['R']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        try:
            assert(len(a) == 1)
            return self._actions[a[0]].copy()
        except:
            return self._actions[a].copy()


#TODO: Set max_steps as a hyperparameter
def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  max_steps=100000,
                  num_frame_stack=None,
                  stats_path=None,
                  hyperparams=None,
                  training=False,
                  norm_obs=False,
                  time=False,
                  use_obs_norm=False):

    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, time=time, max_steps=max_steps)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if env_name in env_hyperparam and hyperparams is not None:
        if stats_path is not None:
            if hyperparams['normalize']:
                print("Loading running average")
                print("with params: {}".format(hyperparams['normalize_kwargs']))
                envs = VecNormalizeBullet(envs, training=False, **hyperparams['normalize_kwargs'])
                envs.load_running_average(stats_path)
    else:
        if len(envs.observation_space.shape) == 1:
            if gamma is None:
                envs = VecNormalize(envs, ret=False, ob=use_obs_norm)
            else:
                envs = VecNormalize(envs, gamma=gamma, ob=use_obs_norm)

    envs = VecPyTorch(envs, device)

    if env_name not in ['duckietown']:
        if num_frame_stack is not None:
            envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
        elif len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        #self.stacked_obs[:, :-self.shape_dim0] = \
        #    self.stacked_obs[:, self.shape_dim0:]
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


# Code taken from stable-baselines
class VecNormalizeBullet(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,
    :param venv: (VecEnv) the vectorized environment to wrap
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_reward: (bool) Whether to normalize rewards or not (default: True)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(self, venv, training=True, norm_obs=True, norm_reward=False,
                 clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if isinstance(self.venv.envs[0], TimeFeatureWrapper):
            # Remove index corresponding to time
            self.old_obs = obs[:,:-1]
        else:
            self.old_obs = obs

        obs = self._normalize_observation(obs)
        if self.norm_reward:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        self.ret[news] = 0
        return obs, rews, news, infos

    def _normalize_observation(self, obs):
        """
        :param obs: (numpy tensor)
        """
        if self.norm_obs:
            if self.training:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
            return obs
        else:
            return obs

    def get_original_obs(self):
          """
          returns the unnormalized observation
          :return: (numpy float)
          """
          return self.old_obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        if len(np.array(obs).shape) == 1:  # for when num_cpu is 1
            #self.old_obs = [obs]
            if isinstance(self.venv.envs[0], TimeFeatureWrapper):
                # Remove index corresponding to time
                self.old_obs = [obs[:,:-1]]
            else:
                self.old_obs = [obs]
        else:
            #self.old_obs = obs
            if isinstance(self.venv.envs[0], TimeFeatureWrapper):
                # Remove index corresponding to time
                self.old_obs = obs[:,:-1]
            else:
                self.old_obs = obs

        self.ret = np.zeros(self.num_envs)
        return self._normalize_observation(obs)

    def save_running_average(self, path):
        """
        :param path: (str) path to log dir
        """
        for rms, name in zip([self.obs_rms], ['obs_rms']):
            with open("{}/{}.pkl".format(path, name), 'wb') as file_handler:
                pickle.dump(rms, file_handler)

    def load_running_average(self, path):
        """
        :param path: (str) path to log dir
        """
        #for name in ['obs_rms', 'ret_rms']:
        for name in ['obs_rms']:
            with open("{}/{}.pkl".format(path, name), 'rb') as file_handler:
                setattr(self, name, pickle.load(file_handler))


# Code taken from stable-baslines
class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode
        self.untimed_obs = None

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def get_original_obs(self):
          """
          returns the unnormalized observation
          :return: (numpy float)
          """
          return  self.untimed_obs[np.newaxis,:]

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        self.untimed_obs = obs
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))
