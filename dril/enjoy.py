import argparse
import os
# workaround to unpickle olf model files
import sys
import time
import numpy as np
from PIL import Image
import glob
import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
from skimage import transform
from torch.distributions import Categorical

import numpy as np
import torch
from dril.a2c_ppo_acktr.model import Policy

from dril.a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from dril.a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from dril.a2c_ppo_acktr.algo.ensemble import Ensemble
import dril.a2c_ppo_acktr.ensemble_models as ensemble_models
from dril.a2c_ppo_acktr.arguments import get_args
from dril.a2c_ppo_acktr.algo.behavior_cloning import BehaviorCloning
from dril.a2c_ppo_acktr.algo.dril import DRIL

import gym, os
import numpy as np
import argparse
import random
import pandas as pd

import sys
import torch
from gym import wrappers
import random

from dril.a2c_ppo_acktr.envs import make_vec_envs
from dril.a2c_ppo_acktr.model import Policy
from dril.a2c_ppo_acktr.utils import get_saved_hyperparams
from dril.a2c_ppo_acktr.arguments import get_args


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--recurrent-policy',
    action='store_true',
    default=False,
    help='use a recurrent policy')
parser.add_argument(
    '--rl_baseline_zoo_dir', 
    type=str, default='', help='directory of rl baseline zoo')
parser.add_argument(
    '--ensemble_hidden_size', default=512,
    help='dril ensemble network number of hidden units (default: 512)')
parser.add_argument(
    '--ensemble_size', type=int, default=5,
    help='numnber of polices in the ensemble (default: 5)')
args, unknown = parser.parse_known_args()

default_args = get_args()

args.det = not args.non_det

device='cpu'
env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic = Policy(
       env.observation_space.shape,
       env.action_space,
       load_expert=False,
       env_name=args.env_name,
       rl_baseline_zoo_dir=args.rl_baseline_zoo_dir,
       expert_algo='a2c',
       base_kwargs={'recurrent': args.recurrent_policy})


try:
    actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir, args.env_name + ".pt"), map_location='cpu')
except:
    params = \
        torch.load(os.path.join(args.load_dir, args.env_name + ".pt"), map_location='cpu')
    actor_critic.load_state_dict(params)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = None

#recurrent_hidden_states = torch.zeros(1,
#                                      actor_critic.recurrent_hidden_state_size)
#masks = torch.zeros(1, 1)

obs = env.reset()

if render_func is not None:
    import gym, os

#if args.env_name.find('Bullet') > -1:
#    import pybullet as p
#
#    torsoId = -1
#        if (p.getBodyInfo(i)[0].decode() == "torso"):
#            torsoId = i

num_inputs  = env.observation_space.shape[0]
try:
    num_actions = env.action_space.n
except:
    num_actions = env.action_space.shape[0]

ensemble_args = (num_inputs, num_actions, args.ensemble_hidden_size, args.ensemble_size)
#if num_inputs) == 3:
#if env_name in ['duckietown']:
#    ensemble_policy = ensemble_models.PolicyEnsembleDuckieTownCNN
#elif uncertainty_reward == 'ensemble':
ensemble_policy = ensemble_models.PolicyEnsembleCNN
#elif uncertainty_reward == 'dropout':
#    ensemble_policy = ensemble_models.PolicyEnsembleCNNDropout
#else:
#    raise Exception("Unknown uncertainty_reward type")
ensemble_policy = ensemble_policy(*ensemble_args).to(device)
best_test_params = torch.load('', map_location=device)
ensemble_policy.load_state_dict(best_test_params)

#if save_traces:
if True:
    traces, u_rewards_raw, u_rewards_quant, actions = [], [], [], []
    variance = np.load('')
    quantile = np.quantile(np.array(variance), .40)
    clip = (lambda x: -1 if x > quantile else 1)

step=0
while True:
    with torch.no_grad():
        value, action, _, _ = actor_critic.act(obs, None, None, deterministic=args.det)

        state = obs.repeat(args.ensemble_size, 1,1,1).float().to(device)
        ensemble_action = ensemble_policy(state).squeeze().detach()

    if isinstance(env.action_space, gym.spaces.Box):
        action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
        ensemble_action = torch.clamp(ensemble_action, env.action_space.low[0],\
                          env. action_space.high[0])

    #action = ensemble_action[[4]].squeeze().max(0)[1].unsqueeze(0).unsqueeze(0)

    cov = np.cov(ensemble_action.T.cpu().numpy())

    # If the env has only one action then we need to reshape cov
    if env.action_space.__class__.__name__ == "Box":
        if env.action_space.shape[0] == 1:
            cov = cov.reshape(-1,1)

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        one_hot_action = torch.FloatTensor(np.eye(num_actions)[int(action.item())])                   
        action_vec = one_hot_action
    elif isinstance(env.action_space, gym.spaces.Box):
        action_vec = action.clone()
    elif isinstance(env.action_space, gym.spaces.MultiBinary):
        #action = actions[[idx]]
        raise Exception('Envrionment shouldnt be MultiBinary')
    else:
        raise Exception("Unknown Action Space")
    
    ensemble_variance = (np.matmul(np.matmul(action_vec, cov), action_vec.T).item())
    print(f'step: {step} ensemble_variance[{action.item()}]: {ensemble_variance} u:{clip(ensemble_variance)}')
    step+=1

    traces.append(obs[0][:3].permute(1,2,0).cpu().numpy().copy())
    u_rewards_raw.append(ensemble_variance)
    actions.append(action.item())
    u_rewards_quant.append(clip(ensemble_variance))

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)


   # masks.fill_(0.0 if done else 1.0)
    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if done:break
    #if step > 1200:
    #    import pdb; pdb.set_trace()

    if render_func is not None:
        render_func('human')

traces_dir = 'video'
os.system(f'mkdir -p {traces_dir}')
for i in range(len(traces)):
    fname = f'{traces_dir}/im{i:05d}.png'#_uq{u_rewards_quant[i]}.png'
    #else:
    #    fname = f'{traces_dir}/im{i:05d}.png'
    #scipy.misc.imsave(fname, traces[i][0].cpu().numpy())
    #im = Image.fromarray(traces[i]*255.999)  
    #im = transform.resize(traces[i].reshape(84,84,3),(252,252))
    #im = transform.resize(traces[i].permute(1,2,0)),(252,252))
    img = Image.fromarray((traces[i]* 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.text((3,3), f"q:{u_rewards_quant[i]} a:{actions[i]} ", fill=(255,255,0))
    img.save(fname)
    
import imageio
images = []
import pdb; pdb.set_trace()
for filename in sorted(glob.glob(f'{traces_dir}/*.png')):
    img = imageio.imread(filename)
    images.append(img)
imageio.mimsave('output.gif', images, fps=10)
