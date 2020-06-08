import numpy as np
import torch
import gym

from dril.a2c_ppo_acktr import utils
from dril.a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, num_episodes=None, atari_max_steps=None):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, atari_max_steps)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        if isinstance(eval_envs.action_space, gym.spaces.Box):
            clip_action = torch.clamp(action, float(eval_envs.action_space.low[0]),\
                         float(eval_envs.action_space.high[0]))
        else:
            clip_action = action

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(clip_action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return np.mean(eval_episode_rewards)
