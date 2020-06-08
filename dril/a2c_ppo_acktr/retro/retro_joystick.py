#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")
import sys, time, pdb
import numpy as np
import os
import pygame
from itertools import count
import argparse
import pandas as pd
import uuid
from pygame.locals import *
from pygame_controller import  InputManager, wait_for_enter, configure_phase
import retro

parser = argparse.ArgumentParser()
parser.add_argument(
  '--env_name',
  choices=['SuperMarioKart-Snes', 'StreetFighterIISpecialChampionEdition-Genesis', 'AyrtonSennasSuperMonacoGPII-Genesis'],
  default='AyrtonSennasSuperMonacoGPII-Genesis' )
parser.add_argument(
  '--fps',
  default=30)
parser.add_argument('--state', default=retro.State.DEFAULT)
args = parser.parse_args()

def interaction_phase(env, input_manager):
    buttons = env.unwrapped.envs[0].unwrapped.buttons
    actions = np.zeros(12)

    for event in input_manager.get_events():
        if event.key == 'A' and event.down:
            pass # weeeeeeee
        if event.key == 'X' and event.up:
            input_manager.quit_attempted = True

    if input_manager.is_pressed('left'):
        actions[buttons.index("LEFT")] = 1
    if input_manager.is_pressed('right'):
        actions[buttons.index("RIGHT")] = 1
    if input_manager.is_pressed('up'):
        actions[buttons.index("UP")] = 1
    if input_manager.is_pressed('down'):
        actions[buttons.index("DOWN")] = 1
    if input_manager.is_pressed('X'):
        actions[buttons.index("X")] = 1
    if input_manager.is_pressed('A'):
        actions[buttons.index("A")] = 1
    if input_manager.is_pressed('B'):
        actions[buttons.index("B")] = 1
    if input_manager.is_pressed('Y'):
        actions[buttons.index("Y")] = 1
    if input_manager.is_pressed('L'):
        actions[buttons.index("L")] = 1
    if input_manager.is_pressed('R'):
        actions[buttons.index("R")] = 1
    return actions

def rollout(env, input_manager):
    import torch
    rtn_obs, rtn_acs, rtn_lens, ep_rewards = [], [], [], []
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0

    while 1:
        start = time.time()
        action = interaction_phase(env, input_manager)
        total_timesteps += 1

        rtn_obs.append(obser.cpu().numpy().copy())
        rtn_acs.append([action])
        obser, env_reward, done, infos = env.step(torch.tensor(action))

        for info in infos or done:
            if 'episode' in info.keys():
                ep_rewards.append(info['episode']['r'])

        total_reward += env_reward
        window_still_open = env.render()

        if done: break
        # maintain frame rate
        difference = start - time.time()
        delay = 1.0 / args.fps - difference
        if delay > 0:
            time.sleep(delay)

    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    rtn_obs_ = np.concatenate(rtn_obs)
    rtn_acs_ = np.concatenate(rtn_acs)
    return (rtn_obs_, rtn_acs_, total_reward)

def setup_controller():
    # configure the pygame controller
    print("Plug in a USB gamepad. Do it! Do it now! Press enter after you have done this.")
    wait_for_enter()
    pygame.init()

    num_joysticks = pygame.joystick.get_count()
    if num_joysticks < 1:
        print("You didn't plug in a joystick. FORSHAME!")
        return

    input_manager = InputManager()

    screen = pygame.display.set_mode((640, 480))
    button_index = 0
    
    is_configured = False
    while not is_configured:
        start = time.time()

        screen.fill((0,0,0))

        # There will be two phases to our "game".
        is_configured = button_index >= len(input_manager.buttons)
 
        # configure the joystrick
        if not is_configured:
            success = configure_phase(screen, input_manager.buttons[button_index], input_manager)
            # if the user pressed a button and configured it...
            if success:
                # move on to the next button that needs to be configured 
                button_index += 1

        pygame.display.flip()

        # maintain frame rate
        difference = start - time.time()
        delay = 1.0 / args.fps - difference
        if delay > 0:
            time.sleep(delay)

    pygame.display.quit()
    return input_manager

def main(input_manager):
    from baselines.common.retro_wrappers import make_retro, wrap_deepmind_retro
    from dril.a2c_ppo_acktr.envs import make_vec_envs
    import torch
    import gym, retro

    log_dir = os.path.expanduser(f'{os.getcwd()}/log')
    env = make_vec_envs(args.env_name, 0, 1, None,
                         log_dir, 'cpu', True, use_obs_norm=False)
    
    pygame.init()
    
    # Initialize the joysticks.
    pygame.joystick.init()
    
    ep_rewards = []

    for num_games in count(1):
        env.render()
        (rtn_obs_, rtn_acs_, reward) = rollout(env, input_manager)
        ep_rewards.append(reward)
    
        demo_data_dir = os.getcwd()
        unique_uuid = uuid.uuid4()
        if os.name == 'nt':
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            obs_path = os.path.join(desktop,f'obs_{args.env_name}_seed=0_ntraj=1_{unique_uuid}.npy')
            acs_path = os.path.join(desktop,f'acs_{args.env_name}_seed=0_ntraj=1_{unique_uuid}.npy')
       else:
            obs_path = f'{demo_data_dir}/obs_{args.env_name}_seed=0_ntraj=1_{unique_uuid}.npy'
            acs_path = f'{demo_data_dir}/acs_{args.env_name}_seed=0_ntraj=1_{unique_uuid}.npy'
 
            
        np.save(obs_path, rtn_obs_)
        np.save(acs_path, rtn_acs_)
    
        to_continue = input('Continue "y" or "n": ')
        if to_continue.lower() == 'y':
            pass
        else:
            break
    
    print(f'expert: {np.mean(ep_rewards)}')
    results_save_path = os.path.join(f'{os.getcwd()}', f'expert_{args.env_name}_seed=0.perf')
    results = [{'total_num_steps':0 , 'train_loss': 0, 'test_loss': 0, 'num_trajs': 0 ,\
         'test_reward':np.mean(ep_rewards), 'u_reward': 0}]
    df = pd.DataFrame(results, columns=np.hstack(['x', 'steps', 'train_loss', 'test_loss',\
                      'train_reward', 'test_reward', 'label', 'u_reward']))
    df.to_csv(results_save_path)

if __name__ == "__main__":
    input_manager = setup_controller()
    main(input_manager)

