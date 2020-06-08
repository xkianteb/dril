import matplotlib.pyplot as plt
import glob, csv, pdb, numpy, torch, os, argparse
import pandas
import matplotlib.pyplot as plt
import os
import numpy as np

import seaborn as sns
import cycler
import matplotlib


parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, default='SpaceInvadersNoFrameskip-v4')
parser.add_argument('-n_bc_epochs', type=int, default=1)
parser.add_argument('-shuffle', type=int, default=2)
parser.add_argument('-lr', type=float, default=0.00025)
parser.add_argument('-quantile', type=float, default=0.98)
parser.add_argument('-decay', type=int, default=1)
parser.add_argument('-exp', type=str, default='exp1')
parser.add_argument('-plot_u_reward', type=int, default=0)
args = parser.parse_args()

data_dir = f'{os.getcwd()}/dril/trained_results/'

def get_results(result_files, filter=False):
    rewards = []
    u_rewards = []
    steps = []
    test_reward = []
    for r in result_files:
        try:
            data = pandas.read_csv(r)
            idx = len(data['test_reward']) - 1
            rewards.append(data['test_reward'][idx])
        except:
            pass

        try:
            u_rewards.append(data['u_reward'].tolist())
            steps.append(data['total_num_steps'].tolist())
            test_reward.append(data['test_reward'].tolist())
        except:
            pass

    return (rewards, u_rewards, steps, test_reward)


def load_results(n_demo):
    # Expert results -----------------
    expert_results = glob.glob(f'{data_dir}/expert/expert_{args.env}_seed=*.perf')
    (expert_reward, _, _, _) = get_results(expert_results)

    # Behavior Cloing results --------
    bc_mse_results = glob.glob(f'{data_dir}/bc/bc_{args.env}_policy_ntrajs={n_demo}_seed=*.perf')
    (bc_mse_reward, _, _,_) = get_results(bc_mse_results)

    # DRIL results -------------------
    exp_name  = f'dril_{args.env}_ntraj={n_demo}_ensemble_lr=0.00025_lr=0.00025_bcep=1001_'
    exp_name += f'shuffle=sample_w_replace_quantile=0.98_cost_-1_to_1_seed=*.perf'

    bc_mse_variance_results = glob.glob(f'{data_dir}/dril/{exp_name}')
    (bc_mse_variance_reward, bc_variance_u_reward, bc_variance_steps, bc_mse_variance_reward_curve) = get_results(bc_mse_variance_results, filter=True)

    # Random results -----------------
    random_reward = []
    random_results = glob.glob(f'{data_dir}/random/{args.env}/random*.perf')
    for r in random_results:
        random_reward.append(pandas.read_csv(r)['test_reward'].max())

    # Gail results --------------------
    params = [(clipped_loss, zero_expert_reward, use_obs_norm, use_bc,      gail_normalized_reward, bc_loss, clamp_gail_action)
         for clipped_loss in [True]
         for zero_expert_reward in [True, False]
         for use_obs_norm in [False]
         for use_bc in [True]
         for gail_normalized_reward in [True]
         for clamp_gail_action in [False]
         for bc_loss in ['mse']]

    gail = {}
    for gail_reward_type in ['unbias', 'favor_zero_reward', 'favor_non_zero_reward']:
        gail_results  = f'gail_{args.env}_ntraj={n_demo}_'
        gail_results += f'gail_lr=0.001_lr=0.00025_bcep=2001_'
        gail_results += f'gail_reward_type={gail_reward_type}_seed=*.perf'
        results = glob.glob(f'{data_dir}/gail/{gail_results}')

        label = f'GAIL {gail_reward_type}'
        (results, _, _, _) = get_results(results)
        if results:
            gail[label] = results
        else:
            gail[label] = []

    return {'expert': numpy.array(expert_reward),
            'bc_mse': numpy.array(bc_mse_reward),
            'bc_mse_variance': numpy.array(bc_mse_variance_reward),
            'bc_variance_u_reward_curve': bc_variance_u_reward,
            'bc_mse_variance_reward_curve': bc_mse_variance_reward_curve,
            'bc_variance_steps': bc_variance_steps,
            'random': numpy.array(random_reward),
            **gail}

def add_line_plot(perf_results, color=None, style=None):
    width = 3
    s = 10
    alpha=0.1

    mean = [numpy.mean(perf) for perf in perf_results]
    for perf in perf_results:
        numpy.std(perf)
    std = [numpy.std(perf) for perf in perf_results]

    plt.plot([1, 3, 5, 10, 15, 20], mean, style, c=color, linewidth=width, markersize=s)
    plt.xticks([1, 3, 5, 10, 15, 20])
    plt.fill_between([1, 3, 5, 10, 15, 20], numpy.array(mean) - numpy.array(std), numpy.array(mean) + numpy.array(std), color=color, alpha=alpha)

styles = {'expert': '--',
      'bc': 'o-',
      'dril': '^-',
      'gail0': 'v-',
      'gail1': 'D-',
      'gail2': '<-',
      'gail3': '*-',
      'random': '.-'}

n = 12
color = numpy.array(sns.color_palette("colorblind", n_colors=n))
matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

c1 = color[7]*0.9
c2 = color[2]
c3 = color[4]
c4 = color[3]
c5 = color[1]
c6 = color[10]
c7 = color[10]
c8 = color[10]

colors = {'expert': c1,
      'bc': c2,
      'dril': c3,
      'gail0': c4,
      'gail1': c5,
      'gail2': c7,
      'gail3': c8,
      'random': c6}

def main():
    # Expert ---------------
    expert = [load_results(n_demo)['expert'] for n_demo in [1, 3, 5, 10, 15, 20]]
    add_line_plot(expert, colors['expert'], styles['expert'])

    # Behavior Cloning ------
    bc_mse = [load_results(n_demo)['bc_mse'] for n_demo in [1, 3, 5, 10, 15, 20]]
    add_line_plot(bc_mse, colors['bc'], styles['bc'])

    # DRIL ------------------
    bc_mse_variance = [load_results(n_demo)['bc_mse_variance'] for n_demo in [1, 3, 5, 10, 15, 20]]
    add_line_plot(bc_mse_variance, colors['dril'], styles['dril'])

    # Random ------------------
    random = [load_results(n_demo)['random'] for n_demo in [1, 3, 5, 10, 15, 20]]
    add_line_plot(random,  colors['random'], styles['random'])

    # GAIL -----------------
    keys = []
    for n_demo in [1, 3, 5, 10, 15, 20]:
        keys += load_results(n_demo).keys()
    gail_keys = sorted(list(set([key for key in keys if 'GAIL' in key])))

    final_keys = []
    for idx, key in enumerate(gail_keys):
        gail_results = [load_results(n_demo)[key] for n_demo in [1, 3, 5, 10, 15, 20]]
        add_line_plot(gail_results, colors[f'gail{idx}'], styles[f'gail{idx}'])
        final_keys.append(key)

    plt.legend(['Expert','BC','DRIL', 'RANDOM']+final_keys, fontsize=6, loc='bottom right')
    fsize=16

    plt.xlabel('Expert Trajectories', fontsize=fsize)
    plt.ylabel('Reward', fontsize=fsize)
    env = args.env.replace('-v4', '').replace('NoFrameskip', '')
    plt.title(env, fontsize=fsize)
    plt.savefig(f'{env}.pdf')

    plt.clf()
    test_rewards = load_results(10)['bc_mse_variance_reward_curve']
    num_values = min([len(test_rewards[0]), len(test_rewards[1])])
    test_rewards = np.array([x[:num_values] for x in test_rewards])

    u_rewards = load_results(10)['bc_variance_u_reward_curve']
    u_rewards = np.array([x[:num_values] for x in u_rewards])

    steps = np.array(load_results(10)['bc_variance_steps'][0][:num_values])

    u_rewards_mean = -numpy.mean(u_rewards, axis=0)
    u_rewards_std = numpy.std(u_rewards, axis=0)

    test_rewards_mean = numpy.mean(test_rewards, axis=0)
    test_rewards_std = numpy.std(test_rewards, axis=0)

    fig, axs = plt.subplots(2, 1)
    ax1 = axs[0]
    ax2 = axs[1]
    box = dict(facecolor='yellow', pad=5, alpha=0.2)
    c1 = color[7]
    c2 = color[10]

    ax1.plot(steps, u_rewards_mean, color='black')
    ax1.fill_between(steps, u_rewards_mean - u_rewards_std, u_rewards_mean + u_rewards_std, color=c1, alpha=0.3)
    ax1.set_xlabel('steps', fontsize=16)
    ax1.set_ylabel('Uncertainty Cost', fontsize=12)
    ax1.set_title(env, fontsize=16)
    ax2.plot(steps, test_rewards_mean, color=c2)
    ax2.fill_between(steps, test_rewards_mean - test_rewards_std, test_rewards_mean + test_rewards_std, color=c2, alpha=0.3)
    ax2.set_xlabel('steps', fontsize=16)
    ax2.set_ylabel('Episode Reward', fontsize=12)
    plt.savefig(f'{env}_u_reward.pdf')


if __name__== "__main__":
  main()

