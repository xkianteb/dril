import os
import numpy as np
import torch
import gym
import pandas as pd
import copy

from dril.a2c_ppo_acktr.algo.behavior_cloning import BehaviorCloning
import dril.a2c_ppo_acktr.ensemble_models as ensemble_models
from baselines.common.running_mean_std import RunningMeanStd

from torch.utils.data import DataLoader, TensorDataset

def Ensemble (uncertainty_reward=None, device=None, envs=None,\
    ensemble_hidden_size=None, ensemble_drop_rate=None, ensemble_size=None, ensemble_lr=None,\
    ensemble_batch_size=None, env_name=None, expert_dataset=None,num_trajs=None, seed=None,\
    num_ensemble_train_epoch=None,training_data_split=None, save_model_dir=None, save_results_dir=None):

    ensemble_size = ensemble_size
    device = device
    env_name = env_name
    observation_space = envs.observation_space

    num_inputs  = envs.observation_space.shape[0]
    try:
        num_actions = envs.action_space.n
    except:
        num_actions = envs.action_space.shape[0]

    ensemble_args = (num_inputs, num_actions, ensemble_hidden_size, ensemble_size)
    if len(observation_space.shape) == 3:
        if env_name in ['duckietown']:
            ensemble_policy = ensemble_models.PolicyEnsembleDuckieTownCNN
        elif uncertainty_reward == 'ensemble':
            ensemble_policy = ensemble_models.PolicyEnsembleCNN
        elif uncertainty_reward == 'dropout':
            ensemble_policy = ensemble_models.PolicyEnsembleCNNDropout
        else:
            raise Exception("Unknown uncertainty_reward type")
    else:
        if uncertainty_reward == 'ensemble':
            ensemble_policy = ensemble_models.PolicyEnsembleMLP
        else:
            raise Exception("Unknown uncertainty_reward type")

    ensemble_policy = ensemble_policy(*ensemble_args).to(device)

    ensemblebc = BehaviorCloning(ensemble_policy,device, batch_size=ensemble_batch_size,\
                   lr=ensemble_lr, envs=envs, training_data_split=training_data_split,\
                   expert_dataset=expert_dataset,ensemble_size=ensemble_size )

    ensemble_model_save_path = os.path.join(save_model_dir, 'ensemble')
    ensemble_file_name = f'ensemble_{env_name}_policy_ntrajs={num_trajs}_seed={seed}'
    ensemble_model_path = os.path.join(ensemble_model_save_path, f'{ensemble_file_name}.model')
    ensemble_results_save_path = os.path.join(save_results_dir, 'ensemble', f'{ensemble_file_name}.perf')
    # Check if model already exist
    best_test_loss, best_test_model = np.float('inf'), None
    if os.path.exists(ensemble_model_path):
        best_test_params = torch.load(ensemble_model_path, map_location=device)
        print(f'*** Loading ensemble policy: {ensemble_model_path} ***')
    else:
        ensemble_results = []
        for ensemble_epoch in range(num_ensemble_train_epoch):
            ensemble_train_loss = ensemblebc.update(update=True, data_loader_type='train')
            with torch.no_grad():
                ensemble_test_loss = ensemblebc.update(update=False, data_loader_type='test')
            print(f'ensemble-epoch {ensemble_epoch}/{num_ensemble_train_epoch} | train loss: {ensemble_train_loss:.4f},  test loss: {ensemble_test_loss:.4f}')
            ensemble_results.append({'epoch': ensemble_epoch, 'trloss':ensemble_train_loss,\
                      'teloss': ensemble_test_loss, 'test_reward': 0})
        best_test_params = copy.deepcopy(ensemble_policy.state_dict())

        # Save the Ensemble model and training results
        torch.save(best_test_params, ensemble_model_path)
        df = pd.DataFrame(ensemble_results, columns=np.hstack(['epoch', 'trloss', 'teloss','test_reward']))
        df.to_csv(ensemble_results_save_path)

    ensemble_policy.load_state_dict(best_test_params)
    return ensemble_policy
