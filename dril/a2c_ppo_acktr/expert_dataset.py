import numpy as np
import torch
import random

from torch.utils.data import DataLoader, TensorDataset

class ExpertDataset:
    def __init__(self, demo_data_dir, env_name, num_trajs, seed, ensemble_shuffle_type):
        self.demo_data_dir = demo_data_dir
        self.env_name = env_name
        self.num_trajs = num_trajs
        self.seed = seed
        self.ensemble_shuffle_type = ensemble_shuffle_type


    def load_demo_data(self, training_data_split, batch_size, ensemble_size):
        obs_file = f'{self.demo_data_dir}/obs_{self.env_name}_seed={self.seed}_ntraj={self.num_trajs}.npy'
        acs_file = f'{self.demo_data_dir}/acs_{self.env_name}_seed={self.seed}_ntraj={self.num_trajs}.npy'

        print(f'loading: {obs_file}')
        obs = torch.from_numpy(np.load(obs_file))
        acs = torch.from_numpy(np.load(acs_file))
        perm = torch.randperm(obs.size(0))
        obs = obs[perm]
        acs = acs[perm]

        n_train = int(obs.size(0)*training_data_split)
        obs_train = obs[:n_train]
        acs_train = acs[:n_train]
        obs_test  = obs[n_train:]
        acs_test  = acs[n_train:]

        if self.ensemble_shuffle_type == 'norm_shuffle' or ensemble_size is None:
            shuffle = True
        elif self.ensemble_shuffle_type == 'no_shuffle' and ensemble_size is not None:
            shuffle = False
        elif self.ensemble_shuffle_type == 'sample_w_replace' and ensemble_size is not None:
            print('***** sample_w_replace *****')
            # sample with replacement
            obs_train_resamp, acs_train_resamp = [], []
            for k in range(n_train * ensemble_size):
                indx = random.randint(0, n_train - 1)
                obs_train_resamp.append(obs_train[indx])
                acs_train_resamp.append(acs_train[indx])
            obs_train = torch.stack(obs_train_resamp)
            acs_train = torch.stack(acs_train_resamp)
            shuffle = False

        tr_batch_size = min(batch_size, len(obs_train))
        # If Droplast is False, insure that that dataset is divisible by
        # the number of polices in the ensemble
        tr_drop_last = (tr_batch_size!=len(obs_train))
        if not tr_drop_last and ensemble_size is not None:
            tr_batch_size = int(ensemble_size * np.floor(tr_batch_size/ensemble_size))
            obs_train = obs_train[:tr_batch_size]
            acs_train = acs_train[:tr_batch_size]
        trdata = DataLoader(TensorDataset(obs_train, acs_train),\
                           batch_size = tr_batch_size, shuffle=shuffle, drop_last=tr_drop_last)

        if len(obs_test) == 0:
            tedata = None
        else:
            te_batch_size = min(batch_size, len(obs_test))
            # If Droplast is False, insure that that dataset is divisible by
            # the number of polices in the ensemble
            te_drop_last = (te_batch_size!=len(obs_test))
            if not te_drop_last and ensemble_size is not None:
                te_batch_size = int(ensemble_size * np.floor(te_batch_size/ensemble_size))
                obs_test = obs_test[:te_batch_size]
                acs_test = acs_test[:te_batch_size]
            tedata = DataLoader(TensorDataset(obs_test, acs_test),\
                                batch_size = te_batch_size, shuffle=shuffle, drop_last=te_drop_last)
        return {'trdata':trdata, 'tedata': tedata}

