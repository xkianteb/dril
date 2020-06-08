import argparse
import os
import uuid

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # Behavior Cloning  ---------------------------------
    parser.add_argument(
        '--bc_lr', type=float, default=2.5e-4, help='behavior cloning learning rate (default: 2.5e-4)')
    parser.add_argument(
        '--bc_batch_size', type=int, default=100, help='behavior cloning batch size (default: 100')
    parser.add_argument(
        '--bc_train_epoch', type=int, default=2001, help='behavior cloning training epochs (default=500)')
    parser.add_argument(
        '--behavior_cloning', default=False, action='store_true',
        help='**_Only_** train model with behavior cloning (default: False)')
    parser.add_argument(
        '--warm_start', default=False, action='store_true',
        help='train model with behavior cloning and then train with reinforcement learning starting with learned policy (default: False)')

    # DRIL  ---------------------------------
    parser.add_argument(
        '--dril', default=False, action='store_true',
        help='train model using dril (default: False)')
    parser.add_argument(
        '--dril_uncertainty_reward', choices=['ensemble', 'dropout'], default='ensemble',
        help='dril uncertainty score to use for the reward function (default: ensemble)')
    parser.add_argument(
        '--pretain_ensemble_only', default=False, action='store_true',
        help='train the ensemble only and then exit')
    parser.add_argument(
        '--ensemble_hidden_size', default=512,
        help='dril ensemble network number of hidden units (default: 512)')
    parser.add_argument(
        '--ensemble_drop_rate', default=0.1,
        help='dril dropout ensemble netwrok rate (default: 0.1)')
    parser.add_argument(
        '--ensemble_size', type=int, default=5,
        help='numnber of polices in the ensemble (default: 5)')
    parser.add_argument(
        '--ensemble_batch_size', type=int, default=100,
        help='dril ensemble training batch size (default: 100)')
    parser.add_argument(
        '--ensemble_lr', type=float, default=2.5e-4,
        help='dril ensemble learning rate (default: 2.5e-4)')
    parser.add_argument(
        '--num_ensemble_train_epoch', type=int, default=2001,
        help='dril ensemble number of training epoch (default: 500)')
    parser.add_argument(
        '--ensemble_quantile_threshold', type=float, default=0.98,
        help='dril reward quantile threshold (default: 0.98)')
    parser.add_argument(
        '--num_dril_bc_train_epoch', type=int, default=1,
        help='number of epochs to do behavior cloning updates after reinforcement learning updates (default: 1)')
    parser.add_argument(
        '--ensemble_shuffle_type',
        choices=['no_shuffle', 'sample_w_replace', 'norm_shuffle'],
        default='sample_w_replace')
    #TODO: Think of better way to handle this
    parser.add_argument(
        '--dril_cost_clip',
        choices=['-1_to_1', 'no_clipping', '-1_to_0'],
        default='-1_to_1',
        help='dril uncertainty reward clipping range "lower bound"_to_"upper bound" (default: -1_to_1)')
    parser.add_argument(
        '--use_obs_norm', default=False, action='store_true',
        help='Normallize the observation (default: False)')
    parser.add_argument(
        '--pretrain_ensemble_only', default=False, action='store_true',
        help='pretrain ensemble only on gpu')

    # GAIL -----------------------------------------------
    parser.add_argument(
        '--gail_reward_type',
        choices=['unbias', 'favor_zero_reward', 'favor_non_zero_reward'],
        default='unbias',
        help='specifiy the reward function used by gail (default: unbias)')

    parser.add_argument(
        '--clip_gail_action', default=True, action='store_true',
        help='continous control actions are clipped, so this clips actions between expert and policy trained (defualt: True)')
    parser.add_argument(
        '--gail-disc-lr', type=float, default=2.5e-3,
        help='learning rate for gail discriminator (default: 2.5e-3)')


    # General Paramteres ---------------------------------
    #TODO: Cleaner way to deal with this
    parser.add_argument(
        '--atari_max_steps', default=100000,
        help='Max steps in atari game')
    parser.add_argument(
        '--default_experiment_params', choices=['atari', 'continous-control', 'None', 'retro'], default='None',
        help='Default params ran in the DRIL experiments')
    parser.add_argument(
        '--rl_baseline_zoo_dir', type=str, default='rl-baselines-zoo', help='directory of rl baseline zoo')
    parser.add_argument(
        '--demo_data_dir', type=str, default=f'{os.getcwd()}/demo_data', help='directory of demonstration data')
    parser.add_argument(
        '--num-trajs', type=int, default=1, help='Number of demonstration trajectories')
    parser.add_argument(
        '--save-model-dir',
        default=f'{os.getcwd()}/trained_models/',
        help='directory to save agents (default: ./trained_models/)')
    parser.add_argument(
        '--save-results-dir',
        default='./trained_results/',
        help='directory to save agent training logs (default: ./trained_results/)')
    parser.add_argument(
        '--training_data_split', type=float, default=0.8,
        help='training split for the behavior cloning data between (0-1) (default: 0.8)')
    parser.add_argument(
        '--load_expert', default=False, action='store_true',
        help='load pretrained expert from rl-baseline-zoo (default: False)')
    parser.add_argument(
        '--subsample_frequency', type=int, default=20,
        help='frequency to subsample demonstration data (default: 20)')
    parser.add_argument(
        '--subsample', action='store_true',
        default=False,
        help='boolean to indicate if the demonstration data will be subsampled (default: False)')
    parser.add_argument(
        '--norm-reward-stable-baseline',
        action='store_true',
        default=False,
        help='Stable-Basline Normalize reward if applicable (trained with VecNormalize) (default: False)')
    parser.add_argument(
        '--num_eval_episodes',
        default=10,
        type=int,
        help='Number of evaluation epsiodes (default: 10)')

    parser.add_argument(
        '--system',
        default='',
        type=str)

    # Original Params ---------------------------------------
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default=f'{os.getcwd()}/tmp/{uuid.uuid4()}/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    if args.env_name in ['AntBulletEnv-v0']:
        args.expert_algo = 'trpo'
    else:
        args.expert_algo = 'ppo2'

    def create_dir(dir):
        try:
            os.makedirs(dir)
        except OSError:
            pass

    if args.env_name in ['duckietown']:
        print('** Duckietown only works with 1 process')
        args.num_processes = 1

    if args.env_name in ['duckietown', 'highway-v0'] and args.load_expert:
        raise Exception("Can not load expert because it does not exist")

    if args.algo == 'acktr':
        raise Exception("Code base was not test with acktr: comment this line!")

    if args.default_experiment_params != 'None':
        # Continous control default settings
        if args.default_experiment_params == 'continous-control':
            args.algo = 'ppo'
            args.use_gae = True
            args.log_interval = 1
            args.num_steps = 2048
            args.num_processes = 1
            args.lr = 3e-4
            #args.clip_param = 0.1
            args.entropy_coef = 0
            args.value_loss_coef = 0.5
            args.ppo_epoch = 10
            args.num_mini_batch = 32
            args.gamma = 0.99
            args.gae_lambda = 0.95
            args.num_env_steps = 20e6
            args.use_linear_lr_decay = True
            args.use_proper_time_limits = True
            args.eval_interval = 200
            args.ensemble_quantile_threshold = 0.98

            args.ensemble_shuffle_type =  'sample_w_replace'
            args.bc_lr = 2.5e-4
            args.ensemble_lr = 2.5e-4
            args.ensemble_size = 5
            args.num_ensemble_train_epoch = 2001
            args.bc_train_epoch = 2001
            args.gail_disc_lr = 1e-3

            if args.gail:
                args.num_env_steps = 10e6

        ## Atari default settings
        elif args.default_experiment_params == 'atari':
            args.algo = 'a2c'
            args.use_gae = True
            args.lr = 2.5e-3
            args.clip_param = 0.1
            args.value_loss_coef = 0.5
            args.num_processes = 8
            args.num_steps = 128
            #args.num_mini_batch = 4
            args.log_interval = 10
            args.use_linear_lr_decay = True
            args.entropy_coef = 0.01
            args.ensemble_quantile_threshold = 0.98
            args.num_env_steps = 20e6
            args.eval_interval = 1000
            args.ensemble_quantile_threshold = 0.98
            args.num_dril_bc_train_epoch = 1

            args.ensemble_shuffle_type =  'sample_w_replace'
            args.bc_lr = 2.5e-4
            args.ensemble_lr = 2.5e-4
            args.ensemble_size = 5
            args.num_ensemble_train_epoch =  1001
            args.bc_train_epoch = 1001
            args.gail_disc_lr = 2.5e-3

        elif args.default_experiment_params == 'retro':
            args.algo = 'ppo'
            args.use_gae = True
            args.log_interval = 1
            args.num_steps = 128
            args.num_processes = 8
            args.lr = 2.5e-4
            args.clip_param = 0.1
            args.entropy_coef = 0
            args.value_loss_coef = 0.5
            args.num_mini_batch = 4
            args.gamma = 0
            args.gae_lambda = 0.95
            args.num_env_steps = 100e6
            args.use_linear_lr_decay = True
            args.use_proper_time_limits = True
            args.eval_interval = 200
            args.ensemble_quantile_threshold = 0.98

            args.ensemble_shuffle_type =  'sample_w_replace'
            args.bc_lr = 2.5e-4
            args.ensemble_lr = 2.5e-4
            args.ensemble_size = 5
            args.num_ensemble_train_epoch =  1001
            args.bc_train_epoch = 1001
            args.gail_disc_lr = 2.5e-3
        else:
            raise Exception('Unknown Defult experiments')

    # Ensure directories are created
    create_dir(os.path.join(args.save_model_dir, args.algo))
    create_dir(os.path.join(args.save_model_dir, 'bc'))
    create_dir(os.path.join(args.save_model_dir, 'ensemble'))
    create_dir(os.path.join(args.save_model_dir, 'gail'))
    create_dir(os.path.join(args.save_model_dir, 'dril'))
    create_dir(os.path.join(args.save_model_dir, 'a2c'))
    create_dir(os.path.join(args.save_model_dir, 'ppo'))

    create_dir(os.path.join(args.save_results_dir, args.algo))
    create_dir(os.path.join(args.save_results_dir, 'bc'))
    create_dir(os.path.join(args.save_results_dir, 'ensemble'))
    create_dir(os.path.join(args.save_results_dir, 'gail'))
    create_dir(os.path.join(args.save_results_dir, 'dril'))
    create_dir(os.path.join(args.save_results_dir, 'a2c'))
    create_dir(os.path.join(args.save_results_dir, 'ppo'))
    create_dir(os.path.join(args.save_results_dir, 'expert'))

    return args

