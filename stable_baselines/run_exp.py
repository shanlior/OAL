#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.mujoco_wrappers import wrap_mujoco

import gym

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines import bench, logger


from stable_baselines.mdal import MDAL_MDPO_OFF
from stable_baselines.gail import ExpertDataset, generate_expert_traj
import os


def train(env_id, algo, num_timesteps, seed, sgd_steps, t_pi, t_c, log, expert_path, pretrain, pretrain_epochs,
          mdpo_update_steps, num_trajectories, expert_model, exploration_bonus, bonus_coef, random_action_len,
          is_action_features):
    """
    Train TRPO model for the mujoco environment, for testing purposes
    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """

    with tf_util.single_threaded_session():
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        env_name = env_id[:-3].lower()
        log_dir = './experiments/' + env_name + '/' + str(algo).lower() + '/'
        log_name = str(algo) + '_updateSteps' + str(mdpo_update_steps) + '_tpi' + str(t_pi) + '_tc' + str(t_c)
        # log_name += '_randLen' + str(random_action_len)
        if exploration_bonus:
            log_name += '_exploration' + str(bonus_coef)
        if pretrain:
            log_name += '_pretrain' + str(pretrain_epochs)
        if not is_action_features:
            log_name += "_states_only"
        log_name+= '_s' + str(seed)

        log_path = log_dir + log_name
        expert_path = './experts/' + expert_path

        num_timesteps = int(num_timesteps)

        if log:
            if rank == 0:
                logger.configure(log_path)
            else:
                logger.configure(log_path, format_strs=[])
                logger.set_level(logger.DISABLED)
        else:
            if rank == 0:
                logger.configure()
            else:
                logger.configure(format_strs=[])
                logger.set_level(logger.DISABLED)


        # workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

        # env = make_mujoco_env(env_id, workerseed)
        def make_env():
            # env_out = gym.make(env_id, reset_noise_scale=1.0)
            env_out = gym.make(env_id)
            env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
            env_out.seed(seed)
            env_out = wrap_mujoco(env_out, random_action_len=random_action_len)
            return env_out
        #


        env = DummyVecEnv([make_env])
        # env = VecNormalize(env)

        if algo == 'Train':
            train = True
        else:
            train = False

        if algo == 'Evaluate':
            eval = True
        else:
            eval = False

        if train:
            from stable_baselines import SAC
            env = VecNormalize(env, norm_reward=False, norm_obs=False)

            if num_timesteps > 0:
                model = SAC('MlpPolicy', env_id, verbose=1, buffer_size=1000000, batch_size=256, ent_coef='auto',
                                train_freq=1, tau=0.01, gradient_steps=1, learning_starts=10000)
            else:
                model = SAC.load(expert_model, env)
            generate_expert_traj(model, expert_path, n_timesteps=num_timesteps, n_episodes=num_trajectories)
            if num_timesteps > 0:
                model.save('sac_' + env_name + '_' + str(num_timesteps))
        elif eval:
            from stable_baselines import SAC
            env = VecNormalize(env, norm_reward=False, norm_obs=False)
            model = SAC.load(expert_model, env)
            generate_expert_traj(model, expert_path, n_timesteps=num_timesteps, n_episodes=100, evaluate=True)
        else:
            expert_path = expert_path + '.npz'
            dataset = ExpertDataset(expert_path=expert_path, traj_limitation=10, verbose=1)

            if algo == 'MDAL':
                model = MDAL_MDPO_OFF('MlpPolicy', env, dataset, verbose=1,
                                      tensorboard_log="./experiments/" + env_name + "/mdal/", seed=seed,
                                      buffer_size=1000000, ent_coef=1.0, learning_starts=10000, batch_size=256, tau=0.01,
                                      gamma=0.99, gradient_steps=sgd_steps, mdpo_update_steps=mdpo_update_steps,
                                      lam=0.0, train_freq=1, tsallis_q=1, reparameterize=True, t_pi=t_pi, t_c=t_c,
                                      exploration_bonus=exploration_bonus, bonus_coef=bonus_coef,
                                      is_action_features=is_action_features)
            elif algo == 'GAIL':
                from mpi4py import MPI
                from stable_baselines import GAIL

                model = GAIL('MlpPolicy', env, dataset, verbose=1,
                             tensorboard_log="./experiments/" + env_name + "/gail/", seed=seed,
                             entcoeff=0.0, adversary_entcoeff=0.001)
            else:
                raise ValueError("Not a valid algorithm.")

            if pretrain:
                model.pretrain(dataset, n_epochs=pretrain_epochs)

            model.learn(total_timesteps=num_timesteps, tb_log_name=log_name)


        env.close()


def main():
    """
    Runs the testd
    """
    args = mujoco_arg_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    log = not args.no_log
    is_action_features = not args.states
    seed_offset = 0
    for seed in range(args.num_seeds):
        train(args.env, algo=args.algo, num_timesteps=args.num_timesteps, seed=(seed+seed_offset), sgd_steps=args.sgd_steps,
              t_pi=args.t_pi, t_c=args.t_c, log=log, expert_path=args.expert_path,
              pretrain=args.pretrain, pretrain_epochs=args.pretrain_epochs, mdpo_update_steps=args.mdpo_update_steps,
              num_trajectories=args.num_trajectories, expert_model=args.expert_model,
              exploration_bonus=args.exploration, bonus_coef=args.bonus_coef, random_action_len=args.random_action_len,
              is_action_features=is_action_features)


if __name__ == '__main__':
    main()