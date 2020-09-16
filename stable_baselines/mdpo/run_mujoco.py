#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.sac_trpo.policies import MlpPolicy
from stable_baselines import bench, logger
from stable_baselines.mdpo import MDPO
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import gym

def train(env_id, num_timesteps, seed, lam, sgd_steps, klcoeff, log):
    """
    Train TRPO model for the mujoco environment, for testing purposes
    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    with tf_util.single_threaded_session():
        rank = MPI.COMM_WORLD.Get_rank()
        log_path = './experiments/'+str(env_id)+'./SAC-M/nips_test19/m'+str(sgd_steps)+'_c'+str(0.5)+'_e'+str(klcoeff)+'_'+str(seed)
        #log_path = './experiments/'+str(env_id)+'./TRPO-3x/TRPOR-oldsampling/noent_klcoeff'+str(sgd_steps)+'_sgdstep_steps5_'+str(seed)
        if not log:
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
        
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

        #env = make_mujoco_env(env_id, workerseed)
        def make_env():
            env_out = gym.make(env_id)
            env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
            env_out.seed(seed)
            return env_out

        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_reward=False, norm_obs=False)
        
        #env = VecNormalize(env)
        model = MDPO(MlpPolicy, env, gamma=0.99, verbose=1, seed=seed, buffer_size=1000000, ent_coef=1.0, gradient_steps=sgd_steps, lam=klcoeff, train_freq=1, tsallis_q=1, reparameterize=True, klconst=0.5)
        model.learn(total_timesteps=int(num_timesteps))#num_timesteps, seed=seed)
        env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.run, lam=args.lam, sgd_steps=args.sgd_steps, klcoeff=args.klcoeff, log=args.log)


if __name__ == '__main__':
    main()