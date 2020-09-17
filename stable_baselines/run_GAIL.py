import gym
#
# from stable_baselines import MDAL, GAIL, SAC, MDAL_SAC, MDAL_MDPO_OFF
from stable_baselines import MDAL_MDPO_OFF
from stable_baselines.gail import ExpertDataset, generate_expert_traj

# Generate expert trajectories (train expert)
# model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
# model = SAC('MlpPolicy', 'Humanoid-v2', verbose=1, buffer_size=1000000, batch_size=256, ent_coef='auto',
#             train_freq=1, tau=0.01, gradient_steps=1, learning_starts=10000)
# # model = SAC('MlpPolicy', 'Humanoid-v2', verbose=1)
# # #
# # # #
# generate_expert_traj(model, 'expert_humanoid_10e6', n_timesteps=10000000, n_episodes=10)
# model.save("sac_humanoid_10e6")

# Load the expert dataset
#
# dataset = ExpertDataset(expert_path='expert_cheetah_2e6.npz', traj_limitation=10, verbose=1)
dataset = ExpertDataset(expert_path='experts/expert_humanoid_10e6.npz', traj_limitation=10, verbose=1)

# #
#
# model = MDAL_MDPO_OFF('MlpPolicy', 'HalfCheetah-v2', dataset, verbose=1,
#                       tensorboard_log="./experiments/cheetah/mdal_mdpo_off_tensorboard/", seed=0,
#                       buffer_size=1000000, ent_coef=1.0, learning_starts=10000, batch_size=256, tau=0.01,
#                       gradient_steps=10, lam=0.0, train_freq=1, tsallis_q=1, reparameterize=True,
#                       t_pi=0.5,
#                       t_c=0.05)
model = MDAL_MDPO_OFF('MlpPolicy', 'Humanoid-v2', dataset, verbose=1,
                      tensorboard_log="./experiments/humanoid/mdal_mdpo_off_tensorboard/", seed=0,
                      buffer_size=1000000, ent_coef=1.0, learning_starts=10000, batch_size=256, tau=0.01,
                      gradient_steps=10, mdpo_gradient_steps=10, lam=0.0, train_freq=1, tsallis_q=1, reparameterize=True,
                      t_pi=0.5, t_c=0.05,
                      n_cpu_tf_sess=4)
# # model = MDAL('MlpPolicy', 'Humanoid-v2', dataset, exploration_bonus=False,
# #              verbose=1, tensorboard_log="./experiments/pendulum/mdal_tensorboard/")
# # # #
# # # # # Note: in practice, you need to train for 1M steps to have a working policy
# model = GAIL('MlpPolicy', 'HalfCheetah-v2', dataset, verbose=1,
#              tensorboard_log="./experiments/cheetah/gail_tensorboard/",
#              entcoeff=0.0, adversary_entcoeff=0.001)
# model.pretrain(dataset, n_epochs=100000)
model.learn(total_timesteps=10000000)
model.save("gail_cheetah")
model.save("mdal_off_humanoid_lr=0.05")


# model = GAIL('MlpPolicy', 'Humanoid-v2', dataset, verbose=1, tensorboard_log="./experiments/humanoid/gail_tensorboard/")
# #
# # # Note: in practice, you need to train for 1M steps to have a working policy
# model.learn(total_timesteps=5000000)
# model.save("gail_humanoid")
# # #
# del model # remove to demonstrate saving and loading
#
# model = SAC.load("sac_walker")
#
# env = gym.make('Walker2d-v2')
# obs = env.reset()
# sum_rew = 0.0
# while True:
#   action, _states = model.predict(obs)
#   obs, rewards, dones, info = env.step(action)
#   env.render()

