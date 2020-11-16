from collections import deque

import numpy as np
import gym
from gym import spaces
import cv2  # pytype:disable=import-error
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action

cv2.ocl.setUseOpenCL(False)


class RandomActionResetEnv(gym.Wrapper):
    def __init__(self, env, random_action_len=0):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.

        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.random_action_len = random_action_len

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)

        if self.random_action_len > 0:
            obs = None
            for _ in range(self.random_action_len):
                unscaled_action = self.env.action_space.sample()
                action = scale_action(self.action_space, unscaled_action)
                obs, _, done, _ = self.env.step(action)
                if done:
                    obs = self.env.reset(**kwargs)

            self._elapsed_steps = 0

        return obs

    def step(self, action):
        return self.env.step(action)

class RandomResetEnv(gym.Wrapper):
    def __init__(self, env, **kwargs):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.

        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        env.reset_noise_scale = 1000.0
        env._reset_noise_scale = env.reset_noise_scale


    def reset(self, **kwargs):

        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
#
#
# class TimeLimit(gym.Wrapper):
#     def __init__(self, env, max_episode_steps=None):
#         super(TimeLimit, self).__init__(env)
#         if max_episode_steps is None and self.env.spec is not None:
#             max_episode_steps = env.spec.max_episode_steps
#         if self.env.spec is not None:
#             self.env.spec.max_episode_steps = max_episode_steps
#         self._max_episode_steps = max_episode_steps
#         self._elapsed_steps = None
#
#     def step(self, action):
#         assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
#         observation, reward, done, info = self.env.step(action)
#         self._elapsed_steps += 1
#         if self._elapsed_steps >= self._max_episode_steps:
#             info['TimeLimit.truncated'] = not done
#             done = True
#         return observation, reward, done, info
#
#     def reset(self, **kwargs):
#         self._elapsed_steps = 0
#         return self.env.reset(**kwargs)

def wrap_mujoco(env, random_reset=False, **kwargs):
    """
    Configure environment for DeepMind-style Atari.

    :param env: (Gym Environment) the atari environment
    :param episode_life: (bool) wrap the episode life wrapper
    :param clip_rewards: (bool) wrap the reward clipping wrapper
    :param frame_stack: (bool) wrap the frame stacking wrapper
    :param scale: (bool) wrap the scaling observation wrapper
    :return: (Gym Environment) the wrapped atari environment
    """
    if random_reset:
        env = RandomResetEnv(env, **kwargs)

    return env
