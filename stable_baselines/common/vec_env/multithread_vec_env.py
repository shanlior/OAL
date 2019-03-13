import threading
from queue import Queue
from collections import OrderedDict

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images


def _worker(remote_in, remove_out, env_fn_wrapper):
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote_in.get()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remove_out.put((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remove_out.put(observation)
            elif cmd == 'render':
                remove_out.put(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                env.close()
                remote_in.task_done()
                break
            elif cmd == 'get_spaces':
                remove_out.put((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remove_out.put(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remove_out.put(getattr(env, data))
            elif cmd == 'set_attr':
                remove_out.put(setattr(env, data[0], data[1]))
            else:
                remote_in.task_done()
                raise NotImplementedError
            remote_in.task_done()
        except EOFError:
            break
        except Exception as e:
            remove_out.put(e)


def queue_get(remote):
    ret = remote.get()
    if isinstance(ret, Exception):
        raise ret
    remote.task_done()
    return ret


class MultithreadVecEnv(VecEnv):
    """
    Creates a multithreaded vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) Environments to run in multiple threads
    """

    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns) - 1

        self.local_env = env_fns[0]()

        self.remote_send = [Queue() for _ in range(n_envs)]
        self.remote_recv = [Queue() for _ in range(n_envs)]
        self.threads = []
        for remote_send, remote_recv, env_fn in zip(self.remote_send, self.remote_recv, env_fns[1:]):
            args = (remote_send, remote_recv, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            thread = threading.Thread(target=_worker, args=args, daemon=True)
            thread.start()
            self.threads.append(thread)

        self.local_step_ret = None
        VecEnv.__init__(self, len(env_fns), self.local_env.observation_space, self.local_env.action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remote_send, actions[1:]):
            remote.put(('step', action))
        self.waiting = True
        self.local_step_ret = self.local_env.step(actions[0])

    def step_wait(self):
        results = [self.local_step_ret] + [queue_get(remote) for remote in self.remote_recv]
        self.waiting = False
        if results[0][2]:
            results[0] = (self.local_env.reset(),) + results[0][1:]
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remote_send:
            remote.put(('reset', None))
        obs = [self.local_env.reset()] + [queue_get(remote) for remote in self.remote_recv]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        for remote in self.remote_recv:
            while not remote.empty():
                queue_get(remote)
        for remote in self.remote_send:
            remote.put(('close', None))
        self.local_env.close()
        for process in self.threads:
            process.join()
        for remote in self.remote_send:
            while not remote.empty():
                queue_get(remote)
            remote.join()
        for remote in self.remote_recv:
            while not remote.empty():
                queue_get(remote)
            remote.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for remote in self.remote_send:
            # gather images from threads
            # `mode` will be taken into account later
            remote.put(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [self.local_env.render(*args, **{'mode': 'rgb_array', **kwargs})] + \
               [queue_get(remote) for remote in self.remote_recv]
        # Create a big image by tiling images from threads
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for remote in self.remote_send:
            remote.put(('render', {"mode": 'rgb_array'}))
        imgs = [self.local_env.render(mode='rgb_array')] + [queue_get(remote) for remote in self.remote_recv]
        return imgs

    def env_method(self, method_name, *method_args, **method_kwargs):
        """
        Provides an interface to call arbitrary class methods of vectorized environments

        :param method_name: (str) The name of the env class method to invoke
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items retured by each environment's method call
        """

        for remote in self.remote_send:
            remote.put(('env_method', (method_name, method_args, method_kwargs)))
        return [getattr(self.local_env, method_name)(*method_args, **method_kwargs)] + \
               [queue_get(remote) for remote in self.remote_recv]

    def get_attr(self, attr_name):
        """
        Provides a mechanism for getting class attribues from vectorized environments
        (note: attribute value returned must be picklable)

        :param attr_name: (str) The name of the attribute whose value to return
        :return: (list) List of values of 'attr_name' in all environments
        """

        for remote in self.remote_send:
            remote.put(('get_attr', attr_name))
        return [getattr(self.local_env, attr_name)] + \
               [queue_get(remote) for remote in self.remote_recv]

    def set_attr(self, attr_name, value, indices=None):
        """
        Provides a mechanism for setting arbitrary class attributes inside vectorized environments
        (note:  this is a broadcast of a single value to all instances)
        (note:  the value must be picklable)

        :param attr_name: (str) Name of attribute to assign new value
        :param value: (obj) Value to assign to 'attr_name'
        :param indices: (list,tuple) Iterable containing indices of envs whose attr to set
        :return: (list) in case env access methods might return something, they will be returned in a list
        """

        if indices is None:
            indices = range(len(self.remote_send))
        elif isinstance(indices, int):
            indices = [indices]
        for remote in [self.remote_send[i] for i in indices]:
            remote.put(('set_attr', (attr_name, value)))
        return [setattr(self.local_env, attr_name, value)] + \
               [queue_get(remote) for remote in [self.remote_recv[i] for i in indices]]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
