"""
Helpers for dealing with vectorized environments.
"""

from collections import OrderedDict
import multiprocessing
import time
import timeit
import sys
import os

import gym
import numpy as np


def _sliencer(call, true_out, true_err, verbose=0):
    def _wrapper(*args, **kwargs):

        if verbose < 2:
            sys.stderr = open(os.devnull, "w")
        if verbose < 3:
            sys.stdout = open(os.devnull, "w")
        ret = call(*args, **kwargs)

        sys.stdout = true_out
        sys.stderr = true_err
        return ret

    return _wrapper


def _timeout_dec(timeout):
    def _decorator(func):
        def _wrapper(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(func, args, kwargs)

            return async_result.get(timeout)
        return _wrapper
    return _decorator


def detect_best_wrapper(env, n_cpu=-1, n_steps=None, n_loops=1, verbose=0, _return_conf=False, check_proc_spawn=False,
                        timeout=5):
    """
    Auto detects and returns the best wrapper for a given environment and parameters
    usage:
    `
    wrapper = detect_best_wrapper(gym.make)("MontezumaRevenge-v4")

    env = wrapper([lambda: gym.make("MontezumaRevenge-v4") for _ in range(n_cpu)])
    `

    :param env: (gym.Env) the environment to benchmark
    :param n_cpu: (int) the number of CPUs to use for the benchmark (default: autodetect)
    :param n_steps: (int) the number of steps for the benchmark (default; ~1's worth of steps over all the tests)
    :param n_loops: (int) the number of loops for the benchmark (default: 1)
    :param verbose: (int) the verbosity of the benchmark (default: 0)
    :param _return_conf: (bool) if the detector should return n_cpu (default: False)
    :param check_proc_spawn: (bool) if the detector is allowed to check SubprocVecEnv with `start_method=spawn`
        (default: False)
    :param timeout: (float) the timeout (in seconds) for the wrapper (default: 5s)
    :return: (VecEnv Class or (VecEnv Class, int)) the best wrapper for the given parameters with optionally
        the number of CPUs
    """
    from . import SubprocVecEnv, MultithreadVecEnv, DummyVecEnv
    if n_cpu == -1:
        try:
            n_cpu = multiprocessing.cpu_count()
        except NotImplementedError:
            n_cpu = 1
        if n_cpu is None:
            n_cpu = 1

    def _wrapper_maker(*args, **kwargs):
        if verbose > 1:
            print("initializing the wrappers...")
        true_out = sys.stdout
        true_err = sys.stderr
        try:
            default = DummyVecEnv([lambda: _sliencer(env, true_out, true_err, verbose=verbose)(*args, **kwargs)])
            proc_fork = SubprocVecEnv([lambda: _sliencer(env, true_out, true_err, verbose=verbose)(*args, **kwargs)
                                       for _ in range(n_cpu)], start_method="fork")
            thread = MultithreadVecEnv([lambda: _sliencer(env, true_out, true_err, verbose=verbose)(*args, **kwargs)
                                        for _ in range(n_cpu)])
            if check_proc_spawn:
                proc_spawn = SubprocVecEnv([lambda: _sliencer(env, true_out, true_err, verbose=verbose)(*args, **kwargs)
                                            for _ in range(n_cpu)], start_method="spawn")
            else:
                proc_spawn = None
        finally:
            sys.stdout = true_out
            sys.stderr = true_err


        if n_steps is None:
            if verbose > 1:
                print("estimating number of steps...")
            default.reset()
            t0 = time.time()
            count = 0
            while time.time() - t0 < 0.25:
                default.step([default.action_space.sample() for _ in range(default.num_envs)])
                count += 1
            max_steps = count
            max_timeout = timeout
        else:
            default.reset()
            t0 = time.time()
            for _ in range(n_steps):
                default.step([default.action_space.sample() for _ in range(default.num_envs)])
            max_timeout = (time.time() - t0) * timeout * 1 / 0.25
            max_steps = n_steps

        @_timeout_dec(max_timeout)
        def bench_proc(env):
            __bench = _bench
            return timeit.timeit("__bench(env)", setup="env.reset()", number=n_loops, globals=locals())

        def _bench(env):
            for _ in range(max_steps):
                env.step([env.action_space.sample() for _ in range(env.num_envs)])

        if verbose > 1:
            print("benchmarking...")

        time_proc_fork = None
        try:
            time_proc_fork = bench_proc(proc_fork)
        except multiprocessing.TimeoutError:
            if verbose > 1:
                print("timeout for proc_fork...")
        proc_fork.close()

        time_thread = None
        try:
            time_thread = bench_proc(thread)
        except multiprocessing.TimeoutError:
            if verbose > 1:
                print("timeout for thread...")
        thread.close()

        time_default = timeit.timeit("_bench(default)", setup="default.reset()", number=n_loops, globals=locals())
        default.close()

        time_proc_spawn = None
        if check_proc_spawn:
            try:
                time_proc_spawn = bench_proc(proc_spawn)
            except multiprocessing.TimeoutError:
                if verbose > 1:
                    print("timeout for proc_spawn...")
            proc_spawn.close()

        fps_proc_fork = fps_thread = -1
        fps_default = (1 * max_steps * n_loops) / time_default
        if time_proc_fork is not None:
            fps_proc_fork = (n_cpu * max_steps * n_loops) / time_proc_fork
        if time_thread is not None:
            fps_thread = (n_cpu * max_steps * n_loops) / time_thread

        if check_proc_spawn and time_proc_spawn is not None:
            fps_proc_spawn = (n_cpu * max_steps * n_loops) / time_proc_spawn
        else:
            fps_proc_spawn = -1

        if fps_proc_fork > fps_default and fps_proc_fork > fps_thread and fps_proc_fork > fps_proc_spawn:
            if verbose > 0:
                print("best is SubprocVecEnv (with fork): ")
            ret = SubprocVecEnv
        elif fps_proc_spawn > fps_default and time_proc_spawn > fps_thread:
            if verbose > 0:
                print("best is SubprocVecEnv (with spawn): ")
            ret = SubprocVecEnv
        elif fps_thread > fps_default:
            if verbose > 0:
                print("best is MultithreadVecEnv: ")
            ret = MultithreadVecEnv
        else:
            if verbose > 0:
                print("best is DummyVecEnv: ")
            ret = DummyVecEnv

        if verbose > 0:
            relative_proc = "{:.2f}%".format(fps_proc_fork/fps_default) if fps_proc_fork != -1 else "timeout"
            relative_thread = "{:.2f}%".format(fps_thread/fps_default) if fps_thread != -1 else "timeout"
            relative_spawn = "{:.2f}%".format(fps_proc_spawn/fps_default) if fps_proc_spawn != -1 else "timeout"
            if check_proc_spawn:
                print("\tproc_fork={},\n \tproc_spawn={},\n \tthread={},\n "
                      .format(relative_proc, relative_spawn, relative_thread) +
                      "fps improvement from DummyVecEnv over {} cpus and {} steps."
                      .format(n_cpu, max_steps))
            else:
                print("\tproc_fork={},\n \tthread={},\n "
                      .format(relative_proc, relative_thread) +
                      "fps improvement from DummyVecEnv over {} cpus and {} steps."
                      .format(n_cpu, max_steps))

        if _return_conf:
            return ret, n_cpu
        return ret

    return _wrapper_maker


def auto_vectorized(env, n_cpu=-1, n_steps=None, n_loops=1, timeout=2, verbose=0):
    """
    Auto detects and returns the vectorized environment
    usage:
    `
    env = auto_vectorized(gym.make)("MontezumaRevenge-v4")
    `

    :param env: (gym.Env) the environment to vectorize
    :param n_cpu: (int) the number of CPUs to use for the benchmark (default: autodetect)
    :param n_steps: (int) the number of steps for the benchmark (default; ~1's worth of steps over all the tests)
    :param n_loops: (int) the number of loops for the benchmark (default: 1)
    :param timeout: (float) the timeout (in seconds) for the wrapper (default: 5s)
    :param verbose: (int) the verbosity of the benchmark (default: 0)
    :return: (VecEnv) the vectorized environment
    """
    from . import DummyVecEnv

    def _wrapper_maker(*args, **kwargs):
        vec_env_type, _n_cpu = detect_best_wrapper(env, n_cpu=n_cpu, n_steps=n_steps, n_loops=n_loops, verbose=verbose,
                                                   timeout=timeout, _return_conf=True)(*args, **kwargs)

        if vec_env_type == DummyVecEnv:
            return DummyVecEnv([lambda: env(*args, **kwargs)])
        return vec_env_type([lambda: env(*args, **kwargs) for _ in range(_n_cpu)])

    return _wrapper_maker


def copy_obs_dict(obs):
    """
    Deep-copy a dict of numpy arrays.

    :param obs: (OrderedDict<ndarray>): a dict of numpy arrays.
    :return (OrderedDict<ndarray>) a dict of copied numpy arrays.
    """
    assert isinstance(obs, OrderedDict), "unexpected type for observations '{}'".format(type(obs))
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def dict_to_obs(space, obs_dict):
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param space: (gym.spaces.Space) an observation space.
    :param obs_dict: (OrderedDict<ndarray>) a dict of numpy arrays.
    :return (ndarray, tuple<ndarray> or dict<ndarray>): returns an observation
            of the same type as space. If space is Dict, function is identity;
            if space is Tuple, converts dict to Tuple; otherwise, space is
            unstructured and returns the value raw_obs[None].
    """
    if isinstance(space, gym.spaces.Dict):
        return obs_dict
    elif isinstance(space, gym.spaces.Tuple):
        assert len(obs_dict) == len(space.spaces), "size of observation does not match size of observation space"
        return tuple((obs_dict[i] for i in range(len(space.spaces))))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


def obs_space_info(obs_space):
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: (gym.spaces.Space) an observation space
    :return (tuple) A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, gym.spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}
    else:
        assert not hasattr(obs_space, 'spaces'), "Unsupported structured space '{}'".format(type(obs_space))
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes
