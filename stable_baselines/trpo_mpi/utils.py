import numpy as np


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def add_successor_features(seg, gamma):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1

    dones = seg["dones"]
    observations = seg["observations"]
    successor_features = np.zeros(observations.shape[1])
    sum_successor_features = np.zeros(observations.shape[1])

    episode_starts = seg["episode_starts"]
    n_episodes = 0
    for episode_start, observation, done in zip(reversed(episode_starts), reversed(observations),reversed(dones)):
        if done:
            sum_successor_features = np.add(sum_successor_features, successor_features)
            successor_features = np.zeros(observation.shape)
        else:
            successor_features = np.add(gamma * successor_features, (1 - gamma) * observation)
        if episode_start:
            n_episodes += 1

    successor_features = sum_successor_features / n_episodes
    return successor_features