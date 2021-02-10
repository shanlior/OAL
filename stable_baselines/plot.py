# DEPRECATED, use baselines.common.plot_util instead
on_policy = True



import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns

sns.set()
import glob2
import argparse
import visdom;

vis = visdom.Visdom()
from PIL import Image

sns.set_style("whitegrid")
sns.set_context("paper")


# sns.set(rc={'figure.figsize':(200, 200)})

def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
                                                                          mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    print("keys", keys, file)
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)
        else:
            padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
            x_padded = np.concatenate([x, padding], axis=0)
            assert x_padded.shape[1:] == x.shape[1:]
            assert x_padded.shape[0] == maxlen
            padded_xs.append(x_padded)
    return np.array(padded_xs)


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--smooth', type=int, default=1)
# parser.add_argument('--name', type=str, default='None')
args = parser.parse_args()

off_policy = not on_policy
if on_policy:
    method = 'on_policy'
else:
    method = 'off_policy'
# Load all data.
data = {}
paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, method, '**', 'progress.csv'))]

for curr_path in paths:
    if not os.path.isdir(curr_path):
        continue
    results = load_results(os.path.join(curr_path, 'progress.csv'))
    if not results:
        print('skipping {}'.format(curr_path))
        continue

    try:
        rewards = np.array(results['ep_rewmean'])
        steps = results['steps']  # np.arange(len(results['EpisodesSoFar'])) + 1
    except:
        rewards = np.array(results['EpTrueRewMean'])
        try:
            steps = results['TimestepsSoFar']  # np.arange(len(results['n_updates'])) + 1
        except:
            steps = results['steps']  # np.arange(len(results['EpisodesSoFar'])) + 1

    env_algo = os.path.split(os.path.split(curr_path)[0])
    env = os.path.split(env_algo[0])[1]
    algo = env_algo[1] #.split("_")[0]

    # Process and smooth data.
    assert rewards.shape == steps.shape
    x = steps
    y = rewards
    if args.smooth:
        x, y = smooth_reward_curve(steps, rewards)
    assert x.shape == y.shape

    if env not in data:
        data[env] = {}
    if algo not in data[env]:
        data[env][algo] = []
    data[env][algo].append((x, y))


legend = True
if on_policy or off_policy:
    color_id = {"mdal_linear": "#296E01FF", "mdal_neural": "#FF6F61FF", "gail": "#135DD8FF",
                "mdal_trpo_linear": "#296E01FF", "mdal_trpo_neural": "#FF6F61FF", "gail_off_policy": "#135DD8FF"}
else:
    color_id = {"mdal_linear": "#4F3466FF", "mdal_neural": "#FF6F61FF", "gail": "#FF0000FF",
                 "mdal_trpo_linear": "#296E01FF", "mdal_trpo_neural": "#FF30AAFF", "gail_off_policy": "#135DD8FF"}
# alg_names = {"mdal_linear": "GAL Linear", "mdal_neural": "GAL Neural", "gail": "GAIL",
#              "mdal_trpo_linear": "GAL Linear TRPO", "mdal_trpo_neural": "GAL Neural TRPO", "gail_off_policy": "GAIL MDPO",
#              "Expert": "Expert"}
alg_names = {"mdal_linear": "OAL Linear", "mdal_neural": "OAL Neural", "gail": "GAIL",
             "mdal_trpo_linear": "OAL Linear", "mdal_trpo_neural": "OAL Neural", "gail_off_policy": "GAIL",
             "Expert": "Expert"}
expert_rewards = {"walker2d": 3464, "hopper": 3053, "halfcheetah": 9052, "humanoid": 6494, "invertedpendulum": 1000}
axes_order = {"walker2d": 0, "hopper": 1, "halfcheetah": 2, "humanoid": 3, "invertedpendulum": 4}

uniform_legend = True
if uniform_legend:
    fig, axs = plt.subplots(ncols=4, figsize=(16,3))
else:
    fig, axs = plt.subplots(ncols=4, figsize=(16,6))
# Plot data.
for env_id in sorted(data.keys()):
    print('exporting {}'.format(env_id))
    # plt.clf()
    # plt.xlim(-0.1, 3)
    legend_entries = []
    if env_id == "invertedpendulum":
        continue
    axes_id = axes_order[env_id]
    ax = axs[axes_id]
    # ax = axs[axes_id // 2][axes_id % 2]
    x_max_total = 0
    for algo in sorted(data[env_id].keys()):
        legend_entries.append(algo)
        xs, ys = zip(*data[env_id][algo])
        # makes sure all trajectories are of the same length (comment out if not required)
        min_len = np.min([l.shape[0] for l in xs])
        xs, ys = [x[:min_len] for x in xs], [y[:min_len] for y in ys]

        if on_policy and env_id == "humanoid":
            xs, ys = [x[:4882] for x in xs], [y[:4882] for y in ys]
            ax.set_xlim([0,5])
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape
        x_max = np.max(xs) / 1e6
        if x_max > x_max_total:
            x_max_total = x_max
        if on_policy and env_id == "humanoid":
            x_max_total = 5
        mean = np.mean(ys, axis=0) / expert_rewards[env_id]
        std = np.nanstd(ys, axis=0)
        nSeeds = ys.shape[0]
        ci_coef = 1.96 / (np.sqrt(nSeeds) * expert_rewards[env_id])

        # if algo == "mdal_neural" and env_id == "walker2d":
        #     entry = 630
        #     xs = xs[:,:entry]
        #     mean = mean[:entry]
        #     std = std[:entry]
        if algo in color_id.keys():
            color = color_id[algo]
        else:
            color = np.random.rand(3,)
        ax.plot(xs[0] / 1e6, mean, label=algo, color=color)
        ax.fill_between(xs[0] / 1e6, mean - ci_coef * std, mean + ci_coef * std, alpha=0.2, color=color)
    expert_line = ax.hlines(1, 0, x_max_total, colors='k', linestyles='dashed')
    ax.text(0.05, 1.01, format(expert_rewards[env_id]), fontsize=11, rotation_mode='anchor')
    if not on_policy:
        ax.set_title(env_id, fontsize=14)
    if not off_policy:
        ax.set_xlabel('Timesteps (1e6)', fontsize=11)
    if axes_id == 0:
        ax.set_ylabel('Mean Episode Reward', fontsize=11)


    # ax.set_xticks(fontsize=11)
    # ax.set_yticks(fontsize=11)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    handles, labels = ax.get_legend_handles_labels()
    handles += [expert_line]
    labels += ['Expert']
    # order = [0, 1, 2, 3, 4]
    # legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=5)
    if not uniform_legend and legend:
        legend = ax.legend(handles, labels, ncol=1, bbox_to_anchor=(0.5, -1.3), loc='lower center')
    # legend = ax.legend(handles, labels, loc='lower center')
        fig.subplots_adjust(bottom=0.5)
    # env_fig.axes[0] = ax

    # Uncomment for separate graphs
    # import pickle
    # p = pickle.dumps(ax)
    # env_ax = pickle.loads(p)
    # env_ax.change_geometry(1,1,1)
    #
    # env_fig = plt.figure(figsize=(13,8))
    # env_fig._axstack.add(env_fig._make_key(env_ax), env_ax)
    # env_fig.axes.append(env_ax)
    # env_fig.tight_layout()
    # plt.gca().set_aspect('equal', adjustable='box', anchor='NW')
    # env_ax.legend(handles, labels, ncol=1, bbox_to_anchor=(0.5, -0.9), loc='lower center')
    #
    # env_fig.subplots_adjust(bottom=0.4)
    # env_fig.savefig(os.path.join(args.dir, 'fig_{}.png'.format(env_id)))


if uniform_legend:
    labels = [alg_names[label] for label in labels]
    if legend:
        # legend = ax.legend(handles, labels, ncol=7, bbox_to_anchor=(2.23, -0.5), loc='lower center', fontsize=11)
        legend = axs[3].legend(handles, labels, ncol=1, loc='right', bbox_to_anchor=(1, 0.33), fontsize=9)

    fig.subplots_adjust(bottom=0.3)
    # fig.set_figwidth(14)


fig.savefig(os.path.join(args.dir, method, '{}.png'.format(method)), bbox_inches='tight')
