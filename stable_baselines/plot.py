# DEPRECATED, use baselines.common.plot_util instead

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns;

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

# Load all data.
data = {}
paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]

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

config_id = {"MDAL": "#4F3466FF", "GAIL": "#FF6F61FF", "MDPO-Tsallis": "#FEAE51FF"}
expert_rewards = {"walker2d": 3464, "humanoid": 6494, "invertedpendulum": 1000, "halfcheetah": 9052}
# Plot data.
for env_id in sorted(data.keys()):
    print('exporting {}'.format(env_id))
    plt.clf()
    # plt.xlim(-0.1, 3)
    legend_entries = []
    for algo in sorted(data[env_id].keys()):
        legend_entries.append(algo)
        xs, ys = zip(*data[env_id][algo])
        # makes sure all trajectories are of the same length (comment out if not required)
        min_len = np.min([l.shape[0] for l in xs])
        xs, ys = [x[:min_len] for x in xs], [y[:min_len] for y in ys]

        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape
        x_max = np.max(xs) / 1e6
        mean = np.mean(ys, axis=0) / expert_rewards[env_id]
        nSeeds = ys.shape[0]
        ci_coef = 1.96 / (np.sqrt(nSeeds) * expert_rewards[env_id])

        plt.plot(xs[0] / 1e6, mean, label=algo)
        std = np.nanstd(ys, axis=0)
        plt.fill_between(xs[0] / 1e6, mean - ci_coef * std, mean + ci_coef * std, alpha=0.2)  # , color=config_id[config])
    plt.hlines(1, 0, x_max, colors='k', linestyles='dashed')
    plt.text(0.1, 1.01, 'Expert Reward = {}'.format(expert_rewards[env_id]), fontsize=11, rotation_mode='anchor')

    plt.title(env_id, fontsize=14)
    plt.xlabel('Timesteps', fontsize=11)
    plt.ylabel('Mean Episode Reward', fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0, 1, 2, 3, 4]
    # legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], ncol=5)
    legend = plt.legend(handles, labels, ncol=2, loc='upper right')
    # legend = plt.legend(loc='upper right')

    plt.savefig(os.path.join(args.dir, 'fig_{}.png'.format(env_id)))


    def export_legend(legend, filename="legend.png", expand=[-10, -10, 10, 10]):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent()
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join(args.dir, filename), dpi="figure", bbox_inches=bbox)


    export_legend(legend)


    # img = Image.open(os.path.join(args.dir, 'legend.png')).convert('RGBA')
    img = Image.open(os.path.join(args.dir, 'fig_{}.png'.format(env_id))).convert('RGBA')
    arr = np.array(img)
    arr = np.transpose(arr, (2, 0, 1))
    # vis.image(arr)
