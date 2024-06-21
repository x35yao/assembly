import yaml
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from process_data import align_trajectoires
from dtw_util import dtw_funcs
from scipy.spatial.transform import Rotation as R
from quaternion_metric import process_quaternions

QUAT_THRES = 0.05
POS_THRES = 3

with open('./data/task_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
project_dir = config['project_path']  # Modify this to your need.
processed_dir = os.path.join(project_dir, 'data', 'processed')
raw_dir = processed_dir.replace('processed', 'raw')
actions = ['action_0', 'action_1','action_2']
dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'grasp_detected']
align_dims = ['grasp_detected', 'z', 'qx', 'qy', 'qz', 'qw']
mask_dims = ['x', 'y', 'qx', 'qy', 'qz', 'qw']

df_trajs = {}
alignment_func = dtw_funcs[config["alignment_method"]]
# alignment_func = dtw_funcs["symmetric1"]
for action in actions:
    demos = os.listdir(os.path.join(processed_dir, action))
    fig, axes = plt.subplots(len(dims), 1, sharex=True, constrained_layout=True)
    trajs = []
    for i, demo in enumerate(demos):
        traj_file = os.path.join(processed_dir, action, demo, f'{demo}.csv')
        df = pd.read_csv(traj_file, index_col=0)
        trajs.append(df[dims].to_numpy())
        for j, dim in enumerate(dims):
            axes[j].plot(df[dim].to_numpy())
            # if j > 2:
            #     axes[j].set_ylim(-1.1, 1.1)
            axes[j].set_title(dim)

    ### Compute stds
    stds = []
    means = []
    trajs = np.array(trajs)
    length = trajs.shape[1]
    for i in range(length):
        stds.append(np.std(trajs[:,i,:], axis = 0))
        means.append(np.mean(trajs[:, i, :], axis = 0))
    stds = np.array(stds)
    means = np.array(means)
    x = np.arange(len(means))

    ### Plot stds
    fig, axes = plt.subplots(stds.shape[1], 1, sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        ax.errorbar(x ,means[:,j],stds[:, j])
        ax.set_title(dims[j])
        # ax.set_ylim(-1.1, 1.1)

    ### Plot pos stds lower than threshold
    fig, axes = plt.subplots(4, 1, sharex=True, constrained_layout=True)
    mask_pos = np.ones(stds.shape[0])
    for j, ax in enumerate(axes):
        if j != len(axes) - 1:
            ax.errorbar(x, means[:, j], stds[:, j])
            ax.set_title(dims[j])
            dim = dims[j]
            mask_new = stds[:, j] < POS_THRES
            mask_pos = mask_pos * mask_new
            inds = np.where(mask_new)[0]
            ax.scatter(inds, means[inds, j], color='red')
        else:
            inds = np.where(mask_pos)[0]
            ax.scatter(inds, mask_pos[inds], color = 'red')

    ### Plot ori stds lower than threshold
    fig, axes = plt.subplots(5, 1, sharex=True, constrained_layout=True)
    mask_ori = np.ones(stds.shape[0])
    for j, ax in enumerate(axes):
        if j != len(axes) - 1:
            j += 3
            ax.errorbar(x, means[:, j], stds[:, j])
            ax.set_title(dims[j])
            dim = dims[j]
            mask_new = stds[:, j] < QUAT_THRES
            mask_ori = mask_ori * mask_new
            inds = np.where(mask_new)[0]
            ax.scatter(inds, means[inds, j], color='red')
        else:
            inds = np.where(mask_ori)[0]
            ax.scatter(inds, mask_ori[inds], color='red')
    plt.show()