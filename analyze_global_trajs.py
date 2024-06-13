import yaml
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from process_data import align_trajectoires
from dtw_util import *
from scipy.spatial.transform import Rotation as R
from quaternion_metric import process_quaternions


def normalize(a):
    a_normlized = np.zeros(a.shape)
    for i in range(a.shape[1]):
        a_normlized[:, i] = 2 * (a[:, i] - np.min(a[:, i]))/(np.max(a[:, i]) - np.min(a[:, i])) - 1
    return a_normlized

with open('./data/task_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
project_dir = config['project_path']  # Modify this to your need.
processed_dir = os.path.join(project_dir, 'data', 'processed')
raw_dir = processed_dir.replace('processed', 'raw')
action = 'action_2'
demos = os.listdir(os.path.join(processed_dir, action))
dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

df_trajs = {}
alignment_func = dtw_funcs[config["alignment_method"]]

for demo in sorted(demos):
    traj_file = os.path.join(processed_dir, action, demo, f'{demo}_new.csv')
    df = pd.read_csv(traj_file)
    df_trajs[demo] = df
df_trajs, median_length = align_trajectoires(df_trajs, alignment_func, align_with=['qx', 'qy', 'qz', 'qw'])

trajs = []
trajs_realigned = []
trajs_raw = []
for demo in sorted(demos):
    traj_file = os.path.join(processed_dir, action, demo, f'{demo}_new.csv')
    df = pd.read_csv(traj_file)
    trajs.append(df[dims].to_numpy())
    # df_raw = pd.read_csv(traj_file_raw)
    df_realigned = df_trajs[demo]
    trajs_realigned.append(df_realigned[dims].to_numpy())
    # traj_file_raw = os.path.join(raw_dir, demo, action, f'{action}.csv')
    # df_raw = pd.read_csv(traj_file_raw)
    # trajs_raw.append(df_raw[['actual_TCP_pose_0', 'actual_TCP_pose_1', 'actual_TCP_pose_2', 'actual_TCP_pose_3', 'actual_TCP_pose_4', 'actual_TCP_pose_5']].to_numpy())

trajs = np.array(trajs)
length = trajs.shape[1]

### Compute stds
stds = []
means = []
for i in range(length):
    stds.append(np.std(trajs[:,i,:], axis = 0))
    means.append(np.mean(trajs[:, i, :], axis = 0))
stds = np.array(stds)
means = np.array(means)
x = np.arange(len(means))



fig2, axes2 = plt.subplots(stds.shape[1], 1)
for i in range(len(trajs)):
    for j, ax in enumerate(axes2):
        ax.plot(trajs[i, :, j])
        if j > 2:
            ax.set_ylim(-1.1, 1.1)

### Plot stds
fig, axes = plt.subplots(stds.shape[1], 1)
for j, ax in enumerate(axes):
    ax.errorbar(x ,means[:,j],stds[:, j])
    ax.set_title(dims[j])
    if j > 2:
        ax.set_ylim(-1.1, 1.1)

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
line = ax.plot(trajs[0, :, 0], trajs[0, :, 1], trajs[0, :, 2],
                   color='blue', label=f'traj')
ax.plot(trajs[0, 0, 0], trajs[0, 0, 1], trajs[0, 0, 2], 'o',
                   color = 'blue')
ax.plot(trajs[0, -1, 0], trajs[0, -1, 1], trajs[0, -1, 2], 'x',
                   color = 'blue', )
lb = 135
ub = -1
ax.scatter(trajs[0, lb:ub, 0], trajs[0, lb:ub, 1], trajs[0, lb:ub, 2], 's',
                   color='red')
# print(np.where(trajs[:, 130, 3]< 0.8))

###
# rotvec = trajs_raw[0][:,3:]
# quat = R.from_rotvec(rotvec).as_quat()
# quat = process_quaternions(quat)
# fig4, axes4 = plt.subplots(4, 1)
# for j, ax in enumerate(axes4):
#     ax.plot(quat[:, j])
#     if j > 2:
#         ax.set_ylim(-1.1, 1.1)

quat_processed = trajs[10][:,3:]
rotvec_processed = R.from_quat(quat_processed).as_rotvec()
fig5, axes5 = plt.subplots(3, 1)
for j, ax in enumerate(axes5):
    ax.plot(rotvec_processed[:, j], 'blue')

# rotvec_new = R.from_quat(quat).as_rotvec()
# fig6, axes6 = plt.subplots(3, 1)
# for j, ax in enumerate(axes6):
#     ax.plot(rotvec[:, j], 'blue')
#     ax.plot(rotvec_new[:, j], 'red')
plt.show()