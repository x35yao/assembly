import yaml
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from process_data import align_trajectoires
from dtw_util import dtw_funcs
from scipy.spatial.transform import Rotation as R
from quaternion_metric import process_quaternions

factor = 25

def normalize(a):
    a_normlized = np.zeros(a.shape)
    maxes = []
    mins = []
    for i in range(a.shape[1]):
        maxes.append(np.max(a[:, i]))
        mins.append(np.min(a[:, i]))
        a_normlized[:, i] = (a[:, i] - np.min(a[:, i]))/(np.max(a[:, i]) - np.min(a[:, i]))
    return a_normlized, maxes, mins

def denormalize(a_normlized, maxes, mins):
    a = np.zeros(a_normlized.shape)
    for i in range(a.shape[1]):
        a[:, i] = a_normlized[:, i] * (maxes[i] - mins[i]) + mins[i]
    return a


with open('./data/task_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
project_dir = config['project_path']  # Modify this to your need.
processed_dir = os.path.join(project_dir, 'data', 'processed')
raw_dir = processed_dir.replace('processed', 'raw')
action = 'action_2'
demos = os.listdir(os.path.join(processed_dir, action))
dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'grasp_detected']
# dims = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'grasp_detected']
align_dims = ['grasp_detected', 'z', 'qx', 'qy', 'qz', 'qw']

df_trajs = {}
alignment_func = dtw_funcs[config["alignment_method"]]
# alignment_func = dtw_funcs["symmetric1"]
trajs = []
gripper_trajs = {}
maxes_all_demo = {}
mins_all_demo = {}
for demo in sorted(demos):
    traj_file_raw = os.path.join(raw_dir, demo, action, f'{action}_new.csv')
    df = pd.read_csv(traj_file_raw, index_col=0)
    df.columns = ['time', 'x', 'y', 'z', 'rx', 'ry',
                  'rz', 'grip_width', 'grasp_detected', 'wrist_camara_capture', 'fx', 'fy', 'fz',
                  'tx', 'ty', 'tz']
    quats = R.from_rotvec(df[['rx', 'ry', 'rz']].values).as_quat()
    quats_processed = process_quaternions(quats)
    df[['qx', 'qy', 'qz', 'qw']] = quats

    df = df.iloc[::factor, :]
    pos = df[['x', 'y', 'z']].to_numpy()
    pos_normalized, maxes, mins = normalize(pos)
    maxes_all_demo[demo] = maxes
    mins_all_demo[demo] = mins
    df[['x', 'y', 'z']] = pos_normalized
    gripper_trajs[demo] = df

# gripper_trajs_aligned, median_len_demo = align_trajectoires(gripper_trajs, alignment_func, align_with=['grasp_detected'])
gripper_trajs_aligned, median_len_demo = align_trajectoires(gripper_trajs, alignment_func, align_with=align_dims)

fig, axes = plt.subplots(len(dims), 1, sharex=True, constrained_layout=True)
trajs = []
for i, demo in enumerate(demos):
    maxes, mins = maxes_all_demo[demo], mins_all_demo[demo]
    gripper_trajs_aligned[demo][['x', 'y', 'z']] = denormalize(gripper_trajs_aligned[demo][['x', 'y', 'z']].to_numpy(), maxes, mins)
    trajs.append(gripper_trajs_aligned[demo][dims].to_numpy())
    for j, dim in enumerate(dims):
        axes[j].plot(gripper_trajs_aligned[demo][dim].to_numpy())
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
    # if j > 2:
    #     ax.set_ylim(-1.1, 1.1)
plt.show()