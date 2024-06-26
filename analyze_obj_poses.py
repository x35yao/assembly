import yaml
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from plot_quaternion import plot_quaternion

POS_THRESH = 5
ANGLE_THRESH = 10

with open('./data/task_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
project_dir = config['project_path']  # Modify this to your need.
processed_dir = os.path.join(project_dir, 'data', 'processed')
raw_dir = processed_dir.replace('processed', 'raw')
actions = ['action_0', 'action_1', 'action_2']
dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
plot_dims = ['x', 'y', 'z', '$\\alpha$', '$\\beta$', '$\\gamma$']
plot_dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
# dims = ['x', 'y', 'z', 'rx', 'ry', 'rz']
objs = config['objects']
colors = {'nut': 'yellow', 'bin':'black', 'jig':'purple', 'bolt':'green'}

obj_poses_all_actions = {action: [] for action in actions}
for action in actions:
    demos = os.listdir(os.path.join(processed_dir, action))
    obj_poses = {obj:[] for obj  in objs}
    bad_demos = []
    for i, demo in enumerate(demos):
        obj_pose_combined = os.path.join(processed_dir, action, demo, f'{demo}_obj_combined.csv')
        try:
            df = pd.read_csv(obj_pose_combined, index_col=0)
        except FileNotFoundError:
            bad_demos.append(demo)
            continue
        for obj in objs:
            unique_obj = [tmp for tmp in df.keys() if obj in tmp][0]
            obj_poses[obj].append(df.loc[dims, unique_obj].to_numpy())
    for demo in bad_demos:
        demos.remove(demo)
    ### Compute mean and std
    stds = []
    means = []
    if action == 'action_0':
        for i in range(len(obj_poses['nut'])):
            fig = plt.figure(figsize=(18, 6))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            for obj in objs:
                quat = obj_poses[obj][i][3:]
                pos = obj_poses[obj][i][:3]
                plot_quaternion(ax, quat, pos, color = colors[obj], length=20)
            fig.suptitle(demos[i], fontsize=16)
        plt.show()


    # for i, obj in enumerate(objs):
    #     obj_poses[obj] = np.array(obj_poses[obj])
    #     tmp = obj_poses[obj].copy()
    #     # tmp = obj_poses[obj].copy()[:, :-1]
    #     # tmp[:, 3:] = R.from_quat(obj_poses[obj][:, 3:]).as_euler('xyz', degrees=True)
    #     mean = np.mean(np.array(tmp), axis = 0)
    #     std = np.std(np.array(tmp), axis = 0)
    #     stds.append(std)
    #     means.append(mean)
    # stds = np.array(stds)
    # means = np.array(means)
    # fig, axes = plt.subplots(len(plot_dims), 1, sharex=True, constrained_layout=True)
    # for i, ax in enumerate(axes):
    #     ax.errorbar(objs, means[:, i], stds[:, i], fmt='o', linewidth=2, capsize=6)
    #     ax.set_title(plot_dims[i])
    # fig.suptitle(action, fontsize=16)
    #
    # print(f'\n {action}: ')
    # for i, obj in enumerate(objs):
    #     pos_std = stds[i,:3]
    #     ori_std = stds[i, 3:]
    #     ori_mean = means[i, 3:]
    #     # outliers = []
    #     # for j,demo in enumerate(demos):
    #     #     qx = obj_poses[obj][j][3]
    #     #     if np.abs(ori_mean[0] - qx) > ori_std[0]:
    #     #         # print(obj, demo, qx)
    #     #         outliers.append(demo)
    #     # print(f'The outliers for {action}, {obj} are {outliers}')
    #
    #     # print(obj, pos_std, ori_std)
    #     augment_pos = (pos_std > POS_THRESH).any()
    #     augment_ori = (ori_std > ANGLE_THRESH).any()
    #     print(f'{obj}: augment position {augment_pos}, augment orientation {augment_ori}')
    # plt.show()



