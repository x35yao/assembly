import yaml
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
from transformations.transformations import homogeneous_transform, inverse_homogeneous_transform, lintrans
from matplotlib import pyplot as plt
import numpy as np

POS_THRESH = 10
ANGLE_THRESH = 10

with open('./data/task_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
project_dir = config['project_path']  # Modify this to your need.
processed_dir = os.path.join(project_dir, 'data', 'processed')
raw_dir = processed_dir.replace('processed', 'raw')
actions = ['action_0', 'action_1', 'action_2']
dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
plot_dims = ['x', 'y', 'z', '$\\alpha$', '$\\beta$', '$\\gamma$']

objs = config['objects']
for action in actions:
    demos = os.listdir(os.path.join(processed_dir, action))
    trajs_ = []
    local_trajs = {obj: [] for obj in objs}
    for demo in demos:
        traj_file = os.path.join(processed_dir, action, demo, f'{demo}_new.csv')
        df = pd.read_csv(traj_file)
        traj = df[dims].to_numpy()
        for obj in objs:
            obj_pose_combined = os.path.join(processed_dir, action, demo, f'{demo}_obj_combined.csv')
            try:
                df_pose = pd.read_csv(obj_pose_combined, index_col=0)
            except FileNotFoundError:
                continue
            unique_obj = [tmp for tmp in df_pose.keys() if obj in tmp][0]
            obj_pose = df_pose.loc[dims, unique_obj].to_numpy()
            rotmat = R.from_quat(obj_pose[3:]).as_matrix()
            H_obj_in_global = homogeneous_transform(rotmat, obj_pose[:3])
            H_global_in_obj = inverse_homogeneous_transform(H_obj_in_global)

            traj_obj = lintrans(traj, H_global_in_obj)
            local_trajs[obj].append(traj_obj)

    # for obj in objs:
    #     pos_all = np.array(local_trajs[obj])
    #     fig = plt.figure(figsize=(9, 5))
    #     ax = fig.add_subplot(1, 1, 1, projection='3d')
    #     for pos in pos_all:
    #         line = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
    #                        color='blue', label=f'traj')
    #         ax.plot(pos[0, 0], pos[ 0, 1], pos[0, 2], 'o',
    #                 color='blue')
    #         ax.plot(pos[-1, 0], pos[-1, 1], pos[-1, 2], 'x',
    #                 color='blue', )
    #     plt.show()
    stds = []
    means = []
    for obj in objs:
        local_trajs[obj] = np.array(local_trajs[obj])
        closest_point = []
        dists = []
        for i, pos in enumerate(local_trajs[obj][:,:,:3]):
            dist = np.linalg.norm(pos, axis = 1)
            ind = np.argmin(dist)
            dists.append(dist[ind])
            closest_point.append(local_trajs[obj][i, ind,:])
        dists = np.array(dists)
        closest_point = np.array(closest_point)
        closest_point_euler = closest_point.copy()[:, :-1]
        closest_point_euler[:, 3:] = R.from_quat(closest_point[:, 3:]).as_euler('xyz', degrees=True)
        mean = np.mean(np.array(closest_point_euler), axis=0)
        std = np.std(np.array(closest_point_euler), axis=0)
        stds.append(std)
        means.append(mean)

    stds = np.array(stds)
    means = np.array(means)
    print(means)
    print('\n')
    print(stds)
    raise

    fig, axes = plt.subplots(len(plot_dims), 1, sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.errorbar(objs, means[:, i], stds[:, i], fmt='o', linewidth=2, capsize=6)
        ax.set_title(plot_dims[i])
    plt.show()
    print(f'\n {action}: ')
    for i, obj in enumerate(objs):
        pos_std = stds[i, :2]
        pos_mean = np.abs(means[i, :2])
        print(pos_mean)
        ori_std = stds[i, -1]
        augment_pos = (pos_mean < POS_THRESH).any()
        print(augment_pos, 'aaaaaaaaaaaa')
        augment_ori = (ori_std > ANGLE_THRESH).any()
        print(f'{obj}: Object position affects trajectory position {augment_pos}, augment orientation {augment_ori}')

