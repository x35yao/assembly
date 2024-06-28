import yaml
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
from transformations.transformations import homogeneous_transform, inverse_homogeneous_transform, lintrans
from matplotlib import pyplot as plt
import numpy as np

POS_THRESH = 20
ANGLE_THRESH = 10

with open('./data/task_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
project_dir = config['project_path']  # Modify this to your need.
processed_dir = os.path.join(project_dir, 'data', 'processed')
raw_dir = processed_dir.replace('processed', 'raw')
actions = ['action_0', 'action_1', 'action_2']
# actions = ['action_2']
dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
oris = ['euler', 'rotvec', 'quat']
ori = oris[1]
plot_action = 'action_2'
if ori == 'euler':
    plot_dims = ['x', 'y', 'z', '$\\alpha$', '$\\beta$', '$\\gamma$']
elif ori == 'rotvec':
    plot_dims = ['x', 'y', 'z', 'rx', 'ry', 'rz']
elif  ori == 'quat':
    plot_dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']


objs = config['objects']
for action in actions:
    if action == 'action_2':
        demos = sorted(os.listdir(os.path.join(processed_dir, action)))[:10]
        demos = [d for d in demos if d not in ['1709242793', '1709244566']]
    else:
        demos = os.listdir(os.path.join(processed_dir, action))
    trajs_ = []
    local_trajs = {obj: [] for obj in objs}
    bad_demos = []
    for demo in demos:
        traj_file = os.path.join(processed_dir, action, demo, f'{demo}.csv')
        df = pd.read_csv(traj_file)
        traj = df[dims].to_numpy()
        obj_pose_combined = os.path.join(processed_dir, action, demo, f'{demo}_obj_combined.csv')
        try:
            df_pose = pd.read_csv(obj_pose_combined, index_col=0)
        except FileNotFoundError:
            bad_demos.append(demo)
            continue
        for obj in objs:
            unique_obj = [tmp for tmp in df_pose.keys() if obj in tmp][0]
            obj_pose = df_pose.loc[dims, unique_obj].to_numpy()
            rotmat = R.from_quat(obj_pose[3:]).as_matrix()
            H_obj_in_global = homogeneous_transform(rotmat, obj_pose[:3])
            H_global_in_obj = inverse_homogeneous_transform(H_obj_in_global)

            traj_obj = lintrans(traj, H_global_in_obj)
            if ori == 'euler':
                traj_obj_euler = traj_obj.copy()[:,:-1]
                traj_obj_euler[:, 3:] = R.from_quat(traj_obj[:, 3:]).as_euler('xyz', degrees=True)
                local_trajs[obj].append(traj_obj_euler)
            elif ori == 'rotvec':
                traj_obj_rotvec = traj_obj.copy()[:, :-1]
                traj_obj_rotvec[:, 3:] = R.from_quat(traj_obj[:, 3:]).as_rotvec()
                local_trajs[obj].append(traj_obj_rotvec)
            else:
                local_trajs[obj].append(traj_obj)
    for demo in bad_demos:
        demos.remove(demo)
    # for obj in objs:
    #     pos_all = np.array(local_trajs[obj])
    #     fig = plt.figure(figsize=(9, 5))
    #     ax = fig.add_subplot(1, 1, 1, projection='3d')
    #     for pos in pos_all:
    #         dist = np.linalg.norm(pos[:, :3], axis=1)
    #         ind = np.argmin(dist)
    #         line = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
    #                        color='blue', label=f'traj')
    #         ax.plot(pos[0, 0], pos[ 0, 1], pos[0, 2], 'o',
    #                 color='blue')
    #         ax.plot(pos[ind, 0], pos[ind, 1], pos[ind, 2], 's',
    #                 color='red', )
    #     plt.show()

    stds = []
    means = []

    # std_obj = {}
    # mean_obj = {}
    var = {}
    for obj in objs:
        local_trajs[obj] = np.array(local_trajs[obj])
        var[obj] = np.std(local_trajs[obj], axis = 0) ** 2
        mean = np.mean(local_trajs[obj], axis=0)
        std = np.std(local_trajs[obj], axis=0)
        if action == plot_action:
            fig, axes = plt.subplots(len(plot_dims), 1, sharex=True, constrained_layout=True)
            for i, ax in enumerate(axes):
                ax.errorbar(np.arange(local_trajs[obj].shape[1]), mean[:, i], std[:, i], fmt='o', linewidth=2, capsize=6)
                ax.set_title(plot_dims[i])
                if i > 2 and ori == 'quat':
                    ax.set_ylim(-1.1, 1.1)
            fig.suptitle(obj, fontsize=16)

            fig, axes = plt.subplots(len(plot_dims), 1, sharex=True, constrained_layout=True)
            for i in range(local_trajs[obj].shape[0]):
                for j, ax in enumerate(axes):
                    ax.plot(local_trajs[obj][i,:,j])
                    ax.set_title(plot_dims[j])
                    if j > 2 and ori == 'quat':
                        ax.set_ylim(-1.1, 1.1)
            fig.suptitle(obj, fontsize=16)
            ind = 60
            n_dim = 4
            print(obj, np.max(std[ind, 3:]), np.max(std[ind, :3]))
            if obj == 'bolt':
                outliers = []
                lb = mean[ind, n_dim] - std[ind, n_dim]
                ub = mean[ind, n_dim] + std[ind, n_dim]
                for k, demo in enumerate(demos):
                    if local_trajs[obj][k, ind, n_dim] > ub or local_trajs[obj][k, ind, n_dim] < lb:
                        outliers.append(demo)
            #     print(outliers)
            # print(obj)
            # print(std[ind, :])
    plt.show()
    # import pickle
    # with open(os.path.join(project_dir, 'transformations', f'variances_{action}.pickle'), 'wb') as f:
    #     pickle.dump(var, f)

        # std_obj[obj] = np.std(local_trajs[obj], axis = 0)
        # mean_obj[obj] = np.mean(local_trajs[obj], axis = 0)
    #     closest_point = []
    #     dists = []
    #     for i, pos in enumerate(local_trajs[obj][:,:,:3]):
    #         dist = np.linalg.norm(pos, axis = 1)
    #         ind = np.argmin(dist)
    #         dists.append(dist[ind])
    #         closest_point.append(local_trajs[obj][i, ind,:])
    #     dists = np.array(dists)
    #     closest_point = np.array(closest_point)
    #     # closest_point_euler = closest_point.copy()[:, :-1]
    #     # closest_point_euler[:, 3:] = R.from_quat(closest_point[:, 3:]).as_euler('xyz', degrees=True)
    #     mean = np.mean(np.array(closest_point), axis=0)
    #     std = np.std(np.array(closest_point), axis=0)
    #     stds.append(std)
    #     means.append(mean)
    #
    # stds = np.array(stds)
    # means = np.array(means)
    # # print(means.shape)
    # # print(means)
    # # print('\n')
    # # print(stds)
    # # print('aaaaaaa')
    # fig, axes = plt.subplots(len(plot_dims), 1, sharex=True, constrained_layout=True)
    # for i, ax in enumerate(axes):
    #     ax.errorbar(objs, means[:, i], stds[:, i], fmt='o', linewidth=2, capsize=6)
    #     ax.set_title(plot_dims[i])
    # plt.show()
    # print(f'\n {action}: ')
    # for i, obj in enumerate(objs):
    #     pos_std = stds[i, :3]
    #     print(pos_std, obj)
    #     ori_std = stds[i, 3:]
    #     print(ori_std)
    #     if (pos_std < POS_THRESH).any():
    #         print(f'{obj} is related')

    # for obj in std_obj.keys():
    #     std = np.array(std_obj[obj])
    #     mean = np.array(mean_obj[obj])
    #     fig, axes = plt.subplots(len(dims), 1, sharex=True, constrained_layout=True)
    #     for i, ax in enumerate(axes):
    #         ax.errorbar(np.arange(len(mean)), mean[:, i], std[:, i], fmt='o', linewidth=2, capsize=6)
    # plt.show()



