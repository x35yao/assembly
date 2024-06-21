import yaml
import os
import numpy as np
import pandas as pd
from dtw_util import *
import pickle
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from quaternion_metric import process_quaternions
from transformations.transformations import lintrans, homogeneous_transform, get_HT_objs_in_base
import shutil
# from .transformations import get_HT_objs_in_base, lintrans, homogeneous_transform
# from .quaternion_metric import process_quaternions
from matplotlib import pyplot as plt

def align_trajectoires(gripper_trajs, alignment_func, align_with = 'speed'):
    '''
    This function will align gripper trajectories with Dynamic time warping based on speed.

    Parameters:
    ----------
    gripper_trajs: dict
        A dictionary that contains the gripper trajectories, where the keys are the demo ids and the values are Dataframes.

    Returns:
    --------
    gripper_trajs_aligned: The gripper trajectories that are aligned with median length trajectory.

    '''

    trajs_len = []
    gripper_trajs_aligned = {}
    demos = list(gripper_trajs.keys())
    for demo in demos:
        df = gripper_trajs[demo]
        if align_with == 'speed':
            time_diff = df['time'].diff(1)
            temp = ((np.sqrt(np.square(df.loc[:, ['x', 'y', 'z']].diff(1)).sum(axis=1)))) / time_diff
            gripper_trajs[demo]['speed'] = np.array(temp)
        elif align_with == 'quat_speed':
            time_diff = df['time'].diff(1)
            temp = ((np.sqrt(np.square(df.loc[:, ['qx', 'qy', 'qz', 'qw']].diff(1)).sum(axis=1))))
            gripper_trajs[demo]['quat_speed'] = np.array(temp)
        gripper_trajs[demo].dropna(inplace=True)
        trajs_len.append(len(gripper_trajs[demo]))
    # get demos with median duration
    median = int(np.median(trajs_len))
    if median not in trajs_len:
        # Deal with the case that there are even number of trajs
        median = min(trajs_len, key=lambda x: abs(x - median))
    median_len_ind = trajs_len.index(median)
    median_len_demo = demos[median_len_ind]

    ref_demo_speed = gripper_trajs[median_len_demo][align_with].to_numpy()
    ref_demo_traj = gripper_trajs[median_len_demo]
    min_cost_demos = {}
    for demo in tqdm(demos):
        test_demo_speed = gripper_trajs[demo][align_with].to_numpy()
        test_demo_traj = gripper_trajs[demo].copy().to_numpy()
        match_indices, min_cost = alignment_func(ref_demo_speed, test_demo_speed)
        match_indices = np.array(match_indices)
        min_cost_demos[demo] = min_cost
        new_demo = np.zeros(ref_demo_traj.shape)
        for match in match_indices:
            new_demo[match[0]] = test_demo_traj[match[1]]
        new_demo[-1] = test_demo_traj[-1]
        new_demo[0] = test_demo_traj[0]
        demo_aligned = ref_demo_traj.copy()
        demo_aligned.iloc[:, :] = new_demo
        gripper_trajs_aligned[demo] = demo_aligned
    return gripper_trajs_aligned, median_len_demo

def _remove_nan_entry_from_tempalte(template):
    objs = template.keys()
    for obj in objs:
        bps = template[obj].keys()
        tmp = template[obj].copy()
        for bp in bps:
            if np.isnan(tmp[bp]).any():
                del tmp[bp]
        template[obj] = tmp
    return template

def transform_df(df, H):
    '''
    :param df: The Dataframe contains the 3d position of bodyparts in reference frame A.
    :param H: The homogeneous tansformation that expresses reference frame A in reference B.
    :return: df_new: The Dataframe contains the 3d position of bodyparts in reference frame B.
    '''
    df_new = df.copy()
    if 'scorer' in df_new.columns.names:
        df_new = df_new.droplevel('scorer', axis=1)
    individuals = df_new.columns.get_level_values(level='individuals').unique()
    for individual in individuals:
        df_individual = df[individual]
        bpts = df_individual.columns.get_level_values(level='bodyparts').unique()
        for bp in bpts:
            pos = df[individual][bp][['x', 'y', 'z']].to_numpy()
            pos_new = lintrans(pos, H)
            df_new.loc[:,(individual, bp, ['x', 'y', 'z'])] = pos_new
    return df_new

# def get_obj_pose_zed(h5_3d_file, destdir, zed_in_base, obj_templates_in_base_for_zed, HT_template_in_base_for_zed):
#     df = pd.read_hdf(h5_3d_file).droplevel('scorer', axis=1)
#     df_in_base = transform_df(df, zed_in_base)
#
#     HT_objs_in_global_demo, dists = get_HT_objs_in_base(df_in_base, obj_templates_in_base_for_zed,
#                                                         HT_template_in_base_for_zed, window_size=5,
#                                                         markers_average=False)
#
#     individuals = df_in_base.columns.get_level_values(level='individuals').unique()
#     obj_poses = {}
#     successful = True
#     for individual in individuals:
#         dist = dists[individual]
#         if dist != np.inf:
#             H = HT_objs_in_global_demo[individual]
#             rotvect = R.from_matrix(H[:3, :3]).as_rotvec()
#             quat = R.from_matrix(H[:3, :3]).as_quat()
#             pos = H[:3, -1]
#             obj_poses[individual] = np.concatenate([pos, rotvect, quat])
#         else:
#             obj_poses[individual] = np.NAN
#     df_pose = pd.DataFrame.from_dict(obj_poses)
#     df_pose.index = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'qx', 'qy', 'qz', 'qw']
#     os.makedirs(destdir, exist_ok=True)
#     fname = os.path.join(destdir, f'obj_pose_zed.csv')
#     df_pose.to_csv(fname)
#     return df_pose

def get_obj_pose(h5_3d_file, destdir, cam_in_base, obj_templates_in_base, HT_template_in_base, suffix, window_size = 5):
    df = pd.read_hdf(h5_3d_file)
    if 'scorer' in df.columns.names:
        df = df.droplevel('scorer', axis=1)
    df_in_base = transform_df(df, cam_in_base)

    HT_objs_in_global_demo, dists = get_HT_objs_in_base(df_in_base, obj_templates_in_base,
                                                        HT_template_in_base, window_size=window_size,
                                                        markers_average=False)

    individuals = df_in_base.columns.get_level_values(level='individuals').unique()
    obj_poses = {}
    successful = True
    for individual in individuals:
        dist = dists[individual]
        if dist != np.inf:
            H = HT_objs_in_global_demo[individual]
            rotvect = R.from_matrix(H[:3, :3]).as_rotvec()
            quat = R.from_matrix(H[:3, :3]).as_quat()
            pos = H[:3, -1]
            obj_poses[individual] = np.concatenate([pos, rotvect, quat])
        else:
            obj_poses[individual] = np.NAN
    df_pose = pd.DataFrame.from_dict(obj_poses)
    df_pose.index = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'qx', 'qy', 'qz', 'qw']
    os.makedirs(destdir, exist_ok=True)
    fname = os.path.join(destdir, f'obj_pose_{suffix}.csv')
    df_pose.to_csv(fname)
    return df_pose

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

if __name__ == '__main__':
    n_actions = 3
    factor1 = 25 ### factor used to downsample the trajectory for dtw
    factor2 = 5 ### further downsampling after dtw
    scale = 1000 ### change the gripper traj's unit from meter to mm
    as_quat = True
    align_dims = ['grasp_detected', 'z', 'qx', 'qy', 'qz', 'qw']
    process_gripper_traj = True
    process_obj_pose = False

    with open('./data/task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config['project_path']  # Modify this to your need.
    raw_dir = os.path.join(project_dir, 'data', 'raw')
    processed_dir = raw_dir.replace('raw', 'processed')
    data_root, demo_dirs, data_files = next(os.walk(raw_dir))
    template_id = str(config['template_video_id'])
    objs = config['objects']
    alignment_func = dtw_funcs[config["alignment_method"]]


    with open(os.path.join(project_dir, 'transformations', 'zed_in_base.pickle'), 'rb') as f:
        zed_in_base = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'wrist_cam_in_tcp.pickle'), 'rb') as f:
        wrist_camera_in_tcp = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'HT_template_in_base_for_zed.pickle'), 'rb') as f:
        HT_template_in_base_for_zed = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'HT_template_in_base_for_wrist.pickle'), 'rb') as f:
        HT_template_in_base_for_wrist = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'template_in_base_for_zed.pickle'), 'rb') as f:
        obj_templates_in_base_for_zed = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'template_in_base_for_wrist.pickle'), 'rb') as f:
        obj_templates_in_base_for_wrist = pickle.load(f)

    wrist_cam_ind = {}
    for i in range(n_actions):
        ### Process gripper information by selecting some of the columns
        action = f'action_{i}'
        wrist_cam_ind[action] = []
        os.makedirs(os.path.join(processed_dir, action), exist_ok=True)
        tcp_poses = {}
        if process_gripper_traj:
            gripper_trajs = {}
            maxes_all_demo = {}
            mins_all_demo = {}
            for demo in sorted(demo_dirs):
                if demo == template_id:
                    continue
                else:
                    fname = os.path.join(raw_dir, demo, action, f'{action}_new.csv')
                    if os.path.isfile(fname):
                        df = pd.read_csv(fname, index_col=False)
                        del df['Unnamed: 0']
                        df.columns = ['time', 'x', 'y', 'z', 'rx', 'ry',
                                      'rz', 'grip_width', 'grasp_detected', 'wrist_camara_capture', 'fx', 'fy', 'fz',
                                      'tx', 'ty', 'tz']
                        if as_quat:
                            quats = R.from_rotvec(df[['rx', 'ry', 'rz']].values).as_quat()
                            quats_processed = process_quaternions(quats)
                            df[['qx', 'qy', 'qz', 'qw']] = quats

                        tcp_poses_demo = []
                        wrist_camera_inds = np.where(df['wrist_camara_capture'] == 1)[0]
                        tmp = 1
                        for ind in wrist_camera_inds:
                            df.loc[ind:, 'wrist_camara_capture'] = tmp
                            tmp += 1
                        for wrist_camera_ind in wrist_camera_inds:
                            tcp_pos = df.iloc[wrist_camera_ind][['x', 'y', 'z']].to_numpy() * scale
                            if as_quat:
                                tcp_ori = R.from_quat(df.iloc[wrist_camera_ind][['qx', 'qy', 'qz', 'qw']].to_numpy())
                            else:
                                tcp_ori = R.from_rotvec(df.iloc[wrist_camera_ind][['rx', 'ry', 'rz']].to_numpy())
                            tcp_ori_mat = tcp_ori.as_matrix()
                            tcp_pose_homo = homogeneous_transform(tcp_ori_mat, tcp_pos)
                            tcp_poses_demo.append(tcp_pose_homo)
                        tcp_poses[demo] = tcp_poses_demo
                        df = df.iloc[::factor1, :]
                        pos = df[['x', 'y', 'z']].to_numpy()
                        pos_normalized, maxes, mins = normalize(pos)
                        maxes_all_demo[demo] = maxes
                        mins_all_demo[demo] = mins
                        df[['x', 'y', 'z']] = pos_normalized
                        gripper_trajs[demo] = df
                    else:
                        continue
            gripper_trajs_aligned, median_len_demo = align_trajectoires(gripper_trajs, alignment_func, align_with=align_dims)

            for demo in gripper_trajs_aligned.keys():
                ## Save gripper trajectory to processed folder
                destdir = os.path.join(processed_dir, action, demo)
                os.makedirs(destdir, exist_ok=True)
                df_aligned = gripper_trajs_aligned[demo].iloc[::factor2, :]
                maxes, mins = maxes_all_demo[demo], mins_all_demo[demo]
                df_aligned.loc[:, ['x', 'y', 'z']] = denormalize(
                    df_aligned[['x', 'y', 'z']].to_numpy(), maxes, mins) * scale
                df_aligned.to_csv(os.path.join(destdir, f'{demo}.csv'))
                if demo == median_len_demo:
                    cam_inds = df_aligned['wrist_camara_capture'][df_aligned['wrist_camara_capture'].diff() != 0].index.tolist()
                    for cam_ind in cam_inds:
                        if cam_ind == 0:
                            continue
                        else:
                            wrist_cam_ind[action].append(np.where(np.array(df_aligned.index.tolist()) == cam_ind)[0][0])

        if process_obj_pose:
            bad_demos = []
            for demo in sorted(demo_dirs):
                if demo == template_id:
                    continue
                print(f'Processing action: {action}, demo: {demo}!!!!!!!')
                tcp_poses_demo = tcp_poses[demo]
                destdir = os.path.join(processed_dir, action, demo)
                ### Get object pose ###
                demo_dir = os.path.join(raw_dir, demo)
                h5_3d_file = os.path.join(demo_dir, action, '3d_combined', 'markers_trajectory_3d.h5')
                h5_3d_wrist_file = os.path.join(demo_dir, action, '3d_combined', 'markers_trajectory_wrist_3d.h5')
                if not os.path.isfile(h5_3d_file):
                    continue
                df = pd.read_hdf(h5_3d_file).droplevel('scorer', axis=1)
                df_in_base = transform_df(df, zed_in_base)

                HT_objs_in_global_demo, dists = get_HT_objs_in_base(df_in_base, obj_templates_in_base_for_zed,
                                                                    HT_template_in_base_for_zed, window_size=5, markers_average=False)

                individuals = df_in_base.columns.get_level_values(level='individuals').unique()
                obj_poses = {}
                for individual in individuals:
                    dist = dists[individual]
                    if dist != np.inf:
                        H = HT_objs_in_global_demo[individual]
                        rotvect = R.from_matrix(H[:3, :3]).as_rotvec()
                        quat = R.from_matrix(H[:3, :3]).as_quat()
                        pos = H[:3, -1]
                        obj_poses[individual] = np.concatenate([pos, rotvect, quat])
                    else:
                        bad_demos.append(demo)
                        obj_poses[individual] = np.NAN
                df_pose = pd.DataFrame.from_dict(obj_poses)
                df_pose.index = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'qx', 'qy', 'qz', 'qw']
                os.makedirs(destdir, exist_ok=True)
                fname = os.path.join(destdir, f'{demo}_obj.csv')
                df_pose.to_csv(fname)

                obj_poses_wrist = {}
                df_wrist = pd.read_hdf(h5_3d_wrist_file)
                for j in range(len(df_wrist)):
                    tcp_in_robot = tcp_poses[demo][j]
                    wrist_cam_in_base = tcp_in_robot @ wrist_camera_in_tcp
                    df_wrist_in_base = transform_df(df_wrist, wrist_cam_in_base)

                    HT_objs_in_global_demo_wrist, dists_wrist = get_HT_objs_in_base(df_wrist_in_base.iloc[[j]],obj_templates_in_base_for_wrist,
                                                                                    HT_template_in_base_for_wrist,
                                                                                    window_size=1, markers_average=False)

                    for individual in individuals:
                        dist = dists_wrist[individual]
                        if j == 1 and individual != 'bin1':
                            continue
                        if dist != np.inf:
                            H = HT_objs_in_global_demo_wrist[individual]
                            rotvect = R.from_matrix(H[:3, :3]).as_rotvec()
                            quat = R.from_matrix(H[:3, :3]).as_quat()
                            pos = H[:3, -1]
                            obj_poses_wrist[individual] = np.concatenate([pos, rotvect, quat])
                        else:
                            obj_poses_wrist[individual] = np.NAN
                df_pose_wrist = pd.DataFrame.from_dict(obj_poses_wrist)
                df_pose_wrist.index = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'qx', 'qy', 'qz', 'qw']
                fname = os.path.join(destdir, f'{demo}_obj_wrist.csv')
                df_pose_wrist.to_csv(fname)
            print(f'bad demos for action {action} are {set(bad_demos)}')
    with open(os.path.join(processed_dir, 'wrist_cam_ind.pickle'), 'wb') as f:
        pickle.dump(wrist_cam_ind, f)
