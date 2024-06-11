import yaml
import os
from glob import glob
import sys
import pickle
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from transformations import homogeneous_transform, inverse_homogeneous_transform, rigid_transform_3D



id = '1711130942'
# analzye_2d = False
# analzye_3d = False
with open('../data/raw/task_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
project_dir = config['project_path']  # Modify this to your need.
transformation_dir = os.path.join(project_dir, 'transformations')
destfolder = f'./{id}'
# DLC3D = config['dlc3d_path']
# DLC2D = config['dlc_path']
# obj = 'robot'
# config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
# config3d = glob(os.path.join(DLC3D, f'*wrist_cam*', 'config.yaml'))[0]
#
#
# # analyze 2d
# if analzye_2d:
#     deeplabcut.analyze_time_lapse_frames(config2d, destfolder, shuffle=1, frametype='.jpeg',
#                             trainingsetindex=0,gputouse=None,save_as_csv=True)
#     # ### modify h5 files
#     h5files = glob(os.path.join(destfolder, '*.h5'))
#     _modify_h5files(h5files, obj=obj, multi=False)
#
# ## analyze 3d
dir_combined_3d = os.path.join(destfolder, '3d_combined')
# if analzye_3d:
#     df_3d = triangulate_images(config3d, h5files[0], destfolder = dir_combined_3d, pcutoff=0)

### Load tcp poses
traj_file = os.path.join(id, 'action_0.csv')
df_traj = pd.read_csv(traj_file)
wrist_cam_inds = np.where(df_traj['wrist_camera_time'] == 1)[0]

h5_3d = glob(os.path.join(dir_combined_3d, '*.h5'))[0]
df_3d = pd.read_hdf(h5_3d).droplevel([0, 1], axis = 1)
n_images = len(df_3d)
n_wrist_cam_captures = len(wrist_cam_inds)
assert n_images == n_wrist_cam_captures, 'n_images should equal to number of captures'

points_base_in_tcp = []
points_base_in_cam = []
offset = np.array([74.5, 0, 0]) ### The red dot position in the robot base reference frame, unit mm
scale = 1000 ### change the gripper traj's unit from meter to mm

for i, ind in enumerate(wrist_cam_inds):
    pos = df_traj.iloc[ind][['actual_TCP_pose_0', 'actual_TCP_pose_1',
       'actual_TCP_pose_2']].to_numpy() * scale - offset ## Subtracting the offset such that we get the tcp position int thre red dot's reference frame
    rot_mat = R.from_rotvec(df_traj.iloc[ind][['actual_TCP_pose_3', 'actual_TCP_pose_4',
       'actual_TCP_pose_5']]).as_matrix()
    homo_matrix = homogeneous_transform(rot_mat, pos)
    inverse_homo_matrix = inverse_homogeneous_transform(homo_matrix)
    points_base_in_tcp.append(inverse_homo_matrix[:-1, -1])
    points_base_in_cam.append(df_3d.iloc[i][['x', 'y', 'z']].to_numpy())

points_base_in_cam = np.array(points_base_in_cam)
points_base_in_tcp = np.array(points_base_in_tcp)
# rotation_matrix, translation = rigid_transform_3D(points_base_in_tcp, points_base_in_cam)
rotation_matrix, translation = rigid_transform_3D(points_base_in_cam, points_base_in_tcp)
H_wrist_cam_in_tcp = homogeneous_transform(rotation_matrix, translation)

with open(os.path.join(transformation_dir, 'wrist_cam_in_tcp.pickle'), 'wb') as f:
    pickle.dump(H_wrist_cam_in_tcp, f)

