import threading
import pygame
import sys
from robot_control import command_camera,open_gripper, close_gripper, move_up, move_down, move_left, move_right, move_foward, move_backward, move_to_defalt_pose, move_to_defalt_ori, rotate_wrist_clockwise, rotate_wrist_counterclockwise, screw, unscrew, get_horizontal, get_vertical, stop, protective_stop, wrist1_plus, wrist2_plus, wrist1_minus, wrist2_minus,servoL
import datetime
import os
import numpy as np
# import nidaqmx
import rtde_receive
from rtde_control import RTDEControlInterface as RTDEControl
import rtde_control
import rtde_io
import time
import ctypes
import pickle
import cv2
import shutil
import yaml
from scipy.spatial.transform import Rotation as R
from transformer import TFEncoderDecoder5, create_tags, normalize_wrapper
from data_processing import get_obj_pose, homogeneous_transform, svo_to_avi, quat_to_rotvect, rotvect_to_quat,inverse_homogeneous_transform, vec2homo, homo2vec, process_quaternions
import pandas as pd
import torch
from matplotlib import pyplot as plt
import urllib
import subprocess

def is_capslock_on():
    return True if ctypes.WinDLL("User32.dll").GetKeyState(0x14) else False

def find_next_greater(a, x):
    for value in a:
        if value > x:
            return value
    return None

def plot_traj_and_obj_pos(traj_pos, obj_data = None, colors = None, all_objs = None):
    if colors is None:
        colors = {}
        colors['traj'] = 'blue'
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    line = ax.plot(traj_pos[:, 0], traj_pos[:, 1], (traj_pos[:, 2]),
                   color=colors['traj'], label=f'traj')
    ax.plot(traj_pos[-1, 0], traj_pos[-1, 1], traj_pos[-1, 2], 'x',
            color=colors['traj'], label=f'end')
    ax.plot(traj_pos[0, 0], traj_pos[0, 1], traj_pos[0, 2], 'o',
            color=colors['traj'], label=f'start')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if obj_data is not None:
        for i in range(len(obj_data)):
            obj_pos = obj_data[i, :3]
            unique_obj = all_objs[i]
            color = colors[unique_obj]
            ax.plot(obj_pos[0], obj_pos[1], obj_pos[2], 's', color=color, label = unique_obj)

    ax.legend()
    return fig, ax

def close_on_key(event):
    #### press any key to close the window.
    global image_key_pressed
    image_key_pressed = event.key
    plt.close()
    return event.key

class Robot():
    def __init__(self, host, frequency):
        self.host = host
        self.frequency = frequency
        self.dt = 1/frequency

    def connect(self):
        try:
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host, self.frequency)
            self.rtde_c = rtde_control.RTDEControlInterface(self.host, self.frequency, RTDEControl.FLAG_CUSTOM_SCRIPT)
            self.rtde = rtde_io.RTDEIOInterface(self.host)
            print("Robot connection success")
        except RuntimeError:
            print('Robot connection failure')

    def get_tcp_pose(self):
        return self.rtde_r.getActualTCPPose()

    def servoL(self, traj):
        servoL(traj, self.rtde_c, self.dt)

    def open_gripper(self):
        open_gripper(self.rtde)

    def close_gripper(self):
        close_gripper(self.rtde)

class Zed():
    def __init__(self, destfolder, videotype = 'avi'):
        self.destfolder = destfolder
        self.videotype = videotype

    def connect_camera(self):
        self.cam = command_camera()
        if self.cam.connected == 1:
            print("Camera connection success")
        else:
            print("Camera connection failure, is the server opened?")

    def capture_video(self, destfolder = None, duration = 1):
        if destfolder is None:
            destfolder = self.destfolder
        os.makedirs(destfolder, exist_ok=True)
        fname = os.path.join(destfolder, 'zed')
        self.cam.start_trial(fname)
        time.sleep(duration)
        self.cam.stop_trial()
        svo_to_avi(fname + '.svo2', outdir = destfolder)

    def analyze_video(self, destfolder = None, visualize = False):
        if destfolder is None:
            destfolder = self.destfolder
        env_name = r"C:/Users/xyao0/anaconda3/envs/DLC"
        script_path = "C:/Users/xyao0/Desktop/project/assembly/analyze_videos.py"
        conda_path = "C:/Users/xyao0/anaconda3/ScripDts/conda.exe"
        subprocess.run(["conda", "run", "-p", env_name, "python", script_path, destfolder, self.videotype, str(visualize)], shell=True)

    def clear_folder(self, destfolder=None):
        if destfolder is None:
            destfolder = self.destfolder
            # Check if the folder exists
        if os.path.exists(destfolder) and os.path.isdir(destfolder):
            for item in os.listdir(destfolder):
                item_path = os.path.join(destfolder, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)  # Remove file or symbolic link
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directory
            print(f"Cleared contents of '{destfolder}'")
        else:
            print(f"The folder '{destfolder}' does not exist or is not a directory.")

class Wrist():
    def __init__(self, destfolder, url_left, url_right, imagetype = 'png'):
        self.url_left = url_left
        self.url_right = url_right
        self.destfolder = destfolder
        self.imagetype = imagetype

    def take_pics(self, destfolder = None):
        if destfolder is None:
            destfolder = self.destfolder
        os.makedirs(destfolder, exist_ok=True)
        wrist_camera_left = urllib.request.urlopen(self.url_left)
        wrist_camera_right = urllib.request.urlopen(self.url_right)
        frame_left = cv2.imdecode(np.array(bytearray(wrist_camera_left.read()), dtype=np.uint8), -1)
        frame_right = cv2.imdecode(np.array(bytearray(wrist_camera_right.read()), dtype=np.uint8), -1)
        fname_left = str(datetime.datetime.now().timestamp()).split('.')[0] + '_left.png'
        fname_right = fname_left.replace('left', 'right')
        path_left = os.path.join(destfolder, fname_left)
        path_right = os.path.join(destfolder, fname_right)
        cv2.imwrite(path_left, frame_left)
        cv2.imwrite(path_right, frame_right)

    def analyze_image(self, destfolder = None, visualize = True):
        if destfolder is None:
            destfolder = self.destfolder
        env_name = r"C:/Users/xyao0/anaconda3/envs/DLC"
        script_path = "C:/Users/xyao0/Desktop/project/assembly/analyze_images.py"
        conda_path = "C:/Users/xyao0/anaconda3/ScripDts/conda.exe"
        subprocess.run(["conda", "run", "-p", env_name, "python", script_path, destfolder, self.imagetype, str(visualize)], shell=True)

    def clear_folder(self, destfolder = None):
        if destfolder is None:
            destfolder = self.destfolder
            # Check if the folder exists
        if os.path.exists(destfolder) and os.path.isdir(destfolder):
            for item in os.listdir(destfolder):
                item_path = os.path.join(destfolder, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)  # Remove file or symbolic link
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directory
            print(f"Cleared contents of '{destfolder}'")
        else:
            print(f"The folder '{destfolder}' does not exist or is not a directory.")

class Network():
    def __init__(self, max_len, train_stat, device, n_dims = 7, n_tasks = 3, n_objs = 5, embed_dim = 64, n_head = 8, n_encoder_layers = 3, n_decoder_layers = 3 ):
        self.traj_seq_dim = n_dims + n_objs
        self.norm_func = normalize_wrapper(train_stat['mean'], train_stat['std'])
        self.mean = train_stat['mean']
        self.std = train_stat['std']
        self.max_len = max_len
        self.model = TFEncoderDecoder5(task_dim=n_dims, target_dim= self.traj_seq_dim, source_dim= self.traj_seq_dim, n_objs = n_objs,
                                      n_tasks=n_tasks,
                                      embed_dim=embed_dim, nhead = n_head, max_len= max_len,
                                      num_encoder_layers=n_encoder_layers,
                                      num_decoder_layers=n_decoder_layers, device=device)
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def get_obj_tags(self, all_objs):
        self.obj_tags = create_tags(all_objs)

    def update_obj_sequence(self, df_pose_zed, df_pose_wrist , tcp_pose, all_objs, task_dims = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']):
        '''
        :param df_pose_zed: Current object pose detected by zed cam
        :param df_pose_wrist: Current object pose detected by wrist cam
        :param tcp_pose: Current tcp pose
        :param task_dims: Dimensions of the task
        :return:
        '''
        obj_pose_seq = []
        if df_pose_wrist is None:
            df_pose_wrist = df_pose_zed
        individuals = list(df_pose_zed.columns)
        for object_ind in all_objs:
            if object_ind != 'trajectory': ### Actual object excluding the trajectory object
                individual = [ind for ind in individuals if object_ind in ind][0]
                obj_pose_zed = df_pose_zed[individual][task_dims].to_numpy()
                obj_pose_wrist = df_pose_wrist[individual][task_dims].to_numpy()
                if not np.isnan(obj_pose_wrist).any():
                    obj_pose = obj_pose_wrist
                else:
                    obj_pose = obj_pose_zed
            else: ### The trajectory object
                obj_pose = tcp_pose
            obj_pose = np.concatenate([obj_pose, self.obj_tags[object_ind]])
            obj_pose_seq.append(obj_pose)
        self.obj_pose = np.array(obj_pose_seq)[:, :len(task_dims)]
        obj_pose_seq = torch.tensor(np.array(obj_pose_seq))
        obj_seq_normalized = self.norm_func(obj_pose_seq.clone())
        self.obj_seq = obj_seq_normalized[None, :, :]

        return self.obj_seq

    def predict_traj_and_action(self):
        self.model.eval()
        traj_hidden = torch.zeros((1, self.max_len, self.traj_seq_dim), dtype=torch.double)
        # traj_hidden[0, 0, :7] = torch.tensor([-1.9606, -0.4157, 0.4109, 0.9999, 0.0132, -0.0105, 0.0031])
        traj_hidden[0, :, -1] = 1
        padding_mask = torch.zeros(1, self.max_len)
        padding_mask = padding_mask > 0
        traj_seq, action_tag = self.model(self.obj_seq, traj_hidden, tgt_padding_mask=padding_mask, predict_action=False)
        traj_seq = traj_seq.detach().numpy()[0]
        traj = quat_to_rotvect(traj_seq)
        traj[:, :3] = (traj[:, :3] * self.std + self.mean) / SCALE
        # action_tag = action_tag.detach().numpy()[0]
        # action = np.argmax(action_tag)
        return traj, action_tag

    # def predict_traj(self, traj_len):
    #     self.model.eval()
    #     traj_hidden = torch.zeros((1, self.max_len, self.traj_seq_dim), dtype=torch.double)
    #     traj_hidden[0, 0, :7] = torch.tensor([-1.9606, -0.4157, 0.4109, 0.9999, 0.0132, -0.0105, 0.0031])
    #     traj_hidden[0, :, 7] = 1
    #     padding_mask = torch.zeros(1, self.max_len)
    #     padding_mask[0, traj_len:] = 1
    #     padding_mask = padding_mask > 0
    #     traj_seq, _ = self.model(self.obj_seq, traj_hidden, tgt_padding_mask=padding_mask, predict_action=True)
    #     traj_seq = traj_seq.detach().numpy()[0]
    #     traj = quat_to_rotvect(traj_seq)
    #     traj[:, :3] = (traj[:, :3] * self.std + self.mean) / SCALE
    #     return traj[:traj_len]
    #
    # def predict_action(self):
    #     self.model.eval()
    #     traj_hidden = torch.zeros((1, self.max_len, self.traj_seq_dim), dtype=torch.double)
    #     traj_hidden[0, 0, :7] = torch.tensor([-1.9606, -0.4157, 0.4109, 0.9999, 0.0132, -0.0105, 0.0031])
    #     traj_hidden[0, :, 7] = 1
    #     padding_mask = torch.zeros(1, self.max_len)
    #     padding_mask = padding_mask > 0
    #     _, action_tag = self.model(self.obj_seq, traj_hidden, tgt_padding_mask=padding_mask, predict_action=True)
    #     action_tag = action_tag.detach().numpy()[0]
    #     action = np.argmax(action_tag)
    #     return action
    
class PoseProcessor():
    def __init__(self, transformation_dir, destfolder_zed, destfolder_wrist):
        with open(os.path.join(transformation_dir, 'zed_in_base.pickle'), 'rb') as f:
            self.zed_in_base = pickle.load(f)
        with open(os.path.join(transformation_dir, 'wrist_cam_in_tcp.pickle'), 'rb') as f:
            self.wrist_camera_in_tcp = pickle.load(f)
        with open(os.path.join(transformation_dir, 'HT_template_in_base_for_zed.pickle'), 'rb') as f:
            self.HT_template_in_base_for_zed = pickle.load(f)
        with open(os.path.join(transformation_dir, 'HT_template_in_base_for_wrist.pickle'), 'rb') as f:
            self.HT_template_in_base_for_wrist = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_base_for_zed.pickle'), 'rb') as f:
            self.obj_templates_in_base_for_zed = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_base_for_wrist.pickle'), 'rb') as f:
            self.obj_templates_in_base_for_wrist = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_obj_for_zed.pickle'), 'rb') as f:
            self.obj_templates_in_obj_for_zed = pickle.load(f)
        with open(os.path.join(transformation_dir, 'template_in_obj_for_wrist.pickle'), 'rb') as f:
            self.obj_templates_in_obj_for_wrist = pickle.load(f)
        self.destfolder_zed = destfolder_zed
        self.destfolder_wrist = destfolder_wrist
        self.obj_pose_zed = None
        self.obj_pose_wrist = None


    def get_pose_zed(self, h5_3d_file = None):
        if h5_3d_file is None:
            h5_3d_file = os.path.join(self.destfolder_zed, '3d_combined', 'markers_trajectory_3d.h5')
        if not os.path.isfile(h5_3d_file):
            self.obj_pose_zed = None
        else:
            self.obj_pose_zed = get_obj_pose(h5_3d_file, self.destfolder_zed, self.zed_in_base, self.obj_templates_in_base_for_zed, self.HT_template_in_base_for_zed, suffix = 'zed',
                         window_size=5)

    def get_pose_wrist(self, current_pose, h5_3d_file = None):
        if h5_3d_file is None:
            h5_3d_file = os.path.join(self.destfolder_wrist, '3d_combined', 'markers_trajectory_wrist_3d.h5')
        if not os.path.isfile(h5_3d_file):
            self.obj_pose_wrist = None
        else:
            pos = np.array(current_pose[:3])
            rotmatrix = R.from_rotvec(current_pose[3:]).as_matrix()
            tcp_in_robot = homogeneous_transform(rotmatrix, pos)
            wrist_cam_in_base = tcp_in_robot @ self.wrist_camera_in_tcp
            self.obj_pose_wrist = get_obj_pose(h5_3d_file, self.destfolder_wrist, wrist_cam_in_base, self.obj_templates_in_base_for_wrist,
                                             self.HT_template_in_base_for_wrist, suffix='wrist',
                                             window_size=1)

class IBVS():
    def __init__(self, ibvs_dir, robot, wrist, max_lmbda = 0.1, thresh = 4, max_run = 4, scale = 1000):
        self.ibvs_dir = ibvs_dir
        self.wrist = wrist
        self.max_lmbda = max_lmbda
        self.thersh = thresh
        self.max_run = max_run
        self.finished = False
        self.robot = robot
        self.scale = scale

    def process_traj(self):
        cam_poses_path = os.path.join(self.ibvs_dir, 'plan//cam_poses.pickle')
        ### Convert tcp poses using wrist camera poses
        if os.path.isfile(cam_poses_path):
            with open(cam_poses_path, 'rb') as f:
                tmp = pickle.load(f)
            cam_poses = tmp[:-1]
            is_finished = tmp[-1]
            with open('C:/Users/xyao0/Desktop/project/assembly/transformations/2024-08-28/wrist_cam_in_tcp.pickle',
                      'rb') as f:
                H_cam_in_tcp = pickle.load(f)

            vec_tcp_in_base = np.array(self.robot.get_tcp_pose())
            H_tcp_in_base = vec2homo(vec_tcp_in_base)
            H_cam_in_tcp[:-1, -1] /= self.scale
            H_tcp_in_cam = inverse_homogeneous_transform(H_cam_in_tcp)

            tcp_poses = []
            tcp_poses_vec = []
            for cam_pose in cam_poses:
                tcp_pose = H_tcp_in_base @ H_cam_in_tcp @ cam_pose @ H_tcp_in_cam
                tcp_poses.append(tcp_pose)
                tcp_poses_vec.append(homo2vec(tcp_pose))
            tcp_poses_vec = np.array(tcp_poses_vec)
            os.remove(cam_poses_path) ### delete the file to avoid repeated movement
        else:
            tcp_poses_vec = None ### Return None if camera pose is not found
            is_finished = False
        return tcp_poses_vec, is_finished

    def run(self, current_dir, target_dir, individuals_grasped, individuals_target):
        self.finished = False
        for i in range(self.max_run):
            if self.finished:
                print('IBVS finished due to threshold statisfied.')
                break
            if i < self.max_run:
                ### Take pics ###
                self.wrist.clear_folder(current_dir)
                if i == 0: ### Copy the analyzed images
                    source_item = os.path.join(self.wrist.destfolder, '3d_combined')
                    destination_item = os.path.join(current_dir, '3d_combined')
                    shutil.copytree(source_item, destination_item)
                    time.sleep(2)
                else: ## Take new images
                    self.wrist.take_pics(current_dir)
                    self.wrist.analyze_image(current_dir)
                env_name = r"C:/Users/xyao0/anaconda3/envs/RVC3"
                # script_path = "C:/Users/xyao0/Desktop/project/assembly/visual_servoing/ibvs_stereo - old.py"
                script_path = "C:/Users/xyao0/Desktop/project/assembly/visual_servoing/ibvs_stereo.py"
                # lmbda = self.max_lmbda * 1 / (self.max_run - i)
                lmbda = self.max_lmbda * 1 / (1 + i)
                args = [current_dir, target_dir, str(individuals_grasped), str(individuals_target),f'{lmbda}', str(self.thersh)]
                subprocess.run(["conda", "run", "-p", env_name, "python", script_path] + args, shell=True)

                ### Handel projectory ###
                planned_traj, is_finished = self.process_traj()
                print(f'length of the traj is {len(planned_traj)}')
                self.finished = is_finished
                fig, ax = plot_traj_and_obj_pos(planned_traj)
                fig.canvas.mpl_connect('key_press_event', close_on_key)  ###  press any key to close the plot window
                plt.show()
                ### Move robot ###
                global image_key_pressed
                if image_key_pressed == 'y' or image_key_pressed == 'Y':
                    self.robot.servoL(planned_traj)
                    image_key_pressed = None
        print('IBVS finished due to max runs')


class Planner():
    def __init__(self, net, ibvs, robot, zed, wrist, pose_processor, action_summary, colors, all_objs):
        self.net = net
        self.ibvs = ibvs
        self.robot = robot
        self.zed = zed
        self.wrist = wrist
        self.pose_processor = pose_processor
        self.action_summary = action_summary
        self.colors = colors
        self.all_objs = all_objs
        self.current_ind = 0

    def step(self, current_ind = None):
        current_tcp_in_base = np.array(robot.get_tcp_pose())
        current_tcp_in_base[:3] = current_tcp_in_base[:3] * SCALE
        current_tcp_in_base_quat = rotvect_to_quat(current_tcp_in_base).flatten()
        if current_ind is None:
            current_ind = self.current_ind
        print(current_ind, 'aaaaaaaaaaaaa')
        if current_ind == 0: ### Take video using zed camera at the start of the trajectory

            # self.zed.clear_folder()
            # self.zed.capture_video()
            # self.zed.analyze_video()

            self.pose_processor.get_pose_zed()
            self.net.update_obj_sequence(self.pose_processor.obj_pose_zed, self.pose_processor.obj_pose_wrist,
                                         current_tcp_in_base_quat,
                                         self.all_objs)
            self.traj_transformer, self.action = self.net.predict_traj_and_action()  ### predict transformer traj after object sequence updated
            self.action = ACTION
            print('Remove this later!!!!!!!!!!!')
            self.current_action_sumamry = self.action_summary[self.action]
            self.wrist_start_inds = self.current_action_sumamry['wrist_start_inds']
            self.wrist_end_inds = self.current_action_sumamry['wrist_end_inds']
            self.traj_len = self.current_action_sumamry['traj_len']
            self.obj_moving = self.current_action_sumamry['obj_moving']
            next_ind = find_next_greater(self.wrist_start_inds, current_ind)
            if next_ind is None:
                next_ind = self.traj_len - 1
            traj_valid = self.traj_transformer[current_ind:next_ind]
            self.current_ind = next_ind
        else:
            self.action = ACTION
            self.current_action_sumamry = self.action_summary[self.action]
            if current_ind in self.current_action_sumamry['wrist_start_inds']:
                self.previous_tcp_in_base_quat = current_tcp_in_base_quat
                wrist_ind = list(self.current_action_sumamry['wrist_start_inds']).index(current_ind)

                print(wrist_ind, 'gggggggggggg')
                individuals_target = []
                individuals_grasped = []
                next_ind = find_next_greater(self.current_action_sumamry['wrist_start_inds'], current_ind)
                for obj in self.all_objs:
                    if obj == 'trajectory':
                        continue
                    if self.current_action_sumamry['obj_moving'][obj][current_ind]: ### Object moving with gripper should not be considered when doing IBVS
                        individuals_grasped.append(obj + '1')
                    if next_ind is not None:
                        if (not self.current_action_sumamry['obj_moving'][obj][current_ind]) and (self.current_action_sumamry['obj_moving'][obj][next_ind]):
                            individuals_target.append(obj + '1')
                    else:
                        if (not self.current_action_sumamry['obj_moving'][obj][current_ind]):
                            individuals_target.append(obj + '1')
                if current_ind == 65:
                    individuals_target = ['nut1', 'jig1']

                self.wrist.clear_folder()
                self.wrist.take_pics()
                self.wrist.analyze_image()

                self.pose_processor.get_pose_wrist(current_tcp_in_base)
                self.pose_processor.get_pose_zed()

                self.net.update_obj_sequence(self.pose_processor.obj_pose_zed, self.pose_processor.obj_pose_wrist, current_tcp_in_base_quat,
                                         self.all_objs)
                self.traj_transformer, _ = self.net.predict_traj_and_action() ### predict transformer traj after object sequence updated
                print('Remove this later!!!!!!!!!!!')
                ### Do ibvs
                print(self.action, 'ffffffffff')
                # individuals_grasped = []
                print(f'Grapsed individuals {individuals_grasped}')
                print(f'Target individuals {individuals_target}')
                current_dir = os.path.join(self.ibvs.ibvs_dir, 'current')
                target_dir = os.path.join(self.ibvs.ibvs_dir, f'target/{self.action}', f'{wrist_ind + 1}')
                self.ibvs.run(current_dir, target_dir, individuals_grasped, individuals_target)
                traj_valid = None
                next_ind = find_next_greater(self.current_action_sumamry['wrist_end_inds'], current_ind)
                self.current_ind = next_ind
            else:
                ### Do transformer
                if self.pose_processor.obj_pose_zed is None:
                    self.pose_processor.obj_pose_zed = pd.read_csv(os.path.join(self.zed.destfolder, 'obj_pose_zed.csv'), index_col=0)
                if self.pose_processor.obj_pose_wrist is None:
                    self.pose_processor.obj_pose_wrist = pd.read_csv(os.path.join(self.wrist.destfolder, 'obj_pose_wrist.csv'), index_col=0)
                self.net.update_obj_sequence(self.pose_processor.obj_pose_zed, self.pose_processor.obj_pose_wrist,
                                             self.previous_tcp_in_base_quat,
                                             self.all_objs)
                self.traj_transformer, _ = self.net.predict_traj_and_action()  ### predict transformer traj after object sequence updated
                current_tcp_in_base = np.array(self.robot.get_tcp_pose()).reshape((1, -1))
                traj = self.traj_transformer[current_ind:]
                diff = traj - traj[0]
                traj = current_tcp_in_base + diff
                # if np.max(diff[:, 3:]) > 0.5 or np.max(diff[:, :3]) > 0.05:
                #     print(f'The difference between tf and ibvs is {diff}')
                #     print('Too big of a difference, are you sure about this')
                #     raise

                # fig2, axes = plt.subplots(3, 1, figsize=(9, 10))
                # for i, ax in enumerate(axes):
                #     ax.plot(self.traj_transformer[current_ind:, i + 3], color='red')
                #     ax.plot(traj[:, 3 + i], color='blue')
                    # ax.set_ylim(-np.sqrt(2), np.sqrt(2))
                self.traj_transformer[current_ind:] = traj
                next_ind = find_next_greater(self.current_action_sumamry['wrist_start_inds'], current_ind)
                print(current_ind, next_ind, 'ffffffffffffffffffffff')

                if next_ind is None:
                    next_ind = self.traj_len - 1
                traj_valid = self.traj_transformer[current_ind:next_ind]
                print(next_ind, 'ggggggggg')
                self.current_ind = next_ind
        if traj_valid is not None:
            traj_valid_plot = traj_valid.copy()
            traj_valid_plot[:, :3] = traj_valid_plot[:, :3] * SCALE
            fig, ax = plot_traj_and_obj_pos(traj_valid_plot, self.net.obj_pose, self.colors, self.all_objs)
            fig.canvas.mpl_connect('key_press_event', close_on_key) ###  press any key to close the plot window
            plt.show()
            global image_key_pressed
            if image_key_pressed == 'y' or image_key_pressed == 'Y':
                self.robot.servoL(traj_valid)
                image_key_pressed = None
        print(self.current_ind, 'bbbbbbbbbbb')
        if self.current_ind in self.current_action_sumamry['gripper_open_inds']:
            self.robot.open_gripper()
        elif self.current_ind in self.current_action_sumamry['gripper_close_inds']:
            self.robot.close_gripper()


if __name__ == "__main__":
    image_key_pressed = None
    SCALE = 1000
    ### Load task config ###
    with open(os.path.join('./data/task_config.yaml')) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    ### Create dirs to reproduce
    project_dir = config['project_path']
    DATE = str(datetime.date.today())
    FOLDER = os.path.join(project_dir, 'data/reproduce')
    if not os.path.isdir(FOLDER):
        os.makedirs(FOLDER, exist_ok=True)

    ### Load transformer data
    max_len = 200
    objs = sorted(config['objects'])
    all_objs = objs + ['trajectory']
    actions = config['actions']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Cuda available: ", torch.cuda.is_available())
    model_folder = 'speed_wrist_weights-hidden-64-smooth_pos'
    model_ind = 10000
    seed = 123
    model_path = os.path.join(project_dir, f'transformer/{model_folder}/{seed}/model_{model_ind}.pth')
    model_stat_path = os.path.join(project_dir, f'transformer/{model_folder}/{seed}/train_stat.pickle')
    with open(model_stat_path, 'rb') as f:
        train_stat = pickle.load(f)
    my_net = Network(max_len, train_stat, device)
    my_net.load_model(model_path)
    my_net.get_obj_tags(all_objs)

    ### Connect to the robot #######
    FREQUENCY = 1
    dt = 1/ FREQUENCY
    host = "192.168.3.5"
    robot = Robot(host, FREQUENCY)
    robot.connect()
    # [0.000323199579725042, -1.1896336714373987, -1.7440336386310022, -0.2002013365374964, 1.5558044910430908, -4.738313380871908]


    ### Connect to the zed camera ###
    destfolder_zed = os.path.join(FOLDER, 'zed')
    zed_cam = Zed(destfolder_zed)
    zed_cam.connect_camera()

    ### Connect to the writst cameras ###
    url_left = 'http://192.168.0.100:8080/shot.jpg'
    url_right = 'http://192.168.0.102:8080/shot.jpg'
    destfolder_wrist = destfolder_zed.replace('zed', 'wrist')
    wrist_cam = Wrist(destfolder_wrist, url_left, url_right)
    # wrist_cam.clear_folder()
    # wrist_cam.take_pics()
    # wrist_cam.analyze_image()
    # raise

    ### Object pose processor ###
    transformation_dir = config['transformation_path']
    pose_processor = PoseProcessor(transformation_dir, destfolder_zed, destfolder_wrist)

    ### IBVS object ####
    img_ind = 3
    ACTION = f'action_1'
    # ACTION = 'action_0'
    # ACTION = 'action_2'

    ibvs_dir = os.path.join(project_dir, 'data/reproduce/ibvs')
    ibvs = IBVS(ibvs_dir, robot, wrist_cam)
    # wrist_cam.analyze_image(os.path.join(ibvs.ibvs_dir, 'target',ACTION, f'{img_ind}'))

    # f = os.path.join(ibvs.ibvs_dir, 'target', ACTION, f'{img_ind}', f'{img_ind}DLC_resnet50_jigNov22shuffle1_80000.h5')
    f = os.path.join(wrist_cam.destfolder, 'wristDLC_resnet50_jigNov22shuffle1_80000.h5')
    df = pd.read_hdf(f)
    # print(df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart1'])
    # print(df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart3'])
    # print(df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart4'])
    # print(df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart2'])
    # raise
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart1']['likelihood'].iloc[0] = 1
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart2']['likelihood'].iloc[0] = 1
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart1']['likelihood'].iloc[1] = 1
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart2']['likelihood'].iloc[1] = 1
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart3']['likelihood'].iloc[0] = 1
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart4']['likelihood'].iloc[0] = 1
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart3']['likelihood'].iloc[1] = 1
    # df['DLC_resnet50_jigNov22shuffle1_80000']['jig1']['bodypart4']['likelihood'].iloc[1] = 1
    # os.remove(f)
    # df.to_hdf(f, key = 'df_with_missing')
    # wrist_cam.analyze_image()
    # # wrist_cam.analyze_image(os.path.join(ibvs.ibvs_dir, 'target', ACTION, '3'))
    # raise
    #
    action_summary_file = os.path.join(project_dir, 'data/action_summary_all.pickle')


    with open(action_summary_file, 'rb') as f:
        action_summary = pickle.load(f)
    # action_summary['action_2']['wrist_start_inds'][1] = 85
    # action_summary['action_2']['wrist_end_inds'][1] = 86
    print(action_summary[ACTION]['wrist_start_inds'])
    print(action_summary[ACTION]['wrist_end_inds'])
    print(action_summary[ACTION]['gripper_open_inds'])
    print(action_summary[ACTION]['gripper_close_inds'])
    print(action_summary[ACTION].keys())
    print(action_summary[ACTION]['median_demo'])

    colors = {'bolt': 'green', 'nut': 'yellow', 'bin': 'black', 'jig': 'purple', 'traj': 'red', 'trajectory': 'pink'}

    my_planner = Planner(my_net, ibvs, robot, zed_cam, wrist_cam, pose_processor, action_summary, colors, all_objs)
    my_planner.current_ind = 0
    # my_planner.traj_len = 198
    # current_tcp_in_base = np.array([0.4646085308729565, -0.1381282317992776, 0.21073617361028618, 2.4689049270946595, 0.06929360866802006, 1.8535178670312829])
    # current_tcp_in_base[:3] = current_tcp_in_base[:3] * SCALE
    # current_tcp_in_base_quat = rotvect_to_quat(current_tcp_in_base).flatten()
    # my_planner.previous_tcp_in_base_quat = current_tcp_in_base_quat

    ##### Set up pygame screen ##############
    n_trial = 0
    n_success = 0
    n_failure = 0
    pygame.init()
    size = [500, 700]
    WIN = pygame.display.set_mode(size)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    FPS = 20
    WIN.fill(WHITE)
    pygame.display.update()
    pygame.display.set_caption("Collect data")

    key_ring = {}

    ### These values could change from keyboard to keyboard
    Caps_lock = '1073741881' ### Used to switch between fast mode or slow mode
    SHIFT = '1073742049' ### Hold left shift key to control the robot and the gripper
    left = '1073741904'
    right = '1073741903'
    up = '1073741906'
    down = '1073741905'
    # page_up = '1073741899'
    # page_down = '1073741902'

    key_ring[SHIFT] = 0  # 1073742049 is the left shift key. This will be displayed in the screen  Caps lock = 1 + keys are the command set
    key_ring[Caps_lock] = 1
    key_pressed = 0  # key press and release will happen one after another
    key_released = 0

    font1 = pygame.font.SysFont('aril', 26)
    font2 = pygame.font.SysFont('aril', 35)
    font3 = pygame.font.SysFont('aril', 150)

    text1 = font1.render('Shift should be 1 to accept any keys', True, BLACK, WHITE)
    text3 = font1.render("'c': close the fingers", True, BLACK, WHITE)
    text4 = font1.render("'o': open the fingers", True, BLACK, WHITE)
    text5 = font1.render("'b': begin ", True, BLACK, WHITE)
    text6 = font1.render("'f': fail", True, BLACK, WHITE)
    text7 = font1.render("'s': success", True, BLACK, WHITE)
    text14 = font1.render("'d': defalt robot position", True, BLACK, WHITE)
    text15 = font1.render("'up arrow': move forward", True, BLACK, WHITE)
    text16 = font1.render("'down arrow': move backward", True, BLACK, WHITE)
    text17 = font1.render("'left arrow': move left", True, BLACK, WHITE)
    text18 = font1.render("'right arrow': move right", True, BLACK, WHITE)
    text19 = font1.render("'page up': move up", True, BLACK, WHITE)
    text20 = font1.render("'page down': move down", True, BLACK, WHITE)
    text22 = font1.render("'1': screw", True, BLACK, WHITE)
    text23 = font1.render("'2': unscrew", True, BLACK, WHITE)
    text24 = font1.render("'v': gripper vertical", True, BLACK, WHITE)
    text25 = font1.render("'h': gripper horizontal", True, BLACK, WHITE)
    text28 = font1.render("'7': slow, '8': medium, '9': fast ", True, BLACK, WHITE)
    text30 = font1.render("Press Caps Lock to change ", True, BLACK, WHITE)
    text21 = font1.render(f"Speed mode: ", True, BLACK, WHITE)
    text26 = font1.render(f"Reference frame:", True, BLACK, WHITE)
    text2 = font1.render(f'Shift : ', True, BLACK, WHITE)

    text8 = font2.render("#Trial", True, BLACK, WHITE)
    text9 = font2.render("#Success", True, BLACK, WHITE)
    text10 = font2.render("#Failure", True, BLACK, WHITE)
    text_counterclockwise = font1.render(f'"[" : counterclockwise rotate ', True, BLACK, WHITE)
    text_clockwize = font1.render(f'"]" : clockwise rotate ', True, BLACK, WHITE)
    text_keypoint = font1.render(f'"t" : keypoint timestamp ', True, BLACK, WHITE)
    text_defalt = font1.render(f'"k" : defalt pose timestamp ', True, BLACK, WHITE)
    text_failed_action = font1.render(f'"r" : falied action timestamp ', True, BLACK, WHITE)
    text_stop = font1.render(f'"enter" : stop movement', True, BLACK, WHITE)
    text_protective_stop = font1.render(f'"ESC" : protective stop ', True, BLACK, WHITE)


    clock = pygame.time.Clock()
    run = True
    speed_mode = 'slow'
    collecting_data = False
    reference_frame = 'base' if is_capslock_on() else 'tool'

    Ds = {'fast': 0.05, 'medium': 0.005, 'slow': 0.001}
    speed = Ds[speed_mode]

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                key_pressed = event.key
                print(key_pressed)
                key_ring[str(key_pressed)] = 1
                demo_dir = FOLDER
                if key_ring[SHIFT] == 1:  # Left shift is pressed
                    if key_pressed == 98:  ## Keyboard 'b' to start a demonstration##
                        demo_id = str(datetime.datetime.now().timestamp()).split('.')[0]
                        demo_dir = os.path.join(FOLDER, demo_id)
                        zed_dir = os.path.join(demo_dir, 'zed_videos')
                        wrist_dir = os.path.join(demo_dir, 'wrist_images')
                        os.makedirs(zed_dir, exist_ok=True)
                        os.makedirs(wrist_dir, exist_ok=True)
                    elif key_pressed == 108:  ## Keyboard 'l' to use the last demonstration##
                        demo_id = sorted(os.listdir(FOLDER))[-1]
                        demo_dir = os.path.join(FOLDER, demo_id)
                    elif key_pressed == 111:  #### Keyboard 'o' to open the gripper ####
                        # print('Open the gripper')
                        open_gripper(robot.rtde)
                    elif key_pressed == 99:  #### Keyboard 'c' to close the gripper####
                        # print('Close the gripper')
                        close_gripper(robot.rtde)
                    elif key_pressed == 100: #### Keyboard 'd' to get back to defalt pose #####
                        move_to_defalt_pose(robot.rtde_c)
                    elif key_pressed == int(up): ### Keyboard up arrow to move forward###
                        move_foward(robot.rtde_r, robot.rtde_c, speed, reference_frame)
                    elif key_pressed == int(down): ### Keyboard down arrow to move backward###
                        move_backward(robot.rtde_r, robot.rtde_c, speed, reference_frame)
                    elif key_pressed == int(left): ### Keyboard left arrow to move left###
                        move_left(robot.rtde_r, robot.rtde_c, speed, reference_frame)
                    elif key_pressed == int(right): ### Keyboard right arrow to move right###
                        move_right(robot.rtde_r, robot.rtde_c, speed, reference_frame)
                    elif key_pressed == 61: ### Keyboard + to move up###
                        move_up(robot.rtde_r, robot.rtde_c, speed, reference_frame)
                    elif key_pressed == 45: ### Keyboard - to move down###
                        move_down(robot.rtde_r, robot.rtde_c, speed, reference_frame)
                    elif key_pressed == 55: ### keyboard 7 to change to slow mode###
                        speed_mode = 'slow'
                        speed = Ds[speed_mode]
                    elif key_pressed == 56: ### keyboard 8 to change to medium speed mode###
                        speed_mode = 'medium'
                        speed = Ds[speed_mode]
                    elif key_pressed == 57: ### keyboard 9 to change to fast speed mode###
                        speed_mode = 'fast'
                        speed = Ds[speed_mode]
                    elif key_pressed == 13: ### keyboard enter to stop the movement###
                        stop(robot.rtde_c)
                    elif key_pressed == 27: ### keyboard space ESC to stop the robot###
                        protective_stop(robot.rtde_c)
                    elif key_pressed == 115: ### keyboard 's' to stop the robot###
                        my_planner.step()
                    elif key_pressed == 112: ### keyboard 'p' to stop the robot###
                        wrist_cam.take_pics()
                    elif key_pressed == int(Caps_lock): ### Use Caps lock to change reference frame
                        if reference_frame == 'base':
                            reference_frame = 'tool'
                        else:
                            reference_frame = 'base'
                    elif key_pressed == 118:### keyboard v to get back to vertical pose###
                        get_vertical(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 104: ### keyboard h to get to horizontal pose####
                        get_horizontal(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 44:### keyboard v to get back to vertical pose###
                        wrist1_plus(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 46: ### keyboard h to get to horizontal pose####
                        wrist1_minus(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 49: ### keyboard 1 to assemble###
                        screw(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 50: ### keyboard 2 to undo assemble###
                        unscrew(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 93:  #### keyboard right bracket to rotate clockwise####
                        rotate_wrist_clockwise(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 91:  #### keyboard left bracket to rotate counter-clockwize#####
                        rotate_wrist_counterclockwise(robot.rtde_r, robot.rtde_c)
                    elif key_pressed == 116: ### keyboard 't' ###
                        print(robot.rtde_r.getActualTCPPose())


            elif event.type == pygame.KEYUP:
                key_released = event.key
                key_ring[str(key_released)] = 0
            else:
                pass  # ignoring other non-logitech joystick event types
        text27 = font2.render(f"{speed_mode}               ", True, RED, WHITE)
        text29 = font2.render(f"{reference_frame}    ", True, RED, WHITE)
        text31 = font2.render(f'{key_ring[SHIFT]}', True, RED, WHITE)
        text11 = font3.render(f"{n_trial}", True, RED, WHITE)
        text12 = font3.render(f"{n_success}", True, RED, WHITE)
        text13 = font3.render(f"{n_failure}", True, RED, WHITE)

        ### Shift
        WIN.blit(text1, (150, 0))
        WIN.blit(text2, (10, 0))
        WIN.blit(text31, (70, -2))


        ### Speed mode
        WIN.blit(text21, (10, 30))
        WIN.blit(text27, (130, 26))
        WIN.blit(text28, (250, 30))

        ### Reference frame
        WIN.blit(text26, (10, 60))
        WIN.blit(text29, (170, 54))
        WIN.blit(text30, (250, 60))

        WIN.blit(text3, (10, 90))
        WIN.blit(text4, (10, 120))
        WIN.blit(text5, (10, 150))
        WIN.blit(text6, (10, 180))
        WIN.blit(text7, (10, 210))
        WIN.blit(text14, (10, 240))
        WIN.blit(text22, (10, 270))
        WIN.blit(text23, (10, 300))

        WIN.blit(text15, (250, 90))
        WIN.blit(text16, (250, 120))
        WIN.blit(text17, (250, 150))
        WIN.blit(text18, (250, 180))
        WIN.blit(text19, (250, 210))
        WIN.blit(text20, (250, 240))
        WIN.blit(text24, (250, 270))
        WIN.blit(text25, (250, 300))


        WIN.blit(text8, (10, 340))
        WIN.blit(text9, (10, 475))
        WIN.blit(text10, (10, 600))

        ### trials, successes, and failures
        WIN.blit(text11, (150, 350))
        WIN.blit(text12, (150, 475))
        WIN.blit(text13, (150, 600))

        ### Rotations
        WIN.blit(text_clockwize, (250, 330))
        WIN.blit(text_counterclockwise, (250, 360))

        ## Keypoints
        WIN.blit(text_keypoint, (250, 390))
        WIN.blit(text_defalt, (250, 420))
        WIN.blit(text_failed_action, (250, 450))

        ## Stops
        WIN.blit(text_stop, (250, 480))
        WIN.blit(text_protective_stop, (250, 510))


        pygame.display.update()
    pygame.quit()


