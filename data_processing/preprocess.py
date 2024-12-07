from glob import glob
import pickle
import sys
import pyzed.sl as sl
import numpy as np
import cv2
import enum
import os
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
import shutil



class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def svo_to_avi(video, outdir, output_as_video = True):
    '''
    This function will convert the .svo file got from ZED to .mp4 file.

    parameters
    ----------
    video: string
        The path to the .svo file
    outdir: string
        The path to the directory where the .mp4 will be saved
    output_as_video: bool
        Not useful for now. Might be useful in the future.

    returns:
    -------
    0
    '''
    svo_input_path = video
    output_path = outdir
    app_type = AppType.LEFT_AND_RIGHT

    # Specify SVO path parameter
    init_params = sl.InitParameters()

    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # Get image size
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height


    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_left_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    svo_image_right_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()

    vid_id = os.path.basename(video).split('.')[0]
    outputdir_left = output_path
    outputdir_right = output_path
    os.makedirs(outputdir_left, exist_ok=True)
    os.makedirs(outputdir_right, exist_ok=True)
    output_path_left = outputdir_left + f'/{vid_id}-left.avi'
    output_path_right = outputdir_right + f'/{vid_id}-right.avi'
    if output_as_video:
        # Create video writer with MPEG-4 part 2 codec
        video_writer_left = cv2.VideoWriter((output_path_left),
                                            cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                            max(zed.get_camera_information().camera_configuration.fps, 25),
                                            (width, height))

        video_writer_right = cv2.VideoWriter((output_path_right),
                                             cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                             max(zed.get_camera_information().camera_configuration.fps, 25),
                                             (width, height))

        if not video_writer_left.isOpened():
            sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                             "permissions.\n")
            zed.close()
            exit()

    rt_param = sl.RuntimeParameters()
    # rt_param.sensing_mode = sl.SENSING_MODE.FILL

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            if app_type == AppType.LEFT_AND_RIGHT:
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            elif app_type == AppType.LEFT_AND_DEPTH:
                zed.retrieve_image(right_image, sl.VIEW.DEPTH)
            elif app_type == AppType.LEFT_AND_DEPTH_16:
                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            if output_as_video:
                # Copy the left image to the left side of SBS image
                svo_image_left_rgba = left_image.get_data()
                # Copy the right image to the right side of SBS image
                svo_image_right_rgba = right_image.get_data()

                # Convert SVO image from RGBA to RGB
                ocv_image_left_rgb = cv2.cvtColor(svo_image_left_rgba, cv2.COLOR_RGBA2RGB)

                ocv_image_right_rgb = cv2.cvtColor(svo_image_right_rgba, cv2.COLOR_RGBA2RGB)
                # fname = svo_input_path.replace('raw', 'processed').replace('svo', 'jpg')

                # Write the RGB image in the video
                video_writer_left.write(ocv_image_left_rgb)
                video_writer_right.write(ocv_image_right_rgb)
            else:
                # Generate file names
                filename1 = output_path / ("left%s.png" % str(svo_position).zfill(6))
                filename2 = output_path / (("right%s.png" if app_type == AppType.LEFT_AND_RIGHT
                                            else "depth%s.png") % str(svo_position).zfill(6))

                # Save Left images
                cv2.imwrite(str(filename1), left_image.get_data())

                if app_type != AppType.LEFT_AND_DEPTH_16:
                    # Save right images
                    cv2.imwrite(str(filename2), right_image.get_data())
                else:
                    # Save depth images (convert to uint16)
                    cv2.imwrite(str(filename2), depth_image.get_data().astype(np.uint16))

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

        # Check if we have reached the end of the video
        elif zed.grab(rt_param) == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:  # End of SVO
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break
    print(f'The video id is {vid_id}')
    if output_as_video:

        # Close the video writer
        video_writer_left.release()
        video_writer_right.release()

    zed.close()
    return [output_path_left, output_path_right]

def get_release_time_ind(df, max_width):
    '''
    :param df: DataFrame. The recorded robot data.
    :param max_width: float. The grip width with the gripper is open.
    :return: Array. The timestamps indices when the robot releases a object.
    '''
    df['shifted'] = df[grip_width_key].shift(-1)
    df['release'] = (df[grip_width_key] < max_width) & (df['shifted'] >= max_width)
    release_time_ind = np.where(df['release'])[0]
    return release_time_ind

def get_release_time(df, max_width):
    '''
    :param df: DataFrame. The recorded robot data.
    :param max_width: float. The grip width with the gripper is open.
    :return: Array. The timestamps indices when the robot releases a object.
    '''
    df['shifted'] = df[grip_width_key].shift(-1)
    df['release'] = (df[grip_width_key] < max_width) & (df['shifted'] >= max_width)
    release_time_ind = np.where(df['release'])[0]
    return df.iloc[release_time_ind]['timestamp'].values

def get_grasp_time_ind(df, max_width):
    '''
    :param df: DataFrame. The recorded robot data.
    :param max_width: float. The grip width with the gripper is open.
    :return: Array. The timestamps indices when the robot grasps a object.
    '''
    df['shifted'] = df[grip_width_key].shift(-1)
    df['release'] = (df[grip_width_key] >= max_width) & (df['shifted'] < max_width)
    grasp_time = np.where(df['release'])[0]
    return grasp_time

def get_defalt_pose_time_ind(df, keypoints):
    timestamp = df['timestamp'].values
    defalt_pose_time = keypoints['defalt_pose_time_stamp']
    defalt_pose_time_ind = []
    for t in defalt_pose_time:
        defalt_pose_time_ind.append(np.argmin(abs(timestamp - t)))
    return defalt_pose_time_ind

def delete_failed_action(df, keypoints):
    timestamp = df['timestamp'].values
    print(df.shape)
    for t in keypoints['failed_action_time_stamp']:
        failed_action_ind = np.argmin(abs(timestamp - t))
        upper_ind = defalt_pose_ind[np.where(defalt_pose_ind > failed_action_ind)[0][0]]
        lower_ind = defalt_pose_ind[np.where(defalt_pose_ind > failed_action_ind)[0][0] - 1]
        df.drop(df.index[list(range(lower_ind, upper_ind))], inplace = False)

def action_successful(action_ind, keypoints):
    failed_action_time = keypoints['failed_action_time_stamp']
    defalt_pose_time = keypoints['defalt_pose_time_stamp']
    current_start_time = defalt_pose_time[action_ind]
    next_start_time = defalt_pose_time[action_ind + 1]
    if failed_action_time == []:
        return True
    else:
        for t in failed_action_time:
            if t > current_start_time and t < next_start_time:
                return False
        return True

def get_timestamp_ind(df, t):
    timestamp = df['timestamp'].values
    return np.argmin(abs(timestamp - t))

def downsample(array, npts, kind = 'linear'):
    interpolated = interp1d(np.arange(len(array)), array, kind = kind, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled


if __name__ == '__main__':
    date = '2024-10-08'
    input_video_type = 'svo2'
    output_video_type = 'avi'
    img_type = 'png'
    n_actions = 2
    MAX_WIDTH = 40
    grip_width_key = 'output_double_register_18'
    grip_detection_key = 'output_int_register_19'

    keys_needed = ['timestamp', 'actual_TCP_pose_0', 'actual_TCP_pose_1', 'actual_TCP_pose_2', 'actual_TCP_pose_3', 'actual_TCP_pose_4',
                        'actual_TCP_pose_5', 'output_double_register_18', 'output_int_register_19', 'wrist_camera_time']
    keys_ft = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
    PROCESS_VIDEO = True
    PROCESS_FORCE = True

    raw_dir = f'../assembly/data//raw/{date}'

    preprocessed_dir = raw_dir.replace('raw', 'preprocessed')
    if not os.path.isdir(preprocessed_dir):
        os.makedirs(preprocessed_dir)
    svo_files = glob(os.path.join(raw_dir, f'*.{input_video_type}'))
    demos = os.listdir(raw_dir)
    process = False
    for demo in demos:
        if demo == '1728409155':
            continue
        # if demo == '1726090435':
        #     process = True
        # if not process:
        #     continue
        src_demo_dir = os.path.join(raw_dir, demo)
        keypoint_file = os.path.join(src_demo_dir, f'{demo}_keypoint.pickle')
        try:
            with open(keypoint_file, 'rb') as f:
                keypoints = pickle.load(f)
        except FileNotFoundError:
            continue
        if not 's' in keypoints['key_time_stamp']:  ## failed demo
            continue

        else: ### Successful demo
            print(f'Processing id: {demo}')
            dest_demo_folder = os.path.join(preprocessed_dir, demo)
            os.makedirs(dest_demo_folder, exist_ok = True)                ### Convert the svo files into mp4 files
            if PROCESS_VIDEO:
                svo_file = keypoint_file.replace('_keypoint.pickle', f'.{input_video_type}')
                svo_to_avi(svo_file, dest_demo_folder)
                output_video_left = os.path.join(dest_demo_folder, f'{demo}-left.{output_video_type}')
                output_video_right = os.path.join(dest_demo_folder, f'{demo}-right.{output_video_type}')
                clip = VideoFileClip(output_video_left)
                video_len = clip.duration
            robot_traj_file = os.path.join(src_demo_dir, demo)
            df = pd.read_csv(robot_traj_file)

            # defalt_pose_time = keypoints['defalt_pose_time_stamp']
            # print(defalt_pose_time)
            # print(int(2304 / 8222 * len(df)))
            # print(len(df))
            # raise
            # del defalt_pose_time[1]
            #
            # keypoints['defalt_pose_time_stamp'] = defalt_pose_time
            # with open(keypoint_file, 'wb') as f:
            #     pickle.dump(keypoints, f)
            # raise

            ## Wrist camera time
            wrist_cam_time = keypoints['wrist_camera_time']
            df['wrist_camera_time'] = 0
            for t in wrist_cam_time:
                wrist_cam_ind = get_timestamp_ind(df, t)
                df['wrist_camera_time'].iloc[wrist_cam_ind] = 1

            ### Chunk trajectory into actions
            release_time = get_release_time(df, MAX_WIDTH)
            robot_traj_len = len(df)
            defalt_pose_time = keypoints['defalt_pose_time_stamp']

            # if len(defalt_pose_time) == 1:
            #     # print('aaaaaaa')
            #     defalt_pose_time.append(defalt_pose_time[0] + 0.01)
            # elif len(defalt_pose_time) == 2:
            #     # print('bbbbbbbb')
            #     if defalt_pose_time[0] == defalt_pose_time[1]:
            #         defalt_pose_time[1] = defalt_pose_time[1] + 0.01

            # print(defalt_pose_time)
            # continue
            n_actions_demo = len(defalt_pose_time)
            # if n_actions_demo != n_actions:
            #     print(f'{demo} has wrong number of actions.')
            #     continue
            for i in range(n_actions_demo):
                if i == 0:
                    continue
                action_folder = os.path.join(dest_demo_folder, f'action_{i}')
                fname_action = os.path.join(action_folder, f'action_{i}.csv')
                os.makedirs(action_folder, exist_ok=True)

                ## Process the robot trajectory file ###
                start_time = defalt_pose_time[i] ### Action starts when the robot get to the defalt position
                try:
                    # end_time = release_time[np.where(release_time > defalt_pose_time[i])[0][0]]
                    end_time = defalt_pose_time[i+1]
                except IndexError:
                    end_time = df.iloc[-1]['timestamp'] ### Action ends the robot get to the default position again or the robot stops recording.
                # end_time =defalt_pose_time[i+1]
                start_id = get_timestamp_ind(df, start_time)
                end_id = get_timestamp_ind(df, end_time)
                df_action = df.iloc[start_id:end_id][keys_needed]
                df_action = df_action.rename(columns = {'actual_TCP_pose_0':'x', 'actual_TCP_pose_1':'y', 'actual_TCP_pose_2':'z', 'actual_TCP_pose_3':'rx', 'actual_TCP_pose_4':'ry',
                    'actual_TCP_pose_5':'rz', "output_double_register_18":"grip_width", "output_int_register_19": "grasp_detected"})

                ### Process the force trajectory file ###
                if PROCESS_FORCE:
                    force_torque_traj_file = robot_traj_file + '.npy'
                    ft_traj = np.load(force_torque_traj_file)
                    ft_traj_len = len(ft_traj)
                    ft_action = ft_traj[int(start_id / robot_traj_len * ft_traj_len):int(end_id / robot_traj_len * ft_traj_len),:] # Chunk the force-torque trajectory
                    for j, key in enumerate(keys_ft): # Downsample the force-torque trajectory
                        tmp = downsample(ft_action[:,j], len(df_action))
                        df_action[key] = tmp
                    np.save(fname_action.replace('.csv', '_ft.npy'), ft_action)
                df_action.to_csv(fname_action)

                ### Copy images for the corresponding action ###
                action_wrist_folder = os.path.join(action_folder, 'wrist_images')
                os.makedirs(action_wrist_folder, exist_ok=True)
                action_wrist_inds = []
                for j, time in enumerate(wrist_cam_time):
                    if time > start_time and time < end_time:
                        action_wrist_inds.append(j)
                imgs_left = sorted(glob(os.path.join(src_demo_dir, f'*_left.{img_type}')))
                imgs_to_move = []
                for k in action_wrist_inds:
                    imgs_to_move.append(imgs_left[k])
                    imgs_to_move.append(imgs_left[k].replace('left', 'right'))
                for img in imgs_to_move:
                    shutil.copy(img, action_wrist_folder)

                ## Cut 1 second of video for the corresponding action ###
                if not PROCESS_VIDEO:
                    output_video_left = os.path.join(dest_demo_folder, f'{demo}-left.{output_video_type}')
                    output_video_right = os.path.join(dest_demo_folder, f'{demo}-right.{output_video_type}')
                    clip = VideoFileClip(output_video_left)
                    video_len = clip.duration
                video_start = start_id / robot_traj_len * video_len
                video_end = video_start + 1  ## Detect object pose at the start of an action(1st second of the action)
                ffmpeg_extract_subclip(output_video_left, video_start, video_end, targetname=os.path.join(action_folder, demo + f'_left.{output_video_type}'))
                ffmpeg_extract_subclip(output_video_right, video_start, video_end, targetname=os.path.join(action_folder, demo + f'_right.{output_video_type}'))
