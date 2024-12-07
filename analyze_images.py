from glob import glob
import os
import pandas as pd
import yaml
import deeplabcut
from dlc_utils.utils import combine_h5files, serch_obj_h5files, _modify_h5file
from dlc_utils.triangulation import triangulate_images, undistort_points
from dlc_utils.visualization import create_video_with_h5file, create_images_with_h5file
from deeplabcut.utils import auxiliaryfunctions
import argparse
from analyze_videos_and_images import compute_shift
import numpy as np
import sys


def analyze_images(images_folder, task_config_path, destfolder = None, image_type = 'png', save_as_csv = True,  overwrite = True, analyze2d = True, analyze3d = True, pcutoff = 0.2, visualize_image = True, force_correspondence = False, pixel_thres = 10):
    with open(task_config_path) as file:
        config = yaml.load(file, Loader = yaml.FullLoader)
    objs = config['objects']
    cams = config['cameras']

    DLC3D = config['dlc3d_path']
    DLC2D = config['dlc_path']
    config3d = glob(os.path.join(DLC3D, f'*wrist_cam*', 'config.yaml'))[0]
    ### Run Deeplabcut to analyze videos
    imgs = glob(os.path.join(images_folder, f'*.{image_type}'))
    if analyze2d:
        for obj in objs:
            config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
            cfg = auxiliaryfunctions.read_config(config2d)
            if cfg['multianimalproject']:  ## multi-animal project
                multi = True
            else:
                multi = False
            ### Analyze 2D
            deeplabcut.analyze_time_lapse_frames(config2d, images_folder, shuffle=1, frametype=f'.{image_type}',
                                                 trainingsetindex=0, gputouse=None, save_as_csv=True)
    h5files = serch_obj_h5files(images_folder)

    for obj in objs:
        h5file_obj = [f for f in h5files if obj in f][0]
        _modify_h5file(h5file_obj, obj=obj, multi=False)

    ### combine h5 files
    if destfolder is None:
        destfolder = images_folder
    dir_combined_2d = os.path.join(destfolder, '2d_combined')
    os.makedirs(dir_combined_2d, exist_ok=True)
    combined_h5file_2d = combine_h5files(h5files, destdir=dir_combined_2d, suffix=f'wrist_2d')
    create_images_with_h5file(imgs, combined_h5file_2d, thresh=pcutoff, over_write=overwrite,
                              destfolder=dir_combined_2d)

    if force_correspondence:
        cfg_3d = auxiliaryfunctions.read_config(config3d)
        cam_names = cfg_3d["camera_names"]
        ### Undistort points such that corresponding points should be on the same horizontal line
        (
            dataFrame_camera1_undistort,
            dataFrame_camera2_undistort,
            stereomatrix,
            path_stereo_file,
        ) = undistort_points(
            config3d, combined_h5file_2d, str(cam_names[0] + "-" + cam_names[1])
        )
        if 'scorer' in dataFrame_camera1_undistort.columns.names:
            dataFrame_camera1_undistort = dataFrame_camera1_undistort.droplevel(level='scorer', axis=1)
            dataFrame_camera2_undistort = dataFrame_camera2_undistort.droplevel(level='scorer', axis=1)
        img_inds = dataFrame_camera2_undistort.index
        individuals = dataFrame_camera1_undistort.columns.get_level_values(level='individuals').unique()
        individuals = ['nut1']
        df_combined = pd.read_hdf(combined_h5file_2d)
        if 'scorer' in df_combined.columns.names:
            df_combined = df_combined.droplevel(level='scorer', axis=1)
        for ind in img_inds:
            for individual in individuals:
                bps = dataFrame_camera1_undistort[individual].columns.get_level_values(level='bodyparts').unique()
                bps_detected = []
                for bp in bps:
                    if dataFrame_camera2_undistort.loc[ind, (individual, bp, 'likelihood')] > pcutoff and \
                            dataFrame_camera1_undistort.loc[
                                ind.replace(cam_names[1], cam_names[0]), (individual, bp, 'likelihood')] > pcutoff:
                        ### Only bodyparts from both camera with likelihood over pcut_2d_image are considered detected
                        bps_detected.append(bp)
                y_left = dataFrame_camera1_undistort.loc[
                    ind.replace(cam_names[1], cam_names[0]), (individual, bps_detected, 'y')].to_numpy()
                y_right = dataFrame_camera2_undistort.loc[ind, (individual, bps_detected, 'y')].to_numpy()
                shift = compute_shift(y_left, y_right)
                df_combined.loc[ind, (individual, bps_detected)] = np.roll(
                    df_combined.loc[ind, (individual, bps_detected)].to_numpy(), shift * 3)
                ### make sure cooreponding points on the same horizontal line
                for i, bp in enumerate(bps_detected):
                    if abs(y_left[i] - y_right[i]) > pixel_thres: ### ### y pixel value higher than pixel_thres is eliminated by setting likelihood value 0
                        df_combined.loc[ind, (individual, bp, 'likelihood')] = 0
                        df_combined.loc[ind.replace(cam_names[1], cam_names[0]), (individual, bp, 'likelihood')] = 0

        ### Save updated dataframe
        # combined_h5file_2d = combined_h5file_2d.replace('.h5', '_shifted.h5')
        os.remove(combined_h5file_2d)
        df_combined.to_hdf(combined_h5file_2d, key='2d_combined')
        df_combined.to_csv(combined_h5file_2d.replace('h5', 'csv'))

    if visualize_image:
        imgs = glob(os.path.join(images_folder, f'*{image_type}'))
        create_images_with_h5file(imgs, combined_h5file_2d, thresh=pcutoff, over_write=overwrite,
                                  destfolder= dir_combined_2d, display_bps=True)
    # ## Triangulate
    if analyze3d:
        dir_combined_3d = dir_combined_2d.replace('2d', '3d')
        os.makedirs(dir_combined_3d, exist_ok=True)
        df_3d = triangulate_images(config3d, combined_h5file_2d, destfolder=dir_combined_3d, pcutoff=pcutoff, save_as_csv = save_as_csv)

def main():
    parser = argparse.ArgumentParser(description="This script does something useful.")

    # Add arguments
    parser.add_argument('-i', '--images_folder', type=str, help='Path of the images directory', required=True)
    parser.add_argument('-2d', '--analyze_2d',  help='Analyse 2d?', required=True)
    parser.add_argument('-3d', '--analyze_3d',  help='Analyse 3d?', required=True)

    args = parser.parse_args()

    if args.analyze_2d:
        analyze2d = True
    else:
        analyze2d = False
    if args.analyze_3d:
        analyze3d = True
    else:
        analyze3d = False

    config_path = './data/task_config.yaml'
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config['project_path']  # Modify this to your need.
    save_as_csv = True
    analyze_images(args.images_folder, config_path, destfolder=None, image_type='png', save_as_csv=save_as_csv,
                   overwrite=True, analyze2d=analyze2d, analyze3d=analyze3d, pcutoff=0.6)



if __name__ == '__main__':
    arguments = sys.argv[1:]
    images_dir = arguments[0]
    imagetype = arguments[1]
    if arguments[2] == 'True':
        visualize = True
    else:
        visualize = False
    config_path = 'C:/Users/xyao0/Desktop/project/assembly/data/task_config.yaml'
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config['project_path']  # Modify this to your need.
    analyze_images(images_dir, config_path, destfolder=None, image_type=imagetype, save_as_csv=True,
                   overwrite=True, analyze2d=True, analyze3d=True, pcutoff=0.6, visualize_image = visualize)

    # config_path = './data/task_config.yaml'
    # with open(config_path) as file:
    #     config = yaml.load(file, Loader=yaml.FullLoader)
    # project_dir = config['project_path']  # Modify this to your need.
    # data_dir = os.path.join(project_dir, 'data', 'raw')
    # filterpredictions = True
    # filtertype = 'median'
    # save_as_csv = True
    # calibrate = False
    # overwrite = True
    # videotype = 'avi'
    # objs = config['objects']
    # cams = config['cameras']
    # template_id = str(config['template_video_id'])
    #
    # data_root, demo_dirs, data_files = next(os.walk(data_dir))
    # DLC3D = config['dlc3d_path']
    # DLC2D = config['dlc_path']
    # # config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
    # config3d = glob(os.path.join(DLC3D, f'*wrist_cam*', 'config.yaml'))[0]
    # analyze_2d = False
    # analyze_3d = True
    #
    # for i, demo in enumerate(sorted(demo_dirs)):
    #     demo_dir = os.path.join(data_root, demo)
    #     _, actions_dir, _ = next(os.walk(demo_dir))
    #     actions = [action for action in actions_dir if 'action' in action]
    #     for action in actions:
    #         imgs = glob(os.path.join(demo_dir, action, 'wrist_images','*.jpeg'))
    #         destfolder = os.path.join(demo_dir, action, 'wrist_images')
    #         if analyze_2d:
    #             for obj in objs:
    #                 config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
    #                 cfg = auxiliaryfunctions.read_config(config2d)
    #                 if cfg['multianimalproject']:  ## multi-animal project
    #                     multi = True
    #                 else:
    #                     multi = False
    #                 ### Analyze 2D
    #                 deeplabcut.analyze_time_lapse_frames(config2d, destfolder, shuffle=1, frametype='.jpeg',
    #                             trainingsetindex=0,gputouse=None,save_as_csv=True)
    #         #
    #         # ### modify h5 files
    #         h5files = serch_obj_h5files(destfolder)
    #         print(h5files)
    #         for obj in objs:
    #             h5file_obj = [f for f in h5files if obj in f]
    #             _modify_h5file(h5file_obj, obj=obj, multi=False)
    #
    #         ### combine h5 files
    #         dir_combined_2d = os.path.join(demo_dir, action, '2d_combined')
    #         combined_h5file_2d = combine_h5files(h5files, destdir=dir_combined_2d, suffix=f'wrist_2d')
    #         create_images_with_h5file(imgs, combined_h5file_2d, thresh=0.6, over_write=overwrite, destfolder=dir_combined_2d)
    #
    #
    #         # ## Triangulate
    #         if analyze_3d:
    #             df_2d = pd.read_hdf(combined_h5file_2d)
    #             dir_combined_3d = dir_combined_2d.replace('2d', '3d')
    #             files_to_remove = glob(os.path.join(dir_combined_3d, '*h5.h5'))
    #             if len(files_to_remove) == 1:
    #                 file_to_remove = files_to_remove[0]
    #                 os.remove(file_to_remove)
    #             df_3d = triangulate_images(config3d, combined_h5file_2d, destfolder = dir_combined_3d, pcutoff=0.5, save_as_csv=save_as_csv)

