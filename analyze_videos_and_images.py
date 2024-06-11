import deeplabcut
import os
from glob import glob
import sys
import shutil
import pandas as pd
from deeplabcut.post_processing import filtering

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
from utils import combine_h5files, serch_obj_h5files
from visualization import create_video_with_h5file, create_images_with_h5file
import yaml
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.refine_training_dataset.stitch import stitch_tracklets
from deeplabcut import analyze_videos, convert_detections2tracklets
import shutil
from triangulation import triangulate_images

def rename_individual(df, obj):
    if 'ind1' in df.columns.get_level_values('individuals'): ## This happens when n_tracks does not equal the defalt value
        n_individuals = len(set(df.columns.get_level_values('individuals')))
        new_level = [f'{obj}{i+1}' for i in range(n_individuals)]  ## Rename the value from 'ind1' to f'{obj}1'
        df.columns = df.columns.set_levels(new_level, level = 'individuals')
    return df

def add_individual_level(df, obj):
    if 'individuals' not in df.columns.names:
        df = pd.concat([df], keys=[f'{obj}1'], axis=1)
        df.columns.set_names('individuals', level=0, inplace=True)
        ## Reorder the names
        df = df.reorder_levels([1, 0, 2, 3], axis=1)
    return df

def analyze_2d(config2d, vids, destfolder, videotype = 'avi', filterpredictions = True, filtertype = 'median', n_tracks = 1):

    successful = True
    cfg = auxiliaryfunctions.read_config(config2d)
    if cfg['multianimalproject']:  ## multi-animal project
        multi = True
    else:
        multi = False
    if multi:
        if n_tracks is None:
            n_tracks = len(cfg['individuals'])
        deeplabcut.analyze_videos(config2d, vids, destfolder=destfolder, auto_track=False)
        convert_detections2tracklets(
            config2d,
            vids,
            videotype,
            destfolder=destfolder,
            calibrate=calibrate,
        )
        for j in reversed(range(n_tracks + 1)):
            print(f'n_tracks is : {j}')
            if j == 0:  ### Does not detect any individual
                successful = False
                break
            try:
                stitch_tracklets(
                    config2d,
                    vids,
                    videotype,
                    destfolder=destfolder,
                    n_tracks=j,
                    save_as_csv=save_as_csv,
                )
                break
            except (ValueError, OSError) as e:
                continue
    else:  ## single-animal project
        deeplabcut.analyze_videos(config2d, vids, destfolder=destfolder, auto_track=True)

    ## Filter result
    if filterpredictions:
        filtering.filterpredictions(
            config2d,
            vids,
            videotype=videotype,
            filtertype=filtertype,
            destfolder=destfolder,
        )
    return successful

def _modify_h5files(h5files, obj, multi):
    for h5file in h5files:
        df = pd.read_hdf(h5file)
        if multi:
            df = rename_individual(df, obj)
        else:
            df = add_individual_level(df, obj)
        if os.path.isfile(h5file):
            os.remove(h5file)
        df.to_hdf(h5file, key='df_with_missing')
        df.to_csv(h5file.replace('.h5', '.csv'))

def analyze_demo(demo_dir, objs, DLC2D, DLC3D, cams, analyze_video_2d = True, analyze_video_3d = True, analyze_image_2d = True,
                 analyze_image_3d = True, filterpredictions = True, filtertype = 'median', save_as_csv = True, calibrate = False,
                 overwrite = True, videotype = 'avi', frametype = '.jpeg', visualize = True):
    is_successful = True  ## Successfully detect all objects
    _, actions_dir, _ = next(os.walk(demo_dir))
    actions = [action for action in actions_dir if 'action' in action]
    config3d_wrist = glob(os.path.join(DLC3D, f'*wrist_cam*', 'config.yaml'))[0]
    for action in sorted(actions):
        vids = []
        h5files_3d = []
        h5files_2d = []
        image_folder = os.path.join(demo_dir, action, 'wrist_images')
        for cam in cams:
            vids.append(glob(os.path.join(demo_dir, action, f'{action}*{cam}.avi'))[0])
        for obj in objs:
            print(f'Processing {demo}, {action}, {obj}!!!!!!!!!!!!!!')
            destfolder = os.path.join(demo_dir, action, obj)
            files_3d = glob(os.path.join(destfolder, '*3d*'))
            for f in files_3d:
                os.remove(f)
            config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
            config3d = glob(os.path.join(DLC3D, f'*{obj}*', 'config.yaml'))[0]
            cfg = auxiliaryfunctions.read_config(config2d)
            if cfg['multianimalproject']:  ## multi-animal project
                multi = True
            else:
                multi = False
            ##### Analyze videos using Deeplabcut ###############
            if analyze_video_2d:
                analyzed = analyze_2d(config2d, vids, destfolder=destfolder, videotype=videotype,
                                      filterpredictions=filterpredictions, filtertype=filtertype)
                if not analyzed:
                    is_successful = False

            ### Modify the h5files obtained
            h5files = serch_obj_h5files(destfolder)
            if filterpredictions:
                h5files_obj_2d = [f for f in h5files if 'filtered' in f]
            _modify_h5files(h5files_obj_2d, obj, multi)
            h5files_2d.extend(h5files_obj_2d)

            ### Create video for each object
            if visualize:
                deeplabcut.create_labeled_video(config2d, vids, filtered=filterpredictions, destfolder=destfolder)

            if analyze_video_3d and is_successful:
                deeplabcut.triangulate(config3d, [vids], filterpredictions=filterpredictions, destfolder=destfolder,
                                       save_as_csv= save_as_csv, video_analyzed=True)
                h5files_new = serch_obj_h5files(destfolder)
                h5files_obj_3d = [f for f in h5files_new if '3d' in f]
                _modify_h5files(h5files_obj_3d, obj, multi)
                h5files_3d.extend(h5files_obj_3d)

            ### Analyze images from wrist camera ###
            if analyze_image_2d:
                deeplabcut.analyze_time_lapse_frames(config2d, image_folder, shuffle=1, frametype= frametype,
                                                     trainingsetindex=0, gputouse=None, save_as_csv=True)
            # ### modify h5 files
            h5files_image = serch_obj_h5files(image_folder)
            h5file_image_obj = [f for f in h5files_image if obj in f]
            _modify_h5files(h5file_image_obj, obj=obj, multi=False)

        ### combine h5file for videos
        demo_dir_combined = os.path.join(demo_dir, action, '2d_combined')
        for cam in cams:
            # ##### Create labeled video ##############
            h5files_2d_cam = [f for f in h5files_2d if cam in f]
            demo_dir_combined = os.path.join(demo_dir, action, '2d_combined')
            combined_h5file_2d_video = combine_h5files(h5files_2d_cam, destdir=demo_dir_combined, suffix=f'{cam}_2d')
            if visualize:
                vid = [v for v in vids if cam in v][0]
                create_video_with_h5file(vid, combined_h5file_2d_video, thresh=-1, over_write=overwrite)

        ### combine h5file for images
        h5files_image = serch_obj_h5files(image_folder)
        combined_h5file_2d_image = combine_h5files(h5files_image, destdir=demo_dir_combined, suffix=f'wrist_2d')
        print(combined_h5file_2d_image)
        if visualize:
            imgs = glob(os.path.join(image_folder,'*.jpeg'))
            create_images_with_h5file(imgs, combined_h5file_2d_image, thresh=-1, over_write=overwrite,
                                  destfolder=demo_dir_combined)

        ### Triangulate for images
        destfolder_demo_3d = os.path.join(data_root, demo, action, '3d_combined')
        if analyze_image_3d:
            df_3d = triangulate_images(config3d_wrist, combined_h5file_2d_image, destfolder=destfolder_demo_3d, pcutoff=0)

        if analyze_video_3d:
            combine_h5files(h5files_3d, destdir=destfolder_demo_3d, suffix='3d')

    return is_successful



if __name__ == '__main__':
    # Load data
    with open('./data/raw/task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config['project_path']  # Modify this to your need.
    data_dir = os.path.join(project_dir, 'data', 'raw')
    filterpredictions = True
    filtertype = 'median'
    save_as_csv = True
    calibrate = False
    overwrite = True
    videotype = 'avi'
    objs = config['objects']
    cams = config['cameras']

    data_root, demo_dirs, data_files = next(os.walk(data_dir))
    DLC3D = config['dlc3d_path']
    DLC2D = config['dlc_path']
    dlc_root, dlc_dirs, dlc_files = next(os.walk(DLC3D))
    bad_demos = []
    analyze_video_2d = False
    analyze_image_2d = False
    analyze_video_3d = True
    analyze_image_3d = True
    visualize = False
    ### Run Deeplabcut to analyze videos
    for i, demo in enumerate(sorted(demo_dirs)):
        if not demo == '1711655345':
            continue
        demo_dir = os.path.join(data_root, demo)
        is_successfuly = analyze_demo(demo_dir, objs, DLC2D, DLC3D, cams, analyze_video_2d, analyze_video_3d, analyze_image_2d,
                 analyze_image_3d, filterpredictions, filtertype, save_as_csv = save_as_csv, calibrate = calibrate,
                 overwrite = overwrite, videotype = 'avi', frametype = '.jpeg', visualize = visualize)
        if not is_successfuly:
            bad_demos.append(demo)
    print(set(bad_demos))



