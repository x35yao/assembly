from glob import glob
import os
import pandas as pd
import yaml
import deeplabcut
from utils import combine_h5files, serch_obj_h5files
from triangulation import triangulate_images
from visualization import create_video_with_h5file, create_images_with_h5file
from analyze_videos_and_images import _modify_h5files
from deeplabcut.utils import auxiliaryfunctions


def analyze_images(images_folder, task_config_path, destfolder = None, save_as_csv = True,  overwrite = True, analyze2d = True, analyze3d = True, create_video = True, pcutoff = 0.2):
    with open(task_config_path) as file:
        config = yaml.load(file, Loader = yaml.FullLoader)
    objs = config['objects']
    cams = config['cameras']

    DLC3D = config['dlc3d_path']
    DLC2D = config['dlc_path']
    config3d = glob(os.path.join(DLC3D, f'*wrist_cam*', 'config.yaml'))[0]
    ### Run Deeplabcut to analyze videos
    imgs = glob(os.path.join(images_folder, '*.jpeg'))
    for obj in objs:
        config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
        cfg = auxiliaryfunctions.read_config(config2d)
        if cfg['multianimalproject']:  ## multi-animal project
            multi = True
        else:
            multi = False
        ### Analyze 2D
        deeplabcut.analyze_time_lapse_frames(config2d, images_folder, shuffle=1, frametype='.jpeg',
                                             trainingsetindex=0, gputouse=None, save_as_csv=True)
    h5files = serch_obj_h5files(images_folder)
    for obj in objs:
        h5file_obj = [f for f in h5files if obj in f]
        _modify_h5files(h5file_obj, obj=obj, multi=False)

    ### combine h5 files
    if destfolder is None:
        destfolder = images_folder
    dir_combined_2d = os.path.join(destfolder, '2d_combined')
    combined_h5file_2d = combine_h5files(h5files, destdir=dir_combined_2d, suffix=f'wrist_2d')
    create_images_with_h5file(imgs, combined_h5file_2d, thresh=0.6, over_write=overwrite,
                              destfolder=dir_combined_2d)

    # ## Triangulate
    if analyze3d:
        df_2d = pd.read_hdf(combined_h5file_2d)
        dir_combined_3d = dir_combined_2d.replace('2d', '3d')
        os.makedirs(dir_combined_3d, exist_ok=True)
        files_to_remove = glob(os.path.join(dir_combined_3d, '*h5.h5'))
        if len(files_to_remove) == 1:
            file_to_remove = files_to_remove[0]
            os.remove(file_to_remove)
        df_3d = triangulate_images(config3d, combined_h5file_2d, destfolder=dir_combined_3d, pcutoff=pcutoff)

if __name__ == '__main__':
    with open('./data/raw/task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    project_dir = config['project_path']  # Modify this to your need.
    data_dir = os.path.join(project_dir, 'data', 'raw', '2024-06-06')
    filterpredictions = True
    filtertype = 'median'
    save_as_csv = True
    calibrate = False
    overwrite = True
    videotype = 'avi'
    objs = config['objects']
    cams = config['cameras']
    template_id = str(config['template_video_id'])

    data_root, demo_dirs, data_files = next(os.walk(data_dir))
    DLC3D = config['dlc3d_path']
    DLC2D = config['dlc_path']
    # config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
    config3d = glob(os.path.join(DLC3D, f'*wrist_cam*', 'config.yaml'))[0]

    data_root, demo_dirs, data_files = next(os.walk(data_dir))
    analyze_2d = False
    analyze_3d = False


    for i, demo in enumerate(sorted(demo_dirs)):
        demo_dir = os.path.join(data_root, demo)
        _, actions_dir, _ = next(os.walk(demo_dir))
        actions = [action for action in actions_dir if 'action' in action]
        for action in actions:
            imgs = glob(os.path.join(demo_dir, action, 'wrist_images','*.jpeg'))
            destfolder = os.path.join(demo_dir, action, 'wrist_images')
            if analyze_2d:
                for obj in objs:
                    config2d = glob(os.path.join(DLC2D, f'*{obj}*', 'config.yaml'))[0]
                    cfg = auxiliaryfunctions.read_config(config2d)
                    if cfg['multianimalproject']:  ## multi-animal project
                        multi = True
                    else:
                        multi = False
                    ### Analyze 2D
                    deeplabcut.analyze_time_lapse_frames(config2d, destfolder, shuffle=1, frametype='.jpeg',
                                trainingsetindex=0,gputouse=None,save_as_csv=True)
            #
            # ### modify h5 files
            h5files = serch_obj_h5files(destfolder)
            for obj in objs:
                h5file_obj = [f for f in h5files if obj in f]
                _modify_h5files(h5file_obj, obj=obj, multi=False)

            ### combine h5 files
            dir_combined_2d = os.path.join(demo_dir, action, '2d_combined')
            combined_h5file_2d = combine_h5files(h5files, destdir=dir_combined_2d, suffix=f'wrist_2d')
            create_images_with_h5file(imgs, combined_h5file_2d, thresh=0.6, over_write=overwrite, destfolder=dir_combined_2d)


            # ## Triangulate
            if analyze_3d:
                df_2d = pd.read_hdf(combined_h5file_2d)
                dir_combined_3d = dir_combined_2d.replace('2d', '3d')
                files_to_remove = glob(os.path.join(dir_combined_3d, '*h5.h5'))
                if len(files_to_remove) == 1:
                    file_to_remove = files_to_remove[0]
                    os.remove(file_to_remove)
                df_3d = triangulate_images(config3d, combined_h5file_2d, destfolder = dir_combined_3d, pcutoff=0)

    # raise
    # destfolder = os.path.join(project_dir, '2024-02-07',  'imgs')
    # deeplabcut.analyze_time_lapse_frames(config2d, destfolder, shuffle=1,
    #                 trainingsetindex=0,gputouse=None,save_as_csv=True)
    # h5files = serch_obj_h5files(destfolder)
    # h5files = [f for f in h5files if obj in f]
    # _modify_h5files(h5files, obj = obj, multi = False)
    # deeplabcut.create_labeled_video(config2d, img, filtered = False,destfolder = destfolder, pcutoff= 0.6, overwrite=True, draw_skeleton= True, save_frames=True)