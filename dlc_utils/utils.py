from glob import glob
import cv2
import os
import pandas as pd

def get_videos(vid_id, obj = None, filtertypes = None, cameras = ['left', 'right'], base_dir = '/home/luke/Desktop/project/make_tea'):
    '''
    This function output video path given the video ID.

    vid_id: The video ID
    obj: The object folder the video is sit in. Options: 'teabag', 'cup', 'pitcher', 'tap'. If None, video in one level higher will be outputed.
    kernel: Given kernel, this function output the video filtered by the kernel. If None, the original non-filtered vildeo path will be outputed.
    base_dir: The path to the make_tea folder.
    '''
    videos = []

    if not isinstance(cameras, list):
        cameras = [cameras]
    for camera in cameras:
        if filtertypes == None:
            if obj == None:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{vid_id}-{camera}.mp4'
            else:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{obj}/{vid_id}-{camera}.mp4'
        else:
            if not isinstance(filtertypes, list):
                suffix = f'_{filtertypes}'
            else:
                temp = '_'.join(filtertypes)
                suffix = f'_{temp}'
            if obj == None:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{vid_id}-{camera}{suffix}.mp4'
            else:
                video = f'{base_dir}/camera-main/videos/{vid_id}/{camera}/{obj}/{vid_id}-{camera}{suffix}.mp4'
        videos.append(video)
    return videos



def get_h5files(dirname, filtertype = None):
    h5files = glob(os.path.join(dirname, '*.h5'))
    return h5files

def video_to_frames(video_path):

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    frames = []
    while success:
      frames.append(image)
      success,image = vidcap.read()
    print(f'The number of frames is {len(frames)}')
    return frames

def serch_obj_h5files(target_dir):
    h5files = glob(os.path.join(target_dir, '*.h5'))
    return h5files

def _multi_ind_to_single_ind(df):
    inds = []
    for ind in df.index:
        inds.append(ind[0])
    df.index = inds
    return df

def combine_h5files(h5files, to_csv = True, destdir = None, suffix = '2d'):
    df_new = pd.DataFrame()
    for h5file in h5files:
        df = pd.read_hdf(h5file)
        if isinstance(df.index, pd.MultiIndex):
            df = _multi_ind_to_single_ind(df)
        df_new = pd.concat([df_new, df], axis = 1)
    if destdir == None:
        destdir = os.path.dirname(os.path.dirname(h5file))
    else:
        if not os.path.isdir(destdir):
            os.makedirs(destdir)
    outputname = destdir + '/' + f'markers_trajectory_{suffix}.h5'
    if os.path.isfile(outputname):
        print('Removing existing file.')
        os.remove(outputname)
    df_new.to_hdf(outputname, mode = 'w', key = f'markers_trajectory_{suffix}')
    if to_csv:
        df_new.to_csv(outputname.replace('.h5', '.csv'))
    print(f'The file is saved at {outputname}')
    return outputname

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
