"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from tqdm import tqdm

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
from deeplabcut.utils import auxfun_multianimal
# from LEAstereo.predictDepth import *

matplotlib_axes_logger.setLevel("ERROR")


def triangulate_images(
    config,
    h5files,
    destfolder=None,
    save_as_csv=True,
    pcutoff = None
):
    """
    This function triangulates the detected DLC-keypoints from the two camera views
    using the camera matrices (derived from calibration) to calculate 3D predictions.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    video_path : string/list of list
        Full path of the directory where videos are saved. If the user wants to analyze
        only a pair of videos, the user needs to pass them as a list of list of videos,
        i.e. [['video1-camera-1.avi','video1-camera-2.avi']]

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n
        Only videos with this extension are analyzed. The default is ``.avi``

    filterpredictions: Bool, optional
        Filter the predictions with filter specified by "filtertype". If specified it
        should be either ``True`` or ``False``.

    filtertype: string
        Select which filter, 'arima' or 'median' filter (currently supported).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi).
        If you do not have a GPU put None.
        See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    destfolder: string, optional
        Specifies the destination folder for analysis data (default is the path of the video)

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``

    Example
    -------
    Linux/MacOS
    To analyze all the videos in the directory:
    >>> deeplabcut.triangulate(config,'/data/project1/videos/')

    To analyze only a few pairs of videos:
    >>> deeplabcut.triangulate(config,[['/data/project1/videos/video1-camera-1.avi','/data/project1/videos/video1-camera-2.avi'],['/data/project1/videos/video2-camera-1.avi','/data/project1/videos/video2-camera-2.avi']])


    Windows
    To analyze all the videos in the directory:
    >>> deeplabcut.triangulate(config,'C:\\yourusername\\rig-95\\Videos')

    To analyze only a few pair of videos:
    >>> deeplabcut.triangulate(config,[['C:\\yourusername\\rig-95\\Videos\\video1-camera-1.avi','C:\\yourusername\\rig-95\\Videos\\video1-camera-2.avi'],['C:\\yourusername\\rig-95\\Videos\\video2-camera-1.avi','C:\\yourusername\\rig-95\\Videos\\video2-camera-2.avi']])
    """
    from deeplabcut.pose_estimation_tensorflow import predict_videos

    cfg_3d = auxiliaryfunctions.read_config(config)
    cam_names = cfg_3d["camera_names"]
    if pcutoff is None:
        pcutoff = cfg_3d["pcutoff"]
    scorer_3d = cfg_3d["scorername_3d"]
    scorer_name = {}


    print("Undistorting...")
    (
        dataFrame_camera_undistort,
        stereomatrix,
        path_stereo_file,
    ) = undistort_points(
        config, h5files, str(cam_names[0] + "-" + cam_names[1])
    )
    # dataFrame_camera1_undistort = dataFrame_camera_undistort.iloc[::2]
    # dataFrame_camera2_undistort = dataFrame_camera_undistort.iloc[1::2]

    dataframe_distort = pd.read_hdf(h5files).droplevel(level= 0, axis = 1)
    dataFrame_camera1_undistort = dataframe_distort.iloc[::2]
    dataFrame_camera2_undistort = dataframe_distort.iloc[1::2]


    if len(dataFrame_camera1_undistort) != len(dataFrame_camera2_undistort):
        import warnings

        warnings.warn(
            "The number of frames do not match in the two videos. Please make sure that your videos have same number of frames and then retry! Excluding the extra frames from the longer video."
        )
        if len(dataFrame_camera1_undistort) > len(dataFrame_camera2_undistort):
            dataFrame_camera1_undistort = dataFrame_camera1_undistort[
                : len(dataFrame_camera2_undistort)
            ]
        if len(dataFrame_camera2_undistort) > len(dataFrame_camera1_undistort):
            dataFrame_camera2_undistort = dataFrame_camera2_undistort[
                : len(dataFrame_camera1_undistort)
            ]
    #                raise Exception("The number of frames do not match in the two videos. Please make sure that your videos have same number of frames and then retry!")

    # df_3d, scorer_3d, bodyparts = auxiliaryfunctions_3d.create_empty_df(
    #     dataFrame_camera_undistort, scorer_3d, flag="3d"
    # )


    df_3d = dataFrame_camera1_undistort.copy()
    columns = df_3d.columns
    names = columns.names
    new_columns = [(c[0], c[1], c[2].replace('likelihood', 'z') )for c in columns]
    new_index = pd.MultiIndex.from_tuples(new_columns, names = names)
    df_3d.columns = new_index
    df_3d.iloc[:] = np.NaN

    P1 = stereomatrix["P1"]
    P2 = stereomatrix["P2"]

    print("Computing the triangulation...")
    individuals = dataFrame_camera_undistort.columns.get_level_values("individuals").unique()
    for individual in individuals:
        X_final = []
        triangulate = []
        bodyparts = dataFrame_camera1_undistort[individual].columns.get_level_values("bodyparts").unique()
        for bpindex, bp in enumerate(bodyparts):
            # Extract the indices of frames where the likelihood of a bodypart for both cameras are less than pvalue
            likelihoods = np.array(
                [
                    dataFrame_camera1_undistort[individual][bp][
                        "likelihood"
                    ].values[:],
                    dataFrame_camera2_undistort[individual][bp][
                        "likelihood"
                    ].values[:],
                ]
            )
            likelihoods = likelihoods.T

            # Extract frames where likelihood for both the views is less than the pcutoff
            low_likelihood_frames = np.any(likelihoods < pcutoff, axis=1)
            # low_likelihood_frames = np.all(likelihoods < pcutoff, axis=1)

            low_likelihood_frames = np.where(low_likelihood_frames == True)[0]
            points_cam1_undistort = np.array(
                [
                    dataFrame_camera1_undistort[individual][bp]["x"].values[:],
                    dataFrame_camera1_undistort[individual][bp]["y"].values[:],
                ]
            )
            points_cam1_undistort = points_cam1_undistort.T


            # For cam1 camera: Assign nans to x and y values of a bodypart where the likelihood for is less than pvalue
            points_cam1_undistort[low_likelihood_frames] = np.nan, np.nan
            # points_cam1_undistort = np.expand_dims(points_cam1_undistort, axis=1)
            points_cam1_undistort = points_cam1_undistort.T

            points_cam2_undistort = np.array(
                [
                    dataFrame_camera2_undistort[individual][bp]["x"].values[:],
                    dataFrame_camera2_undistort[individual][bp]["y"].values[:],
                ]
            )
            points_cam2_undistort = points_cam2_undistort.T

            # For cam2 camera: Assign nans to x and y values of a bodypart where the likelihood is less than pvalue
            points_cam2_undistort[low_likelihood_frames] = np.nan, np.nan
            # points_cam2_undistort = np.expand_dims(points_cam2_undistort, axis=1)
            points_cam2_undistort = points_cam2_undistort.T

            X_l = auxiliaryfunctions_3d.triangulatePoints(
                P1, P2, points_cam1_undistort, points_cam2_undistort
            )

            # ToDo: speed up func. below by saving in numpy.array
            X_final.append(X_l)
        triangulate.append(X_final)
        triangulate = np.asanyarray(triangulate)
        metadata = {}
        metadata["stereo_matrix"] = stereomatrix
        metadata["stereo_matrix_file"] = path_stereo_file
        df_inds = df_3d.index
        for bpindex, bp in enumerate(bodyparts):
            for j in range(triangulate.shape[-1]):
                df_3d.loc[df_inds[j], (individual, bp, "x")] = triangulate[0, bpindex, 0, j]
                df_3d.loc[df_inds[j], (individual, bp, "y")] = triangulate[0, bpindex, 1, j]
                df_3d.loc[df_inds[j],(individual, bp, "z")] = triangulate[0, bpindex, 2, j]
    if destfolder == None:
        destfolder = os.path.dirname(os.path.dirname(os.path.dirname(h5files[0])))
    filename =  os.path.basename(h5files).replace('2d', '3d')
    output_filename = os.path.join(destfolder, filename)
    if isinstance(df_3d.index, pd.MultiIndex):
        new_index = [ind[0] for ind in df_3d.index]
        df_3d.index = new_index
    df_3d.to_hdf(
        output_filename,
        "df_with_missing",
        format="table",
        mode="w",
    )
    auxiliaryfunctions_3d.SaveMetadata3d(
        str(output_filename + "_meta.pickle"), metadata
    )

    if save_as_csv:
        df_3d.to_csv(output_filename.replace('.h5', '.csv'))

    print("Results are saved under: ", destfolder)



"""
ToDo: speed up func. below and check only for one cam individually
PredicteData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
pdindex = pd.MultiIndex.from_product([[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],names=['scorer', 'bodyparts', 'coords'])
auxiliaryfunctions.SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, framelist,save_as_csv)
"""


def undistort_points(config, dataframe, camera_pair):
    cfg_3d = auxiliaryfunctions.read_config(config)
    (
        img_path,
        path_corners,
        path_camera_matrix,
        path_undistort,
        _,
    ) = auxiliaryfunctions_3d.Foldernames3Dproject(cfg_3d)
    """
    path_undistort = destfolder
    filename_cam1 = Path(dataframe[0]).stem
    filename_cam2 = Path(dataframe[1]).stem

    #currently no interm. saving of this due to high speed.
    # check if the undistorted files are already present
    if os.path.exists(os.path.join(path_undistort,filename_cam1 + '_undistort.h5')) and os.path.exists(os.path.join(path_undistort,filename_cam2 + '_undistort.h5')):
        print("The undistorted files are already present at %s" % os.path.join(path_undistort,filename_cam1))
        dataFrame_cam1_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam1 + '_undistort.h5'))
        dataFrame_cam2_undistort = pd.read_hdf(os.path.join(path_undistort,filename_cam2 + '_undistort.h5'))
    else:
    """
    if True:
        # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
        dataframe_cam = pd.read_hdf(dataframe)
        dataframe_cam.columns = dataframe_cam.columns.droplevel(0)
        path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
        stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
        mtx_l = stereo_file[camera_pair]["cameraMatrix1"]
        dist_l = stereo_file[camera_pair]["distCoeffs1"]

        mtx_r = stereo_file[camera_pair]["cameraMatrix2"]
        dist_r = stereo_file[camera_pair]["distCoeffs2"]

        R1 = stereo_file[camera_pair]["R1"]
        P1 = stereo_file[camera_pair]["P1"]

        R2 = stereo_file[camera_pair]["R2"]
        P2 = stereo_file[camera_pair]["P2"]

        # Create an empty dataFrame to store the undistorted 2d coordinates and likelihood
        dataFrame_cam_undistort = dataframe_cam.copy()
        individuals = dataFrame_cam_undistort.columns.get_level_values("individuals").unique()
        for individual in individuals:
            bodyparts = dataFrame_cam_undistort[individual].columns.get_level_values("bodyparts").unique()
            for bpindex, bp in tqdm(enumerate(bodyparts)):
                points_cam1 = dataframe_cam[individual][bp].values[:,:-1]
                points_cam1_remapped = cv2.undistortPoints(
                    src=points_cam1.astype(np.float32),
                    cameraMatrix=mtx_l,
                    distCoeffs=dist_l,
                    P=P1,
                    R=R1,
                )


                dataFrame_cam_undistort.loc(axis=1)[individual, bp, ["x", "y"]
                ] = points_cam1_remapped.squeeze()
                dataFrame_cam_undistort.loc(axis=1)[
                     individual, bp, "likelihood"
                ] = dataframe_cam.xs(
                    (individual, bp, "likelihood"), level=["individuals","bodyparts", "coords"], axis=1
                ).values

        # Save the undistorted files
        dataFrame_cam_undistort.sort_index(inplace=True)

    return (
        dataFrame_cam_undistort,
        stereo_file[camera_pair],
        path_stereo_file,
    )
