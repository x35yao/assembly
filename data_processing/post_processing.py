import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pandas as pd
import numpy as np
from glob import glob
import os
from  deeplabcut.utils import auxiliaryfunctions,auxfun_multianimal
from scipy.interpolate import interp1d
from pathlib import Path
import deeplabcut
from deeplabcut.refine_training_dataset.outlier_frames import FitSARIMAXModel
from scipy import signal, ndimage
from sklearn.linear_model import LinearRegression,RANSACRegressor
from skimage.measure import LineModelND, ransac
# import sws
import scipy.stats as st
# from utils import get_videos, get_h5files
from tqdm import trange



def normalize(x):
    norm = np.linalg.norm(x, axis = 1)
    return x / norm[:, None]

def get_pos_mean(df, bodyparts, lr_model = 'ransac'):
    '''
    Given the trajectories of different bodyparts, get the average object postion.
    '''
    df_new = df.copy()
    if lr_model == 'ransac':
        n_data = len(df)
        pos = np.zeros((n_data, 3))
        for i in range(len(df)):
            data = df_new.iloc[i].values.reshape(-1, 3)
            X = data[:,:-1]
            y = data[:, -1]
            model_robust, inliers = ransac(data, LineModelND, min_samples=3,
                               residual_threshold=.6, max_trials=1000)
            print(i, inliers)
            outliers = inliers == False

            inds_in = np.where(inliers == 1)
            inds_out = np.where(outliers == 1)
            x = np.arange(data.shape[0])
            print(data)
            data[outliers, :] = np.nan

            spl_x = interp1d(x[inds_in], data[inds_in, 0],kind = 'linear', fill_value='extrapolate')
            spl_y = interp1d(x[inds_in], data[inds_in, 1],kind = 'linear', fill_value='extrapolate')
            spl_z = interp1d(x[inds_in], data[inds_in, 2],kind = 'linear', fill_value='extrapolate')

            x_interp = spl_x(inds_out).flatten()
            y_interp = spl_y(inds_out).flatten()
            z_interp = spl_z(inds_out).flatten()

            data[inds_out, 0] = x_interp
            data[inds_out, 1] = y_interp
            data[inds_out, 2] = z_interp

            print([x_interp, y_interp, z_interp])
            df_new.iloc[i] = data.flatten()

    pos_mean = df_new.mean(axis = 1, level = 'coords')
    pos_mean = np.nan_to_num(pos_mean)
    return pos_mean


def get_ori_mean(df, bodyparts, lr_model = 'ransac'):
    '''
    Given the trajectories of different bodyparts, get the average object orientation.
    '''
    n_data = len(df)
    n_bodyparts = len(bodyparts)
    n_pairs = int(n_bodyparts/2)
    ori = np.zeros((n_data, 3))
    for i in range(len(df)):
        data = df.iloc[i].values.reshape(-1, 3)
        X = data[:,:-1]
        y = data[:, -1]
        if lr_model == 'ransac':
            model_robust, inliers = ransac(data, LineModelND, min_samples=3,
                               residual_threshold=1.5, max_trials=1000)
            print(i, inliers)
            # if i < 10:
            #     print(inliers)
            # else:
            #     raise

        elif lr_model == 'linear':
            lr = LinearRegression()
            lr.fit(X, y)
            m = lr.coef_
        vec = model_robust.params[1]
        vec_normalized = vec / np.linalg.norm(vec)
        if i > 0:
            if np.dot(vec_normalized, ori[i-1,:]) < 0:
                # This is the situation when 2 consecutive frames have opposite direction
                vec_normalized = -1 * vec_normalized
        ori[i,:] = vec_normalized
    return ori


def get_interval(x, confidence):

    v = np.nan_to_num(sws.get_gradient(x))
    a = np.nan_to_num(sws.get_gradient(v))
    mean_a = np.mean(a)
    std_a = np.std(a)
    z = st.norm.ppf(confidence)
    low = mean_a - z * std_a
    high = mean_a + z * std_a

    return low, high

def get_outlier_inds(x, low, high):
    v = np.nan_to_num(sws.get_gradient(x))
    a = np.nan_to_num(sws.get_gradient(v))
    mean_a = np.mean(a)
    std_a = np.std(a)
    mask = np.where(np.logical_and(abs(a) > high, abs(a) > 3))
    inds = mask[0][mask[0] > 10]
    return inds

def acc_smooth(x, window_size, confidence):
    low, high = get_interval(x, confidence)
    inds = get_outlier_inds(x, low, high)
    count = 0
    half_window = int(window_size/2)
    print(inds)
    while inds !=[]:
        inds_previous = inds
        ind = inds[0]
        start = ind - 1 - half_window
        end = ind + half_window
        np.median(x[start:end])
        x[ind] = x[ind - 1] = np.median(x[start:end])
        # x[ind] = x[ind - 1] = x[ind - 2]
        # x[ind] = x[ind - 1]
        inds = get_outlier_inds(x, low, high)
        if not set(inds_previous) == set(inds):
            print(inds)
        count += 1
        if count > 5000:
            confidence += 0.01
            low, high = get_interval(x, confidence = confidence)
    return x

def filter_3D_data(file_path, filtertype = 'median', window_size = 5, confidence = 0.95, thresh = 3):

    df= pd.read_hdf(file_path)
    # df = remove_nans(df_with_nans, remove_method = remove_method)
    df_filtered = df.copy()

    if filtertype == 'median':
        f = signal.medfilt
        df_filtered = df.apply(
                                f, args=(window_size,), axis=0)
    elif filtertype == 'gaussian':
        f = ndimage.gaussian_filter
        df_filtered = df.apply(
                                f, args=(window_size,), axis=0)
    elif filtertype == 'acc_smooth':
        n_columns = df.shape[1]
        for j in np.arange(n_columns):
            df_new = df.copy()
            x = df_new.values[:, j]
            low, high = get_interval(x, confidence)
            temp = ndimage.median_filter(x, size = window_size)
            inds = get_outlier_inds(temp, low , high)
            while inds != []:
                window_size += 1
                temp = ndimage.median_filter(x, size = window_size)
                inds = get_outlier_inds(temp, low, high)
            df_filtered.values[:,j] = temp
    if 'DLC' in file_path:
        outputname = file_path.split('DLC')[0] + f'_{filtertype}.h5'
    else:
        outputname = file_path.replace('.h5', f'_{filtertype}.h5')
    if os.path.isfile(outputname):
        os.remove(outputname)
    df_filtered.to_hdf(outputname, key = 'filtered_3d')
    print(f'File is saved at {outputname}.')
    return df_filtered

def get_pose_mean(df, lr_model = 'ransac', residual = 0.8):
    '''
    df is the dataframe for a object(pitcher, teabag, cup, exc.).
    '''
    df_new = df.copy()
    bodyparts = df.columns.get_level_values('bodyparts').unique()
    mid_ind = int(len(bodyparts) / 2)
    if lr_model == 'ransac':
        n_data = len(df)
        pose = np.empty((n_data, 6))
        pose[:] = np.nan
        for i in trange(len(df)):
            data = df_new.iloc[i].values.reshape(-1,3)
            data_temp = df_new.iloc[i].dropna().values.reshape(-1,3)
            n_valid_samples = data_temp.shape[0]
            mask_nan = np.isnan(data).any(axis = 1)
            inds_valid = np.where(~mask_nan)[0]
            if n_valid_samples < 2: # Less than 2 markers available
                continue
            elif n_valid_samples == 2: # 2 markers available, simple draw a line between them
                lm = LineModelND()
                lm.estimate(data_temp)
                model_robust = lm
                inds_inlier = inds_valid
            else: # More than 2 markers available, find a robust line using ransac
                model_robust, inliers = ransac(data_temp, LineModelND, min_samples= 2, residual_threshold = residual,
                                    max_trials=50)
                if model_robust == None:
                    # No robust model is found
                    continue
                else: # Find out if the middle marker is a inlier
                    inds_inlier = inds_valid[inliers]
            if mid_ind in inds_inlier: # Directly use the position of the middle marker if it is an inlier
                pose[i,:3] = data[mid_ind]
            else: # Interpolate middle marker if it is not an inlier
                x = np.arange(data.shape[0])
                spl_x = interp1d(x[inds_inlier], data[inds_inlier, 0],kind = 'linear', fill_value='extrapolate')
                spl_y = interp1d(x[inds_inlier], data[inds_inlier, 1],kind = 'linear', fill_value='extrapolate')
                spl_z = interp1d(x[inds_inlier], data[inds_inlier, 2],kind = 'linear', fill_value='extrapolate')
                pose[i,:3] = spl_x(mid_ind),spl_y(mid_ind),spl_z(mid_ind)
            # Get the direction vector to represent the orientation of the object
            vec = model_robust.params[1]
            vec_normalized = vec / np.linalg.norm(vec)
            if i > 0:
                if np.dot(vec_normalized, pose[i-1,3:]) < 0:
                    # This is the situation when 2 consecutive frames have opposite direction
                    vec_normalized = -1 * vec_normalized
            pose[i,3:] = vec_normalized
    return pose

def get_obj_pose(file_path, residual = 1, save_to_csv = True, fps = 30):

    '''
    This function will take the file(obtained from trangulation) that containes the trajectories of the markers.
    Markers of the same object will be used to compute the position and orientation of each object at each frame.

    file_path: path to the file that containes the 3D trajectories of the makers.

    '''
    print(f'Processing the file {file_path}')
    df = pd.read_hdf(file_path).droplevel(0, axis = 1)
    individuals = df.columns.get_level_values('individuals').unique()
    df_new = pd.DataFrame()

    for individual in individuals:
        print(f'Processing {individual}: ')
        df_individual = df[individual]
        pose_mean = get_pose_mean(df_individual,lr_model = 'ransac', residual = residual)
        pdindex = pd.MultiIndex.from_product(
                    [[individual], ["x", "y", "z", "X", "Y","Z"]],
                    names=["individuals","pose"],
                )
        frame = pd.DataFrame(pose_mean, columns=pdindex)
        df_new = pd.concat([frame, df_new], axis=1)
    df_new['time_stamp'] = df.index * 1 / fps
    dirname = os.path.dirname(file_path)
    filename = dirname + '/obj_pose_trajectory.h5'
    if os.path.isfile(filename):
        print('Removing existing file.')
        os.remove(filename)
    df_new.to_hdf(filename, key = 'obj_pose')
    print(f'The file is saved at {filename}.')
    if save_to_csv:
        filename_csv = filename.replace('.h5', '.csv')
        df_new.to_csv(filename_csv)
    return df_new

def columnwise_interp(data, filtertype, max_gap=0):
    """
    Perform cubic spline interpolation over the columns of *data*.
    All gaps of size lower than or equal to *max_gap* are filled,
    and data slightly smoothed.

    Parameters
    ----------
    data : array_like
        2D matrix of data.
    max_gap : int, optional
        Maximum gap size to fill. By default, all gaps are interpolated.

    Returns
    -------
    interpolated data with same shape as *data*
    """
    if np.ndim(data) < 2:
        data = np.expand_dims(data, axis=1)
    nrows, ncols = data.shape
    temp = data.copy()
    valid = ~np.isnan(temp)

    x = np.arange(nrows)
    for i in range(ncols):
        mask = valid[:, i]
        if (
            np.sum(mask) > 3
        ):  # Make sure there are enough points to fit the cubic spline
            spl = interp1d(x[mask], temp[mask, i],kind = filtertype, fill_value='extrapolate')
            y = spl(x)
            if max_gap > 0:
                inds = np.flatnonzero(np.r_[True, np.diff(mask), True])
                count = np.diff(inds)
                inds = inds[:-1]
                to_fill = np.ones_like(mask)
                for ind, n, is_nan in zip(inds, count, ~mask[inds]):
                    if is_nan and n > max_gap:
                        to_fill[ind : ind + n] = False
                y[~to_fill] = np.nan
            temp[:, i] = y
    return temp

def interpolate_data(h5file,
    kernel ="linear",
    window_size = 0,
    p_bound=0.001,
    save_as_csv=True,):

    df = pd.read_hdf(h5file)
    data = df.copy()
    outdataname = h5file.replace('.h5', '_interpolated.h5')
    xy = data.values
    xy_filled = columnwise_interp(xy, kernel, window_size)
    filled = ~np.isnan(xy_filled)
    data[filled] = xy_filled
    if os.path.isfile(outdataname):
        os.remove(outdataname)
    data.to_hdf(outdataname, "df_interpolated", format="table", mode="w")
    print(f'The h5 file is saved at: {outdataname}')
    if save_as_csv:
        print("Saving interpolated csv poses!")
        data.to_csv(outdataname.split(".h5")[0] + ".csv")
    return data


def remove_low_likelihood_points(df_src, df_tgt, threshold):
    df_src = df_src.copy()
    df_tgt = df_tgt.copy()
    mask = df_src.xs('likelihood', level = 'coords', axis = 1, drop_level=False ) < threshold
    temp = df_src.xs('likelihood', level = 'coords', axis = 1, drop_level=False )[~mask]
    idx = pd.IndexSlice
    df_src.loc[:,idx[:,:,:,'likelihood']] = temp
    bpts = df_src.columns.get_level_values('bodyparts').unique()
    for bpt in bpts:
        df_bpt = df_src.loc[:, idx[:,:,bpt]]
        mask_nan = df_bpt.isnull().any(axis = 1)
        df_tgt.loc[mask_nan,idx[:,:,bpt]] = np.nan
    return df_tgt
