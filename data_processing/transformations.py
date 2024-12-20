import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from itertools import combinations
import os
import pickle

def quat_to_rotvect(traj):
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    traj_new = np.zeros((len(traj), 6))
    traj_new[:, :3] = traj[:, :3]
    traj_new[:, 3:] = R.from_quat(traj[:, 3:]).as_rotvec()
    return traj_new

def rotvect_to_quat(traj):
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    traj_new = np.zeros((len(traj), 7))
    traj_new[:, :3] = traj[:, :3]
    traj_new[:, 3:] = R.from_rotvec(traj[:, 3:]).as_quat()
    return traj_new

def quat_conjugate(quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]
    return [-qx, -qy, -qz, qw]

def get_quat_matrix(quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]
    result = np.zeros((4, 4))
    result[0,:] = [qw, -qz, qy, qx]
    result[1, :] = [qz, qw, -qx, qy]
    result[2, :] = [-qy, qx, qw, qz]
    result[3, :] = [-qx, -qy, -qz, qw]
    return result

def get_rot_quat_matrix(r):
    rotmatrix = r.as_matrix()
    quat = r.as_quat()
    result = np.zeros((7, 7))
    result[:3, :3] = rotmatrix
    result[3:, 3:] = get_quat_matrix(quat)
    return result

def get_HT_for_grouped_object(grouped_obj, HT_objs_in_ndi_demo):
    obj1 = grouped_obj.split('-')[0]
    obj2 = grouped_obj.split('-')[1]
    pos1 = HT_objs_in_ndi_demo[obj1][:-1, 3]
    pos2 = HT_objs_in_ndi_demo[obj2][:-1, 3]
    axis_obj = pairwise_constrained_axis3d(pos1, pos2, up_axis=0)
    rot_matrix = axis3d_to_rotmatrix(axis_obj)
    HT = homogeneous_transform(rot_matrix, pos1)
    return HT

def pairwise_constrained_axis3d(pos1, pos2, up_axis=0):
    '''
    Generates local object's 3D axis given pos1 and pos2 with x-axis parallel to the line between pos1 and pos2
    and z-axis perpendicular such that its unit vector's up-axis value is maximized.

    Parameters:
    ----------
    pos1: numpy.array
        A 3D numpy array representing position of object 1.
    pos2: numpy.array
        A 3D numpy array representing position of object 2.
    up_axis: int
        dimension of the z-axis' unit vector that should be maximized

    Returns:
    -------
        Three 3D unit vectors representing the direction xyz-axis are pointing.
    '''
    vec = pos2 - pos1
    x_axis_norm = vec/np.linalg.norm(vec)
    up = np.zeros(3)
    up[up_axis] = 1
    x_comp_of_up = np.dot(up, x_axis_norm)*x_axis_norm

    z_axis_norm = (up-x_comp_of_up)/np.linalg.norm(up-x_comp_of_up)
    y_axis = np.cross(z_axis_norm, x_axis_norm)
    y_axis_norm = y_axis/np.linalg.norm(y_axis)
    return (x_axis_norm, y_axis_norm, z_axis_norm)

def axis3d_to_rotmatrix(axis3d):
    '''
    Convert the object xyz-axis vectors into rotation matrix

    Parameters:
    ----------
    axis3d: numpy.array
        A set of 3D numpy array representing the direction xyz-axis are pointing.
    Returns:
    -------
        A 3 by 3 rotation matrix.
    '''
    rot = np.eye(3)
    for i, axis_i in enumerate(axis3d):
        rot[:, i] = axis_i
    return rot

def axis3d_to_quat(axis3d):
    '''
    Convert the object xyz-axis vectors to a rotation using quaternion and return the transformation
    representing the rotation to standard unit vector (Extremely rare fail case happens when axis object
    and world axis are exactly the same or opposite)

    Parameters:
    ----------
    axis3d: numpy.array
        A set of 3D numpy array representing the directions xyz-axis are pointing.
    Returns:
    -------
        A scipy.spatial.transform.Rotation object that rotates the xyz-axis vectors into standard unit vector.
    '''
    # start by matching x-axis:
    x_axis = axis3d[0]/np.linalg.norm(axis3d[0])
    x_world = np.array([1,0,0])
    rot_axis1 = np.cross(x_axis, x_world)
    n1 = rot_axis1/np.linalg.norm(rot_axis1)
    theta1 = np.arccos(np.dot(x_axis, x_world))
    q1 = R.from_quat((n1*np.sin(theta1/2)).tolist() + [np.cos(theta1/2)])

    # followed by y-axis
    y_axis = q1.apply(axis3d[1])/np.linalg.norm(axis3d[1])
    y_world = np.array([0,1,0])
    rot_axis2 = np.cross(y_axis, y_world)
    n2 = rot_axis2/np.linalg.norm(rot_axis2)
    theta2 = np.arccos(np.dot(y_axis, y_world))
    q2 = R.from_quat((n2*np.sin(theta2/2)).tolist() + [np.cos(theta2/2)])
    # combine into single rotation quaternion
    q3 = q2*q1
    return q3


def homogeneous_position(vect):
    '''
    This function will make sure the coordinates in the right shape(4 by N) that could be multiplied by a
    homogeneous transformation(4 by 4).

    Parameters
    ----------
    vect: list or np.array
        A N by 3 array. x, y, z coordinates of the N points

    Return
    ------
    A 4 by N array.
    '''
    temp = np.array(vect)
    if temp.ndim == 1:
        ones = np.ones(1)
        return np.r_[temp, ones].reshape(-1, 1)
    elif temp.ndim == 2:
        num_rows, num_cols = temp.shape
        if num_cols != 3:
            raise Exception(f"vect is not N by 3, it is {num_rows}x{num_cols}")

        ones = np.ones(num_rows).reshape(-1, 1)
        return np.c_[temp, ones].T

def lintrans(a, H):
    '''
    This function applies the linear transformation expressed with the homogeneous transformation H.

    Parameters:
    -----------
    a: np.array
        The trajectory data with shape (N by D) where N is the number of datapoints and D is the dimension.
    H: np.array
        A 4 by 4 homogeneous transformation matrix.

    Returns:
    --------
    a_transformed: np.array
        A N by D array that contains the transformed trajectory
    '''
    D = a.shape[1]
    if D == 3: # position only
        a_homo = homogeneous_position(a)
        a_transformed = (H @ a_homo).T[:,:3]
    elif D == 4: # orientation only
        r_quat = R.from_matrix(H[:3, :3]).as_quat()
        quat_matrix = get_quat_matrix(r_quat)
        ori = (quat_matrix @ a.T).T
        ori_new = R.from_matrix(R.from_quat(ori).as_matrix()).as_quat()
        a_transformed = ori_new
    elif D == 7: # position + orientation(quaternion)
        a_homo = homogeneous_position(a[:,:3])
        pos = (H @ a_homo).T[:,:3]
        r = R.from_matrix(H[:3, :3])
        quat_matrix = get_quat_matrix(r.as_quat())
        ori = (quat_matrix @ a[:, 3:].T).T
        try:
            ori_new = R.from_matrix(R.from_quat(ori).as_matrix()).as_quat()
        except ValueError:
            ori_new = ori
        a_transformed = np.concatenate((pos, ori_new), axis = 1)
    return a_transformed

def lintrans_cov(sigmas, H):
    '''
    This function applies the linear transformation on a covirance matrix expressed with the homogeneous transformation H.

    Parameters:
    -----------
    sigmas: np.array
        The covariance matrices with shape (N by D by D) where N is the number of datapoints and D is the dimension.
    H: np.array
        A 4 by 4 homogeneous transformation matrix.

    Returns:
    --------
    a_transformed: np.array
        A N by D array that contains the transformed trajectory
    '''
    rotmatrix = H[:3, :3]
    D = sigmas.shape[1]
    if D == 3: # position only
        sigmas_transformed = [rotmatrix @ cov @ rotmatrix.T for cov in sigmas]
    elif D == 4: # orientation only
        r_quat = R.from_matrix(rotmatrix).as_quat()
        r_quat_inv = quat_conjugate(r_quat)
        quat_matrix = get_quat_matrix(r_quat)
        quat_matrix_inv = get_quat_matrix(r_quat_inv)
        sigmas_transformed = [quat_matrix @ cov @ quat_matrix_inv for cov in sigmas]
        # sigmas_transformed = [-cov for cov in sigmas_transformed]
    elif D == 7: # position + orientation(quaternion)
        r = R.from_matrix(rotmatrix)
        r_inv = r.inv()
        rot_quat_matrix = get_rot_quat_matrix(r)
        rot_quat_matrix_inv = get_rot_quat_matrix(r_inv)
        sigmas_transformed = [rot_quat_matrix @ cov @ rot_quat_matrix_inv for cov in sigmas]
    return sigmas_transformed

def rigid_transform_3D(A, B):
    '''
    This function finds the rotation and translation that moves A to B if different points are in the same reference frame.
    or
    This function finds the rotation and translation that expresses A in B if the same points are in the different reference frames.

    Parameters
    ----------
    A: array
        N * 3 array, where N is the number of datapoints
    B: array
        N * 3 array, where N is the number of datapoints

    Returns
    -------
    R: array
        3 * 3 array, the rotation matrix to transform A to match with B
    t: array
        3 array, the translation that transforms A to match with B

    '''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_cols != 3:
        raise Exception(f"matrix A is not Nx3, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_cols != 3:
        raise Exception(f"matrix B is not Nx3, it is {num_rows}x{num_cols}")
    A = A.T
    B = B.T
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)

    # sanity check
    # if np.linalg.matrix_rank(H) < 3:
    #     print("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))
       # raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

#     special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t.flatten()

def homogeneous_transform(rotmat,vect):

    '''
    :param rotmat: 3x3 matrix
    :param vect: list x,y,z
    :return:homogeneous transformation 4x4 matrix using R and vect
    '''
    if not isinstance(vect, list):
        vect = list(vect)
    H = np.zeros((4,4))
    H[0:3,0:3] = rotmat
    frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1,4)
    H[:,3] = D
    return H

def inverse_homogeneous_transform(H):

    '''
    :param H: homogeneous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    '''


    rotmat = H[0:3,0:3]
    origin = H[:-1,3]
    origin.shape = (3,1)

    rotmat = rotmat.T
    origin = -rotmat.dot(origin)
    return homogeneous_transform(rotmat,list(origin.flatten()))

def vec2homo(vec):
    rotmat = R.from_rotvec(vec[3:]).as_matrix()
    t = vec[:3]
    return homogeneous_transform(rotmat, t)

def homo2vec(H):
    rotmat = H[:3, :3]
    rotvec = R.from_matrix(rotmat).as_rotvec()
    t = H[:3, -1]
    return np.concatenate([t, rotvec])



def get_HT_template_in_obj(template, markers_coords_in_camera):
    '''
    This function will align the object with template and output the homogeneous transformation
    and the markers' average distance

    Parameters
    ----------
    template: dict
        The obj template that has the markers position in robot base reference frame
    markers_coords_in_camera: dict or Dataframe
        The dict or Dataframe that contains the marker's coordinates in robot bse reference frame for a frame of the video

    Returns
    -------
    H: np.array
        A 4 by 4 homogeneous transformation that moves templte to object in the global reference frame
    dist: float
        The lowest average markers' distance after aligning object with the template

    '''
    points_template = []
    points_obj = []
    if not isinstance(markers_coords_in_camera, dict):
        bps = markers_coords_in_camera.index.get_level_values('bodyparts').unique()
        for bodypart in bps:
            df_bodypart = markers_coords_in_camera.loc[bodypart]
            if not df_bodypart.isnull().values.any():
                points_template.append(template[bodypart])
                points_obj.append(df_bodypart.to_numpy())
    else:
        keys = markers_coords_in_camera.keys()
        for bodypart in keys:
            df_bodypart = markers_coords_in_camera[bodypart]
            if not df_bodypart.isnull().values.any():
                points_template.append(template[bodypart])
                points_obj.append(df_bodypart.to_numpy())


    points_template_in_base = np.array(points_template)
    points_obj_in_base = np.array(points_obj)

    rotmatrix, translation = rigid_transform_3D(points_template_in_base, points_obj_in_base)
    H = homogeneous_transform(rotmatrix, translation)

    points_template_transformed = lintrans(points_template_in_base, H)
    dists = []
    for i, point in enumerate(points_template_transformed):
        point = point
        dist = np.linalg.norm(point- points_obj_in_base[i])
        dists.append(dist)

    dist_average = np.mean(np.array(dists), axis=0)

    return H, dist_average

def get_HT_obj_in_template(template, markers_coords_in_camera):
    '''
    This function will align the object with template and output the homogeneous transformation
    and the markers' average distance

    Parameters
    ----------
    template: dict
        The obj template that has the markers position in camera reference frame
    markers_coords_in_camera: dict or Dataframe
        The dict or Dataframe that contains the marker's coordinates in camera reference frame for a frame of the video

    Returns
    -------
    template_in_obj: np.array
        A 4 by 4 homogeneous transformation that represents template in the object's reference frame
    dist: float
        The lowest average markers' distance after aligning object with the template

    '''
    points_template = []
    points_obj = []
    if not isinstance(markers_coords_in_camera, dict):
        bps = markers_coords_in_camera.index.get_level_values('bodyparts').unique()
        for bodypart in bps:
            df_bodypart = markers_coords_in_camera.loc[bodypart]
            if not df_bodypart.isnull().values.any():
                points_template.append(template[bodypart])
                points_obj.append(df_bodypart.to_numpy())
    else:
        keys = markers_coords_in_camera.keys()
        for bodypart in keys:
            df_bodypart = markers_coords_in_camera[bodypart]
            if not df_bodypart.isnull().values.any():
                points_template.append(template[bodypart])
                points_obj.append(df_bodypart.to_numpy())


    points_template = np.array(points_template)
    points_obj = np.array(points_obj)

    rotmatrix, translation = rigid_transform_3D(points_template, points_obj)
    H = homogeneous_transform(rotmatrix, translation)
    points_template_transformed = lintrans(points_template, H)
    dists = []
    for i, point in enumerate(points_template_transformed):
        point = point
        dist = np.linalg.norm(point- points_obj[i])
        dists.append(dist)
    dist_average = np.mean(np.array(dists), axis=0)
    return H, dist_average

def if_visible(marker_coords, combs):
    for comb in combs:
        partial_marker_coords = marker_coords.loc[list(comb)]
        if not partial_marker_coords.isnull().any():
            return True
    return False

def match_template_to_obj(obj_template, df_individual, window_size, n_markers=3): ###v3
    '''
    Search backwards the first window_size rows of df_individual where at least n_markers are visible, and output the ho`mogeneous transformation.

    Parameters
    ----------
    obj_template: dict
        The obj template that has the markers position in camera reference frame
    df_individual: Dataframe
        The Dataframe that contains the DLC + LEAstereo output for an individual
    camera_in_template: np.array
        A 4 by 4 array that represents camera in the template's reference frame
    window_size: int
        The first window_size rows that at least n_markers are visible.
    n_markers: int
        The number of markers that will be used to align object with template.

    Returns
    -------
    template_in_obj: np.array
        A 4 by 4 homogeneous transformation that represents template in the object's reference frame
    dist: float
        The lowest average markers' distance after aligning object with the template

    '''
    dist = np.inf
    n_consecutive_frames = 0
    bps = list(obj_template.keys())
    if len(df_individual) == 0:
        template_to_obj = None
        dist = np.inf
    for i in range(len(df_individual)):
        if n_consecutive_frames < window_size:
            marker_coords = df_individual.iloc[i]
            bps_valid = [bp for bp in bps if not marker_coords[bp].isnull().any()]
            if len(bps_valid) >= n_markers: ### There are enough markers vidible to match object to template
                n_consecutive_frames +=1
                combs = combinations(bps_valid, n_markers)
                for comb in combs:
                    partial_marker_coords = marker_coords.loc[list(comb)]
                    H, dist_average = get_HT_template_in_obj(obj_template, partial_marker_coords)
                    # print(list(comb), dist_average, 'aaaaaaaaa')
                    if dist_average < dist:
                        dist = dist_average
                        template_to_obj = H
            else:
                n_consecutive_frames = 0
                template_to_obj = None
                dist = np.inf
                continue
        else:
            return template_to_obj, dist
    return template_to_obj, dist

def get_HT_template_in_camera(obj_template, base_in_camera):
    '''
    This function will assign a frames to the template where the x,y,z axes are aligned with camera axes, and the origin will
    be at the mean of the markers of the template.

    Parameters:
    -----------
    obj_template: dict
        A dictionary that contains the makers' positions in camera's reference frame.
    Returns:
    --------
    template_in_camera: np.array
        A 4 by 4 array that contains the homogeneous transformation that expresses template in camera reference frame.
    '''
    rot_matrix = base_in_camera[:3, :3]
    # rot_matrix = np.eye(3)
    markers = []
    for bp in obj_template:
        markers.append(obj_template[bp])
    markers = np.array(markers)
    center = np.mean(markers, axis=0)
    template_in_camera = homogeneous_transform(rot_matrix, list(center))
    return template_in_camera




def get_HT_objs_in_base(obj_traj_in_base, templates_in_base, HT_templates_in_base, window_size = 5, markers_average = False, thresh = np.inf):
    '''
    This function will get homogeneous transformation for each object in each demonstration.

    Parameters:
    -----------
    obj_traj_in_base: Dataframe
        A dataframe that contains the object's trajectories for 1 demonstration in the robot base reference frame.
    obj_templates: dict
        The dictionary that contains the objects' templates in the robot base reference frame.
    window_size: int
        Number of rows that the object needs to be visible so that the HT computed will be seen as valid.
    markers_average: bool
        If true, the object position will be computed as the mean of all visible markers. Else, it will be using the objects' templates.
    Returns
    -------
    HTs: dict
        A dictionary that contains the homogeneous transformation for object in each demonstration
    dists: list
        The list distances when matching object with template for each object.
    '''

    template_objs = templates_in_base.keys()
    individuals = obj_traj_in_base.columns.get_level_values(level='individuals').unique()

    HTs = {}
    dists = {}
    for individual in individuals:
        if not markers_average:
            obj = [obj for obj in template_objs if obj in individual][0]
            obj_template_in_base = templates_in_base[obj]
            df_individual = obj_traj_in_base.loc[:,individual]
            df_individual = df_individual.dropna(axis = 0, how = 'all') ### remove a row of nans
            bps = df_individual.columns.get_level_values('bodyparts').unique()
            bps_valid = []
            for bp in bps:
                if (not np.isnan(df_individual[bp].to_numpy()).any()) and (not df_individual[bp].empty):
                    bps_valid.append(bp)
            n_markers = np.max([3, len(bps_valid)])
            HT_template_in_base = HT_templates_in_base[obj]
            dist = np.inf

            while n_markers >= 3: ### If the dist is greater than thresh, remove one marker and match again
                if dist < thresh:
                    break
                else:
                    template_to_obj_in_base, dist = match_template_to_obj(obj_template_in_base, df_individual, window_size = window_size, n_markers = n_markers)
                    # print(individual, n_markers, dist)
                    n_markers = n_markers - 1
            if not dist == np.inf:
                dists[individual] = dist
                obj_in_base =  template_to_obj_in_base @ HT_template_in_base
                HTs[individual] = obj_in_base
            else:
                dists[individual] = dist
                # print(f'Could not find a match for object {individual}')

        else:
            A = np.eye(3)
            df_individual = obj_traj_in_base[individual]
            bps = df_individual.columns.get_level_values('bodyparts').unique()
            coords = df_individual.columns.get_level_values('coords').unique()
            coords_average = np.zeros(coords.shape)
            for i in range(len(coords)):
                coord_sum = 0
                j = 0
                for bp in bps:
                    df_bp = df_individual[bp].iloc[:window_size]
                    if not df_bp.isnull().values.any(): #body parts don't contain nans
                        coord_sum += np.mean(df_bp[coords[i]])
                        j += 1
                coord_average = coord_sum / j
                coords_average[i] = coord_average
            H_obj_in_base = homogeneous_transform(A, coords_average)
            HTs[individual] = H_obj_in_base
            dists[individual] = 0
    return HTs, dists


