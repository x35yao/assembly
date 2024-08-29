from scipy.spatial.transform import Rotation as R
import numpy as np
from glob import glob
import os
import pickle
from transformations.transformations import homogeneous_transform, lintrans
import pandas as pd
import yaml

# def get_templates_in_base(templates, camera_in_base):
#     templates_in_base = templates.copy()
#     objs = templates.keys()
#     for obj in objs:
#         bps = templates[obj].keys()
#         for bp in bps:
#             tmp = lintrans(templates[obj][bp].reshape(-1, 3), camera_in_base)
#             templates_in_base[obj][bp] = tmp.flatten()
#     return templates_in_base

def get_template_in_base(template, camera_in_base):
    template_in_base = template.copy()
    bps = template.keys()
    for bp in bps:
        tmp = lintrans(template[bp].reshape(-1, 3), camera_in_base)
        template_in_base[bp] = tmp.flatten()
    return template_in_base

def get_HT_templates_in_base(templates_in_base):
    objs = templates_in_base.keys()
    HT_templates_in_base = {}
    rot_matrix = np.eye(3)
    for obj in objs:
        template = templates_in_base[obj]
        markers = []
        for bp in template:
            markers.append(template[bp])
        markers = np.array(markers)
        center = np.mean(markers, axis=0)
        HT_templates_in_base[obj] = homogeneous_transform(rot_matrix, list(center))
    return HT_templates_in_base

if __name__ == '__main__':
    date = '2024-08-28'
    # read config file
    with open('./data/task_config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    scale = 1000
    vid_id = config['template_video_id']
    objs = config['objects']

    obj_templates = {}
    obj_templates_wrist = {}
    project_dir = config['project_path']
    template_id = str(config['template_video_id'])

    template_h5_path = os.path.join(project_dir, 'data', 'preprocessed', template_id, 'action_0', '3d_combined','markers_trajectory_3d.h5')
    template_wrist_h5_path = template_h5_path.replace('markers_trajectory_3d.h5', 'markers_trajectory_wrist_3d.h5')
    traj_path = os.path.join(project_dir, 'data', 'preprocessed', template_id, 'action_0', 'action_0.csv')

    df = pd.read_hdf(template_h5_path).droplevel(0, axis = 1)
    df_wrist = pd.read_hdf(template_wrist_h5_path).sort_index() ### sort the index to make sure the rows are corresponding to images taken
    df_traj = pd.read_csv(traj_path)
    wrist_obj_inds = {}
    for obj in objs:
        individual = f'{obj}1'
        idx = pd.IndexSlice
        bps = df.loc[:, individual].columns.get_level_values('bodyparts').unique()
        bp_3d = {}
        bp_3d_wrist = {}
        df_wrist_individual = df_wrist.loc[:, individual]
        ind = np.where(~df_wrist_individual.isnull().any(axis=1))[0][0] # Index of the picture where this object is detected
        wrist_obj_inds[obj] = ind
        df_wrist_obj = df_wrist_individual.iloc[ind] ### This picks up the picture for this certain object
        for bp in bps:
            bp_3d[bp] = df.iloc[0][individual, bp].values
            bp_3d_wrist[bp] = df_wrist_obj[bp].values
        obj_templates[obj] = bp_3d
        obj_templates_wrist[obj] = bp_3d_wrist

    with open(os.path.join(project_dir, 'transformations', 'zed_in_base.pickle'), 'rb') as f:
        zed_in_base = pickle.load(f)
    with open(os.path.join(project_dir, 'transformations', 'wrist_cam_in_tcp.pickle'), 'rb') as f:
        wrist_in_tcp = pickle.load(f)

    ### get base template for the zed camera
    # template_in_base_for_zed = get_templates_in_base(obj_templates, zed_in_base)
    template_in_base_for_zed = {}
    for obj in obj_templates.keys():
        template = obj_templates[obj]
        template_in_base_for_zed[obj] = get_template_in_base(template, zed_in_base)
    HT_template_in_base_for_zed = get_HT_templates_in_base(template_in_base_for_zed)
    transfomration_folder = os.path.join('./transformations', date)
    os.makedirs(transfomration_folder, exist_ok=True)
    fname_HT_zed = os.path.join(transfomration_folder, 'HT_template_in_base_for_zed.pickle')
    with open(fname_HT_zed, 'wb') as f:
        pickle.dump(HT_template_in_base_for_zed,f)

    fname_template_zed = fname_HT_zed.replace('HT_template_in_base_for_zed.pickle', 'template_in_base_for_zed.pickle')
    with open(fname_template_zed, 'wb') as f:
        pickle.dump(template_in_base_for_zed,f)
    df_template_in_base_for_wrist = pd.DataFrame.from_dict(template_in_base_for_zed)
    df_template_in_base_for_wrist.to_csv(fname_template_zed.replace('.pickle', '.csv'))

    ### get base template for the wrist camera
    wrist_inds = np.where(df_traj['wrist_camera_time']==1)[0]
    template_in_base_for_wrist = {}
    for obj in obj_templates_wrist.keys():
        template_wrist = obj_templates_wrist[obj]
        wrist_ind = wrist_inds[wrist_obj_inds[obj]] ### Find the robot position when taking a picture for this obj
        pos = df_traj.iloc[wrist_ind][['x', 'y', 'z']].to_numpy() * scale
        rotmat = R.from_rotvec(df_traj.iloc[wrist_ind][['rx', 'ry', 'rz']].to_numpy()).as_matrix()
        tcp_in_base = homogeneous_transform(rotmat, pos)
        wrist_in_base = tcp_in_base @ wrist_in_tcp
        template_in_base_for_wrist[obj] = get_template_in_base(template_wrist, wrist_in_base)
    HT_template_in_base_for_wrist = get_HT_templates_in_base(template_in_base_for_wrist)

    fname_HT_wrist = os.path.join(transfomration_folder, 'HT_template_in_base_for_wrist.pickle')
    with open(fname_HT_wrist, 'wb') as f:
        pickle.dump(HT_template_in_base_for_wrist,f)
    fname_template_wrist= fname_HT_wrist.replace('HT_template_in_base_for_wrist.pickle', 'template_in_base_for_wrist.pickle')
    with open(fname_template_wrist, 'wb') as f:
        pickle.dump(template_in_base_for_wrist,f)
    df_template_in_base_for_wrist = pd.DataFrame.from_dict(template_in_base_for_wrist)
    df_template_in_base_for_wrist.to_csv(fname_template_wrist.replace('.pickle', '.csv'))
