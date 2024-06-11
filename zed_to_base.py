import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from transformations.transformations import homogeneous_transform, lintrans, inverse_homogeneous_transform, rigid_transform_3D
import pickle
from matplotlib import pyplot as plt

# from analyze_videos import analyze_video
from create_dataset import create_dataset

def get_translation(ndi_points, vid_points, rot_matrix):
    translations = []
    for ndi, vid in zip(np.array(ndi_points).reshape(-1, 3), np.array(vid_points).reshape(-1,3)):
        diff = ndi - rot_matrix @ vid
        translations.append(diff)
    translation_mean = np.array(translations).mean(axis=0)
    return translation_mean

def get_rotation(ndi_pairs, vid_pairs):
    ndi_vecs = np.array(ndi_pairs)[:,1,:] - np.array(ndi_pairs)[:,0,:]
    vid_vecs = np.array(vid_pairs)[:,1,:] - np.array(vid_pairs)[:,0,:]
    r, e = R.align_vectors(ndi_vecs, vid_vecs)
    return r, e


def solve_A(x, b):

    A = b @ x.T @ np.linalg.inv(x @ x.T )

    return A


id = '1702495370'
# analyze_video(id)
create_dataset(id)

data_file = os.path.join(id, 'data.pickle')
with open(data_file, 'rb') as f:
    data = pickle.load(f)
data_train = data['train']
data_val = data['valid']
data_test = data['test']

src_points = np.concatenate([data_train[:,:3], data_val[:, :3]])
dst_points = np.concatenate([data_train[:, 3:], data_val[:, 3:]])


# src_points = data_train[:,:3]
# dst_points = data_train[:, 3:]
# np.set_printoptions(threshold=sys.maxsize)

inds = np.arange(dst_points.shape[0])
# inds = random.choices(inds, k = 10000)
# combs = combinations(inds, 2)
#
# dst_point_pairs = []
# src_point_pairs = []
# for comb_ind in tqdm(list(combs)):
#     dst_point_pairs.append(([dst_points[comb_ind[0]], dst_points[comb_ind[1]]]))
#     src_point_pairs.append(([src_points[comb_ind[0]], src_points[comb_ind[1]]]))
# r, e = get_rotation(dst_point_pairs, src_point_pairs)
# rot_matrix = r.as_matrix()
#
# t = get_translation(dst_points, src_points, rot_matrix)
rot_matrix, t = rigid_transform_3D(src_points, dst_points)

print(f'Translation is :{t}')
print(f'Rotation is: {rot_matrix}')

H = homogeneous_transform(rot_matrix, t.flatten())
H_inverse = inverse_homogeneous_transform(H)

src_points_test = data_test[:,:3]
dst_points_test = data_test[:,3:]
src_points_test_transformed = lintrans(src_points_test, H)
dists = np.linalg.norm(dst_points_test - src_points_test_transformed, axis = 1)

print(f'The average distance is {np.array(dists).mean()} mm ')
fig = plt.figure(figsize=(18, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(dst_points_test[:,0], dst_points_test[:,1], dst_points_test[:,2], color = 'blue')
ax.scatter(src_points_test_transformed[:, 0], src_points_test_transformed[:, 1], src_points_test_transformed[:, 2], color = 'red')
plt.show()

with open(os.path.join('../transformations', 'zed_in_base.pickle'), 'wb') as f1:
    pickle.dump(H, f1)
with open(os.path.join('../transformations', 'base_in_zed.pickle'), 'wb') as f2:
    pickle.dump(H_inverse, f2)