from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import numpy as np

quat = [0, 0, 0, 1]
position = [0, 0, 0]

def plot_quaterinon(ax, quat, position, length = 1):
    rotmatrix = R.from_quat(quat).as_matrix()
    endpoint = rotmatrix @ np.eye(3)
    ax.quiver(position[0], position[1], position[2], endpoint[0, 0], endpoint[1,0], endpoint[2, 0], color = 'red', label = 'x', length = length)
    ax.quiver(position[0], position[1], position[2], endpoint[0, 1], endpoint[1,1], endpoint[2, 1], color = 'green', label = 'y', length = length)
    ax.quiver(position[0], position[1], position[2],endpoint[0, 2], endpoint[1,2], endpoint[2, 2], color = 'blue', label = 'z', length = length)
    return ax

if __name__ == '__main__':
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    quat1 = [0, 0, 0, 1]
    position1 = [0, 0, 0]


    quat2 = [0.574579226927182,
0.068339885140753,
0.799908239099365,
0.159170289638686,
]
    position2 = [1, 1, 1]

    ax = plot_quaterinon(ax, quat1, position1, length = 1)
    ax = plot_quaterinon(ax, quat2, position2, length = 1)
    plt.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-3,3)
    plt.show()

