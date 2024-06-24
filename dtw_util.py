import numpy as np
from dtw import dtw


# basic implementation
def dynamic_time_warp(data1, data2):
    """
    Takes two sequences and returns the list of matching indices 
    representing the path and its minimum cost

    Parameters:
    -----------
    data1: dict
        First trajectory
    data2: dict 
        second trajectory
    Returns:
    --------
        an array of realigned indice and the min cost value of the path.
    """
    len1, len2 = len(data1), len(data2)
    # Initialize grid
    dtw = np.full((len1, len2), fill_value=float('inf'))
    dtw[0, 0] = 0
    for i in range(1, len1):
        for j in range(1, len2):
            cost = np.linalg.norm(data1[i] - data2[j])
            dtw[i,j] = cost + min([dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1]])
    index_match = []

    # get optimal matching moving backwards.
    cur_pair = [len1 - 1, len2 - 1]
    while cur_pair != [0, 0]:
        index_match.append(cur_pair.copy())
        min_pair_val = min([dtw[cur_pair[0] - 1, cur_pair[1]], dtw[cur_pair[0], cur_pair[1] - 1],
                            dtw[cur_pair[0] - 1, cur_pair[1] - 1]])
        if dtw[cur_pair[0] - 1, cur_pair[1] - 1] <= min_pair_val:
            cur_pair[0] -= 1
            cur_pair[1] -= 1
        elif dtw[cur_pair[0], cur_pair[1] - 1] <= min_pair_val:
            cur_pair[1] -= 1
        elif dtw[cur_pair[0] - 1, cur_pair[1]] <= min_pair_val:
            cur_pair[0] -= 1
    index_match.reverse()
    return np.array(index_match) - 1, dtw[len1 - 1, len2 - 1]


def get_slice_w_padding(i, seq, window):
    """
        Get the local window of sequence at position i.
        Parameters:
        -----------
        i: int
            position of the center of the window
        seq: list
            a list of values
        window: int
            size of the window in either direction
        Returns:
        --------
            an array of elements of the window at i with padded missing values.
    """
    result = np.full(1 + window * 2, -1)
    for k, i in enumerate(range(i - window, i + window + 1)):
        if i < 0:
            result[k] = seq[0]
        elif i >= len(seq):
            result[k] = seq[-1]
    return result


def shapedDTW(data1, data2, shape_descriptor, window=5):
    """
    matched trajectory using the shapedDTW that takes into consideration the local structure when matching
    Parameters:
    -----------
    data1: dict
        First trajectory
    data2: dict
        second trajectory
    Returns:
    --------
        an array of realigned indice and the min cost value of the path.
    """
    len1, len2 = len(data1), len(data2)
    # Initialize grid
    dtw = np.full((len1, len2), fill_value=float('inf'))
    dtw[0, 0] = 0
    for i in range(1, len1):
        local_i = get_slice_w_padding(i, data1, window)
        descrip_i = shape_descriptor(local_i)
        for j in range(1, len2):
            local_j = get_slice_w_padding(j, data2, window)
            descrip_j = shape_descriptor(local_j)
            cost = np.linalg.norm(descrip_i - descrip_j)
            dtw[i, j] = cost + min([dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]])
    index_match = []

    # get optimal matching moving backwards.
    cur_pair = [len1 - 1, len2 - 1]
    while cur_pair != [0, 0]:
        index_match.append(cur_pair.copy())
        min_pair_val = min([dtw[cur_pair[0] - 1, cur_pair[1]], dtw[cur_pair[0], cur_pair[1] - 1],
                            dtw[cur_pair[0] - 1, cur_pair[1] - 1]])
        if dtw[cur_pair[0] - 1, cur_pair[1] - 1] <= min_pair_val:
            cur_pair[0] -= 1
            cur_pair[1] -= 1
        elif dtw[cur_pair[0], cur_pair[1] - 1] <= min_pair_val:
            cur_pair[1] -= 1
        elif dtw[cur_pair[0] - 1, cur_pair[1]] <= min_pair_val:
            cur_pair[0] -= 1
    index_match.reverse()
    return np.array(index_match) - 1, dtw[len1 - 1, len2 - 1]

def get_slice_w_padding(i, seq, window):
    """
        Get the local window of sequence at position i.
        Parameters:
        -----------
        i: int
            position of the center of the window
        seq: list
            a list of values
        window: int
            size of the window in either direction
        Returns:
        --------
            an array of elements of the window at i with padded missing values.
    """
    result = np.full(1+window*2, -1)
    for k, i in enumerate(range(i-window, i+window+1)):
        if i < 0: result[k] = seq[0]
        elif i >= len(seq): result[k] = seq[-1]
    return result

def shapedDTW(data1, data2, shape_descriptor, window=5):
    """
    matched trajectory using the shapedDTW that takes into consideration the local structure when matching
    Parameters:
    -----------
    data1: dict
        First trajectory
    data2: dict
        second trajectory
    Returns:
    --------
        an array of realigned indice and the min cost value of the path.
    """
    len1, len2 = len(data1), len(data2)
    # Initialize grid
    dtw = np.full((len1, len2), fill_value=float('inf'))
    dtw[0, 0] = 0
    for i in range(1, len1):
        local_i = get_slice_w_padding(i, data1, window)
        descrip_i = shape_descriptor(local_i)
        for j in range(1, len2):
            local_j = get_slice_w_padding(j, data2, window)
            descrip_j = shape_descriptor(local_j)
            cost = np.linalg.norm(descrip_i - descrip_j)
            dtw[i, j] = cost + min([dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]])
    index_match = []

    # get optimal matching moving backwards.
    cur_pair = [len1 - 1, len2 - 1]
    while cur_pair != [0, 0]:
        index_match.append(cur_pair.copy())
        min_pair_val = min([dtw[cur_pair[0] - 1, cur_pair[1]], dtw[cur_pair[0], cur_pair[1] - 1],
                            dtw[cur_pair[0] - 1, cur_pair[1] - 1]])
        if dtw[cur_pair[0] - 1, cur_pair[1] - 1] <= min_pair_val:
            cur_pair[0] -= 1
            cur_pair[1] -= 1
        elif dtw[cur_pair[0], cur_pair[1] - 1] <= min_pair_val:
            cur_pair[1] -= 1
        elif dtw[cur_pair[0] - 1, cur_pair[1]] <= min_pair_val:
            cur_pair[0] -= 1
    index_match.reverse()
    return np.array(index_match) - 1, dtw[len1 - 1, len2 - 1]


# Implementation by another library with constrains.
def dtw_typeId(traj1, traj2):
    traj_align = dtw(traj1, traj2, step_pattern="typeId", keep_internals=True)
    # traj_align.plot('threeway')
    return np.array([traj_align.index1, traj_align.index2]).transpose(), traj_align.normalizedDistance

# Implementation by another library with constrains.
def dtw_symmetric1(traj1, traj2):
    traj_align = dtw(traj1, traj2, step_pattern="symmetric1", keep_internals=True)
    # traj_align.plot('threeway')
    return np.array([traj_align.index1, traj_align.index2]).transpose(), traj_align.normalizedDistance

def dtw_shaped(traj1, traj2):
    traj_align = shapedDTW(traj1, traj2, shape_descriptor=ident_shape_descriptor, window=5)
    return traj_align

def ident_shape_descriptor(seq):
    return seq

dtw_funcs = {"typeId": dtw_typeId,
             "symmetric1": dtw_symmetric1,
             "shaped": dtw_shaped}
