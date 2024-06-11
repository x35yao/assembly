import torch

def create_tags(objs, dtype = torch.float32):
    one_hots = torch.eye(len(objs), dtype = dtype)
    tag_dict = {}
    for i, obj in enumerate(objs):
        tag_dict[obj] = one_hots[i]
    return tag_dict

def normalize_wrapper(average, std):
    """normalize for multiprocessing"""
    return lambda x: normalize_3d(x, average, std)


def normalize_3d(entry, average, std):
    entry[:, :3] = (entry[:, :3] - average) / std
    return entry