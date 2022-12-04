import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
import torch


def get_query_point(bd=0.55, resolution=129):
    vxs = np.linspace(-bd, bd, resolution)
    vys = np.linspace(-bd, bd, resolution)
    vzs = np.linspace(-bd, bd, resolution)
    pxs, pys, pzs = np.meshgrid(vxs, vys, vzs)
    p = np.concatenate((pxs[:, :, :, None], pys[:, :, :, None], pzs[:, :, :, None]), axis=-1)
    print(p.shape)
    return p


if __name__ == '__main__':
    get_query_point(0.55, 129)
