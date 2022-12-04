import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
import torch


def get_query_point(bd=0.55, resolution=128):
    shape = (resolution, resolution, resolution)
    vxs = torch.arange(-bd, bd, bd * 2 / resolution)
    vys = torch.arange(-bd, bd, bd * 2 / resolution)
    vzs = torch.arange(-bd, bd, bd * 2 / resolution)
    pxs = vxs.view(-1, 1, 1).expand(*shape).contiguous().view(resolution ** 3)
    pys = vys.view(1, -1, 1).expand(*shape).contiguous().view(resolution ** 3)
    pzs = vzs.view(1, 1, -1).expand(*shape).contiguous().view(resolution ** 3)
    p = torch.stack([pxs, pys, pzs], dim=1).reshape(resolution, resolution ** 2, 3)
    print(np.asarray(p))
    print(p.shape)
    return


if __name__ == '__main__':
    get_query_point(0.55, 128)
