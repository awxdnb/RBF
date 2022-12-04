import numpy as np
import torch

import open3d as o3d
import trimesh

from scipy.spatial import distance
from skimage.measure import marching_cubes


def normalize_mesh_export(mesh, file_out=None):
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    if file_out is not None:
        mesh.export(file_out)
    return mesh


def get_query_point(bd=0.51, resolution=64):
    shape = (resolution, resolution, resolution)
    vxs = torch.arange(-bd, bd, bd * 2 / resolution)
    vys = torch.arange(-bd, bd, bd * 2 / resolution)
    vzs = torch.arange(-bd, bd, bd * 2 / resolution)
    pxs = vxs.view(-1, 1, 1).expand(*shape).contiguous().view(resolution ** 3)
    pys = vys.view(1, -1, 1).expand(*shape).contiguous().view(resolution ** 3)
    pzs = vzs.view(1, 1, -1).expand(*shape).contiguous().view(resolution ** 3)
    q = torch.stack([pxs, pys, pzs], dim=1).numpy().reshape(-1, 3)
    return q


if __name__ == '__main__':
    pts_o3d = o3d.io.read_point_cloud("C:/Users/82733/Desktop/RBF-master/CAD_0.ply", format="ply")
    points = np.array(pts_o3d.points)
    normals = np.array(pts_o3d.normals)

    sample_num = 2000
    sample_idx = np.random.choice(points.shape[0], sample_num, replace=False)
    points = points[sample_idx]
    normals = normals[sample_idx]

    eps = 0.01
    points_pos = points + normals * eps
    points_neg = points - normals * eps

    all_points = np.concatenate((points, points_pos, points_neg), axis=0)
    dist = distance.cdist(all_points, all_points, 'euclidean')
    A = dist ** 3  # Triharmonic RBF kernel
    # A = np.exp(-dist ** 2)  # Gauissan RBF kernel
    A = dist / np.sum(dist, axis=1, keepdims=True)

    b0 = np.zeros((points.shape[0], 1))
    b1 = np.full((points.shape[0], 1), fill_value=eps)
    b2 = np.full((points.shape[0], 1), fill_value=-eps)
    b = np.concatenate((b0, b1, b2), axis=0)

    x = np.linalg.solve(A, b)

    # generate grid points
    grid_size = 64
    grid = get_query_point(bd=0.55, resolution=grid_size)
    # interpolate
    dist = distance.cdist(grid, all_points, 'euclidean')
    A = dist ** 3  # Triharmonic RBF kernel
    # # A = np.exp(-dist ** 2)  # Gauissan RBF kernel
    A = dist / np.sum(dist, axis=1, keepdims=True)
    grid_values = np.dot(A, x)

    v, f, _, _ = marching_cubes(grid_values.reshape(grid_size, grid_size, grid_size), 0)
    mesh = trimesh.Trimesh(v, f)

    normalize_mesh_export(mesh, file_out="C:/Users/82733/Desktop/RBF-master/CAD_0_RBF_{}.obj".format(grid_size))