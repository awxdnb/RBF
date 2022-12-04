import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes


def get_query_point(bd=0.51, resolution=64):
    vxs = np.linspace(-bd, bd, resolution)
    vys = np.linspace(-bd, bd, resolution)
    vzs = np.linspace(-bd, bd, resolution)
    pxs, pys, pzs = np.meshgrid(vxs, vys, vzs)
    p = np.concatenate((pxs[:, :, :, None], pys[:, :, :, None], pzs[:, :, :, None]), axis=-1)
    return p


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("C:/Users/82733/Desktop/RBF_master/CAD_0.ply", format="ply")  # 读ply
    pcd = pcd.random_down_sample(0.05)
    p_xyz = np.asarray(pcd.points)
    p_n = np.asarray(pcd.normals)
    eps = 0.01
    # beta = 0.01
    p_xyz_pos = p_xyz + eps * p_n
    p_xyz_neg = p_xyz - eps * p_n

    d_points = np.concatenate((p_xyz, p_xyz_pos, p_xyz_neg), axis=0)
    dist = cdist(d_points, d_points)
    A = dist ** 3
    # A = np.exp(- beta * (np.power(dist,2)))

    b1 = np.zeros((p_xyz.shape[0], 1))
    b2 = np.full((p_xyz.shape[0], 1), fill_value=eps)
    b3 = np.full((p_xyz.shape[0], 1), fill_value=-eps)

    b = np.concatenate((b1, b2, b3), axis=0)
    X = np.linalg.solve(A, b)
    # print(X)

    # generate grid points
    grid_size = 64
    grid = get_query_point(bd=0.55, resolution=grid_size)
    # print(grid)

    # interpolate
    dist = cdist(grid.reshape(-1, 3), d_points, 'euclidean')
    # print(dist)
    A = dist ** 3  # Triharmonic RBF kernel
    # A = np.exp(-dist ** 2)  # Gauissan RBF kernel
    A = A / np.sum(A, axis=1, keepdims=True)
    grid_values = np.dot(A, X)
    print(grid_values)

    v, f, _, _ = marching_cubes(grid_values.reshape(grid_size, grid_size, grid_size), 0)
    mesh = trimesh.Trimesh(v, f)
    mesh.export("C:/Users/82733/Desktop/RBF_master/CAD_1.obj")
