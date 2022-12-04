import math
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.linalg import norm, pinv


def _basisfunc(x, y, beta):
    return np.exp(beta * norm(x - y) ** 2)


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud("C:/Users/82733/Desktop/RBF-master/CAD_0.ply", format="ply")  # 读ply
    pcd = pcd.uniform_down_sample(20)
    p_xyz = np.asarray(pcd.points)
    p_n = np.asarray(pcd.normals)
    eps = 0.005
    p_xyz_pos = p_xyz + eps * p_n
    p_xyz_neg = p_xyz - eps * p_n
    A = np.concatenate((p_xyz, p_xyz_pos, p_xyz_neg), axis=0)
    dist = cdist(A, A)
    A = dist ** 3
    b1 = np.zeros((p_xyz.shape[0], 1))
    b2 = np.full((p_xyz.shape[0], 1), fill_value=eps)
    b3 = np.full((p_xyz.shape[0], 1), fill_value=-eps)
    B = np.concatenate((b1, b2, b3), axis=0)
    W = np.linalg.solve(A, B)
    # p_3xyz = []
    # Y = []
    # for i in range(2000):
    #     p_3xyz.append(p_xyz[i])
    #     p_3xyz.append(p_xyz[i] + 0.0005 * p_n[i])
    #     p_3xyz.append(p_xyz[i] - 0.0005 * p_n[i])
    #
    # for i in range(2000):
    #     Y.append(0)
    #     Y.append(0.0005)
    #     Y.append(-0.0005)
    #
    # X = [[] for i in range(6000)]
    #
    # for i in range(6000):
    #     for j in range(6000):
    #         print(1)
    #         X[i].append(_basisfunc(p_3xyz[j], p_3xyz[i], 8))
    # W = np.dot(pinv(X), Y)
    # print(W)
