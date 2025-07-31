from typing import Tuple
import numpy as np


def fit_arc(points:np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    '''
    points: (N, 3)
    C: center, (3,)
    A: right-hand normal, (3,)
    r: radius, float
    error: float
    '''
    # from https://blog.csdn.net/jiangjjp2812/article/details/106937333
    N = points.shape[0]
    assert N >= 3

    # fit plane, ax+by+cz=1
    A = np.linalg.pinv(points.T @ points) @ points.T @ np.ones((N, 1))
    
    # fit circle, C * (P1+P2)/2 = 0
    indices = np.triu_indices(N, k=1)
    i_indices, j_indices = indices
    B = points[j_indices] - points[i_indices] # (N*(N-1)//2, 3)
    points_squared_norms = np.sum(points**2, axis=1) # (N,)
    L = (points_squared_norms[j_indices] - points_squared_norms[i_indices]).reshape(-1, 1) / 2 # (N*(N-1)//2, 1)
    coefficient_matrix = np.zeros((4, 4))
    coefficient_matrix[:3, :3] = B.T @ B
    coefficient_matrix[:3, 3:] = A
    coefficient_matrix[3:, :3] = A.T
    coefficient_matrix[3, 3] = 0
    rhs = np.zeros((4, 1))
    rhs[:3, 0:] = B.T @ L
    rhs[3, 0] = 1
    C_lambda = np.linalg.pinv(coefficient_matrix) @ rhs
    C = C_lambda[:3, 0]
    A = (A / (np.linalg.norm(A) + 1e-8)).squeeze(axis=-1)
    rs = np.linalg.norm(points - C, axis=-1)
    r = np.mean(rs)
    
    # finetune direction
    projected = points - ((points - C[None, :]) @ A[:, None]) * A[None, :]
    v1 = projected[:-1] - C[None, :]
    v2 = projected[1:] - C[None, :]
    cross = np.cross(v1, v2)
    accum_cross = np.sum(cross, axis=0)
    if np.dot(accum_cross, A) < 0:
        A = -A
    
    # calculate error
    projected_offset = projected - C[None, :]
    projected_offset = projected_offset / (np.linalg.norm(projected_offset, axis=-1, keepdims=True) + 1e-8)
    predicted = C[None, :] + r * projected_offset
    error = np.mean(np.linalg.norm(points - predicted, axis=-1))

    return (C, A, r, error)


if __name__ == '__main__':
    points = np.array([[-0.3, 0.46, 0.83],
                    [-0.2, 0.254, 0.946],
                    [-0.1, 0.111, 0.989],
                    [0, 0, 1],
                    [0.1, -0.0910, 0.991],
                    [0.2, -0.166, 0.966],
                    [0.3, -0.227, 0.927],
                    [0.4, -0.275, 0.874],
                    [0.5, -0.309, 0.809],
                    [0.6, -0.329, 0.729],
                    [0.7, -0.332, 0.632],
                    [0.8, -0.312, 0.512],
                    [0.9, -0.254, 0.354],
                    [0.8, 0.512, -0.312],
                    [0.7, 0.632, -0.332],
                    [0.6, 0.729, -0.329],
                    [0.5, 0.809, -0.309],
                    [0.4, 0.874, -0.274],
                    [0.3, 0.927, -0.227],
                    [0.2, 0.966, -0.166],
                    [0.1, 0.991, -0.091],
                    [0, 1, 0],
                    [-0.1, 0.989, 0.111],
                    [-0.2, 0.946, 0.254],
                    [-0.3, 0.83, 0.45]])
    points = np.array([[11.5713, 6.9764, 10.4685],
                    [11.5859, 9.088, 13.5831],
                    [11.5802, 11.1949, 14.6103],
                    [11.5542, 13.312, 14.7279],
                    [11.5692, 15.3806, 14.0576],
                    [11.5632, 17.4873, 12.1397],
                    [11.5598, 17.8894, 6.4025],
                    [11.5577, 15.8714, 4.0703],
                    [11.5729, 13.8578, 3.232],
                    [11.5657, 11.8711, 3.1326],
                    [11.5706, 9.8797, 3.7866],
                    [11.5663, 7.8676, 5.5348]])
    C, A, r, error = fit_arc(points)
    print(f'center: {C}, normal: {A}, radius: {r}, error: {error}')

    import open3d as o3d
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    geometries.append(pcd)
    
    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1)
    joint.paint_uniform_color([1, 0, 0])
    rotation = np.zeros((3, 3))
    temp2 = np.cross(A, np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-6:
        temp1 = np.cross(np.array([0., 1., 0.]), A)
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(A, temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, A)
        temp1 /= np.linalg.norm(temp1)
    rotation[:, 0] = temp1
    rotation[:, 1] = temp2
    rotation[:, 2] = A
    joint.rotate(rotation, np.array([[0], [0], [0]]))
    joint.translate(C.reshape((3, 1)))
    geometries.append(joint)

    o3d.visualization.draw_geometries(geometries)
