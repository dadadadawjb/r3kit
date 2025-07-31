from typing import Tuple
import numpy as np


def fit_line(points:np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    '''
    points: (N, 3)
    pivot: starting (3,)
    direction: motion (3,)
    error: float
    '''
    N = points.shape[0]
    assert N >= 2

    # PCA main direction
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered)
    direction = Vt[0]

    # finetune direction
    motion_dir = np.sum(points[1:] - points[:-1], axis=0)
    if np.dot(direction, motion_dir) < 0:
        direction = -direction

    # choose starting pivot
    projected = centered @ direction
    pivot = centroid + np.min(projected) * direction

    # calculate error
    predicted = pivot[None, :] + ((points - pivot[None, :]) @ direction[:, None]) * direction[None, :]
    error = np.mean(np.linalg.norm(points - predicted, axis=1))

    return (pivot, direction, error)


if __name__ == '__main__':
    points = np.array([[2., 2., 1.],
                       [2.5, 2.5, 1.5],
                       [3., 3., 2.],
                       [3.5, 3.5, 2.5],
                       [4., 4., 3.],
                       [4.5, 4.5, 3.5],
                       [5., 5., 4.],
                       [5.5, 5.5, 4.5]])
    points = np.array([[2., 2., 1.],
                       [2.5, 2.5, 1.6],
                       [3.2, 3.1, 1.9],
                       [3.5, 3.6, 2.5],
                       [3.8, 4., 3.1],
                       [4.5, 4.6, 3.6],
                       [5., 5., 4.],
                       [5.51, 5.48, 4.6]])
    pivot, direction, error = fit_line(points)
    print(f'pivot: {pivot}, direction: {direction}, error: {error}')

    import open3d as o3d
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    geometries.append(pcd)

    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1)
    joint.paint_uniform_color([1, 0, 0])
    rotation = np.zeros((3, 3))
    temp2 = np.cross(direction, np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-6:
        temp1 = np.cross(np.array([0., 1., 0.]), direction)
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(direction, temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, direction)
        temp1 /= np.linalg.norm(temp1)
    rotation[:, 0] = temp1
    rotation[:, 1] = temp2
    rotation[:, 2] = direction
    joint.rotate(rotation, np.array([[0], [0], [0]]))
    joint.translate(pivot.reshape((3, 1)))
    geometries.append(joint)

    o3d.visualization.draw_geometries(geometries)
