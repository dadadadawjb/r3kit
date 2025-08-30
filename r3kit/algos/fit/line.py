from typing import Tuple, Optional
import numpy as np


def fit_line(points:np.ndarray, query_idx:Optional[int]=None) -> Tuple[np.ndarray, np.ndarray, float, Optional[float], Optional[float]]:
    '''
    points: (N, 3)
    query_idx: int
    pivot: starting (3,)
    direction: motion (3,)
    error: float
    min_dist: min line distance w.r.t. query, float
    max_dist: max line distance w.r.t. query, float
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

    # query distance
    if query_idx is not None:
        min_dist = np.min(projected) - projected[query_idx]
        max_dist = np.max(projected) - projected[query_idx]
    else:
        min_dist, max_dist = None, None

    return (pivot, direction, error, min_dist, max_dist)


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
    pivot, direction, error, min_dist, max_dist = fit_line(points, query_idx=3)
    print(f'pivot: {pivot}, direction: {direction}, error: {error}, min_dist: {min_dist}, max_dist: {max_dist}')

    import open3d as o3d
    from r3kit.utils.transformation import align_dir
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    geometries.append(pcd)

    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1)
    joint.paint_uniform_color([1, 0, 0])
    joint.rotate(align_dir(np.array([0., 0., 1.]), direction), np.array([[0], [0], [0]]))
    joint.translate(pivot.reshape((3, 1)))
    geometries.append(joint)

    o3d.visualization.draw_geometries(geometries)
