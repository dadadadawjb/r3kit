from typing import Tuple, Optional
import numpy as np


def _build_plane_basis(n:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # (u, v, n) right-hand coordinate system
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = a - np.dot(a, n) * n
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-12)
    return (u, v)


def fit_arc_2d(xy:np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    xy: (N, 2)
    center: (2,)
    r: radius, float
    '''
    # least squares on x^2 + y^2 + a x + b y + c = 0
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1) # (N, 3)
    b = -(x**2 + y**2) # (N,)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = p
    center = np.array([-a / 2.0, -b_ / 2.0])
    r = np.sqrt(max(center[0] * center[0] + center[1] * center[1] - c, 0.0))
    return (center, r)

def fit_arc(points:np.ndarray, query_idx:Optional[int]=None) -> Tuple[np.ndarray, np.ndarray, float, float, Optional[float], Optional[float]]:
    '''
    points: (N, 3)
    query_idx: int
    center: (3,)
    normal: right-hand normal, (3,)
    r: radius, float
    error: float
    min_angle: min arc angle w.r.t. query, float
    max_angle: max arc angle w.r.t. query, float
    '''
    N = points.shape[0]
    assert N >= 3

    # PCA minimal direction
    centroid = np.mean(points, axis=0)
    centered = points - centroid[None, :]
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]
    d = np.dot(normal, centroid) # n * (x - c) = 0, d = n * c, (N,)

    # project to plane
    t = points @ normal - d # (N,)
    projected = points - t[:, None] * normal[None, :] # (N, 3)
    u, v = _build_plane_basis(normal) # (3,), (3,)
    xy = np.stack([(projected - centroid[None, :]) @ u, (projected - centroid[None, :]) @ v], axis=1) # (N, 2)
    center_2d, r = fit_arc_2d(xy)
    center = centroid + center_2d[0] * u + center_2d[1] * v

    # finetune direction
    v1 = projected[:-1] - center[None, :]
    v2 = projected[1:] - center[None, :]
    cross = np.cross(v1, v2)
    accum_cross = np.sum(cross, axis=0)
    if np.dot(accum_cross, normal) < 0:
        normal = -normal
    
    # calculate error
    projected_offset = projected - center[None, :]
    projected_offset = projected_offset / (np.linalg.norm(projected_offset, axis=-1, keepdims=True) + 1e-8)
    predicted = center[None, :] + r * projected_offset
    error = np.mean(np.linalg.norm(points - predicted, axis=-1))

    # query angle
    if query_idx is not None:
        ref_x = projected_offset[query_idx]
        ref_x /= (np.linalg.norm(ref_x) + 1e-8)
        ref_y = np.cross(normal, ref_x)
        ref_y /= (np.linalg.norm(ref_y) + 1e-8)
        coords = np.stack([projected_offset @ ref_x, projected_offset @ ref_y], axis=1)   # (N,2)
        angles = np.arctan2(coords[:, 1], coords[:, 0])
        angles = np.unwrap(angles)
        min_angle = np.min(angles)
        max_angle = np.max(angles)
    else:
        min_angle, max_angle = None, None

    return (center, normal, r, error, min_angle, max_angle)


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
    points = np.array([[0.5713, 0.9764, 10.4685],
                    [0.5859, 3.088, 13.5831],
                    [0.5802, 5.1949, 14.6103],
                    [0.5542, 7.312, 14.7279],
                    [0.5692, 9.3806, 14.0576],
                    [0.5632, 11.4873, 12.1397],
                    [0.5598, 11.8894, 6.4025],
                    [0.5577, 9.8714, 4.0703],
                    [0.5729, 7.8578, 3.232],
                    [0.5657, 5.8711, 3.1326],
                    [0.5706, 3.8797, 3.7866],
                    [0.5663, 1.8676, 5.5348]])
    center, normal, r, error, min_angle, max_angle = fit_arc(points, query_idx=5)
    print(f'center: {center}, normal: {normal}, radius: {r}, error: {error}, min_angle: {min_angle * 180 / np.pi}, max_angle: {max_angle * 180 / np.pi}')

    import open3d as o3d
    from r3kit.utils.transformation import align_dir
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    geometries.append(pcd)
    
    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1)
    joint.paint_uniform_color([1, 0, 0])
    joint.rotate(align_dir(np.array([0., 0., 1.]), normal), np.array([[0], [0], [0]]))
    joint.translate(center.reshape((3, 1)))
    geometries.append(joint)

    o3d.visualization.draw_geometries(geometries)
