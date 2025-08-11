from typing import Tuple, Optional
import numpy as np
from pycpd import RigidRegistration


def cpd_align(sources:np.ndarray, targets:np.ndarray, source_outlier:float=0.1, with_scale:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    sources: (N, 3)
    targets: (N', 3)
    '''
    reg = RigidRegistration(X=targets, Y=sources, w=source_outlier, scale=with_scale)
    aligned_sources, (s, R, t) = reg.register()
    align_transformation = np.eye(4)
    align_transformation[:3, :3] = s * R.T
    align_transformation[:3, 3] = t
    return align_transformation, aligned_sources


if __name__ == "__main__":
    import open3d as o3d
    from r3kit.utils.transformation import transform_pc
    from r3kit.utils.vis import vis_pc
    from r3kit.utils.transformation import delta_smat

    mesh = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=3.0, resolution=20)
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    sources = np.asarray(pcd.points).copy()
    pcd.scale(0.5, center=(0, 0, 0))
    pcd.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
    sources = np.concatenate([sources, np.asarray(pcd.points)], axis=0)
    targets = sources.copy() + np.random.randn(*sources.shape) * 0.01
    transformation = np.eye(4)
    transformation[:3, :3] = 0.5 * np.array([[0.6124, -0.7891, 0.0474], [0.6124, 0.4356, -0.6597], [0.5, 0.433, 0.75]])
    transformation[:3, 3] = np.array([0.5, -0.25, 2.0])
    targets = transform_pc(targets, transformation)
    vis_pc(np.concatenate([sources, targets]),
           np.concatenate([np.array([[1, 0, 0]] * len(sources)), np.array([[0, 1, 0]] * len(targets))]))

    align_transformation, aligned_sources = cpd_align(sources, targets, source_outlier=0.0, with_scale=True)
    vis_pc(np.concatenate([aligned_sources, targets]),
           np.concatenate([np.array([[1, 0, 0]] * len(aligned_sources)), np.array([[0, 1, 0]] * len(targets))]))
    print(delta_smat(align_transformation, transformation))
