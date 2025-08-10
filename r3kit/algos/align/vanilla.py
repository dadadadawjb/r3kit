from typing import Tuple, Optional
import numpy as np

from r3kit.utils.transformation import transform_pc


def vanilla_align(sources:np.ndarray, targets:np.ndarray, method:str="p95", return_aligned:bool=False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # sources, targets: (N, 3)
    # cannot estimate rotation
    src_c = sources.mean(axis=0, keepdims=True)
    tgt_c = targets.mean(axis=0, keepdims=True)
    r_src = np.linalg.norm(sources - src_c, axis=1)
    r_tgt = np.linalg.norm(targets - tgt_c, axis=1)

    if method == "rms":
        s = np.sqrt((r_tgt ** 2).mean()) / np.sqrt((r_src ** 2).mean())
    elif method == "median":
        s = np.median(r_tgt) / np.median(r_src)
    elif method == "p95":
        q = 0.95
        s = np.quantile(r_tgt, q) / np.quantile(r_src, q)
    else:
        raise NotImplementedError
    t = tgt_c - s * src_c
    align_transformation = np.eye(4)
    align_transformation[:3, :3] *= s
    align_transformation[:3, 3] = t.squeeze()

    if return_aligned:
        aligned_sources = transform_pc(sources, align_transformation)
    else:
        aligned_sources = None
    return align_transformation, aligned_sources


if __name__ == "__main__":
    import open3d as o3d
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

    align_transformation, aligned_sources = vanilla_align(sources, targets, method='p95', return_aligned=True)
    vis_pc(np.concatenate([aligned_sources, targets]),
           np.concatenate([np.array([[1, 0, 0]] * len(aligned_sources)), np.array([[0, 1, 0]] * len(targets))]))
    print(delta_smat(align_transformation, transformation))
