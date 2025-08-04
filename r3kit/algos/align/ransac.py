from typing import Tuple, Optional
import numpy as np
import open3d as o3d

from r3kit.utils.pc import cal_avg_nn_dist


def ransac_align(sources:np.ndarray, targets:np.ndarray, method:str='point', with_scale:bool=False, return_aligned:bool=False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    '''
    sources: (N, 3)
    targets: (N', 3)
    method: 'point' or 'plane', only 'point' supports `with_scale`
    Actually gives a rough transformation
    '''
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(sources)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(targets)

    src_avg_nn_dist = cal_avg_nn_dist(sources)
    tgt_avg_nn_dist = cal_avg_nn_dist(targets)
    max_correspondence_distance = 0.5 * (src_avg_nn_dist + tgt_avg_nn_dist) * 1.5 # heuristic

    src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    src_feat = o3d.pipelines.registration.compute_fpfh_feature(
        src_pcd, o3d.geometry.KDTreeSearchParamKNN(knn=100))
    tgt_feat = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_pcd, o3d.geometry.KDTreeSearchParamKNN(knn=100))

    if method == 'point':
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scale)
    elif method == 'plane':
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError(f"Unknown method: {method}")
    reg_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd, tgt_pcd, src_feat, tgt_feat, 
        mutual_filter=True, max_correspondence_distance=max_correspondence_distance, estimation_method=method,
    )
    align_transformation = reg_result.transformation
    if return_aligned:
        aligned_sources = np.asarray(src_pcd.transform(align_transformation).points)
    else:
        aligned_sources = None
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

    align_transformation, aligned_sources = ransac_align(sources, targets, method='point', with_scale=True, return_aligned=True)
    vis_pc(np.concatenate([aligned_sources, targets]),
           np.concatenate([np.array([[1, 0, 0]] * len(aligned_sources)), np.array([[0, 1, 0]] * len(targets))]))
    print(delta_smat(align_transformation, transformation))
