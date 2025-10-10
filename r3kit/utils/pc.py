from typing import Tuple, Dict
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import yourdfpy
import fpsample


def voxelize(pc_xyz:np.ndarray, pc_rgb:np.ndarray, voxel_size:float) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_pc_xyz = np.asarray(downsampled_pcd.points)
    downsampled_pc_rgb = np.asarray(downsampled_pcd.colors)
    return (downsampled_pc_xyz, downsampled_pc_rgb)

def farthest_point_sample(point:np.ndarray, npoint:int) -> Tuple[np.ndarray, np.ndarray]:
    N, D = point.shape
    xyz = point[:,:3]
    # centroids = np.zeros((npoint,), dtype=np.uint64)
    # distance = np.ones((N,)) * 1e10
    # farthest = np.random.randint(0, N)
    # for i in range(npoint):
    #     centroids[i] = farthest
    #     centroid = xyz[farthest, :]
    #     dist = np.sum((xyz - centroid) ** 2, -1)
    #     mask = dist < distance
    #     distance[mask] = dist[mask]
    #     farthest = np.argmax(distance, -1)
    h = max(3, min(9, int(round(0.5 * np.log2(N) - 1)))) # heuristic
    centroids = fpsample.bucket_fps_kdline_sampling(xyz, npoint, h=h)
    point = point[centroids]
    return (point, centroids)


def remove_outlier(pc:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    nb_neighbors = max(10, min(int(np.sqrt(pc.shape[0])), 50)) # heuristic
    std_ratio = 2.0
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_pc = np.asarray(pcd.select_by_index(ind).points)
    return (inlier_pc, np.asarray(ind))


def nearest_point_query(src:np.ndarray, query:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(src)
    distances, indices = tree.query(query)
    result_pc = src[indices]
    return (result_pc, indices)


def cal_avg_nn_dist(point:np.ndarray) -> float:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distances = []
    for point in pcd.points:
        _, idx, dists = pcd_tree.search_knn_vector_3d(point, 2)
        distances.append(np.sqrt(dists[1]))
    return np.mean(distances)


def mesh2pc(obj_path:str, num_points:int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(obj_path)
    pc = mesh.sample_points_uniformly(number_of_points=num_points)
    pc = np.asarray(pc.points)
    return pc

def urdf2pc(urdf_path:str, joints:Dict[str, float], num_points:int) -> np.ndarray:
    urdf = yourdfpy.URDF.load(urdf_path)
    urdf.update_cfg(joints)
    mesh = urdf.scene.to_mesh()
    geometry = o3d.geometry.TriangleMesh()
    geometry.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    geometry.triangles = o3d.utility.Vector3iVector(mesh.faces)
    pc = geometry.sample_points_uniformly(number_of_points=num_points)
    pc = np.asarray(pc.points)
    return pc
