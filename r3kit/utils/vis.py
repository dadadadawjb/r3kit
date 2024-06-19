from typing import Optional
import numpy as np
import open3d as o3d


def vis_pc(xyz:np.ndarray, rgb:Optional[np.ndarray]=None, show_frame:bool=True) -> None:
    '''
    xyz: (N, 3) in meter in camera frame
    rgb: (N, 3) in [0, 1]
    '''
    geometries = []
    if show_frame:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)
    else:
        pass
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        pass
    geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)
