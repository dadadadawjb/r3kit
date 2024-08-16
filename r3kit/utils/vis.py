import os
from typing import List, Optional
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import concurrent
import concurrent.futures


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


def draw_time(timestamps:List[float], path:str) -> None:
    num = len(timestamps) - 1
    x = list(range(num))
    y = [timestamps[idx+1] - timestamps[idx] for idx in range(num)]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('data')
    plt.ylabel('time')
    plt.savefig(path)


def save_img(idx:int, path:str, frame_list:List[np.ndarray], suffix:str='png', normalize:bool=False) -> None:
    if normalize:
        cv2.imwrite(os.path.join(path, f"{str(idx).zfill(16)}.{suffix}"), ((frame_list[idx] - frame_list[idx].min()) / (frame_list[idx].max() - frame_list[idx].min()) * 255).astype(np.uint8))
    else:
        cv2.imwrite(os.path.join(path, f"{str(idx).zfill(16)}.{suffix}"), frame_list[idx])

def save_imgs(path:str, frame_list:List[np.ndarray], suffix:str='png', normalize:bool=False) -> None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                save_img,
                idx,
                path,
                frame_list,
                suffix,
                normalize
            )
            for idx in range(len(frame_list))
        ]

        for future in concurrent.futures.as_completed(futures):
            pass
