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


def rotation_vec2mat(vec:np.ndarray) -> np.ndarray:
    mat = np.zeros((3, 3))
    temp2 = np.cross(vec, np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-3:
        temp1 = np.cross(np.array([0., 1., 0.]), vec)
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(vec, temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, vec)
        temp1 /= np.linalg.norm(temp1)
    mat[:, 0] = temp1
    mat[:, 1] = temp2
    mat[:, 2] = vec
    return mat


def draw_time(timestamps:List[float], path:str) -> None:
    num = len(timestamps) - 1
    x = list(range(num))
    y = [timestamps[idx+1] - timestamps[idx] for idx in range(num)]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('data')
    plt.ylabel('time')
    plt.savefig(path)

def draw_items(items:np.ndarray, path:str) -> None:
    assert len(items.shape) == 1 or len(items.shape) == 2
    if len(items.shape) == 1:
        T = items.shape[0]
        x = list(range(T))
        y = items
        plt.figure()
        plt.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.savefig(path)
    else:
        plt.figure()
        T, N = items.shape
        x = list(range(T))
        if N <= 3:
            for i in range(N):
                plt.subplot(1, N, i+1)
                plt.plot(x, items[:, i])
                plt.xlabel('time')
                plt.ylabel(f'value_{i}')
        else:
            for i in range(N):
                plt.subplot(int(np.ceil(N/3)), 3, i+1)
                plt.plot(x, items[:, i])
                plt.xlabel('time')
                plt.ylabel(f'value_{i}')
        plt.savefig(path)


def save_img(idx:int, path:str, frame:np.ndarray, suffix:str='png', normalize:bool=False, idx_bias:int=0) -> None:
    if normalize:
        cv2.imwrite(os.path.join(path, f"{str(idx+idx_bias).zfill(16)}.{suffix}"), ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8))
    else:
        cv2.imwrite(os.path.join(path, f"{str(idx+idx_bias).zfill(16)}.{suffix}"), frame)

def save_imgs(path:str, frame_list:List[np.ndarray], suffix:str='png', normalize:bool=False, idx_bias:int=0) -> None:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                save_img,
                idx,
                path,
                frame_list[idx],
                suffix,
                normalize,
                idx_bias
            )
            for idx in range(len(frame_list))
        ]

        for future in concurrent.futures.as_completed(futures):
            pass

def save_video(path:str, frame_list:List[np.ndarray], fps:int=30) -> None:
    height, width, _ = frame_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frame_list:
        out.write(frame)
    out.release()
