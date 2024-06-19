from typing import List, Tuple, Optional
import numpy as np


def get_point_cloud(depth:np.ndarray, intrinsics:List[float], color:Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    height, weight = depth.shape
    [pixX, pixY] = np.meshgrid(np.arange(weight), np.arange(height))
    x = (pixX - intrinsics[0]) * depth / intrinsics[2]
    y = (pixY - intrinsics[1]) * depth / intrinsics[3]
    z = depth
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    if color is None:
        rgb = None
    else:
        rgb = color.reshape(-1, 3)
    return (xyz, rgb)
