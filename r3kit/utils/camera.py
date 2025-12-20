import numpy as np


def compute_K(f_mm:float, sensor_width:float, sensor_height:float, image_width:int, image_height:int) -> np.ndarray:
    """
    f_mm: focal length in mm
    sensor_width: sensor width in mm
    sensor_height: sensor height in mm
    image_width: image width in pixels
    image_height: image height in pixels
    
    fx: focal length in pixels along x-axis
    fy: focal length in pixels along y-axis
    cx: principal point x-coordinate in pixels
    cy: principal point y-coordinate in pixels
    """
    fx = f_mm / sensor_width * image_width
    fy = f_mm / sensor_height * image_height
    cx = image_width / 2
    cy = image_height / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K


def depth2pc(depth:np.ndarray, K:np.ndarray) -> np.ndarray:
    """
    depth: (H, W)
    K: (3, 3)
    pc: (H, W, 3), follow OpenCV convention, (x, y, z) = (right, down, in)
    """
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (x - K[0, 2]) * z / K[0, 0]
    y = (y - K[1, 2]) * z / K[1, 1]
    points = np.dstack((x, y, z))
    return points

def pc2pix(pc:np.ndarray, K:np.ndarray) -> np.ndarray:
    """
    pc: (N, 3) or (H, W, 3), follow OpenCV convention, (x, y, z) = (right, down, in)
    K: (3, 3)
    pix: (N, 3) or (H, W, 3)
    """
    x = pc[..., 0] * K[0, 0] / pc[..., 2] + K[0, 2]
    y = pc[..., 1] * K[1, 1] / pc[..., 2] + K[1, 2]
    z = pc[..., 2]
    pix = np.stack((x, y, z), axis=-1)
    return pix


def opengl2opencv(pose:np.ndarray) -> np.ndarray:
    pose[:3, :3] = pose[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return pose


def K2P(K:np.ndarray, H:int, W:int, n:float=0.1, f:float=1000) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.array([[2 * fx / W, 0, -2 * cx / W + 1, 0],
                  [0, 2 * fy / H, 2 * cy / H - 1, 0],
                  [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                  [0, 0, -1, 0]])
    return P


import numpy as np

def lookat(eye:np.ndarray=np.array([1.0, 1.0, 1.0]), center:np.ndarray=np.array([0.0, 0.0, 0.0]), up:np.ndarray=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """
    Return camera-to-world transform (c2w) following OpenCV camera convention:
    - camera x: right
    - camera y: down
    - camera z: forward (towards the object)
    """
    # camera forward (z axis) in world coords
    z_cam = center - eye
    z_cam = z_cam / np.linalg.norm(z_cam)

    # camera right (x axis) in world coords
    x_cam = np.cross(z_cam, up)
    x_cam = x_cam / np.linalg.norm(x_cam)

    # camera down (y axis) in world coords (right-handed)
    y_cam = np.cross(z_cam, x_cam)

    # Rotation: camera -> world (columns are camera axes expressed in world)
    R_wc = np.stack([x_cam, y_cam, z_cam], axis=1)  # 3x3

    # c2w
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_wc
    T_c2w[:3, 3] = eye
    return T_c2w
