from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import distance as Dist
from scipy.linalg import polar


def transform_pc(pc_camera:np.ndarray, c2w:np.ndarray) -> np.ndarray:
    # pc_camera: (N, 3), c2w: (4, 4)
    # c2w support (s * R @ x + t), where s is scalar, R is (3, 3), t is (3,).
    pc_camera_hm = np.concatenate([pc_camera, np.ones((pc_camera.shape[0], 1), dtype=pc_camera.dtype)], axis=-1)        # (N, 4)
    pc_world_hm = pc_camera_hm @ c2w.T                                                          # (N, 4)
    pc_world = pc_world_hm[:, :3]                                                               # (N, 3)
    return pc_world

def transform_dir(dir_camera:np.ndarray, c2w:np.ndarray) -> np.ndarray:
    # dir_camera: (N, 3), c2w: (4, 4)
    # c2w support (s * R @ x + t), where s is scalar, R is (3, 3), t is (3,).
    dir_camera_hm = np.concatenate([dir_camera, np.zeros((dir_camera.shape[0], 1), dtype=dir_camera.dtype)], axis=-1)   # (N, 4)
    dir_world_hm = dir_camera_hm @ c2w.T                                                        # (N, 4)
    dir_world = dir_world_hm[:, :3]                                                             # (N, 3)
    dir_world /= np.linalg.norm(dir_world, axis=-1, keepdims=True)
    return dir_world

def transform_quat(quat_camera:np.ndarray, c2w:np.ndarray) -> np.ndarray:
    # quat_camera: (N, 4) (xyzw), c2w: (4, 4)
    quat_world = Rot.from_matrix(c2w[:3, :3] @ Rot.from_quat(quat_camera).as_matrix()).as_quat()
    return quat_world

def transform_frame(ref_T:np.ndarray, new_T:np.ndarray) -> np.ndarray:
    """
    ref_T: (4, 4)
    new_T: (4, 4)
    new_in_ref: (4, 4)
    """
    new_in_ref = np.linalg.inv(ref_T) @ new_T
    return new_in_ref

def forward_frame(ref_T:np.ndarray, new_in_ref:np.ndarray) -> np.ndarray:
    """
    ref_T: (4, 4)
    new_in_ref: (4, 4)
    new_T: (4, 4)
    """
    new_T = ref_T @ new_in_ref
    return new_T


def align_dir(src:np.ndarray, dst:np.ndarray) -> np.ndarray:
    """
    src: (3,) normalized
    dst: (3,) normalized
    rot: (3, 3) rotation matrix from src to dst
    """
    au = np.linalg.svd(src.reshape((-1, 1)))[0]
    bu = np.linalg.svd(dst.reshape((-1, 1)))[0]
    if np.linalg.det(au) < 0:
        au[:, -1] *= -1.0
    if np.linalg.det(bu) < 0:
        bu[:, -1] *= -1.0
    rot = bu.dot(au.T)
    return rot


def mean_xyz(xyzs:np.ndarray) -> np.ndarray:
    # xyzs: (N, 3)
    return np.mean(xyzs, axis=0)

def median_xyz(xyzs:np.ndarray) -> np.ndarray:
    # xyzs: (N, 3)
    distances = Dist.cdist(xyzs, xyzs, metric='euclidean') # (N, N)
    total_distance = distances.sum(axis=1) # (N,)
    j_star = np.argmin(total_distance)
    return xyzs[j_star]

def mean_dir(dirs:np.ndarray) -> np.ndarray:
    # dirs: (N, 3), normalized
    dir_sum = np.sum(dirs, axis=0)
    norm = np.linalg.norm(dir_sum)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return dir_sum / norm

def median_dir(dirs:np.ndarray) -> np.ndarray:
    # dirs: (N, 3), normalized
    dots = 1.0 - Dist.cdist(dirs, dirs, metric='cosine') # (N, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    total_angle = angles.sum(axis=1) # (N,)
    j_star = np.argmin(total_angle)
    return dirs[j_star]


def rot6d2R(rot6d:np.ndarray=np.array([1., 0., 0., 0., 1., 0.]), eps:float=1e-8) -> np.ndarray:
    """
    rot6d without constraint
    """
    a, b = rot6d[:3], rot6d[3:]

    a_n = np.linalg.norm(a)
    if a_n < eps:
        r1 = a * 0.0
    else:
        r1 = a / a_n

    # Remove projection of b onto r1 to enforce orthogonality
    b_ortho = b - np.dot(r1, b) * r1
    b_ortho_n = np.linalg.norm(b_ortho)
    if b_ortho_n < eps:
        r2 = b_ortho * 0.0
    else:
        r2 = b_ortho / b_ortho_n

    # If degenerate (a≈0 or b parallel to a), provide a fallback r2
    if np.linalg.norm(r2) < eps:
        # Pick a vector that is not parallel to r1
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(fallback, r1)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b_ortho = fallback - np.dot(r1, fallback) * r1
        r2 = b_ortho / np.linalg.norm(b_ortho)

    r3 = np.cross(r1, r2)

    R = np.stack([r1, r2, r3], axis=1)
    return R

def R2rot6d(R:np.ndarray=np.eye(3)) -> np.ndarray:
    r1 = R[:, 0]
    r2 = R[:, 1]
    return np.concatenate([r1, r2], axis=0)


def xyzrot2mat(xyz:np.ndarray=np.zeros(3), rot:np.ndarray=np.eye(3)) -> np.ndarray:
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = rot
    pose_4x4[:3, 3] = xyz
    return pose_4x4

def mat2xyzrot(pose_4x4:np.ndarray=np.eye(4)) -> Tuple[np.ndarray, np.ndarray]:
    xyz = pose_4x4[:3, 3]
    rot = pose_4x4[:3, :3]
    return (xyz, rot)

def xyzquat2mat(xyz:np.ndarray=np.zeros(3), quat:np.ndarray=np.array([0., 0., 0., 1.])) -> np.ndarray:
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = Rot.from_quat(quat).as_matrix()
    pose_4x4[:3, 3] = xyz
    return pose_4x4

def mat2xyzquat(pose_4x4:np.ndarray=np.eye(4)) -> Tuple[np.ndarray, np.ndarray]:
    xyz = pose_4x4[:3, 3]
    quat = Rot.from_matrix(pose_4x4[:3, :3]).as_quat()
    return (xyz, quat)

def xyzrot6d2mat(xyz:np.ndarray=np.zeros(3), rot6d:np.ndarray=np.array([1., 0., 0., 0., 1., 0.])) -> np.ndarray:
    """
    rot6d without constraint
    """
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = rot6d2R(rot6d)
    pose_4x4[:3, 3] = xyz
    return pose_4x4

def mat2xyzrot6d(pose_4x4:np.ndarray=np.eye(4)) -> Tuple[np.ndarray, np.ndarray]:
    xyz = pose_4x4[:3, 3]
    rot6d = R2rot6d(pose_4x4[:3, :3])
    return (xyz, rot6d)

def sRt2smat(s:float, R:np.ndarray, t:np.ndarray) -> np.ndarray:
    smat = np.eye(4)
    smat[:3, :3] = s * R
    smat[:3, 3] = t
    smat[3, 3] = 1.0
    return smat

def smat2sRt(smat:np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    s = np.linalg.norm(smat[:3, :3], ord='fro') / np.sqrt(3.0)
    R_approx = smat[:3, :3] / s
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = smat[:3, 3]
    return (s, R, t)


def xyz2len(xyz:np.ndarray) -> float:
    return np.linalg.norm(xyz)
def delta_xyz(xyz1:np.ndarray, xyz2:np.ndarray) -> float:
    delta = xyz1 - xyz2
    return xyz2len(delta)

def delta_dir(dir1:np.ndarray, dir2:np.ndarray) -> float:
    dot = np.dot(dir1, dir2)
    theta = np.arccos(np.clip(dot, -1, 1))
    theta = np.clip(theta, 0, np.pi)
    return theta
def delta_quat(quat1:np.ndarray, quat2:np.ndarray) -> float:
    delta = Rot.from_quat(quat1).inv() * Rot.from_quat(quat2) # represented in quat1
    delta_value = delta.magnitude()
    return delta_value

def rot2angle(rot:np.ndarray) -> float:
    trace_rot = np.trace(rot)
    theta = np.arccos(np.clip((trace_rot - 1) / 2, -1, 1))
    theta = np.clip(theta, 0, np.pi)
    return theta
def delta_rot(rot1:np.ndarray, rot2:np.ndarray) -> float:
    delta = rot1.T @ rot2 # represented in rot1
    return rot2angle(delta)

def delta_mat(mat1:np.ndarray, mat2:np.ndarray) -> Tuple[float, float]:
    delta = np.linalg.inv(mat1) @ mat2 # represented in mat1
    return (xyz2len(delta[:3, 3]), rot2angle(delta[:3, :3]))
def delta_smat(sm1:np.ndarray, sm2:np.ndarray) -> Tuple[float, float, float]:
    t1, t2 = sm1[:3, 3], sm2[:3, 3]
    rot1, s1 = polar(sm1[:3, :3])
    rot2, s2 = polar(sm2[:3, :3])
    return (delta_xyz(t1, t2), delta_rot(rot1, rot2), s2[0, 0] / s1[0, 0])
