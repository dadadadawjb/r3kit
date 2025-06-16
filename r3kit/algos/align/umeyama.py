from typing import Tuple, Optional
import numpy as np


def umeyama_align(sources:np.ndarray, targets:np.ndarray, with_scale:bool=False, return_aligned:bool=False) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    '''
    sources, targets: (N, 3) or (N, 4, 4)
    '''
    assert sources.shape == targets.shape
    is_pose = len(sources.shape[1:]) == 2 and len(sources.shape[1:]) == 2
    if is_pose:
        t_src = sources[:, :3, 3]
        t_tgt = targets[:, :3, 3]
    else:
        t_src = sources
        t_tgt = targets
    
    mu_src = np.mean(t_src, axis=0)
    mu_tgt = np.mean(t_tgt, axis=0)

    src_centered = t_src - mu_src
    tgt_centered = t_tgt - mu_tgt

    cov = src_centered.T @ tgt_centered / sources.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S_mat = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S_mat[2, 2] = -1

    R = U @ S_mat @ Vt

    if with_scale:
        var_src = np.sum(src_centered ** 2) / sources.shape[0]
        s = np.sum(D * np.diag(S_mat)) / var_src
    else:
        s = 1.0

    t = mu_tgt - s * R @ mu_src

    if return_aligned:
        if is_pose:
            aligned_sources = []
            for pose in sources:
                aligned_pose = np.eye(4)
                aligned_pose[:3, :3] = R @ pose[:3, :3]
                aligned_pose[:3, 3] = s * (R @ pose[:3, 3]) + t
                aligned_sources.append(aligned_pose)
            aligned_sources = np.stack(aligned_sources)
        else:
            aligned_sources = s * (R @ sources.T).T + t
    else:
        aligned_sources = None
    return R, t, s, aligned_sources
