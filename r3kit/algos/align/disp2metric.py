from typing import Tuple
import numpy as np


def disp2metric(gt_metric:np.ndarray, pred_disp:np.ndarray) -> Tuple[float, float, np.ndarray]:
    '''
    gt_metric, pred_disp: (N,) or (N, 1) or (H, W, 1) or (H, W)
    1.0 / gt_metric = b + s * pred_disp
    '''
    metric = gt_metric.reshape((-1, 1))
    disp = pred_disp.reshape((-1, 1))
    b_s = np.hstack([np.ones_like(disp), disp]) # (N, 2)
    target = 1.0 / metric                       # (N, 1)

    solution, _, _, _ = np.linalg.lstsq(b_s, target, rcond=None)    # b_s @ [b, s] ≈ 1 / metric
    b, s = solution[:2].squeeze()
    pred_metric = 1.0 / (b + s * pred_disp)
    return float(b), float(s), pred_metric
