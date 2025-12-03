from typing import Dict, Optional
import numpy as np
import cv2

from r3kit.utils.transformation import delta_mat
from r3kit.algos.calib.chessboard import ChessboardExtCalibor
from r3kit.algos.calib.aruco import ArucoExtCalibor
from r3kit.algos.calib.config import *


class HandEyeCalibor(object):
    def __init__(self, marker_type:str=HANDEYE_MARKER_TYPE, 
                 ext_calib_params:dict={'patter_size': CHESSBOARD_PATTERN_SIZE, 'square_size': CHESSBOARD_SQUARE_SIZE}) -> None:
        if marker_type == 'chessboard':
            self.ext_calibor = ChessboardExtCalibor(**ext_calib_params)
        elif marker_type == 'aruco':
            self.ext_calibor = ArucoExtCalibor(**ext_calib_params)
        else:
            raise NotImplementedError
        
        self.b2g = []
    
    def add_image_pose(self, img:np.ndarray, pose:np.ndarray, vis:bool=True) -> bool:
        '''
        img: the image of chessboard in [0, 255] (h, w, 3) BGR
        pose: 4x4 transformation matrix from robot base to gripper in eye-in-hand mode, from gripper to robot base in eye-to-hand mode
        '''
        ret = self.ext_calibor.add_image(img, vis)
        if ret:
            self.b2g.append(pose)
        return ret
    
    def run(self, intrinsics:Optional[np.ndarray]=None, opt_intrinsics:bool=True, opt_distortion:bool=False) -> Optional[Dict[str, np.ndarray]]:
        '''
        b2w: 4x4 transformation matrix from robot base to world in eye-in-hand mode, from gripper to world in eye-to-hand mode
        g2c: 4x4 transformation matrix from gripper to camera in eye-in-hand mode, from robot base to camera in eye-to-hand mode
        error: (reprojection error, translation error, rotation error)
        '''
        result = self.ext_calibor.run(intrinsics, opt_intrinsics, opt_distortion)
        if result is None:
            return None
        
        w2c = result['extrinsics']
        R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(w2c[:, :3, :3], w2c[:, :3, 3], 
                                                                    np.array(self.b2g)[:, :3, :3], np.array(self.b2g)[:, :3, 3])
        b2w = np.eye(4)
        b2w[:3, :3] = R_b2w
        b2w[:3, 3:] = t_b2w
        g2c = np.eye(4)
        g2c[:3, :3] = R_g2c
        g2c[:3, 3:] = t_g2c

        trans_errors, rot_errors = [], []
        for w2c_i, b2g_i in zip(w2c, self.b2g):
            T_left = b2w @ w2c_i
            T_right = b2g_i @ g2c
            trans_error, rot_error = delta_mat(T_left, T_right)
            trans_errors.append(trans_error)
            rot_errors.append(rot_error)
        trans_error = np.mean(trans_errors)
        rot_error = np.mean(rot_errors)
        reproj_error = result['error']
        return {
            'b2w': b2w,
            'g2c': g2c,
            'error': np.array([reproj_error, trans_error, rot_error])
        }


if __name__ == '__main__':
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./calib_data')
    parser.add_argument('--marker_type', type=str, default='chessboard')
    args = parser.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'image_*.png')))
    
    ext_calib_params = {'pattern_size': (11, 8), 'square_size': 15}
    calibor = HandEyeCalibor(marker_type=args.marker_type, ext_calib_params=ext_calib_params)

    b2g = np.load(os.path.join(args.data_dir, 'b2g.npy'))
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        calibor.add_image_pose(img, b2g[idx], vis=True)
    
    result = calibor.run()
    np.set_printoptions(precision=4, suppress=True)
    print(result)
    import pdb; pdb.set_trace()
