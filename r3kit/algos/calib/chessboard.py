from typing import Tuple, Dict, Optional
import numpy as np
import cv2

from r3kit.algos.calib.config import *
from r3kit.algos.calib.utils import rodrigues_rvec2mat


class ChessboardExtCalibor(object):
    CRITERIA:tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    def __init__(self, pattern_size:Tuple[int, int]=CHESSBOARD_PATTERN_SIZE, square_size:float=CHESSBOARD_SQUARE_SIZE) -> None:
        '''
        pattern_size: (block_num_x - 1, block_num_y - 1)
        square_size: the size of each block in mm
        '''
        self.pattern_size = pattern_size
        self.square_size = square_size

        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = square_size * np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        for i in range(pattern_size[0] * pattern_size[1]):
            x, y = self.objp[i, 0], self.objp[i, 1]
            self.objp[i, 0], self.objp[i, 1] = y, x
        
        self.obj_points = []    # 3D points in real world space
        self.img_points = []    # 2D points in image plane
    
    def add_image(self, img:np.ndarray, vis:bool=True) -> bool:
        '''
        img: the image of chessboard in [0, 255] (h, w, 3) BGR
        ret: whether detected
        '''
        if not hasattr(self, 'image_size'):
            self.image_size = img.shape[:2]
        else:
            assert img.shape[:2] == self.image_size, f'Image size {img.shape[:2]} does not match the camera intrinsics {self.image_size}'
        
        _img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, CHESSBOARD_CORNER_WINDOW_SIZE, (-1, -1), self.CRITERIA)
            
            self.obj_points.append(self.objp.copy())
            self.img_points.append(corners2)

            if vis:
                cv2.drawChessboardCorners(_img, self.pattern_size, corners2, ret)
                cv2.imshow('chessboard', _img)
                cv2.waitKey(500)
        return ret
    
    def run(self, intrinsics:Optional[np.ndarray]=None, opt_intrinsics:bool=True, opt_distortion:bool=False) -> Optional[Dict[str, np.ndarray]]:
        '''
        extrinsics: Nx4x4 transformation matrices from world to camera in the detected added order
        intrinsics: 3x3 camera intrinsic matrix
        error: reprojection error in pixel
        '''
        assert len(self.obj_points) >= 1, "Not enough valid images for chessboard calibration."

        if opt_distortion:
            distortion = None
            flags = 0
        else:
            distortion = np.zeros((1, 8), dtype=np.float32)
            flags = (
                cv2.CALIB_ZERO_TANGENT_DIST |
                cv2.CALIB_FIX_K1 |
                cv2.CALIB_FIX_K2 |
                cv2.CALIB_FIX_K3 |
                cv2.CALIB_FIX_K4 |
                cv2.CALIB_FIX_K5 |
                cv2.CALIB_FIX_K6
            )
        if opt_intrinsics:
            if intrinsics is None:
                pass
            else:
                flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        else:
            assert intrinsics is not None, "Intrinsics must be provided if opt_intrinsics is False."
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        
        ret, mtx, _dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.image_size[::-1], intrinsics, distortion, flags=flags)
        if ret:
            w2c = []
            for rvec, tvec in zip(rvecs, tvecs):
                mat = rodrigues_rvec2mat(rvec, tvec / 1000.0)
                w2c.append(mat.astype(np.float32))
            extrinsics = np.stack(w2c, axis=0)
            return {
                "extrinsics": extrinsics,
                "intrinsics": mtx.astype(np.float32),
                "error": ret
            }
        return None


if __name__ == '__main__':
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./calib_data')
    args = parser.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'image_*.png')))
    
    calibor = ChessboardExtCalibor(pattern_size=(11, 8), square_size=10)

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        calibor.add_image(img, vis=True)
    
    result = calibor.run(opt_distortion=False, opt_intrinsics=False, intrinsics=np.array([[605.95, 0., 312.68], [0., 604.91, 232.92], [0., 0., 1.]]))
    np.set_printoptions(precision=4, suppress=True)
    print(result)
    print(np.linalg.inv(result['extrinsics'][0]))
    import pdb; pdb.set_trace()
