from typing import Optional, List, Union, Dict
import numpy as np
import cv2
from cv2 import aruco

from r3kit.algos.calib.config import *
from r3kit.algos.calib.utils import rodrigues_rvec2mat


def resolve_aruco_dict_id(dict_type:Union[str, int]) -> int:
    if isinstance(dict_type, int):
        return dict_type
    if not isinstance(dict_type, str):
        raise TypeError(f"dict_type must be str or int, got {type(dict_type)}.")

    name = dict_type.upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name

    if not hasattr(aruco, name):
        available = [k for k in dir(aruco) if k.startswith("DICT_")]
        raise ValueError(
            f"Unknown ArUco dictionary name '{dict_type}'. "
            f"Resolved name '{name}' not found in cv2.aruco. "
            f"Available examples: {available[:10]} ..."
        )
    return getattr(aruco, name)


class ArucoExtCalibor(object):
    CRITERIA:tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, dict_type:Union[str, int]=ARUCO_DICT_TYPE, marker_length:float=ARUCO_MARKER_LENGTH) -> None:
        """
        dict_type: aruco dictionary spec. Examples: "DICT_4X4_50", "4x4_50", or an OpenCV enum int.
        marker_length: the size of the marker in mm.
        """
        dict_id = resolve_aruco_dict_id(dict_type)
        self.dictionary = aruco.getPredefinedDictionary(dict_id)

        self.detector_params = aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        self.detector_params.cornerRefinementWinSize = 5

        self.detector = aruco.ArucoDetector(self.dictionary, self.detector_params)

        # 3D template points for a single marker (4 corners, in mm)
        self.marker_length = marker_length
        L = marker_length
        self.objp_marker = np.array(
            [
                [0.0, 0.0, 0.0],
                [L, 0.0, 0.0],
                [L, L, 0.0],
                [0.0, L, 0.0],
            ],
            dtype=np.float32,
        )

        self.obj_points = []    # 3D points in real world space
        self.img_points = []    # 2D points in image plane

    def add_image(self, img:np.ndarray, vis:bool=True) -> bool:
        '''
        img: the image of chessboard in [0, 255] (h, w, 3) BGR
        ret: whether detected
        '''
        if not hasattr(self, "image_size"):
            self.image_size = img.shape[:2]
        else:
            assert img.shape[:2] == self.image_size, f'Image size {img.shape[:2]} does not match the camera intrinsics {self.image_size}'
        
        _img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        ret = ids is not None and len(ids) > 0
        if ret:
            assert len(ids) == 1, "Only one marker per image is expected."

            corners2 = corners[0].reshape(-1, 2).astype(np.float32)
            corners2 = cv2.cornerSubPix(gray, corners2, ARUCO_CORNER_WINDOW_SIZE, (-1, -1), self.CRITERIA)

            self.obj_points.append(self.objp_marker.copy())
            self.img_points.append(corners2)

            if vis:
                aruco.drawDetectedMarkers(_img, corners, ids, (0, 255, 0))
                cv2.imshow("aruco", _img)
                cv2.waitKey(500)
        return ret

    def run(self, intrinsics:Optional[np.ndarray]=None, opt_intrinsics:bool=True, opt_distortion:bool=False) -> Optional[Dict[str, np.ndarray]]:
        '''
        extrinsics: Nx4x4 transformation matrices from world to camera in the detected added order
        intrinsics: 3x3 camera intrinsic matrix
        error: reprojection error in pixel
        '''
        if opt_intrinsics and opt_distortion:
            if len(self.obj_points) < ARUCO_CALIB_FULL_MIN_NUM:
                raise RuntimeError(f"Not enough valid images for Aruco calibration. Minimum required is {ARUCO_CALIB_FULL_MIN_NUM}, but got {len(self.obj_points)}.")
        elif not opt_intrinsics and not opt_distortion:
            assert len(self.obj_points) >= 1, "Not enough valid images for Aruco calibration."
        else:
            if len(self.obj_points) < ARUCO_CALIB_HALF_MIN_NUM:
                raise RuntimeError(f"Not enough valid images for Aruco calibration. Minimum required is {ARUCO_CALIB_HALF_MIN_NUM}, but got {len(self.obj_points)}.")

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
    
    calibor = ArucoExtCalibor(dict_type="6X6_1000", marker_length=80)

    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        calibor.add_image(img, vis=True)
    
    result = calibor.run(opt_distortion=False, opt_intrinsics=False, intrinsics=np.array([[605.95, 0., 312.68], [0., 604.91, 232.92], [0., 0., 1.]]))
    np.set_printoptions(precision=4, suppress=True)
    print(result)
    print(np.linalg.inv(result['extrinsics'][0]))
    import pdb; pdb.set_trace()
