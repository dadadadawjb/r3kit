import os
from typing import List, Tuple
from rich import print
from tap import Tap
import numpy as np
import cv2

from r3kit.devices.encoder.pdcd.angler import Angler
from r3kit.devices.camera.realsense.general import RealSenseCamera
from r3kit.algos.kinematics.urdf import URDFKinematics
from r3kit.algos.calib.handeye import HandEyeCalibor
from r3kit.utils.vis import Sequence2DVisualizer, Sequence3DVisualizer
from r3kit.utils.transformation import transform_pc, transform_dir
from r3kit.utils.transformation import delta_xyz, delta_dir

class ArgumentParser(Tap):
    encoder_id: str = '/dev/ttyUSB0'
    encoder_index: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    encoder_baudrate: int = 1000000
    encoder_name: str = 'Angler'

    camera_id: str = '319522062799'
    camera_streams: List[Tuple[str, int, int, int, int]] = [('color', -1, 640, 480, 30)]
    camera_name: str = 'D415'

    urdf_path: str = './urdf/exo.urdf'
    end_link: str = 'flange'
    base_link: str = 'link0'

    calib_type: str = 'aruco'  # 'chessboard' or 'aruco'
    # calib_params: dict = {'pattern_size': (11, 8), 'square_size': 10}
    calib_params: dict = {'dict_type': '6x6_1000', 'marker_length': 80}

    gt_xyz: Tuple[float, float, float] = (1.292, 0.0025, 0.665)
    gt_dx: Tuple[float, float, float] = (0., 1., 0.)
    gt_dy: Tuple[float, float, float] = (np.cos(35./180.*np.pi), 0., -np.sin(35./180.*np.pi))
    gt_dz: Tuple[float, float, float] = (-np.sin(35./180.*np.pi), 0., -np.cos(35./180.*np.pi))
    save_path: str = "./data"
    gui: bool = False


def joint_mapping(angles:np.ndarray, middles:np.ndarray=np.array([77.08, 318.34, 285.21, 207.33, 238.27, 348.22, 312.89, 79.5])) -> Tuple[np.ndarray, float]:
    assert len(angles) == 8

    joints = angles[:7].copy()
    joints[5] += 90.0
    joints = (joints - middles[:7] + 180.0) % 360.0 - 180.
    joints[5] -= 90.0
    joints *= -1.0 * np.pi / 180.

    width = angles[7:].item()
    width = (width - middles[7]) * np.pi / 180. * 18. / 1000.
    if width < 0:
        width = 0

    return (joints, width)


def main(args:ArgumentParser):
    exist_data = os.path.exists(args.save_path)
    print(f"Exist data: {exist_data}")

    # initialize devices
    if not exist_data:
        encoder = Angler(id=args.encoder_id, index=args.encoder_index, fps=0, baudrate=args.encoder_baudrate, 
                         gap=-1, strict=True, name=args.encoder_name)
        camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)

        os.makedirs(args.save_path, exist_ok=True)

    # initialize kinematics
    urdf = URDFKinematics(urdf_path=args.urdf_path, end_link=args.end_link, base_link=args.base_link)
    
    # initialize calibor
    calibor = HandEyeCalibor(marker_type=args.calib_type, ext_calib_params=args.calib_params)

    # initialize visualizers
    if args.gui:
        vis2d = Sequence2DVisualizer()
        vis3d = Sequence3DVisualizer()

    # loop
    i = 0
    while True:
        print(f"{i}th")

        # get data
        if not exist_data:
            angles = encoder.get()['angle']
            color = camera.get()['color']
        else:
            angles = np.load(os.path.join(args.save_path, f'angle_{i}.npy'))
            color = cv2.imread(os.path.join(args.save_path, f'rgb_{i}.png'), cv2.IMREAD_COLOR)

        # forward kinematics
        joints, width = joint_mapping(angles)
        tcp_pose = urdf.fk(joints)

        # visualization
        if args.gui:
            vis2d.update_image(name='color', image=color, type='bgr')
            vis3d.update_urdf(name='exo', path=args.urdf_path, joints=joints)
            vis3d.update_view()

        # add data
        if not exist_data:
            cmd = input("whether save? (y/n): ")
            if cmd == 'y':
                np.save(os.path.join(args.save_path, f'angle_{i}.npy'), angles)
                cv2.imwrite(os.path.join(args.save_path, f"rgb_{i}.png"), color)
                calibor.add_image_pose(color, tcp_pose, vis=args.gui)
                i += 1
            elif cmd == 'n':
                cmd = input("whether quit? (y/n): ")
                if cmd == 'y':
                    break
                elif cmd == 'n':
                    pass
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            calibor.add_image_pose(color, tcp_pose, vis=args.gui)
            i += 1
            if not os.path.exists(os.path.join(args.save_path, f'angle_{i}.npy')):
                break

    # run calibration
    if not exist_data:
        intrinsics = camera.color_intrinsics
    else:
        intrinsics = np.loadtxt(os.path.join(args.save_path, 'intrinsics.txt'))
    K = np.array([[intrinsics[2], 0., intrinsics[0]], [0., intrinsics[3], intrinsics[1]], [0., 0., 1.]])
    result = calibor.run(intrinsics=K, opt_intrinsics=False, opt_distortion=False)
    e2m = result['b2w']
    b2c = result['g2c']
    error = result['error']
    c2b = np.linalg.inv(b2c)

    # results
    print(f"c2b: {c2b}")
    print(f"error: {error}")
    xyz = transform_pc(np.array([[0., 0., 0.]]), c2b)[0]
    dx = transform_dir(np.array([[1., 0., 0.]]), c2b)[0]
    dy = transform_dir(np.array([[0., 1., 0.]]), c2b)[0]
    dz = transform_dir(np.array([[0., 0., 1.]]), c2b)[0]
    gt_xyz = np.array(gt_xyz)
    gt_x = np.array(gt_x)
    gt_y = np.array(gt_y)
    gt_z = np.array(gt_z)
    print("xyz", delta_xyz(xyz, gt_xyz))
    print("dx", delta_dir(dx, gt_x) * 180.0 / np.pi)
    print("dy", delta_dir(dy, gt_y) * 180.0 / np.pi)
    print("dz", delta_dir(dz, gt_z) * 180.0 / np.pi)
    np.savetxt(os.path.join(args.save_path, 'extrinsics.txt'), c2b, fmt="%.16f")
    np.savetxt(os.path.join(args.save_path, 'intrinsics.txt'), intrinsics, fmt="%.16f")


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
