import os
import numpy as np
import cv2

from r3kit.devices.encoder.pdcd.angler import Angler
from r3kit.devices.camera.realsense.d415 import D415
from r3kit.algos.kinematics.urdf import URDFKinematics
from r3kit.algos.calib.handeye import HandEyeCalibor
from r3kit.utils.transformation import transform_pc, transform_dir
from r3kit.utils.transformation import delta_xyz, delta_dir


def mapping(angles:np.ndarray, middles:np.ndarray=np.array([77.08, 318.34, 285.21, 207.33, 238.27, 348.22, 312.89])) -> np.ndarray:
    # angles as degrees
    # output as radians
    angles[5] += 90.0
    angles = (angles - middles + 180.0) % 360.0 - 180.0
    angles[5] -= 90.0
    angles *= -1.0
    return angles * np.pi / 180.0


encoder_id = '/dev/ttyUSB0'
encoder_index = (1, 2, 3, 4, 5, 6, 7, 8)
encoder_baudrate = 1000000
encoder_name = 'Angler'

camera_id = '319522062799'
camera_depth = False
camera_name = 'D415'

save_path = './data'
urdf_path = "./urdf/exo.urdf"
end_link = "flange"
base_link = "link0"

gt_xyz = (1.292, 0.0025, 0.665)
gt_dx = (0., 1., 0.)
gt_dy = (np.cos(35./180.*np.pi), 0., -np.sin(35./180.*np.pi))
gt_dz = (-np.sin(35./180.*np.pi), 0., -np.cos(35./180.*np.pi))

gui = False


def main():
    exist_data = os.path.exists(save_path)
    print(f"Exist data: {exist_data}")

    # initialize devices
    if not exist_data:
        encoder = Angler(id=encoder_id, index=encoder_index, fps=0, baudrate=encoder_baudrate, 
                         gap=-1, strict=True, name=encoder_name)
        camera = D415(id=camera_id, depth=camera_depth, name=camera_name)

        os.makedirs(save_path, exist_ok=True)

    # initialize kinematics
    urdf = URDFKinematics(urdf_path=urdf_path, end_link=end_link, base_link=base_link)
    
    # initialize calibor
    # ext_calib_params = {'pattern_size': (11, 8), 'square_size': 10}
    ext_calib_params = {'dict_type': '6x6_1000', 'marker_length': 80}
    calibor = HandEyeCalibor(marker_type='aruco', ext_calib_params=ext_calib_params)

    i = 0
    while True:
        print(f"{i}th")

        # get data
        if not exist_data:
            data = encoder.get()
            color, _depth = camera.get()
            
            angle = np.copy(data['angle'])
            angle = mapping(angle)
        else:
            angle = np.load(os.path.join(save_path, f'angle_{i}.npy'))
            color = cv2.imread(os.path.join(save_path, f'rgb_{i}.png'), cv2.IMREAD_COLOR)

        # forward kinematics
        tcp_pose = urdf.fk(angle)

        # visualization
        if gui:
            cv2.imshow('color', color)
            while True:
                if cv2.getWindowProperty('color', cv2.WND_PROP_VISIBLE) <= 0:
                    break
                cv2.waitKey(1)
            cv2.destroyAllWindows()

        if not exist_data:
            cmd = input("whether save? (y/n): ")
            if cmd == 'y':
                np.save(os.path.join(save_path, f'angle_{i}.npy'), angle)
                cv2.imwrite(os.path.join(save_path, f"rgb_{i}.png"), color)
                calibor.add_image_pose(color, tcp_pose, vis=gui)
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
            calibor.add_image_pose(color, tcp_pose, vis=gui)
            i += 1
            if not os.path.exists(os.path.join(save_path, f'angle_{i}.npy')):
                break

    # run calibration
    if not exist_data:
        intrinsics = camera.color_intrinsics
    else:
        intrinsics = np.loadtxt(os.path.join(save_path, 'intrinsics.txt'))
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
    print("x", delta_dir(dx, gt_x) * 180.0 / np.pi)
    print("y", delta_dir(dy, gt_y) * 180.0 / np.pi)
    print("z", delta_dir(dz, gt_z) * 180.0 / np.pi)
    np.savetxt(os.path.join(save_path, 'extrinsics.txt'), c2b, fmt="%.16f")
    np.savetxt(os.path.join(save_path, 'intrinsics.txt'), intrinsics, fmt="%.16f")


if __name__ == '__main__':
    main()
