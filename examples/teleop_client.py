import numpy as np
from typing import Tuple
from tap import Tap

from r3kit.devices.base import DeviceBase
from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.devices.gripper.xense.xense import Xense
from r3kit.algos.kinematics.urdf import URDFKinematics
from r3kit.utils.vis import SequenceKeyboardListener

class ArgumentParser(Tap):
    robot_id: str = 'Rizon4s-063231'
    gripper_id: str = '5e77ff097831'
    mode: str = 'joint'     # 'joint' or 'tcp'
    urdf_path: str = './urdf/rizon4s.urdf'  # only used if mode is 'tcp'
    end_link: str = 'flange'                # only used if mode is 'tcp'
    base_link: str = 'link0'                # only used if mode is 'tcp'
    shm_name: str = 'Angler'


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

def tcp_mapping(angles:np.ndarray, urdf_path:str, end_link:str, base_link:str) -> Tuple[np.ndarray, float]:
    joints, width = joint_mapping(angles)

    if not hasattr(tcp_mapping, 'robot'):
        robot = URDFKinematics(urdf_path, end_link=end_link, base_link=base_link)
        tcp_mapping.robot = robot
    else:
        robot = tcp_mapping.robot
    pose = robot.fk(joints)

    return (pose, width)


def main(args:ArgumentParser):
    # initialize robot
    robot = Rizon(id=args.robot_id, gripper=False)
    robot.motion_mode(args.mode)
    robot.block(blocking=False)
    gripper = Xense(id=args.gripper_id)
    gripper.block(blocking=False)

    # read shm
    angler = DeviceBase()
    angler_streaming_array, angler_shared_memory, angler_streaming_lock = angler.get_streaming_memory(shm=args.shm_name)

    # collect
    keyboard_listener = SequenceKeyboardListener(verbose=False)
    while not keyboard_listener.quit:
        with angler_streaming_lock:
            angles = np.copy(angler_streaming_array["angle"])
            timestamp_ms = angler_streaming_array["timestamp_ms"].item()
        
        if args.mode == 'joint':
            joints, width = joint_mapping(angles)
            robot.joint_move(joints)
        elif args.mode == 'tcp':
            pose, width = tcp_mapping(angles, urdf_path=args.urdf_path, end_link=args.end_link, base_link=args.base_link)
            robot.tcp_move(pose=pose)
        else:
            raise NotImplementedError
        gripper.move(width)
    keyboard_listener.stop()

    # disconnect
    angler_shared_memory.close()


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
