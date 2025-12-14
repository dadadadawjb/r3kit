import os
from typing import Tuple, List
import json
import time
from rich import print
from tap import Tap
import numpy as np

from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.devices.gripper.xense.xense import Xense
from r3kit.devices.camera.realsense.general import RealSenseCamera
from r3kit.utils.buffer import ObsBuffer, ActBuffer
from r3kit.utils.vis import SequenceKeyboardListener
from r3kit.utils.vis import Sequence3DVisualizer, Sequence2DVisualizer, Sequence1DVisualizer

class ArgumentParser(Tap):
    robot_id: str = 'Rizon4s-063231'
    robot_name: str = 'Rizon4s'

    gripper_id: str = '5e77ff097831'
    gripper_name: str = 'Xense'

    camera_id: str = '104122061018'
    camera_streams: List[Tuple[str, int, int, int, int]] = [('color', -1, 640, 480, 30)]
    camera_name: str = 'D415'

    mode: str = 'joint'     # 'joint' or 'tcp'
    block: bool = False
    wait_time: float = 0.1

    num_obs: int = 1
    num_actions: int = 20
    num_cmd: int = 10
    num_steps: int = -1
    sleep_time: float = 0.01

    meta_path: str = './.meta/'
    vis: bool = False
    vis_urdf_path: str = './urdf/rizon4s.urdf'


def main(args:ArgumentParser):
    obs_dict, act_dict = {}, {}
    # initialize robot
    robot = Rizon(id=args.robot_id, gripper=False, name=args.robot_name)
    robot.motion_mode(args.mode)
    robot.block(args.block)
    if args.mode == 'joint':
        obs_dict['joints'] = ((7,), np.float32.__name__)
        act_dict['joints'] = ((7,), np.float32.__name__)
    elif args.mode == 'tcp':
        obs_dict['pose'] = ((4, 4), np.float32.__name__)
        act_dict['pose'] = ((4, 4), np.float32.__name__)
    else:
        raise NotImplementedError
    gripper = Xense(id=args.gripper_id, name=args.gripper_name)
    gripper.block(args.block)
    obs_dict['width'] = ((1,), np.float32.__name__)
    act_dict['width'] = ((1,), np.float32.__name__)
    # initialize camera
    camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)
    obs_dict['color'] = (camera.color_image_shape, np.uint8.__name__)
    # initialize buffers
    obs_buffer = ObsBuffer(num_obs=args.num_obs, obs_dict=obs_dict, create=True)
    act_buffer = ActBuffer(num_act=args.num_actions, act_dict=act_dict, create=True)
    os.makedirs(args.meta_path, exist_ok=True)
    with open(os.path.join(args.meta_path, 'obs_dict.json'), 'w') as f:
        json.dump(obs_dict, f, indent=4)
    with open(os.path.join(args.meta_path, 'act_dict.json'), 'w') as f:
        json.dump(act_dict, f, indent=4)
    print("=========> Initialized")
    
    # rollout
    if args.num_steps < 0:
        listener = SequenceKeyboardListener(verbose=False)
    if args.vis:
        visualizer3d = Sequence3DVisualizer(left=0, top=0)
        visualizer2d = Sequence2DVisualizer(left=0, top=400)
        visualizer1d = Sequence1DVisualizer(width=320, height=240, left=640, top=0)
    step_idx = 0
    quit = False
    while not quit:
        if step_idx % args.num_cmd == 0:
            # get obs
            o = {}
            if args.mode == 'joint':
                joints = robot.joint_read()
                o['joints'] = joints
            elif args.mode == 'tcp':
                pose = robot.tcp_read()
                o['pose'] = pose
            else:
                raise NotImplementedError
            width = gripper.read()
            o['width'] = np.array([width], dtype=np.float32)
            color_image = camera.get()["color"]
            o['color'] = color_image
            # visualize
            if args.vis:
                visualizer1d.update_item('width_o', item=np.array([width], dtype=np.float32))
                visualizer2d.update_image('color_o', image=color_image, type='bgr')
                if args.mode == 'joint':
                    visualizer3d.update_urdf('joints_o', path=args.vis_urdf_path, joints=joints)
                elif args.mode == 'tcp':
                    visualizer3d.update_frame('pose_o', pose=pose)
                else:
                    raise NotImplementedError
                visualizer3d.update_view()
            # add obs
            act_buffer.setf(False)
            obs_buffer.add1(o)
            obs_buffer.setf(True)
            print(f"=========> Add obs {step_idx}")
        
        # get act
        while not act_buffer.getf():
            time.sleep(args.sleep_time)
        a = act_buffer.get1()
        if args.mode == 'joint':
            joints = a['joints']
        elif args.mode == 'tcp':
            pose = a['pose']
        else:
            raise NotImplementedError
        width = a['width'][0]
        print(f"=========> Get act {step_idx}")
        # visualize
        if args.vis:
            visualizer1d.update_item('width_a', item=np.array([width], dtype=np.float32))
            if args.mode == 'joint':
                visualizer3d.update_urdf('joints_a', path=args.vis_urdf_path, joints=joints)
            elif args.mode == 'tcp':
                visualizer3d.update_frame('pose_a', pose=pose)
            else:
                raise NotImplementedError
            visualizer3d.update_view()
        # execute act
        if args.mode == 'joint':
            robot.joint_move(joints)
        elif args.mode == 'tcp':
            robot.tcp_move(pose)
        else:
            raise NotImplementedError
        gripper.move(width)
        if not args.block:
            time.sleep(args.wait_time)

        # step
        if args.num_steps == -1:
            quit = listener.quit
        else:
            quit = step_idx >= args.num_steps
        step_idx += 1


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
