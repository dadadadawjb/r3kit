import os
import json
import time
import numpy as np
import cv2

from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.devices.camera.realsense.d415 import D415
from r3kit.utils.buffer import ObsBuffer, ActBuffer
from r3kit.utils.vis import SequenceKeyboardListener
from r3kit.utils.vis import Sequence2DVisualizer, Sequence1DVisualizer

robot_id = 'Rizon4s-063231'
robot_name = 'Rizon4s'
block = False
wait_time = 0.1

camera_id = '104122061018'
camera_name = 'D415'
image_size = 224

num_obs = 1
num_actions = 20
num_cmd = 10
num_steps = -1
sleep_time = 0.01

meta_path = './.meta/'
vis = True


def main():
    obs_dict, act_dict = {}, {}
    # initialize robot
    robot = Rizon(id=robot_id, gripper=False, name=robot_name)
    robot.motion_mode('joint')
    robot.block(block)
    obs_dict['joints'] = ((7,), np.float32.__name__)
    act_dict['joints'] = ((7,), np.float32.__name__)
    # initialize camera
    camera = D415(id=camera_id, depth=False, name=camera_name)
    obs_dict['color'] = ((image_size, image_size, 3), np.uint8.__name__)
    # initialize buffers
    obs_buffer = ObsBuffer(num_obs=num_obs, obs_dict=obs_dict, create=True)
    act_buffer = ActBuffer(num_act=num_actions, act_dict=act_dict, create=True)
    os.makedirs(meta_path, exist_ok=True)
    with open(os.path.join(meta_path, 'obs_dict.json'), 'w') as f:
        json.dump(obs_dict, f, indent=4)
    with open(os.path.join(meta_path, 'act_dict.json'), 'w') as f:
        json.dump(act_dict, f, indent=4)
    print("=========> Initialized")
    
    # rollout
    if num_steps < 0:
        listener = SequenceKeyboardListener(verbose=False)
    if vis:
        visualizer2d = Sequence2DVisualizer(left=0, top=400)
        visualizer1d = Sequence1DVisualizer(width=320, height=240, left=640, top=0)
    step_idx = 0
    quit = False
    while not quit:
        if step_idx % num_cmd == 0:
            # get obs
            o = {}
            joints = robot.joint_read()
            o['joints'] = joints
            if vis:
                visualizer1d.update_item('joints_o', item=joints)
            color_image, _ = camera.get()
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image = cv2.resize(color_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            o['color'] = color_image
            if vis:
                visualizer2d.update_image('color', image=color_image, type='rgb', scale=1)
            # add obs
            act_buffer.setf(False)
            obs_buffer.add1(o)
            obs_buffer.setf(True)
            print(f"=========> Add obs {step_idx}")
        
        # get act
        while not act_buffer.getf():
            time.sleep(sleep_time)
        a = act_buffer.get1()
        print(f"=========> Get act {step_idx}")
        # execute act
        joints = a['joints']
        if vis:
            visualizer1d.update_item('joints_a', item=joints)
        robot.joint_move(joints)
        if not block:
            time.sleep(wait_time)

        # step
        if num_steps == -1:
            quit = listener.quit
        else:
            quit = step_idx >= num_steps
        step_idx += 1


if __name__ == '__main__':
    main()
