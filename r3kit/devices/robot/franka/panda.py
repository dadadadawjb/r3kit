from typing import Optional
import time
import numpy as np
import transformations as tf

try:
    from frankx import Robot, JointMotion, Affine, LinearMotion, ImpedanceMotion
except ImportError:
    print("Robot Franka Panda needs `frankx")
    raise ImportError

from r3kit.devices.robot.base import RobotBase
from r3kit.devices.gripper.base import GripperBase
from r3kit.devices.robot.franka.config import *


class Panda(RobotBase):
    DOF:int = 7

    def __init__(self, ip:str=PANDA_IP, gripper:Optional[GripperBase]=None, name:str='') -> None:
        super().__init__(name)
        
        self.robot = Robot(ip)
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(PANDA_DYNAMIC_REL)
        self.gripper = gripper

        self.in_impedance_control = False
    
    def homing(self) -> None:
        self.joint_move(PANDA_HOME_JOINTS, relative=False)
    
    def joint_read(self) -> np.ndarray:
        '''
        joints: 7 DoF joint angles in radian
        '''
        if not self.in_impedance_control:
            state = self.robot.read_once()
            joints = np.array(state.q)
        else:
            raise NotImplementedError
        return joints

    def joint_move(self, joints:np.ndarray, relative:bool=False) -> None:
        '''
        joints: 7 DoF joint angles in radian
        relative: if True, move relative to current joints; if False, move absolutely
        '''
        if not relative:
            action = JointMotion(joints.tolist())
        else:
            current_joints = self.joint_read().tolist()
            action = JointMotion([current_joints[i] + joints[i] for i in range(self.DOF)])
        
        if not self.in_impedance_control:
            self.robot.move(action)
        else:
            raise NotImplementedError

    def tcp_read(self) -> np.ndarray:
        '''
        tcp: 4x4 transformation matrix tcp2rb
        '''
        if not self.in_impedance_control:
            tcp = np.array(self.robot.read_once().O_T_EE).reshape(4, 4).T
        else:
            # NOTE: not supported in frankx, need to rebuild
            tcp = np.array(self.impedance_motion.get_robotstate().O_T_EE).reshape(4, 4).T
        return tcp
    
    def tcp_move(self, tcp:np.ndarray, relative:bool=False) -> None:
        '''
        tcp: 4x4 transformation matrix tcp2rb
        relative: if True, move relative to current pose; if False, move absolutely in robot base frame
        '''
        if not relative:
            tr = tcp[:3, 3]
            rot = tf.euler_from_matrix(tcp, axes='rzyx')
            if not self.in_impedance_control:
                action = LinearMotion(Affine(tr[0], tr[1], tr[2], rot[0], rot[1], rot[2]))
                self.robot.move(action)
            else:
                self.impedance_motion.target = Affine(tr[0], tr[1], tr[2], rot[0], rot[1], rot[2])
        else:
            raise NotImplementedError
    
    def start_impedance_control(self, tr_stiffness:float=PANDA_TR_STIFFNESS, rot_stiffness:float=PANDA_ROT_STIFFNESS) -> None:
        self.impedance_motion = ImpedanceMotion(tr_stiffness, rot_stiffness)
        self.robot_thread = self.robot.move_async(self.impedance_motion)
        time.sleep(0.5)
        self.in_impedance_control = True
    
    def end_impedance_control(self) -> None:
        self.impedance_motion.finish()
        self.robot_thread.join()
        self.impedance_motion = None
        self.robot_thread = None
        self.in_impedance_control = False


if __name__ == "__main__":
    robot = Panda(ip='172.16.0.2', name='panda', gripper=None)

    robot.homing()
    print("homing")
    joint = robot.joint_read()
    print("current joint:", joint)
    pose = robot.tcp_read()
    print("current pose:", pose)
    target_pose = pose.copy()
    target_pose[:3, 3] += np.array([0.05, 0.05, 0.05])
    robot.tcp_move(target_pose, relative=False)
    print("move")
    joint = robot.joint_read()
    print("current joint:", joint)
    pose = robot.tcp_read()
    print("current pose:", pose)
    robot.start_impedance_control(tr_stiffness=1000.0, rot_stiffness=20.0)
    print("start impedance control")
    for i in range(10):
        current_pose = robot.tcp_read()
        print(i, current_pose)
        target_pose = current_pose.copy()
        target_pose[:3, 3] += np.array([0., 0.02, 0.])
        robot.tcp_move(target_pose, relative=False)
        time.sleep(0.3)
    pose = robot.tcp_read()
    print("current pose:", pose)
    robot.end_impedance_control()
    robot.homing()
    print("homing")
