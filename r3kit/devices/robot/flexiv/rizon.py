import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from r3kit.devices.robot.base import RobotBase
from r3kit.devices.robot.flexiv.config import *
from r3kit.devices.robot.flexiv.utils import parse_pt_states

try:
    import sys
    sys.path.insert(0, FLEXIV_RDK_PATH)
    import flexivrdk
except ImportError:
    print("Robot Flexiv Rizon needs `flexivrdk`")
    sys.exit(1)


class Rizon(RobotBase):
    DOF:int = 7

    def __init__(self, robot_ip:str=RIZON_ROBOT_IP, local_ip:str=RIZON_LOCAL_IP, name:str='Rizon', verbose:bool=True) -> None:
        super().__init__(name)

        self.verbose = verbose
        if self.verbose:
            self.log = flexivrdk.Log()
        self.mode = flexivrdk.Mode
        self.robot = flexivrdk.Robot(robot_ip, local_ip)
        if self.robot.isFault():
            if self.verbose:
                self.log.warn("Fault occurred on robot server, trying to clear ...")
            self.robot.clearFault()
            time.sleep(RIZON_CLEAR_FAULT_WAIT_TIME)
            if self.robot.isFault():
                if self.verbose:
                    self.log.error("Fault cannot be cleared, exiting ...")
                raise ValueError
            if self.verbose:
                self.log.info("Fault on robot server is cleared")
        if self.verbose:
            self.log.info("Enabling robot ...")
        self.robot.enable()
        seconds_waited = 0
        while not self.robot.isOperational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == RIZON_OPERATIONAL_WAIT_TIME:
                if self.verbose:
                    self.log.warn(
                        "Still waiting for robot to become operational, please check that the robot 1) "
                        "has no fault, 2) is in [Auto (remote)] mode")
                raise ValueError
        if self.verbose:
            self.log.info("Robot is now operational")
        self.robot.setMode(self.mode.NRT_PRIMITIVE_EXECUTION)
    
    def homing(self) -> None:
        if self.verbose:
            self.log.info("Executing primitive: Home")
        self.robot.executePrimitive("Home()")
        while (self.robot.isBusy()):
            time.sleep(1)
    
    def joint_read(self) -> np.ndarray:
        '''
        joints: 7 DoF joint angles in radian
        '''
        robot_states = flexivrdk.RobotStates()
        self.robot.getRobotStates(robot_states)
        joints = np.deg2rad(np.array(robot_states.q))
        return joints
    
    def joint_move(self, joints:np.ndarray, relative:bool=False) -> None:
        '''
        joints: 7 DoF joint angles in radian
        relative: if True, move relative to current joints; if False, move absolutely
        '''
        if not relative:
            action = np.rad2deg(joints).tolist()
        else:
            current_joints = self.joint_read()
            action = np.rad2deg(current_joints + joints).tolist()

        self.robot.executePrimitive("MoveJ(target={} {} {} {} {} {} {})".format(*action))
        while (parse_pt_states(self.robot.getPrimitiveStates(), "reachedTarget") != "1"):
            time.sleep(1)
    
    def flange_read(self) -> np.ndarray:
        '''
        f2b: 4x4 transformation matrix from flange to robot base
        '''
        robot_states = flexivrdk.RobotStates()
        self.robot.getRobotStates(robot_states)
        flange_pose = robot_states.flangePose

        f2b = np.identity(4)
        ori = flange_pose[3:]       # (w, x, y, z)
        rot = Rot.from_quat([ori[1], ori[2], ori[3], ori[0]]).as_matrix()
        f2b[:3, :3] = rot
        f2b[:3, 3] = np.array(flange_pose[:3])
        return f2b


if __name__ == "__main__":
    robot = Rizon(robot_ip='192.168.2.112', local_ip='192.168.2.220', name='Rizon', verbose=True)

    robot.homing()
    print("homing")
    joint = robot.joint_read()
    print("current joint:", joint)
    pose = robot.flange_read()
    print("current pose:", pose)
