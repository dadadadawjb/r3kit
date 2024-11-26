import os
import numpy as np

FLEXIV_RDK_PATH = os.path.join(os.path.dirname(__file__), "flexiv_rdk", "lib_py")

RIZON_ID = 'Rizon4s-123456'
RIZON_OPERATIONAL_WAIT_TIME = 10
RIZON_HOME_JOINTS = np.deg2rad([0, -40, 0, 90, 0, 40, 0])
RIZON_HOME_POSE = np.eye(4) # TODO: figure out the home pose
RIZON_JOINT_MAX_VEL = 1.0
RIZON_JOINT_MAX_ACC = 5.0
RIZON_TCP_MAX_VEL = (0.5, 1.0)
RIZON_TCP_MAX_ACC = (2.0, 5.0)
RIZON_JOINT_EPSILON = 0.02
RIZON_GRIPPER_EPSILON = 0.01
RIZON_TCP_POSE_EPSILON = (0.01, 0.01)
RIZON_BLOCK_WAIT_TIME = 0.01
