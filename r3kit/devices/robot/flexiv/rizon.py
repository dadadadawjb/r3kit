from typing import List, Dict, Union, Optional
import time
import gc
from rich import print
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from threading import Thread, Lock, Event
from multiprocessing import shared_memory, Manager
from copy import deepcopy
from functools import partial
import flexivrdk
assert flexivrdk.__version__ == '1.8.0', "Only support Flexiv RDK v1.8.0, current version is {flexivrdk.__version__}"

from r3kit.devices.robot.base import RobotBase
from r3kit.devices.robot.flexiv.config import *
from r3kit.utils.transformation import xyzquat2mat, mat2xyzquat, delta_xyz, delta_quat
from r3kit.utils.vis import draw_time, draw_items
from r3kit import DEBUG, INFO


class Rizon(RobotBase):
    DOF:int = 7

    def __init__(self, id:str=RIZON_ID, fps:int=RIZON_FPS, gripper:bool=True, tool_name:str='Flange', name:str='Rizon') -> None:
        super().__init__(name)
        self._fps = fps

        self.robot = flexivrdk.Robot(id)
        if self.robot.fault():
            if not self.robot.ClearFault():
                raise RuntimeError(f"Failed to clear fault for robot {name}")
        self.robot.Enable()
        seconds_waited = 0
        while not self.robot.operational():
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited == RIZON_OPERATIONAL_WAIT_TIME:
                raise RuntimeError(f"Failed to enable robot {name}")
        self.robot.SwitchMode(flexivrdk.Mode.IDLE)
        tool = flexivrdk.Tool(self.robot)
        if tool.exist(tool_name):
            tool.Switch(tool_name)
        else:
            raise RuntimeError(f"{tool_name} tool not found")
        if INFO:
            print(f"[INFO-r3kit] {tool_name} tool switched")
        self.motion_mode('primitive')
        self.block(True)
        info = self.robot.info()
        self._joint_limits = (np.array(info.q_min), np.array(info.q_max))
        if gripper:
            self.gripper = flexivrdk.Gripper(self.robot)
            self.gripper.Init()
            info = self.gripper.states()
            self._gripper_limits = (0, float(info.max_width))
        else:
            self.gripper = None
        
        # config
        self.joints_dtype = np.dtype(np.float64)
        self.joints_shape = (self.DOF,)
        self.pose_dtype = np.dtype(np.float64)
        self.pose_shape = (4, 4)

        # stream
        self.in_streaming = Event()
    
    def __del__(self) -> None:
        if hasattr(self, 'robot') and self.robot is not None:
            self.robot.Stop()
        if hasattr(self, 'gripper') and self.gripper is not None:
            self.gripper.Stop()
    
    def sleep(self, sleep_time:float, wait_stopped_interval:float=0.001) -> None:
        while not self.robot.stopped():
            time.sleep(wait_stopped_interval)
        time.sleep(sleep_time)

    def motion_mode(self, mode:str) -> None:
        if mode == 'primitive':
            self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        elif mode == 'joint':
            self.robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_IMPEDANCE)
        elif mode == 'tcp':
            self.robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        else:
            raise ValueError(f"Invalid motion mode: {mode}")
        self.mode = mode
    
    def block(self, blocking:bool) -> None:
        self.blocking = blocking
    
    def homing(self) -> None:
        '''
        Move robot to home position
        '''
        if self.mode == 'primitive':
            self.robot.ExecutePrimitive("Home", dict())
            if self.blocking:
                time_start = time.time()
                while not self.robot.primitive_states()["reachedTarget"]:
                    if time.time() - time_start > RIZON_BLOCK_TIMEOUT:
                        return
                    time.sleep(RIZON_BLOCK_WAIT_TIME)
            else:
                pass
        elif self.mode == 'joint':
            self.joint_move(RIZON_HOME_JOINTS)
        elif self.mode == 'tcp':
            self.tcp_move(RIZON_HOME_POSE)
        else:
            raise ValueError(f"Invalid motion mode: {self.mode}")
    
    def _read(self) -> Dict[str, Union[float, np.ndarray]]:
        data = self.robot.states()
        receive_time = time.time() * 1000
        joints = np.array(data.q, dtype=np.float64)
        vec = np.array(data.tcp_pose, dtype=np.float64)
        xyz = vec[:3]
        quat = vec[3:]                                          # (w, x, y, z)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])   # (x, y, z, w)
        pose = xyzquat2mat(xyz, quat)
        result = {
            'joints': joints,
            'pose': pose,
            'timestamp_ms': receive_time
        }
        return result
    
    def joint_read(self) -> np.ndarray:
        '''
        joints: 7 DoF joint angles in radian
        '''
        joints = np.array(self.robot.states().q)
        return joints
    
    def joint_move(self, joints:np.ndarray, velocities:Optional[np.ndarray]=None) -> None:
        '''
        joints: 7 DoF joint angles in radian
        velocities: 7 DoF joint velocities in radian/s
        '''
        if velocities is None:
            velocities = np.array([0.0]*self.DOF)
        joints = np.clip(joints, self._joint_limits[0], self._joint_limits[1])
        max_vel = np.array([RIZON_JOINT_MAX_VEL]*self.DOF)
        max_acc = np.array([RIZON_JOINT_MAX_ACC]*self.DOF)
        
        if self.mode == 'primitive':
            self.robot.ExecutePrimitive("MoveJ", {"target": np.rad2deg(joints).tolist()})
            if self.blocking:
                time_start = time.time()
                while not self.robot.primitive_states()["reachedTarget"]:
                    if time.time() - time_start > RIZON_BLOCK_TIMEOUT:
                        return
                    time.sleep(RIZON_BLOCK_WAIT_TIME)
            else:
                pass
        elif self.mode == 'joint':
            self.robot.SendJointPosition(joints.tolist(), velocities.tolist(), max_vel.tolist(), max_acc.tolist())
            if self.blocking:
                error = float('inf')
                time_start = time.time()
                while error > RIZON_JOINT_EPSILON:
                    if time.time() - time_start > RIZON_BLOCK_TIMEOUT:
                        return
                    time.sleep(RIZON_BLOCK_WAIT_TIME)
                    error = np.abs(self.joint_read() - joints).max()
            else:
                pass
        elif self.mode == 'tcp':
            raise ValueError("Cannot move joints in tcp mode")
        else:
            raise ValueError(f"Invalid motion mode: {self.mode}")
    
    def gripper_read(self) -> float:
        '''
        width: gripper full width in meter
        '''
        if self.gripper is None:
            raise ValueError("Gripper is not initialized")
        width = float(self.gripper.states().width)
        return width
    
    def gripper_move(self, width:float, velocity:float=0.05) -> None:
        '''
        width: gripper full width in meter
        velocity: gripper velocity in m/s
        '''
        if self.gripper is None:
            raise ValueError("Gripper is not initialized")
        width = np.clip(width, self._gripper_limits[0], self._gripper_limits[1])
        self.gripper.Move(width, velocity)
        if self.blocking:
            error = float('inf')
            time_start = time.time()
            while error > RIZON_GRIPPER_EPSILON:
                if time.time() - time_start > RIZON_BLOCK_TIMEOUT:
                    return
                time.sleep(RIZON_BLOCK_WAIT_TIME)
                error = np.abs(self.gripper_read() - width)
        else:
            pass
    
    def tcp_read(self) -> np.ndarray:
        '''
        pose: 4x4 transformation matrix of tcp relative to robot base
        '''
        vec = np.array(self.robot.states().tcp_pose)

        xyz = vec[:3]
        quat = vec[3:]                                          # (w, x, y, z)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])   # (x, y, z, w)
        pose = xyzquat2mat(xyz, quat)
        return pose
    
    def tcp_move(self, pose:Optional[np.ndarray]=None, wrench:Optional[np.ndarray]=None) -> None:
        '''
        pose: 4x4 transformation matrix of tcp relative to robot base
        wrench: 6D force/torque in N/Nm in the force control reference frame
        '''
        if pose is None:
            pose = self.tcp_read()
            pure_force = True
        if wrench is None:
            wrench = np.zeros(6)
            pure_motion = True
        
        if self.mode == 'primitive':
            raise NotImplementedError("Primitive mode is not implemented for tcp control")
        elif self.mode == 'joint':
            raise ValueError("Cannot move tcp in joint mode")
        elif self.mode == 'tcp':
            vec = np.zeros(7)
            xyz, quat = mat2xyzquat(pose)
            vec[:3] = xyz
            vec[3:] = [quat[3], quat[0], quat[1], quat[2]]  # (w, x, y, z)
            self.robot.SendCartesianMotionForce(vec.tolist(), wrench.tolist(), 
                                                max_linear_vel=RIZON_TCP_MAX_VEL[0], max_linear_acc=RIZON_TCP_MAX_ACC[0], 
                                                max_angular_vel=RIZON_TCP_MAX_VEL[1], max_angular_acc=RIZON_TCP_MAX_ACC[1])
            if self.blocking:
                if pure_motion:
                    error_xyz, error_quat = float('inf'), float('inf')
                    time_start = time.time()
                    while error_xyz > RIZON_TCP_POSE_EPSILON[0] or error_quat > RIZON_TCP_POSE_EPSILON[1]:
                        if time.time() - time_start > RIZON_BLOCK_TIMEOUT:
                            return
                        time.sleep(RIZON_BLOCK_WAIT_TIME)
                        current_pose = self.tcp_read()
                        target_pose = pose
                        current_xyz, current_quat = mat2xyzquat(current_pose)
                        target_xyz, target_quat = mat2xyzquat(target_pose)
                        error_xyz = delta_xyz(current_xyz, target_xyz)
                        error_quat = delta_quat(current_quat, target_quat)
                elif pure_force:
                    raise NotImplementedError("Blocking pure force control is not implemented")
                else:
                    raise NotImplementedError("Blocking hybrid motion and force control is not implemented")
            else:
                pass
        else:
            raise ValueError(f"Invalid motion mode: {self.mode}")
    
    def start_streaming(self, callback:Optional[callable]=None, **kwargs) -> None:
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        if not hasattr(self, "_shm"):
            self._shm = None
        
        self.in_streaming.set()
        if self._shm is None:
            if callback is None:
                self.streaming_mutex = Lock()
                self.streaming_data = {
                    "joints": [], 
                    "pose": [], 
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                joints_memory_size = self.joints_dtype.itemsize * np.prod(self.joints_shape).item()
                pose_memory_size = self.pose_dtype.itemsize * np.prod(self.pose_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = joints_memory_size + pose_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "joints": np.ndarray(self.joints_shape, dtype=self.joints_dtype, buffer=self.streaming_memory.buf[0:joints_memory_size]), 
                    "pose": np.ndarray(self.pose_shape, dtype=self.pose_dtype, buffer=self.streaming_memory.buf[joints_memory_size:joints_memory_size+pose_memory_size]),
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[joints_memory_size+pose_memory_size:])
                }
                self.streaming_array_meta = {
                    "joints": (self.joints_shape, self.joints_dtype.name, (0, joints_memory_size)), 
                    "pose": (self.pose_shape, self.pose_dtype.name, (joints_memory_size, joints_memory_size+pose_memory_size)),
                    "timestamp_ms": ((1,), np.float64.__name__, (joints_memory_size+pose_memory_size, streaming_memory_size))
                }
                self._save_streaming_meta(self.streaming_array_meta)
            else:
                pass
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def stop_streaming(self) -> Dict[str, Union[List[np.ndarray], List[float]]]:
        self.in_streaming.clear()
        self.thread.join()
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} stop_streaming data size: {len(streaming_data['timestamp_ms'])}")
            self.streaming_data = {
                "joints": [], 
                "pose": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "joints": [np.copy(self.streaming_array["joints"])], 
                "pose": [np.copy(self.streaming_array["pose"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
            self.streaming_memory.close()
            self.streaming_memory.unlink()
            del self.streaming_memory
            del self.streaming_array, self.streaming_array_meta
            del self.streaming_manager
            del self.streaming_lock
        else:
            raise AttributeError
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            if INFO:
                draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
            else:
                np.savetxt(os.path.join(save_path, f"freq_{freq}.txt"), np.array([]))
        else:
            freq = 0
        np.save(os.path.join(save_path, "joints.npy"), np.array(streaming_data["joints"], dtype=float))
        np.save(os.path.join(save_path, "pose.npy"), np.array(streaming_data["pose"], dtype=float))
        if INFO:
            draw_items(np.array(streaming_data["joints"], dtype=float), os.path.join(save_path, "joints.png"))
    
    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect
    
    def shm_streaming(self, shm:Optional[str]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert (not self.in_streaming.is_set()) or (not self._collect_streaming_data)
        self._shm = shm
    
    def get_streaming(self) -> Dict[str, Union[List[np.ndarray], List[float]]]:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} get_streaming data size: {len(streaming_data['timestamp_ms'])}")
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "joints": [np.copy(self.streaming_array["joints"])], 
                "pose": [np.copy(self.streaming_array["pose"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self, **kwargs) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['joints'].clear()
            self.streaming_data['pose'].clear()
            self.streaming_data['timestamp_ms'].clear()
            del self.streaming_data
            del self.streaming_mutex
            gc.collect()
        elif hasattr(self, "streaming_array"):
            self.streaming_memory.close()
            self.streaming_memory.unlink()
            del self.streaming_memory
            del self.streaming_array, self.streaming_array_meta
            del self.streaming_manager
            del self.streaming_lock
        else:
            raise AttributeError
        
        if self._shm is None:
            self.streaming_mutex = Lock()
            self.streaming_data = {
                "joints": [], 
                "pose": [], 
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            joints_memory_size = self.joints_dtype.itemsize * np.prod(self.joints_shape).item()
            pose_memory_size = self.pose_dtype.itemsize * np.prod(self.pose_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = joints_memory_size + pose_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "joints": np.ndarray(self.joints_shape, dtype=self.joints_dtype, buffer=self.streaming_memory.buf[0:joints_memory_size]), 
                "pose": np.ndarray(self.pose_shape, dtype=self.pose_dtype, buffer=self.streaming_memory.buf[joints_memory_size:joints_memory_size+pose_memory_size]),
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[joints_memory_size+pose_memory_size:])
            }
            self.streaming_array_meta = {
                "joints": (self.joints_shape, self.joints_dtype.name, (0, joints_memory_size)), 
                "pose": (self.pose_shape, self.pose_dtype.name, (joints_memory_size, joints_memory_size+pose_memory_size)),
                "timestamp_ms": ((1,), np.float64.__name__, (joints_memory_size+pose_memory_size, streaming_memory_size))
            }
            self._save_streaming_meta(self.streaming_array_meta)
    
    def _streaming_data(self, callback:Optional[callable]=None):
        DT = 1.0 / self._fps
        while self.in_streaming.is_set():
            t_start = time.perf_counter()

            # get data
            if not self._collect_streaming_data:
                continue

            data = self._read()
            
            if callback is None:
                if hasattr(self, "streaming_data"):
                    self.streaming_mutex.acquire()
                    self.streaming_data['joints'].append(data['joints'])
                    self.streaming_data['pose'].append(data['pose'])
                    self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                    self.streaming_mutex.release()
                elif hasattr(self, "streaming_array"):
                    with self.streaming_lock:
                        self.streaming_array["joints"][:] = data['joints'][:]
                        self.streaming_array["pose"][:] = data['pose'][:]
                        self.streaming_array["timestamp_ms"][:] = data['timestamp_ms']
                else:
                    raise AttributeError
            else:
                callback(deepcopy(data))
            
            elapsed = time.perf_counter() - t_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def zero_ft(self) -> None:
        '''
        Zero FT sensor, must blocking
        '''
        if self.mode != 'primitive':
            original_mode = self.mode
            self.motion_mode('primitive')
        else:
            original_mode = None
        
        self.robot.ExecutePrimitive("ZeroFTSensor", dict())
        while not self.robot.primitive_states()["terminated"]:
            time.sleep(RIZON_BLOCK_WAIT_TIME)
        
        if original_mode is not None:
            self.motion_mode(original_mode)
        else:
            pass
    
    def ft_read(self, tcp:bool=True, filtered:bool=True, raw:bool=False) -> np.ndarray:
        '''
        ft: 6D force/torque in N/Nm
        tcp: if True, return force/torque in tcp frame; if False, return in robot base frame
        filtered: if True, return filtered force/torque; if False, return unfiltered force/torque
        raw: if True, return raw force/torque reading; if False, return processed force/torque
        '''
        if raw:
            ft = np.array(self.robot.states().ft_sensor_raw)
        else:
            if tcp:
                if filtered:
                    ft = np.array(self.robot.states().ext_wrench_in_tcp)
                else:
                    ft = np.array(self.robot.states().ext_wrench_in_tcp_raw)
            else:
                if filtered:
                    ft = np.array(self.robot.states().ext_wrench_in_world)
                else:
                    ft = np.array(self.robot.states().ext_wrench_in_world_raw)
        return ft
    
    def set_force_control_config(self, frame_type:str, relative_transformation:np.ndarray=np.eye(4), enabled_axes:List[bool]=[False, False, False, False, False, False]) -> None:
        '''
        frame_type: 'tcp' (moving) or 'world' (fixed)
        relative_transformation: 4x4 transformation matrix of force control frame relative to selected frame
        '''
        if self.mode != 'tcp':
            original_mode = self.mode
            self.motion_mode('tcp')
        else:
            original_mode = None
        
        if frame_type == 'tcp':
            force_ctrl_frame = flexivrdk.CoordType.TCP
        elif frame_type == 'world':
            force_ctrl_frame = flexivrdk.CoordType.WORLD
        else:
            raise ValueError(f"Invalid force control frame type: {frame_type}")
        relative_vec = np.zeros(7)
        relative_vec[:3] = relative_transformation[:3, 3]
        relative_quat = Rot.from_matrix(relative_transformation[:3, :3]).as_quat()
        relative_vec[3:] = [relative_quat[3], relative_quat[0], relative_quat[1], relative_quat[2]] # (w, x, y, z)
        self.robot.SetForceControlFrame(force_ctrl_frame, relative_vec)
        self.robot.SetForceControlAxis(enabled_axes, max_linear_vel=[RIZON_TCP_MAX_VEL[0]] * 3)

        if original_mode is not None:
            self.motion_mode(original_mode)
        else:
            pass
    
    @staticmethod
    def raw2tare(raw_ft:np.ndarray, tare:Dict[str, Union[float, np.ndarray]], pose:np.ndarray) -> np.ndarray:
        '''
        raw_ft: raw force torque data
        pose: 3x3 rotation matrix from ft to base
        '''
        raw_f, raw_t = raw_ft[:3], raw_ft[3:]
        f = raw_f - tare['f0']
        f -= np.linalg.inv(pose) @ np.array([0., 0., -9.8 * tare['m']])
        t = raw_t - tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(np.linalg.inv(pose) @ np.array(tare['c']), np.array([0., 0., -9.8 * tare['m']]))
        return np.concatenate([f, t])


if __name__ == "__main__":
    robot = Rizon(id='Rizon4s-12345', gripper=True, name='Rizon')

    robot.homing()
    print("homing")
    joint = robot.joint_read()
    print("current joint:", joint)
    pose = robot.tcp_read()
    print("current pose:", pose)
