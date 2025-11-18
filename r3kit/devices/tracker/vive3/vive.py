import numpy as np
import time
from threading import Lock, Thread
from multiprocessing import shared_memory, Manager
import json
from typing import List, Dict, Union, Optional
from scipy.spatial.transform import Rotation as Rot

from r3kit.devices.tracker.base import TrackerBase
from r3kit.devices.tracker.vive3.vive_tracker.track import ViveTrackerModule
from r3kit.devices.tracker.vive3.config import VIVE_TRACKER_FPS
from r3kit.utils.vis import draw_time


class ViveTracker(TrackerBase):
    def __init__(
        self,
        names: List[str] = ["tracker_1"],
        shm_name: str = "vive_tracker_shm",
        fps: int = VIVE_TRACKER_FPS,
        name: str = 'ViveTracker'
    ):
        super().__init__(name=name)
        self.names = names
        self.fps = fps
        self.vive_tracker_module = ViveTrackerModule()
        self.tracking_devices = self.vive_tracker_module.return_selected_devices(self.names)
        self.tracking_device = self.tracking_devices[self.names[0]] if self.names else None
        self.shm_name = shm_name
        self.in_streaming = False

        initial_data = self._read()
        if initial_data is not None:
            self.xyz_dtype = initial_data['xyz'].dtype
            self.xyz_shape = initial_data['xyz'].shape
            self.quat_dtype = initial_data['quat'].dtype
            self.quat_shape = initial_data['quat'].shape
        else:
            self.xyz_dtype = np.float32
            self.xyz_shape = (3,)
            self.quat_dtype = np.float32
            self.quat_shape = (4,)

        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        if not hasattr(self, "_shm"):
            self._shm = None

    def _read(self) -> Optional[Dict[str, Union[np.ndarray, float]]]:
        try:
            pose = self.tracking_device.get_T()
            xyz = pose[:3, 3]
            quat = Rot.from_matrix(pose[:3, :3]).as_quat()
            receive_time = time.time() * 1000
            return {"xyz": xyz, "quat": quat, "timestamp_ms": receive_time}
        except:
            return None

    def get_pose(self):
        pose = self.tracking_device.get_T()
        return pose

    def get_poses(self):
        poses = [self.tracking_devices[key].get_T() for key in self.tracking_devices]
        return poses

    def get_pose_quat_time(self):
        pose = self.get_pose()
        xyz = pose[:3, 3]
        quat = Rot.from_matrix(pose[:3, :3]).as_quat()
        receive_time = time.time() * 1000
        return xyz, quat, receive_time

    def get(self) -> Optional[List[Dict[str, Union[np.ndarray, float]]]]:
        if not self.in_streaming:
            data = self._read()
            if data is None:
                return None
            return [data]
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                if len(self.streaming_data["xyz"]) == 0:
                    self.streaming_mutex.release()
                    return None
                xyz = self.streaming_data["xyz"][-1]
                quat = self.streaming_data["quat"][-1]
                timestamp_ms = self.streaming_data["timestamp_ms"][-1]
                self.streaming_mutex.release()
                return [{"xyz": xyz, "quat": quat, "timestamp_ms": timestamp_ms}]
            elif hasattr(self, "streaming_array"):
                with self.streaming_lock:
                    xyz = np.copy(self.streaming_array["xyz"])
                    quat = np.copy(self.streaming_array["quat"])
                    timestamp_ms = self.streaming_array["timestamp_ms"].item()
                return [{"xyz": xyz, "quat": quat, "timestamp_ms": timestamp_ms}]
            else:
                raise AttributeError

    def start_streaming(self, callback: Optional[callable] = None) -> None:
        self.in_streaming = True
        if self._shm is None:
            if callback is None:
                self.streaming_mutex = Lock()
                self.streaming_data = {
                    "xyz": [],
                    "quat": [],
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                xyz_memory_size = self.xyz_dtype.itemsize * np.prod(self.xyz_shape).item()
                quat_memory_size = self.quat_dtype.itemsize * np.prod(self.quat_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                total_mem = xyz_memory_size + quat_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=total_mem)
                offset = 0
                self.streaming_array = {}
                self.streaming_array_meta = {}
                self.streaming_array["xyz"] = np.ndarray(
                    self.xyz_shape,
                    dtype=self.xyz_dtype,
                    buffer=self.streaming_memory.buf[offset:offset+xyz_memory_size]
                )
                self.streaming_array_meta["xyz"] = (self.xyz_shape, self.xyz_dtype.__name__, (offset, offset+xyz_memory_size))
                offset += xyz_memory_size
                self.streaming_array["quat"] = np.ndarray(
                    self.quat_shape,
                    dtype=self.quat_dtype,
                    buffer=self.streaming_memory.buf[offset:offset+quat_memory_size]
                )
                self.streaming_array_meta["quat"] = (self.quat_shape, self.quat_dtype.__name__, (offset, offset+quat_memory_size))
                offset += quat_memory_size
                self.streaming_array["timestamp_ms"] = np.ndarray(
                    (1,),
                    dtype=np.float64,
                    buffer=self.streaming_memory.buf[offset:offset+timestamp_memory_size]
                )
                self.streaming_array_meta["timestamp_ms"] = ((1,), np.float64.__name__, (offset, offset+timestamp_memory_size))
            else:
                pass
        self.capture_thread = Thread(target=self._capture_thread, daemon=True)
        self.capture_thread.start()

    def stop_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        self.in_streaming = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()

        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            self.streaming_data = {"xyz": [], "quat": [], "timestamp_ms": []}
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "xyz": [np.copy(self.streaming_array["xyz"])],
                "quat": [np.copy(self.streaming_array["quat"])],
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

    def save_streaming(self, save_path: str, streaming_data: dict) -> None:
        assert len(streaming_data["xyz"]) == len(streaming_data["quat"]) == len(streaming_data["timestamp_ms"])
        import os
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        np.save(os.path.join(save_path, "xyz.npy"), np.array(streaming_data["xyz"], dtype=float))
        np.save(os.path.join(save_path, "quat.npy"), np.array(streaming_data["quat"], dtype=float))

    def collect_streaming(self, collect: bool = True) -> None:
        self._collect_streaming_data = collect

    def shm_streaming(self, shm: Optional[str] = None) -> None:
        assert (not self.in_streaming) or (not self._collect_streaming_data)
        self._shm = shm or self.shm_name
        if self._shm is not None:
            import os
            os.makedirs('.temp', exist_ok=True)
            with open(os.path.join('.temp', f"{self._shm}_array_meta.json"), 'w') as f:
                if hasattr(self, 'streaming_array_meta'):
                    json.dump(self.streaming_array_meta, f, indent=4)

    def get_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "xyz": [np.copy(self.streaming_array["xyz"])],
                "quat": [np.copy(self.streaming_array["quat"])],
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data

    def reset_streaming(self) -> None:
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['xyz'].clear()
            self.streaming_data['quat'].clear()
            self.streaming_data['timestamp_ms'].clear()
            del self.streaming_data
            del self.streaming_mutex
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
                "xyz": [],
                "quat": [],
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            xyz_memory_size = self.xyz_dtype.itemsize * np.prod(self.xyz_shape).item()
            quat_memory_size = self.quat_dtype.itemsize * np.prod(self.quat_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            total_mem = xyz_memory_size + quat_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=total_mem)
            offset = 0
            self.streaming_array = {}
            self.streaming_array_meta = {}
            self.streaming_array["xyz"] = np.ndarray(
                self.xyz_shape,
                dtype=self.xyz_dtype,
                buffer=self.streaming_memory.buf[offset:offset+xyz_memory_size]
            )
            self.streaming_array_meta["xyz"] = (self.xyz_shape, self.xyz_dtype.__name__, (offset, offset+xyz_memory_size))
            offset += xyz_memory_size
            self.streaming_array["quat"] = np.ndarray(
                self.quat_shape,
                dtype=self.quat_dtype,
                buffer=self.streaming_memory.buf[offset:offset+quat_memory_size]
            )
            self.streaming_array_meta["quat"] = (self.quat_shape, self.quat_dtype.__name__, (offset, offset+quat_memory_size))
            offset += quat_memory_size
            self.streaming_array["timestamp_ms"] = np.ndarray(
                (1,),
                dtype=np.float64,
                buffer=self.streaming_memory.buf[offset:offset+timestamp_memory_size]
            )
            self.streaming_array_meta["timestamp_ms"] = ((1,), np.float64.__name__, (offset, offset+timestamp_memory_size))

    def _capture_thread(self):
        t0 = time.time()
        t_dur = 1 / self.fps
        while self.in_streaming:
            if self._collect_streaming_data:
                self.callback()
            t1 = time.time()
            t_01 = t_dur - (t1 - t0)
            if t_01 > 0:
                time.sleep(t_01)
            t0 = time.time()
            time.sleep(0.001)

    def callback(self):
        if not self._collect_streaming_data:
            return
        xyz, quat, ts = self.get_pose_quat_time()
        if hasattr(self, "streaming_data"):
            self.streaming_mutex.acquire()
            if len(self.streaming_data["timestamp_ms"]) != 0 and ts == self.streaming_data["timestamp_ms"][-1]:
                self.streaming_mutex.release()
                return
            self.streaming_data["xyz"].append(xyz.copy())
            self.streaming_data["quat"].append(quat.copy())
            self.streaming_data["timestamp_ms"].append(ts)
            self.streaming_mutex.release()
        elif hasattr(self, "streaming_array"):
            with self.streaming_lock:
                self.streaming_array["xyz"][:] = xyz[:]
                self.streaming_array["quat"][:] = quat[:]
                self.streaming_array["timestamp_ms"][:] = ts

    def release(self):
        if self.in_streaming:
            self.stop_streaming()

    def __del__(self):
        self.release()

