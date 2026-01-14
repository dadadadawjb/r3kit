import os
from rich import print
import time
import numpy as np
from threading import Lock, Thread, Event
from multiprocessing import shared_memory, Manager
from typing import List, Dict, Union, Optional
from copy import deepcopy
from functools import partial

from r3kit.devices.tracker.base import TrackerBase
from r3kit.devices.tracker.vive.vr_module import ViveTrackerModule
from r3kit.devices.tracker.vive.config import *
from r3kit.utils.transformation import mat2xyzquat, xyzquat2mat
from r3kit.utils.vis import draw_time
from r3kit import DEBUG, INFO


class VIVE3(TrackerBase):
    def __init__(self, id:List[str]=VIVE3_ROLE, name:str='VIVE3') -> None:
        super().__init__(name=name)

        self._num = len(id)

        self.vive_tracker_module = ViveTrackerModule()
        self.tracking_devices = self.vive_tracker_module.return_selected_devices(id)

        # config
        initial_start_time = time.time()
        data = self._read()
        while data is None:
            data = self._read()
            if time.time() - initial_start_time > 3:
                raise RuntimeError("VIVE3 cannot read data")
        self.xyz_dtype = data['xyz'].dtype
        self.xyz_shape = data['xyz'].shape
        self.quat_dtype = data['quat'].dtype
        self.quat_shape = data['quat'].shape

        self.in_streaming = Event()

    def _read(self) -> Dict[str, Union[np.ndarray, float]]:
        xyzs = []
        quats = []
        receive_times = []
        for tracking_device in self.tracking_devices:
            pose = tracking_device.get_T()
            xyz, quat = mat2xyzquat(pose)
            receive_time = time.time() * 1000
            xyzs.append(xyz)
            quats.append(quat)
            receive_times.append(receive_time)
        return {"xyz": np.array(xyzs), "quat": np.array(quats), "timestamp_ms": np.array(receive_times).mean().item()}

    def get(self) -> Dict[str, Union[np.ndarray, float]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                data = {}
                data['xyz'] = self.streaming_data["xyz"][-1]
                data['quat'] = self.streaming_data["quat"][-1]
                data['timestamp_ms'] = self.streaming_data["timestamp_ms"][-1]
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                data = {}
                with self.streaming_lock:
                    data['xyz'] = np.copy(self.streaming_array["xyz"])
                    data['quat'] = np.copy(self.streaming_array["quat"])
                    data['timestamp_ms'] = self.streaming_array["timestamp_ms"].item()
            else:
                raise AttributeError
        return data

    def start_streaming(self, callback:Optional[callable]=None) -> None:
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        if not hasattr(self, "_shm"):
            self._shm = None
        
        self.in_streaming.set()
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
                streaming_memory_size = xyz_memory_size + quat_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "xyz": np.ndarray(self.xyz_shape, dtype=self.xyz_dtype, buffer=self.streaming_memory.buf[:xyz_memory_size]), 
                    "quat": np.ndarray(self.quat_shape, dtype=self.quat_dtype, buffer=self.streaming_memory.buf[xyz_memory_size:xyz_memory_size+quat_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[xyz_memory_size+quat_memory_size:])
                }
                self.streaming_array_meta = {
                    "xyz": (self.xyz_shape, self.xyz_dtype.name, (0, xyz_memory_size)), 
                    "quat": (self.quat_shape, self.quat_dtype.name, (xyz_memory_size, xyz_memory_size+quat_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (xyz_memory_size+quat_memory_size, xyz_memory_size+quat_memory_size+timestamp_memory_size))
                }
                self._save_streaming_meta(self.streaming_array_meta)
            else:
                pass
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()

    def stop_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        self.in_streaming.clear()
        self.thread.join()

        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} stop_streaming data size: {len(streaming_data['timestamp_ms'])}")
            self.streaming_data = {
                "xyz": [], 
                "quat": [], 
                "timestamp_ms": []
            }
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

    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        assert len(streaming_data["xyz"]) == len(streaming_data["quat"]) == len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            if INFO:
                draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
            else:
                np.savetxt(os.path.join(save_path, f"freq_{freq}.txt"), np.array([]))
        else:
            freq = 0
        np.save(os.path.join(save_path, "xyz.npy"), np.array(streaming_data["xyz"], dtype=float))
        np.save(os.path.join(save_path, "quat.npy"), np.array(streaming_data["quat"], dtype=float))

    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect

    def shm_streaming(self, shm:Optional[str]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert (not self.in_streaming.is_set()) or (not self._collect_streaming_data)
        self._shm = shm

    def get_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} get_streaming data size: {len(streaming_data['timestamp_ms'])}")
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
        # NOTE: only valid for non-custom-callback
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
            streaming_memory_size = xyz_memory_size + quat_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "xyz": np.ndarray(self.xyz_shape, dtype=self.xyz_dtype, buffer=self.streaming_memory.buf[:xyz_memory_size]), 
                "quat": np.ndarray(self.quat_shape, dtype=self.quat_dtype, buffer=self.streaming_memory.buf[xyz_memory_size:xyz_memory_size+quat_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[xyz_memory_size+quat_memory_size:])
            }
            self.streaming_array_meta = {
                "xyz": (self.xyz_shape, self.xyz_dtype.name, (0, xyz_memory_size)), 
                "quat": (self.quat_shape, self.quat_dtype.name, (xyz_memory_size, xyz_memory_size+quat_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (xyz_memory_size+quat_memory_size, xyz_memory_size+quat_memory_size+timestamp_memory_size))
            }
            self._save_streaming_meta(self.streaming_array_meta)

    def _streaming_data(self, callback:Optional[callable]=None):
        while self.in_streaming.is_set():
            # get data
            if not self._collect_streaming_data:
                continue
            data = self._read()
            if data is not None:
                if callback is None:
                    if hasattr(self, "streaming_data"):
                        self.streaming_mutex.acquire()
                        self.streaming_data['xyz'].append(data['xyz'])
                        self.streaming_data['quat'].append(data['quat'])
                        self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                        self.streaming_mutex.release()
                    elif hasattr(self, "streaming_array"):
                        with self.streaming_lock:
                            self.streaming_array["xyz"][:] = data['xyz'][:]
                            self.streaming_array["quat"][:] = data['quat'][:]
                            self.streaming_array["timestamp_ms"][:] = data['timestamp_ms']
                    else:
                        raise AttributeError
                else:
                    callback(deepcopy(data))
    
    @staticmethod
    def raw2pose(xyz:np.ndarray, quat:np.ndarray) -> np.ndarray:
        return xyzquat2mat(xyz, quat)


if __name__ == "__main__":
    tracker = VIVE3(id=['tracker_1', 'tracker_2'], name='VIVE3')
    streaming = False
    shm = False

    if not streaming:
        while True:
            data = tracker.get()
            print(data)
            time.sleep(0.1)
    else:
        tracker.collect_streaming(collect=True)
        tracker.shm_streaming(shm='VIVE3' if shm else None)
        tracker.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = tracker.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        xyz = streaming_data["xyz"][-1]
        quat = streaming_data["quat"][-1]
        print(xyz, quat)
