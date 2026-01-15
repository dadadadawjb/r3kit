from typing import Tuple, List, Dict, Optional
import os
import time
import gc
from rich import print
import numpy as np
from threading import Thread, Lock, Event
from multiprocessing import shared_memory, Manager
from copy import deepcopy
from functools import partial
from xensegripper import XenseGripper

from r3kit.devices.gripper.base import GripperBase
from r3kit.devices.gripper.xense.config import *
from r3kit.utils.vis import draw_items, draw_time
from r3kit import INFO, DEBUG


class Xense(GripperBase):
    MAX_WIDTH:float = 0.085
    MAX_VELOCITY:float = 0.35
    MAX_FORCE:float = 60.0

    def __init__(self, id:str=XENSE_ID, fps:int=XENSE_FPS, name:str='Xense') -> None:
        super().__init__(name)
        self._fps = fps

        self.gripper = XenseGripper.create(id)
        self.block(-1)

        # config
        self.width_dtype = np.dtype(np.float64)
        self.width_shape = (1,)

        # stream
        self.in_streaming = Event()
    
    def block(self, blocking:float) -> None:
        self.blocking = blocking
    
    def _read(self) -> Dict[str, float]:
        status = self.gripper.get_gripper_status()
        receive_time = time.time() * 1000
        data = {
            'width': status['position'] / 1000.,
            'timestamp_ms': receive_time
        }
        return data
    
    def read(self) -> float:
        '''
        width: gripper width in meter
        '''
        status = self.gripper.get_gripper_status()
        return status['position'] / 1000.

    def move(self, width:float, vmax:float=0.08, fmax:float=27.0) -> bool:
        '''
        width: gripper width in meter
        reached: whether the gripper reaches the target width within timeout
        '''
        target_width = min(max(width, 0.0), self.MAX_WIDTH) * 1000.
        target_vmax = min(max(vmax, 0.0), self.MAX_VELOCITY) * 1000.
        target_fmax = min(max(fmax, 0.0), self.MAX_FORCE)
        if self.blocking <= 0:
            self.gripper.set_position(target_width, target_vmax, target_fmax)
            reached = True
        else:
            reached = self.gripper.set_position_sync(target_width, target_vmax, target_fmax, 
                                                    timeout=self.blocking, tolerance=XENSE_TOLERANCE)
        return reached
    
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
                    "width": [], 
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                width_memory_size = self.width_dtype.itemsize * np.prod(self.width_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = width_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "width": np.ndarray(self.width_shape, dtype=self.width_dtype, buffer=self.streaming_memory.buf[0:width_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[width_memory_size:])
                }
                self.streaming_array_meta = {
                    "width": (self.width_shape, self.width_dtype.name, (0, width_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (width_memory_size, width_memory_size+timestamp_memory_size))
                }
                self._save_streaming_meta(self.streaming_array_meta)
            else:
                pass
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def stop_streaming(self) -> Dict[str, List[float]]:
        self.in_streaming.clear()
        self.thread.join()
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} stop_streaming data size: {len(streaming_data['timestamp_ms'])}")
            self.streaming_data = {
                "width": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "width": [self.streaming_array["width"].item()], 
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
        assert len(streaming_data["width"]) ==  len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            if INFO:
                draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
            else:
                np.savetxt(os.path.join(save_path, f"freq_{freq}.txt"), np.array([]))
        else:
            freq = 0
        np.save(os.path.join(save_path, "width.npy"), np.array(streaming_data["width"], dtype=float))
        if INFO:
            draw_items(np.array(streaming_data["width"], dtype=float), os.path.join(save_path, "width.png"))
    
    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect
    
    def shm_streaming(self, shm:Optional[str]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert (not self.in_streaming.is_set()) or (not self._collect_streaming_data)
        self._shm = shm
    
    def get_streaming(self) -> Dict[str, List[float]]:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} get_streaming data size: {len(streaming_data['timestamp_ms'])}")
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "width": [self.streaming_array["width"].item()], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self, **kwargs) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['width'].clear()
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
                "width": [], 
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            width_memory_size = self.width_dtype.itemsize * np.prod(self.width_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = width_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "width": np.ndarray(self.width_shape, dtype=self.width_dtype, buffer=self.streaming_memory.buf[0:width_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[width_memory_size:])
            }
            self.streaming_array_meta = {
                "width": (self.width_shape, self.width_dtype.name, (0, width_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (width_memory_size, width_memory_size+timestamp_memory_size))
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
                    self.streaming_data['width'].append(data['width'])
                    self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                    self.streaming_mutex.release()
                elif hasattr(self, "streaming_array"):
                    with self.streaming_lock:
                        self.streaming_array["width"][:] = data['width']
                        self.streaming_array["timestamp_ms"][:] = data['timestamp_ms']
                else:
                    raise AttributeError
            else:
                callback(deepcopy(data))
            
            elapsed = time.perf_counter() - t_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def set_led_color(self, rgb:Tuple[int, int, int]) -> None:
        self.gripper.set_led_color(*rgb)


if __name__ == "__main__":
    gripper = Xense(id="5e77ff097831", name="Xense")

    gripper.block(5)
    gripper.move(0.08)
    gripper.move(0.005)
    gripper.move(0.05)
    gripper_width = gripper.read()
    print("gripper width:", gripper_width)
