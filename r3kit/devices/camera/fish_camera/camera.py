import os
import time
import numpy as np
from threading import Lock, Thread
from multiprocessing import shared_memory, Manager
import cv2
import json
from typing import Tuple, List, Union, Dict, Optional

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.fish_camera.config import FISH_CAMERA_FPS
from r3kit.utils.vis import draw_time, save_imgs


class FishCamera(CameraBase):
    def __init__(self, id: int = 0, name: str = 'FishCamera', fps: int = FISH_CAMERA_FPS) -> None:
        super().__init__(name=name)
        self.cap = cv2.VideoCapture(id)
        self.in_streaming = False
        self.id = id
        self.fps = fps

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to initialize camera")
        self.color_image_dtype = frame.dtype
        self.color_image_shape = frame.shape

        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        if not hasattr(self, "_shm"):
            self._shm = None

    def get(self) -> Optional[np.ndarray]:
        if not self.in_streaming:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                if len(self.streaming_data["color"]) == 0:
                    self.streaming_mutex.release()
                    return None
                color_image = self.streaming_data["color"][-1]
                self.streaming_mutex.release()
                return color_image
            elif hasattr(self, "streaming_array"):
                with self.streaming_lock:
                    color_image = np.copy(self.streaming_array["color"])
                return color_image
            else:
                raise AttributeError

    def start_streaming(self, callback: Optional[callable] = None) -> None:
        self.cap.release()
        self.cap = cv2.VideoCapture(self.id)

        if self._shm is None:
            if callback is None:
                self.streaming_mutex = Lock()
                self.streaming_data = {
                    "color": [],
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                color_memory_size = self.color_image_dtype.itemsize * np.prod(self.color_image_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = color_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "color": np.ndarray(
                        self.color_image_shape,
                        dtype=self.color_image_dtype,
                        buffer=self.streaming_memory.buf[:color_memory_size]
                    ),
                    "timestamp_ms": np.ndarray(
                        (1,),
                        dtype=np.float64,
                        buffer=self.streaming_memory.buf[color_memory_size:]
                    )
                }
                self.streaming_array_meta = {
                    "color": (self.color_image_shape, self.color_image_dtype.name, (0, color_memory_size)),
                    "timestamp_ms": ((1,), np.float64.__name__, (color_memory_size, color_memory_size + timestamp_memory_size))
                }
                self._save_streaming_meta(self.streaming_array_meta)
            else:
                pass
        self.in_streaming = True
        self.capture_thread = Thread(target=self._capture_thread, daemon=True)
        self.capture_thread.start()

    def stop_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        self.in_streaming = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()

        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            self.streaming_data = {"color": [], "timestamp_ms": []}
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "color": [np.copy(self.streaming_array["color"])],
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
        self.cap.release()
        self.cap = cv2.VideoCapture(self.id)
        return streaming_data

    def save_streaming(self, save_path: str, streaming_data: dict) -> None:
        assert len(streaming_data["color"]) == len(streaming_data["timestamp_ms"])
        os.makedirs(save_path, exist_ok=True)
        if "color" in streaming_data and len(streaming_data["color"]) > 0:
            os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)
            save_imgs(os.path.join(save_path, 'color'), streaming_data["color"])
        if "timestamp_ms" in streaming_data and len(streaming_data["timestamp_ms"]) > 0:
            np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
            if len(streaming_data["timestamp_ms"]) > 1:
                freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
                draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))

    def collect_streaming(self, collect: bool = True) -> None:
        self._collect_streaming_data = collect

    def shm_streaming(self, shm: Optional[str] = None) -> None:
        assert (not self.in_streaming) or (not self._collect_streaming_data)
        self._shm = shm

    def get_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "color": [np.copy(self.streaming_array["color"])],
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data

    def reset_streaming(self) -> None:
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['color'].clear()
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
                "color": [],
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            color_memory_size = self.color_image_dtype.itemsize * np.prod(self.color_image_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = color_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "color": np.ndarray(
                    self.color_image_shape,
                    dtype=self.color_image_dtype,
                    buffer=self.streaming_memory.buf[:color_memory_size]
                ),
                "timestamp_ms": np.ndarray(
                    (1,),
                    dtype=np.float64,
                    buffer=self.streaming_memory.buf[color_memory_size:]
                )
            }
            self.streaming_array_meta = {
                "color": (self.color_image_shape, self.color_image_dtype.name, (0, color_memory_size)),
                "timestamp_ms": ((1,), np.float64.__name__, (color_memory_size, color_memory_size + timestamp_memory_size))
            }
            self._save_streaming_meta(self.streaming_array_meta)

    def _capture_thread(self):
        t0 = time.perf_counter()
        t_dur = 1 / self.fps
        while self.in_streaming:
            if self._collect_streaming_data:
                self.callback()
            t1 = time.perf_counter()
            t_01 = t_dur - (t1 - t0)
            if t_01 > 0:
                time.sleep(t_01)
            t0 = time.perf_counter()

    def callback(self):
        ts = time.time() * 1000
        if not self._collect_streaming_data:
            return

        ret, frame = self.cap.read()
        if not ret:
            return
        color_image = frame

        if hasattr(self, "streaming_data"):
            self.streaming_mutex.acquire()
            if len(self.streaming_data["timestamp_ms"]) != 0 and ts == self.streaming_data["timestamp_ms"][-1]:
                self.streaming_mutex.release()
                return
            self.streaming_data["color"].append(color_image.copy())
            self.streaming_data["timestamp_ms"].append(ts)
            self.streaming_mutex.release()
        elif hasattr(self, "streaming_array"):
            with self.streaming_lock:
                self.streaming_array["color"][:] = color_image[:]
                self.streaming_array["timestamp_ms"][:] = ts

    def __del__(self) -> None:
        if hasattr(self, 'cap'):
            self.cap.release()

