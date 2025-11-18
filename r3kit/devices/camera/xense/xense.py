import os
import time
import numpy as np
from threading import Lock, Thread
from multiprocessing import shared_memory, Manager
import json
from typing import Dict, Union, List, Optional
from xensesdk import Sensor

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.xense.config import XENSE_FPS
from r3kit.utils.vis import draw_time, save_imgs


def precise_sleep(duration_s: float) -> None:
    start_ns = time.perf_counter_ns()
    duration_ns = duration_s * 1e9
    while (time.perf_counter_ns() - start_ns) < duration_ns:
        pass


class Xense(CameraBase):
    def __init__(
        self,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        config_path: Optional[str] = None,
        video_path: Optional[str] = None,
        name: str = 'Xense',
        use_gpu: bool = False,
        fps: int = XENSE_FPS
    ) -> None:
        super().__init__(name=name)
        self.device_id = device_id
        self.ip_address = ip_address
        self.config_path = config_path
        self.video_path = video_path
        self.in_streaming = False
        self.fps = fps

        self.sensor = Sensor.create(
            device_id,
            ip_address=ip_address,
            use_gpu=use_gpu,
            config_path=config_path,
            video_path=video_path
        )

        rectify = self.sensor.selectSensorInfo(Sensor.OutputType.Rectify)
        self.rectify_dtype = rectify.dtype
        self.rectify_shape = rectify.shape

        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        if not hasattr(self, "_shm"):
            self._shm = None

    def get(self) -> Dict[str, np.ndarray]:
        if not self.in_streaming:
            rectify = self.sensor.selectSensorInfo(Sensor.OutputType.Rectify)
            return {"rectify": rectify}
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                if len(self.streaming_data["rectify"]) == 0:
                    self.streaming_mutex.release()
                    return None
                rectify = self.streaming_data["rectify"][-1]
                timestamp_ms = self.streaming_data["timestamp_ms"][-1]
                self.streaming_mutex.release()
                return {"rectify": rectify, "timestamp_ms": timestamp_ms}
            elif hasattr(self, "streaming_array"):
                with self.streaming_lock:
                    rectify = np.copy(self.streaming_array["rectify"])
                    timestamp_ms = self.streaming_array["timestamp_ms"].item()
                return {"rectify": rectify, "timestamp_ms": timestamp_ms}
            else:
                raise AttributeError

    def start_streaming(self, callback: Optional[callable] = None) -> None:
        self.in_streaming = True
        if self._shm is None:
            if callback is None:
                self.streaming_mutex = Lock()
                self.streaming_data = {
                    "rectify": [],
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()

                rectify_memory_size = self.rectify_dtype.itemsize * np.prod(self.rectify_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = rectify_memory_size + timestamp_memory_size

                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)

                offset = 0
                self.streaming_array = {}
                self.streaming_array_meta = {}

                self.streaming_array["rectify"] = np.ndarray(
                    self.rectify_shape,
                    dtype=self.rectify_dtype,
                    buffer=self.streaming_memory.buf[offset:offset+rectify_memory_size]
                )
                self.streaming_array_meta["rectify"] = (
                    self.rectify_shape,
                    self.rectify_dtype.name,
                    (offset, offset+rectify_memory_size)
                )
                offset += rectify_memory_size

                self.streaming_array["timestamp_ms"] = np.ndarray(
                    (1,),
                    dtype=np.float64,
                    buffer=self.streaming_memory.buf[offset:offset+timestamp_memory_size]
                )
                self.streaming_array_meta["timestamp_ms"] = (
                    (1,),
                    np.float64.__name__,
                    (offset, offset+timestamp_memory_size)
                )
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
            self.streaming_data = {"rectify": [], "timestamp_ms": []}
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "rectify": [np.copy(self.streaming_array["rectify"])],
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
        assert len(streaming_data["rectify"]) == len(streaming_data["timestamp_ms"])
        os.makedirs(save_path, exist_ok=True)
        if "rectify" in streaming_data and len(streaming_data["rectify"]) > 0:
            os.makedirs(os.path.join(save_path, 'rectify'), exist_ok=True)
            save_imgs(os.path.join(save_path, 'rectify'), streaming_data["rectify"])
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
        if self._shm is not None:
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
                "rectify": [np.copy(self.streaming_array["rectify"])],
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data

    def reset_streaming(self) -> None:
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['rectify'].clear()
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
                "rectify": [],
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()

            rectify_memory_size = self.rectify_dtype.itemsize * np.prod(self.rectify_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = rectify_memory_size + timestamp_memory_size

            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)

            offset = 0
            self.streaming_array = {}
            self.streaming_array_meta = {}

            self.streaming_array["rectify"] = np.ndarray(
                self.rectify_shape,
                dtype=self.rectify_dtype,
                buffer=self.streaming_memory.buf[offset:offset+rectify_memory_size]
            )
            self.streaming_array_meta["rectify"] = (
                self.rectify_shape,
                self.rectify_dtype.name,
                (offset, offset+rectify_memory_size)
            )
            offset += rectify_memory_size

            self.streaming_array["timestamp_ms"] = np.ndarray(
                (1,),
                dtype=np.float64,
                buffer=self.streaming_memory.buf[offset:offset+timestamp_memory_size]
            )
            self.streaming_array_meta["timestamp_ms"] = (
                (1,),
                np.float64.__name__,
                (offset, offset+timestamp_memory_size)
            )

    def _capture_thread(self):
        t0 = time.perf_counter()
        t_dur = 1 / self.fps
        while self.in_streaming:
            if self._collect_streaming_data:
                self.callback()
            t1 = time.perf_counter()
            t_01 = t_dur - (t1 - t0)
            if t_01 > 0:
                precise_sleep(t_01)
            t0 = time.perf_counter()

    def callback(self):
        ts = time.time() * 1000
        if not self._collect_streaming_data:
            return

        rectify = self.sensor.selectSensorInfo(Sensor.OutputType.Rectify)

        if hasattr(self, "streaming_data"):
            self.streaming_mutex.acquire()
            if len(self.streaming_data["timestamp_ms"]) != 0 and ts == self.streaming_data["timestamp_ms"][-1]:
                self.streaming_mutex.release()
                return
            self.streaming_data["rectify"].append(rectify.copy())
            self.streaming_data["timestamp_ms"].append(ts)
            self.streaming_mutex.release()
        elif hasattr(self, "streaming_array"):
            with self.streaming_lock:
                self.streaming_array["rectify"][:] = rectify[:]
                self.streaming_array["timestamp_ms"][:] = ts

    def release(self):
        if self.in_streaming:
            self.stop_streaming()
        if hasattr(self, 'sensor'):
            self.sensor.release()

    def __del__(self) -> None:
        self.release()

