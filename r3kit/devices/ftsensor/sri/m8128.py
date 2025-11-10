import os
from typing import Tuple, List, Dict, Union, Optional
import struct
import time
import gc
import tqdm
import numpy as np
from threading import Thread, Lock, Event
from multiprocessing import shared_memory, Manager
from copy import deepcopy
from functools import partial
import serial

from r3kit.devices.ftsensor.base import FTSensorBase
from r3kit.devices.ftsensor.sri.config import *
from r3kit.utils.vis import draw_time, draw_items


class M8128(FTSensorBase):
    def __init__(self, id:str=M8128_ID, baudrate:int=M8128_BAUDRATE, fps:int=M8128_FPS, name:str='M8128') -> None:
        super().__init__(name=name)

        self._id = id
        self._baudrate = baudrate
        self._fps = fps

        # serial
        self.ser = serial.Serial(id, baudrate=baudrate, timeout=M8128_TIMEOUT)
        if not self.ser.is_open:
            raise RuntimeError('Fail to open the serial port, please check your settings again.')
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(M8128_STARTTIME)
        self._cmd(f"AT+SMPF={fps}")
        self._cmd("AT+DCKMD=SUM")   # SUM check
        
        # config
        self.ft_dtype = np.dtype(np.float64)
        self.ft_shape = (6,)

        # stream
        self.in_streaming = Event()
    
    def __del__(self):
        if self.in_streaming.is_set():
            self._cmd("AT+GSD=STOP")
        self.ser.close()
    
    def _cmd(self, cmd:str) -> None:
        if not cmd.endswith("\r\n"):
            cmd += "\r\n"
        self.ser.write(cmd.encode("ascii"))
    
    def _read(self) -> Dict[str, Union[float, np.ndarray]]:
        self.ser.reset_input_buffer()
        self._cmd("AT+GOD")

        headbuf = self.ser.read_until(M8128_HDR)
        if not headbuf.endswith(M8128_HDR):
            return self._read()
        len_bytes = self.ser.read(2)
        if len(len_bytes) < 2:
            return self._read()
        length = (len_bytes[0] << 8) | len_bytes[1]
        if length != M8128_LEN_SUM:
            return self._read()
        payload = self.ser.read(length)
        if len(payload) < length:
            return self._read()
        pkgno = (payload[0] << 8) | payload[1]
        data_bytes = payload[2:2+24]
        sum_recv = payload[2+24]
        if (sum(data_bytes) & 0xFF) != sum_recv:
            return self._read()
        fx, fy, fz, mx, my, mz = struct.unpack("<6f", data_bytes)
        receive_time = time.time() * 1000
        return {'ft': np.array([fx, fy, fz, mx, my, mz]), 'timestamp_ms': receive_time}
    
    def get(self) -> Dict[str, Union[float, np.ndarray]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                data = {}
                data['ft'] = self.streaming_data['ft'][-1]
                data['timestamp_ms'] = self.streaming_data['timestamp_ms'][-1]
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                data = {}
                with self.streaming_lock:
                    data['ft'] = np.copy(self.streaming_array['ft'])
                    data['timestamp_ms'] = self.streaming_array['timestamp_ms'].item()
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
                    "ft": [], 
                    "timestamp_ms": []
                }
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()
                ft_memory_size = self.ft_dtype.itemsize * np.prod(self.ft_shape).item()
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size = ft_memory_size + timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                self.streaming_array = {
                    "ft": np.ndarray(self.ft_shape, dtype=self.ft_dtype, buffer=self.streaming_memory.buf[0:ft_memory_size]), 
                    "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[ft_memory_size:])
                }
                self.streaming_array_meta = {
                    "ft": (self.ft_shape, self.ft_dtype.name, (0, ft_memory_size)), 
                    "timestamp_ms": ((1,), np.float64.__name__, (ft_memory_size, ft_memory_size+timestamp_memory_size))
                }
            else:
                pass
        self._cmd("AT+GSD")
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def stop_streaming(self) -> Dict[str, Union[List[np.ndarray], List[float]]]:
        self.in_streaming.clear()
        self.thread.join()
        self._cmd("AT+GSD=STOP")
        if hasattr(self, "streaming_data"):
            streaming_data = self.streaming_data
            self.streaming_data = {
                "ft": [], 
                "timestamp_ms": []
            }
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "ft": [np.copy(self.streaming_array["ft"])], 
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
        assert len(streaming_data["ft"]) ==  len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        else:
            freq = 0
        np.save(os.path.join(save_path, "ft.npy"), np.array(streaming_data["ft"], dtype=float))
        draw_items(np.array(streaming_data["ft"], dtype=float), os.path.join(save_path, "ft.png"))
    
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
        elif hasattr(self, "streaming_array"):
            streaming_data = {
                "ft": [np.copy(self.streaming_array["ft"])], 
                "timestamp_ms": [self.streaming_array["timestamp_ms"].item()]
            }
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            self.streaming_data['ft'].clear()
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
                "ft": [], 
                "timestamp_ms": []
            }
        else:
            self.streaming_manager = Manager()
            self.streaming_lock = self.streaming_manager.Lock()
            ft_memory_size = self.ft_dtype.itemsize * np.prod(self.ft_shape).item()
            timestamp_memory_size = np.dtype(np.float64).itemsize
            streaming_memory_size = ft_memory_size + timestamp_memory_size
            self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
            self.streaming_array = {
                "ft": np.ndarray(self.ft_shape, dtype=self.ft_dtype, buffer=self.streaming_memory.buf[0:ft_memory_size]), 
                "timestamp_ms": np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[ft_memory_size:])
            }
            self.streaming_array_meta = {
                "ft": (self.ft_shape, self.ft_dtype.name, (0, ft_memory_size)), 
                "timestamp_ms": ((1,), np.float64.__name__, (ft_memory_size, ft_memory_size+timestamp_memory_size))
            }
    
    def _streaming_data(self, callback:Optional[callable]=None):
        buf = bytearray()
        while self.in_streaming.is_set():
            # get data
            if not self._collect_streaming_data:
                continue
            chunk = self.ser.read(M8128_CHUNKSIZE)
            receive_time = time.time() * 1000
            if chunk:
                buf.extend(chunk)
            else:
                continue
            # parse data
            fts = []
            while True:
                parsed, buf = self._parse(buf)
                if parsed is None:
                    break
                pkgno, fx, fy, fz, mx, my, mz = parsed
                fts.append([fx, fy, fz, mx, my, mz])
            datas = []
            for idx, ft in enumerate(fts):
                data = {'ft': np.array(ft), 'timestamp_ms': receive_time - (len(fts) - 1 - idx) * (1000 / self._fps)}
                datas.append(data)
            if callback is None:
                if hasattr(self, "streaming_data"):
                    self.streaming_mutex.acquire()
                    for data in datas:
                        self.streaming_data['ft'].append(data['ft'])
                        self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                    self.streaming_mutex.release()
                elif hasattr(self, "streaming_array"):
                    with self.streaming_lock:
                        self.streaming_array["ft"][:] = datas[-1]['ft'][:]
                        self.streaming_array["timestamp_ms"][:] = datas[-1]['timestamp_ms']
                else:
                    raise AttributeError
            else:
                for data in datas:
                    callback(deepcopy(data))
    
    @staticmethod
    def _parse(buf:bytearray) -> Tuple[Optional[Tuple[int, float, float, float, float, float, float]], bytearray]:
        i = 0
        n = len(buf)
        while i + M8128_FRAME_TOTAL <= n:
            # find frame header
            if buf[i] != M8128_HDR0 or buf[i+1] != M8128_HDR1:
                i += 1
                continue
            # check length
            length = (buf[i+2] << 8) | buf[i+3]
            if length != M8128_LEN_SUM:
                i += 1
                continue
            start = i
            end = i + 4 + M8128_LEN_SUM
            payload = buf[i+4:end]
            # package number
            pkgno = (payload[0] << 8) | payload[1]
            data_bytes = payload[2:2+24]
            # SUM check
            checksum = payload[2+24]
            if (sum(data_bytes) & 0xFF) != checksum:
                i += 1
                continue
            # valid frame
            fx, fy, fz, mx, my, mz = struct.unpack("<6f", data_bytes)
            return (pkgno, fx, fy, fz, mx, my, mz), buf[end:]
        # no valid frame
        keep = buf[-64:] if n > 64 else buf
        return None, keep
    
    @staticmethod
    def raw2tare(raw_ft:np.ndarray, tare:Dict[str, Union[float, np.ndarray]], pose:np.ndarray) -> np.ndarray:
        '''
        raw_ft: raw force torque data
        pose: 3x3 rotation matrix from ft300 to base
        '''
        raw_f, raw_t = raw_ft[:3], raw_ft[3:]
        f = raw_f - tare['f0']
        f -= np.linalg.inv(pose) @ np.array([0., 0., -9.8 * tare['m']])
        t = raw_t - tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(np.linalg.inv(pose) @ np.array(tare['c']), np.array([0., 0., -9.8 * tare['m']]))
        return np.concatenate([f, t])


if __name__ == '__main__':
    sensor = M8128(id='COM9', baudrate=115200, fps=100, name='M8128')
    streaming = True
    shm = False

    if not streaming:
        np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
        with tqdm.tqdm() as pbar:
            while True:
                data = sensor.get()
                pbar.update()
                pbar.set_description(str(data['ft']))
    else:
        sensor.collect_streaming(collect=True)
        sensor.shm_streaming(shm='M8128' if shm else None)
        sensor.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = sensor.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        print(streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
        data = {
            "ft": streaming_data["ft"][-1]
        }
        print(data)
