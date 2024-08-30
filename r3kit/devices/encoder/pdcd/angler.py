import os
from typing import Dict, Optional
import struct
import time
import numpy as np
from threading import Thread, Lock, Event
from copy import deepcopy
from functools import partial
import serial

from r3kit.devices.encoder.base import EncoderBase
from r3kit.devices.encoder.pdcd.config import *
from r3kit.utils.vis import draw_time, draw_items

'''
Modified from: https://github.com/Galaxies99/easyrobot/blob/main/easyrobot/encoder/angle.py
'''


def hex2dex(e_hex):
    return int(e_hex, 16)

def hex2bin(e_hex):
    return bin(int(e_hex, 16))

def dex2bin(e_dex):
    return bin(e_dex)

def crc16(hex_num):
    """
    CRC16 verification
    :param hex_num:
    :return:
    """
    crc = '0xffff'
    crc16 = '0xA001'
    test = hex_num.split(' ')

    crc = hex2dex(crc)  
    crc16 = hex2dex(crc16) 
    for i in test:
        temp = '0x' + i
        temp = hex2dex(temp) 
        crc ^= temp  
        for i in range(8):
            if dex2bin(crc)[-1] == '0':
                crc >>= 1
            elif dex2bin(crc)[-1] == '1':
                crc >>= 1
                crc ^= crc16

    crc = hex(crc)
    crc_H = crc[2:4]
    crc_L = crc[-2:]

    return crc, crc_H, crc_L


class Angler(EncoderBase):
    def __init__(self, id:str=ANGLER_ID, index:int=ANGLER_INDEX, fps:int=ANGLER_FPS, 
                 baudrate:int=ANGLER_BAUDRATE, gap:float=ANGLER_GAP, name:str='Angler') -> None:
        super().__init__(name=name)

        self._id = id
        self._index = index
        self._fps = fps
        self._baudrate = baudrate
        self._gap = gap

        # serial
        self.ser = serial.Serial(id, baudrate=baudrate)
        if not self.ser.is_open:
            raise RuntimeError('Fail to open the serial port, please check your settings again.')
        self.ser.flushInput()
        self.ser.flushOutput()

        # stream
        self.in_streaming = Event()
    
    def _read(self) -> Optional[Dict[str, float]]:
        self.ser.flushInput()

        sendbytes = str(self._index).zfill(2) + " 03 00 41 00 03"
        crc, crc_H, crc_L = crc16(sendbytes)
        sendbytes = sendbytes + ' ' + crc_L + ' ' + crc_H
        sendbytes = bytes.fromhex(sendbytes)
        self.ser.write(sendbytes)
        receive_time = time.time() * 1000
        time.sleep(self._gap)
        re = self.ser.read(11)
        if self.ser.inWaiting() > 0:
            se = self.ser.read_all()
            re += se
        
        receive = False
        ret = 0
        for b in range(len(re) - 10):
            if re[b + 1] == 3 and re[b + 2] == 6 and re[b] == self._index:
                angle = 360 * (re[b + 3] * 256 + re[b + 4]) / 4096
                ret = angle
                receive = True
        if not receive:
            return None
        return {'angle': ret, 'timestamp_ms': receive_time}
    
    def get(self) -> Optional[Dict[str, float]]:
        if not self.in_streaming.is_set():
            data = self._read()
        else:
            self.streaming_mutex.acquire()
            data = {}
            data['angle'] = self.streaming_data['angle'][-1]
            data['timestamp_ms'] = self.streaming_data['timestamp_ms'][-1]
            self.streaming_mutex.release()
        return data
    
    def get_mean_data(self, n=10, name='angle') -> float:
        assert name in ['angle'], 'name must be one of [angle]'
        tare_list = []
        count = 0
        while count < n:
            data = self.get()
            if data is not None:
                tare_list.append(data[name])
                count += 1
        tare = sum(tare_list) / n
        return tare
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
        self.in_streaming.set()
        self.streaming_mutex = Lock()
        self.streaming_data = {
            "angle": [], 
            "timestamp_ms": []
        }
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def stop_streaming(self) -> dict:
        self.in_streaming.clear()
        self.thread.join()
        self.streaming_mutex = None
        streaming_data = self.streaming_data
        self.streaming_data = {
            "angle": [], 
            "timestamp_ms": []
        }
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        assert len(streaming_data["angle"]) == len(streaming_data["timestamp_ms"])
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
        draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
        np.save(os.path.join(save_path, "angle.npy"), np.array(streaming_data["angle"], dtype=float))
        draw_items(np.array(streaming_data["angle"], dtype=float), os.path.join(save_path, "angle.png"))
    
    def collect_streaming(self, collect:bool=True) -> None:
        self._collect_streaming_data = collect
    
    def _streaming_data(self, callback:Optional[callable]=None):
        while self.in_streaming.is_set():
            # fps
            time.sleep(1/self._fps)

            # get data
            if not self._collect_streaming_data:
                continue
            data = self._read()
            if data is not None:
                if callback is None:
                    self.streaming_mutex.acquire()
                    self.streaming_data['angle'].append(data['angle'])
                    self.streaming_data['timestamp_ms'].append(data['timestamp_ms'])
                    self.streaming_mutex.release()
                else:
                    callback(deepcopy(data))
    
    @staticmethod
    def raw2angle(raw:np.ndarray) -> np.ndarray:
        result = []
        assert len(raw) > 100
        if np.any(raw[:10] < 1) and np.any(raw[:10] > 359):
            initial_angle = raw[0]
        else:
            assert np.quantile(raw[:10], 0.75) - np.quantile(raw[:10], 0.25) < 1, np.quantile(raw[:10], 0.75) - np.quantile(raw[:10], 0.25)
            initial_angle = np.median(raw[:10])
        count = 0
        result.append(raw[0] - initial_angle + 360 * count)
        for i in range(1, len(raw)):
            if abs(raw[i] - raw[i-1]) > 100:
                count +=  1 if raw[i] - raw[i-1] < 0 else -1
            result.append(raw[i] - initial_angle + 360 * count)
        return np.array(result)


if __name__ == "__main__":
    encoder = Angler(id='/dev/ttyUSB0', index=1, baudrate=115200, fps=30, gap=0.002, name='Angler')

    while True:
        data = encoder.get()
        print(data)
        time.sleep(0.1)
