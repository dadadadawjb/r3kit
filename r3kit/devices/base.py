from typing import Tuple, Dict, Optional
import os
import json
from abc import ABC
from multiprocessing import shared_memory, Manager
from threading import Lock
import numpy as np

from .config import META_DIR


class DeviceBase(ABC):
    name: str
    
    def __init__(self, name:str='') -> None:
        self.name = name
        pass

    def _save_streaming_meta(self, meta:dict, suffix:str="", meta_path:Optional[str]=None) -> None:
        assert hasattr(self, '_shm'), "Device does not have shared memory attribute."
        if meta_path is None:
            os.makedirs(META_DIR, exist_ok=True)
            meta_path = os.path.join(META_DIR, f'{self._shm}{suffix}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

    @staticmethod
    def _load_streaming_meta(shm:str, suffix:str="", meta_path:Optional[str]=None) -> dict:
        if meta_path is None:
            meta_path = os.path.join(META_DIR, f'{shm}{suffix}.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        return meta

    @staticmethod
    def get_streaming_memory(shm:str, suffix:str="", meta_path:Optional[str]=None) -> Tuple[Dict[str, np.ndarray], shared_memory.SharedMemory, Lock]:
        streaming_array_meta = DeviceBase._load_streaming_meta(shm, suffix, meta_path)

        streaming_manager = Manager()
        this_streaming_lock = streaming_manager.Lock()
        this_shared_memory = shared_memory.SharedMemory(name=shm+suffix)
        this_streaming_array = {k: np.ndarray(streaming_array_meta[k][0], dtype=getattr(np, streaming_array_meta[k][1]), 
                                    buffer=this_shared_memory.buf[streaming_array_meta[k][2][0]:streaming_array_meta[k][2][1]]) for k in streaming_array_meta.keys()}
        return this_streaming_array, this_shared_memory, this_streaming_lock
