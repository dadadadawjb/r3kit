from typing import List, Dict
import time
import numpy as np
from multiprocessing import shared_memory, Manager


class ObsBuffer:
    def __init__(self, num_obs:int, obs_dict:Dict[str, tuple], create:bool=False) -> None:
        self.num_obs = num_obs
        self.obs_dict = obs_dict
        self.obs_names = set(obs_dict.keys())

        item_memory_size = {}
        for name in sorted(self.obs_dict.keys()):
            shape, dtype_name = self.obs_dict[name]
            dtype = np.dtype(getattr(np, dtype_name))
            item_memory_size[name] = dtype.itemsize * np.prod(shape).item()
        step_memory_size = sum(item_memory_size.values())
        flag_shape, flag_dtype = (1,), np.dtype('bool')
        flag_memory_size = flag_dtype.itemsize * np.prod(flag_shape).item()
        
        self.shm_manager = Manager()
        self.shm_lock = self.shm_manager.Lock()
        if create:
            self.shm_memory = shared_memory.SharedMemory(create=True, name='obs_buffer', size=step_memory_size * self.num_obs + flag_memory_size)
        else:
            while True:
                try:
                    self.shm_memory = shared_memory.SharedMemory(name='obs_buffer')
                    break
                except FileNotFoundError:
                    time.sleep(0.1)
        self.shm_arrays = []
        offset = 0
        for i in range(self.num_obs):
            shm_array = {}
            for name in sorted(self.obs_dict.keys()):
                shape, dtype_name = self.obs_dict[name]
                dtype = np.dtype(getattr(np, dtype_name))
                shm_array[name] = np.ndarray(shape, dtype=dtype, buffer=self.shm_memory.buf[offset:offset+item_memory_size[name]])
                offset += item_memory_size[name]
            self.shm_arrays.append(shm_array)
        self.shm_flag = np.ndarray(flag_shape, dtype=flag_dtype, buffer=self.shm_memory.buf[-flag_memory_size:])
        
        self.length = 0
        self.idx = 0
        self.setf(False)
    
    def __len__(self) -> int:
        return self.length
    
    def add1(self, obs:Dict[str, np.ndarray]) -> None:
        assert set(obs.keys()) == self.obs_names
        with self.shm_lock:
            for name in obs.keys():
                self.shm_arrays[self.idx][name][:] = obs[name]
            self.idx = (self.idx + 1) % self.num_obs
            self.length = min(self.length + 1, self.num_obs)
    
    def getn(self) -> List[Dict[str, np.ndarray]]:
        with self.shm_lock:
            obs = []
            for i in range(self.length):
                step_obs = {}
                for name in self.obs_names:
                    step_obs[name] = np.copy(self.shm_arrays[(self.idx + i) % self.num_obs][name])
                obs.append(step_obs)
        return obs
    
    def setf(self, flag:bool) -> None:
        with self.shm_lock:
            self.shm_flag[:] = flag
    
    def getf(self) -> bool:
        with self.shm_lock:
            return self.shm_flag[:].item()


class ActBuffer:
    def __init__(self, num_act:int, act_dict:Dict[str, tuple], create:bool=False) -> None:
        self.num_act = num_act
        self.act_dict = act_dict
        self.act_names = set(act_dict.keys())

        item_memory_size = {}
        for name in sorted(self.act_dict.keys()):
            shape, dtype_name = self.act_dict[name]
            dtype = np.dtype(getattr(np, dtype_name))
            item_memory_size[name] = dtype.itemsize * np.prod(shape).item()
        step_memory_size = sum(item_memory_size.values())
        flag_shape, flag_dtype = (1,), np.dtype('bool')
        flag_memory_size = flag_dtype.itemsize * np.prod(flag_shape).item()
        
        self.shm_manager = Manager()
        self.shm_lock = self.shm_manager.Lock()
        if create:
            self.shm_memory = shared_memory.SharedMemory(create=True, name='act_buffer', size=step_memory_size * self.num_act + flag_memory_size)
        else:
            while True:
                try:
                    self.shm_memory = shared_memory.SharedMemory(name='act_buffer')
                    break
                except FileNotFoundError:
                    time.sleep(0.1)
        self.shm_arrays = []
        offset = 0
        for i in range(self.num_act):
            shm_array = {}
            for name in sorted(self.act_dict.keys()):
                shape, dtype_name = self.act_dict[name]
                dtype = np.dtype(getattr(np, dtype_name))
                shm_array[name] = np.ndarray(shape, dtype=dtype, buffer=self.shm_memory.buf[offset:offset+item_memory_size[name]])
                offset += item_memory_size[name]
            self.shm_arrays.append(shm_array)
        self.shm_flag = np.ndarray(flag_shape, dtype=flag_dtype, buffer=self.shm_memory.buf[-flag_memory_size:])
        
        self.length = 0
        self.idx = 0
        self.setf(False)
    
    def __len__(self) -> int:
        return self.length
    
    def addn(self, obs:List[Dict[str, np.ndarray]]) -> None:
        assert len(obs) == self.num_act
        with self.shm_lock:
            for i in range(len(obs)):
                assert set(obs[i].keys()) == self.obs_names
                for name in obs[i].keys():
                    self.shm_arrays[self.idx][name][:] = obs[i][name]
        self.length = self.num_act
    
    def get1(self) -> Dict[str, np.ndarray]:
        with self.shm_lock:
            obs = {}
            for name in self.obs_names:
                obs[name] = np.copy(self.shm_arrays[self.idx][name])
        self.idx += 1
        return obs
    
    def setf(self, flag:bool) -> None:
        with self.shm_lock:
            self.shm_flag[:] = flag
    
    def getf(self) -> bool:
        with self.shm_lock:
            return self.shm_flag[:].item()
