import os
import shutil
from typing import Tuple, List, Dict, Union, Sequence, Iterable, Optional
import time
import gc
from rich import print
import numpy as np
from threading import Lock, Thread
from multiprocessing import shared_memory, Manager
from copy import deepcopy
from functools import partial
import pyrealsense2 as rs

from r3kit.devices.config import *
from r3kit.devices.camera.realsense.config import *
from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.realsense.query import RealSenseQuery
from r3kit.utils.vis import draw_time, save_imgs, save_img
from r3kit.utils.transformation import xyzrot2mat, xyzquat2mat
from r3kit import DEBUG, INFO


class RealSenseCamera(CameraBase):
    STREAM_TYPE_FORMAT_DEFAULTS = {
        rs.stream.color:   rs.format.bgr8,
        rs.stream.depth:   rs.format.z16,
        rs.stream.fisheye: rs.format.y8,
        rs.stream.pose:    rs.format.six_dof,
    }

    def __init__(self, id:Optional[str]=None, streams:Iterable[Sequence]=(), name:str='RealSense') -> None:
        """
        streams = [
            ('color',),                         # only type
            ('color', 0),                       # type + index
            ('color', 640, 480),                # type + width + height
            ('color', 640, 480, 'bgr8'),        # type + width + height + format
            ('color', 640, 480, 'bgr8', 30),    # type + width + height + format + fps
            ('color', 'bgr8'),                  # type + format
            ('color', 'bgr8', 30),              # type + format + fps
            ('color', 0, 640, 480),             # type + index + width + height
            ('color', 0, 640, 480, 'bgr8'),     # type + index + width + height + format
            ('color', 0, 640, 480, 30),         # type + index + width + height + fps
            ('color', 0, 640, 480, 'bgr8', 30)  # explicitly specify all
        ]
        examples:
        D415: [('depth', -1, 640, 480, 30), ('color', -1, 640, 480, 30)]
        T265: [('pose', 'six_dof', 200), ('fisheye', 1, 848, 800, 30), ('fisheye', 2, 848, 800, 30)]
        assume:
        1 color, 1 color + 1 depth, 1 pose, 1 pose + 2 fisheye
        query:
        use `query.py` to get serial number as id and available streams
        """
        super().__init__(name)

        # enable
        self.color_idx = []
        self.depth_idx = []
        self.fisheye_idx = []
        self.pose_idx = []
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.query = RealSenseQuery()
        if id is not None:
            try:
                self.config.enable_device(id)
            except RuntimeError as e:
                device_sns = list(self.query.devices().keys())
                raise RuntimeError(f"Failed to enable RealSense device with serial number {id}. Available devices: {device_sns}.") from e
        else:
            device_sns = list(self.query.devices().keys())
            assert len(device_sns) == 1
            id = device_sns[0]
        for stream in streams:
            rs_stream, index, width, height, fmt, fps = self._resolve_stream(stream)
            if DEBUG:
                print(f"[DEBUG-r3kit] {self.name}: enabling stream {rs_stream}, index={index}, width={width}, height={height}, format={fmt}, fps={fps}")
            try:
                self.config.enable_stream(rs_stream, index, width, height, fmt, fps)
            except RuntimeError as e:
                sensors = self.query.sensors(id)
                available_streams = []
                for sensor_name, profiles in sensors.items():
                    available_streams += profiles
                raise RuntimeError(f"Failed to enable RealSense stream {rs_stream}, index={index}, width={width}, height={height}, format={fmt}, fps={fps}. Available streams: {available_streams}") from e
            if rs_stream == rs.stream.color:
                self.color_idx.append(index)
            if rs_stream == rs.stream.depth:
                self.depth_idx.append(index)
            if rs_stream == rs.stream.fisheye:
                self.fisheye_idx.append(index)
            if rs_stream == rs.stream.pose:
                self.pose_idx.append(index)
        assert len(self.color_idx) in [0, 1], "Only support at most 1 color stream"
        assert len(self.depth_idx) in [0, 1], "Only support at most 1 depth stream"
        assert len(self.pose_idx) in [0, 1], "Only support at most 1 pose stream"
        self.has_img = len(self.color_idx) > 0 or len(self.depth_idx) > 0 or len(self.fisheye_idx) > 0
        if INFO:
            print(f"[INFO-r3kit] {self.name}: configured with {len(self.color_idx)} color, {len(self.depth_idx)} depth, {len(self.fisheye_idx)} fisheye, {len(self.pose_idx)} pose streams")
        
        # option
        # NOTE: hard code pose options to balance accuracy and smoothness
        if len(self.pose_idx) > 0:
            pipeline_profile = self.pipeline.start(self.config)
            pose_sensor = pipeline_profile.get_device().first_pose_sensor()
            self.pipeline.stop()
            # pose_sensor.set_option(rs.option.enable_mapping, 0)
            pose_sensor.set_option(rs.option.enable_pose_jumping, 0)
            # pose_sensor.set_option(rs.option.enable_relocalization, 0)

        # config
        pipeline_profile = self.pipeline.start(self.config)
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            if DEBUG:
                print(f"[DEBUG-r3kit] {self.name} failed to get frames, resetting...")
            device = pipeline_profile.get_device()
            device.hardware_reset()
            time.sleep(REALSENSE_RESET_TIME)
            frames = self.pipeline.wait_for_frames()
        if len(self.depth_idx) > 0:
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            if len(self.color_idx) > 0:
                # NOTE: hard code depth alignment to color
                self.align = rs.align(rs.stream.color)
                color_frame = frames.get_color_frame().as_video_frame()
                depth_frame = frames.get_depth_frame().as_depth_frame()
                depth2color = depth_frame.get_profile().get_extrinsics_to(color_frame.get_profile())
                self.depth2color = xyzrot2mat(depth2color.translation, np.array(depth2color.rotation).reshape((3, 3)))
                frames = self.align.process(frames)
            
            depth_frame = frames.get_depth_frame().as_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            self.depth_image_dtype = depth_image.dtype
            self.depth_image_shape = depth_image.shape
        if len(self.color_idx) > 0:
            color_frame = frames.get_color_frame()
            color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = [color_intrinsics.ppx, color_intrinsics.ppy, color_intrinsics.fx, color_intrinsics.fy]
            color_frame = color_frame.as_video_frame()
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            self.color_image_dtype = color_image.dtype
            self.color_image_shape = color_image.shape
        if len(self.pose_idx) > 0:
            pose_frame = frames.get_pose_frame()
            pose_data = pose_frame.get_pose_data()
            xyz = np.array([pose_data.translation.x, pose_data.translation.y, pose_data.translation.z], dtype=np.float64)
            quat = np.array([pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w], dtype=np.float64)
            self.xyz_dtype = xyz.dtype
            self.xyz_shape = xyz.shape
            self.quat_dtype = quat.dtype
            self.quat_shape = quat.shape
        if len(self.fisheye_idx) > 0:
            self.fisheye_image_dtypes = []
            self.fisheye_image_shapes = []
            for fisheye_index in self.fisheye_idx:
                fisheye_frame = frames.get_fisheye_frame(fisheye_index).as_video_frame()
                fisheye_image = np.asanyarray(fisheye_frame.get_data(), dtype=np.uint8)
                self.fisheye_image_dtypes.append(fisheye_image.dtype)
                self.fisheye_image_shapes.append(fisheye_image.shape)
        
        self.in_streaming = False
        self._collect_streaming_data = True
        self._shm = None
        self._streaming_save_path = None
    
    def _read(self) -> Dict[str, Union[np.ndarray, float]]:
        result = {}
        frames = self.pipeline.wait_for_frames()
        result["timestamp_ms"] = time.time() * 1000
        if len(self.depth_idx) > 0 and len(self.color_idx) > 0:
            frames = self.align.process(frames)
        if len(self.color_idx) > 0:
            color_frame = frames.get_color_frame().as_video_frame()
            result["color"] = np.asanyarray(color_frame.get_data(), dtype=np.uint8).copy()
        if len(self.depth_idx) > 0:
            depth_frame = frames.get_depth_frame().as_depth_frame()
            result["depth"] = np.asanyarray(depth_frame.get_data(), dtype=np.uint16).copy()
        if len(self.fisheye_idx) > 0:
            for fisheye_index in self.fisheye_idx:
                fisheye_frame = frames.get_fisheye_frame(fisheye_index).as_video_frame()
                result[f"fisheye_{fisheye_index}"] = np.asanyarray(fisheye_frame.get_data(), dtype=np.uint8).copy()
        if len(self.pose_idx) > 0:
            pose_frame = frames.get_pose_frame()
            pose_data = pose_frame.get_pose_data()
            xyz = np.array([pose_data.translation.x, pose_data.translation.y, pose_data.translation.z], dtype=np.float64)
            quat = np.array([pose_data.rotation.x, pose_data.rotation.y, pose_data.rotation.z, pose_data.rotation.w], dtype=np.float64)
            result["xyz"] = xyz
            result["quat"] = quat
        return result
    
    def get(self) -> Dict[str, Union[np.ndarray, float]]:
        result = {}
        if not self.in_streaming:
            result = self._read()
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                if len(self.color_idx) > 0:
                    result["color"] = self.streaming_data["color"][-1].copy()
                if len(self.depth_idx) > 0:
                    result["depth"] = self.streaming_data["depth"][-1].copy()
                if len(self.fisheye_idx) > 0:
                    for fisheye_index in self.fisheye_idx:
                        result[f"fisheye_{fisheye_index}"] = self.streaming_data[f"fisheye_{fisheye_index}"][-1].copy()
                if len(self.pose_idx) > 0:
                    result["xyz"] = self.streaming_data["xyz"][-1].copy()
                    result["quat"] = self.streaming_data["quat"][-1].copy()
                result["timestamp_ms"] = self.streaming_data["timestamp_ms"][-1]
                self.streaming_mutex.release()
            elif hasattr(self, "streaming_array"):
                with self.streaming_lock:
                    if len(self.color_idx) > 0:
                        result["color"] = np.copy(self.streaming_array["color"])
                    if len(self.depth_idx) > 0:
                        result["depth"] = np.copy(self.streaming_array["depth"])
                    if len(self.fisheye_idx) > 0:
                        for fisheye_index in self.fisheye_idx:
                            result[f"fisheye_{fisheye_index}"] = np.copy(self.streaming_array[f"fisheye_{fisheye_index}"])
                    if len(self.pose_idx) > 0:
                        result["xyz"] = np.copy(self.streaming_array["xyz"])
                        result["quat"] = np.copy(self.streaming_array["quat"])
                    result["timestamp_ms"] = self.streaming_array["timestamp_ms"].item()
            else:
                raise AttributeError
        return result
    
    def start_streaming(self, streaming_save_path:Optional[str]=None, callback:Optional[callable]=None) -> None:
        self._prepare_streaming(streaming_save_path=streaming_save_path, callback=callback)
        self.in_streaming = True
        self.thread = Thread(target=partial(self._streaming_data, callback=callback), daemon=True)
        self.thread.start()
    
    def _prepare_streaming(self, streaming_save_path:Optional[str]=None, callback:Optional[callable]=None) -> None:
        if self._shm is None:
            if callback is None:
                self.streaming_mutex = Lock()

                self.streaming_data = {
                    "timestamp_ms": []
                }
                if len(self.color_idx) > 0:
                    self.streaming_data["color"] = []
                if len(self.depth_idx) > 0:
                    self.streaming_data["depth"] = []
                if len(self.pose_idx) > 0:
                    self.streaming_data["xyz"] = []
                    self.streaming_data["quat"] = []
                if len(self.fisheye_idx) > 0:
                    for fisheye_index in self.fisheye_idx:
                        self.streaming_data[f"fisheye_{fisheye_index}"] = []
                
                if self.has_img:
                    if streaming_save_path is not None:
                        self._streaming_save_path = streaming_save_path if streaming_save_path != "" else os.path.join(TEMP_DIR, self.name)
                        if os.path.exists(self._streaming_save_path):
                            shutil.rmtree(self._streaming_save_path)
                        self._write_flag = True
                        self._write_idx = 0
                        self._image_streaming_data_writer = Thread(target=self._write_image_streaming_data, args=(self._streaming_save_path,))
                        self._image_streaming_data_writer.start()
                    else:
                        self._streaming_save_path = None
            else:
                pass
        else:
            if callback is None:
                self.streaming_manager = Manager()
                self.streaming_lock = self.streaming_manager.Lock()

                streaming_memory_size = 0
                if len(self.color_idx) > 0:
                    color_memory_size = self.color_image_dtype.itemsize * np.prod(self.color_image_shape).item()
                    streaming_memory_size += color_memory_size
                if len(self.depth_idx) > 0:
                    depth_memory_size = self.depth_image_dtype.itemsize * np.prod(self.depth_image_shape).item()
                    streaming_memory_size += depth_memory_size
                if len(self.fisheye_idx) > 0:
                    fisheye_memory_size = []
                    for shape, dtype in zip(self.fisheye_image_shapes, self.fisheye_image_dtypes):
                        fisheye_memory_size.append(dtype.itemsize * np.prod(shape).item())
                    streaming_memory_size += sum(fisheye_memory_size)
                if len(self.pose_idx) > 0:
                    xyz_memory_size = self.xyz_dtype.itemsize * np.prod(self.xyz_shape).item()
                    quat_memory_size = self.quat_dtype.itemsize * np.prod(self.quat_shape).item()
                    streaming_memory_size += xyz_memory_size + quat_memory_size
                timestamp_memory_size = np.dtype(np.float64).itemsize
                streaming_memory_size += timestamp_memory_size
                self.streaming_memory = shared_memory.SharedMemory(name=self._shm, create=True, size=streaming_memory_size)
                
                offset = 0
                self.streaming_array = {}
                self.streaming_array_meta = {}
                if len(self.color_idx) > 0:
                    self.streaming_array["color"] = np.ndarray(self.color_image_shape, dtype=self.color_image_dtype, buffer=self.streaming_memory.buf[offset:offset+color_memory_size])
                    self.streaming_array_meta["color"] = (self.color_image_shape, self.color_image_dtype.name, (offset, offset+color_memory_size))
                    offset += color_memory_size
                if len(self.depth_idx) > 0:
                    self.streaming_array["depth"] = np.ndarray(self.depth_image_shape, dtype=self.depth_image_dtype, buffer=self.streaming_memory.buf[offset:offset+depth_memory_size])
                    self.streaming_array_meta["depth"] = (self.depth_image_shape, self.depth_image_dtype.name, (offset, offset+depth_memory_size))
                    offset += depth_memory_size
                if len(self.fisheye_idx) > 0:
                    for fisheye_memory_size, fisheye_shape, fisheye_dtype, fisheye_index in zip(fisheye_memory_size, self.fisheye_image_shapes, self.fisheye_image_dtypes, self.fisheye_idx):
                        self.streaming_array[f"fisheye_{fisheye_index}"] = np.ndarray(fisheye_shape, dtype=fisheye_dtype, buffer=self.streaming_memory.buf[offset:offset+fisheye_memory_size])
                        self.streaming_array_meta[f"fisheye_{fisheye_index}"] = (fisheye_shape, fisheye_dtype.name, (offset, offset+fisheye_memory_size))
                        offset += fisheye_memory_size
                if len(self.pose_idx) > 0:
                    self.streaming_array["xyz"] = np.ndarray(self.xyz_shape, dtype=self.xyz_dtype, buffer=self.streaming_memory.buf[offset:offset+xyz_memory_size])
                    self.streaming_array_meta["xyz"] = (self.xyz_shape, self.xyz_dtype.name, (offset, offset+xyz_memory_size))
                    offset += xyz_memory_size
                    self.streaming_array["quat"] = np.ndarray(self.quat_shape, dtype=self.quat_dtype, buffer=self.streaming_memory.buf[offset:offset+quat_memory_size])
                    self.streaming_array_meta["quat"] = (self.quat_shape, self.quat_dtype.name, (offset, offset+quat_memory_size))
                    offset += quat_memory_size
                self.streaming_array["timestamp_ms"] = np.ndarray((1,), dtype=np.float64, buffer=self.streaming_memory.buf[offset:offset+timestamp_memory_size])
                self.streaming_array_meta["timestamp_ms"] = ((1,), np.float64.__name__, (offset, offset+timestamp_memory_size))
                self._save_streaming_meta(self.streaming_array_meta)
            else:
                pass
    
    def stop_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        if hasattr(self, "streaming_data"):
            if hasattr(self, "_write_flag"):
                self._write_flag = False
                if DEBUG:
                    writer_time = time.time()
                self._image_streaming_data_writer.join()
                if DEBUG:
                    writer_time = time.time() - writer_time
                    print(f"[DEBUG-r3kit] {self.name} stop_writer join time: {writer_time} seconds")

            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} stop_streaming data size: {len(streaming_data['timestamp_ms'])}")
            del self.streaming_data
            del self.streaming_mutex
        elif hasattr(self, "streaming_array"):
            streaming_data = {}
            with self.streaming_lock:
                for key in self.streaming_array.keys():
                    if key == "timestamp_ms":
                        streaming_data[key] = self.streaming_array[key].item()
                    else:
                        streaming_data[key] = np.copy(self.streaming_array[key])
            self.streaming_memory.close()
            self.streaming_memory.unlink()
            del self.streaming_memory
            del self.streaming_array, self.streaming_array_meta
            del self.streaming_manager
            del self.streaming_lock
        else:
            raise AttributeError
        self.in_streaming = False
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:Dict[str, List[Union[np.ndarray, float]]]) -> None:
        has_writer = False
        if self._streaming_save_path and os.path.exists(self._streaming_save_path):
            for root, dirs, files in os.walk(self._streaming_save_path):
                if len(files) > 0:
                    has_writer = True
                    break
        
        if len(self.color_idx) > 0:
            np.savetxt(os.path.join(save_path, "intrinsics.txt"), self.color_intrinsics, fmt="%.16f")
        if len(self.depth_idx) > 0:
            np.savetxt(os.path.join(save_path, "depth_scale.txt"), [self.depth_scale], fmt="%.16f")
            if len(self.color_idx) > 0:
                np.savetxt(os.path.join(save_path, "depth2color.txt"), self.depth2color, fmt="%.16f")
        np.save(os.path.join(save_path, "timestamps.npy"), np.array(streaming_data["timestamp_ms"], dtype=float))
        if len(streaming_data["timestamp_ms"]) > 1:
            freq = len(streaming_data["timestamp_ms"]) / (streaming_data["timestamp_ms"][-1] - streaming_data["timestamp_ms"][0])
            if INFO:
                draw_time(streaming_data["timestamp_ms"], os.path.join(save_path, f"freq_{freq}.png"))
            else:
                np.savetxt(os.path.join(save_path, f"freq_{freq}.txt"), np.array([]))
        else:
            freq = 0
        
        idx_bias = self._write_idx if has_writer else 0
        if has_writer and not os.path.samefile(save_path, self._streaming_save_path):
            if DEBUG:
                clean_time = time.time()
            if len(self.color_idx) > 0:
                shutil.move(os.path.join(self._streaming_save_path, 'color'), os.path.join(save_path, 'color'))
            if len(self.depth_idx) > 0:
                shutil.move(os.path.join(self._streaming_save_path, 'depth'), os.path.join(save_path, 'depth'))
            if len(self.fisheye_idx) > 0:
                for fisheye_index in self.fisheye_idx:
                    shutil.move(os.path.join(self._streaming_save_path, f'fisheye_{fisheye_index}'), os.path.join(save_path, f'fisheye_{fisheye_index}'))
            shutil.rmtree(self._streaming_save_path)
            if DEBUG:
                clean_time = time.time() - clean_time
                print(f"[DEBUG-r3kit] {self.name} clean time: {clean_time} seconds")
        else:
            if len(self.color_idx) > 0:
                os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)
            if len(self.depth_idx) > 0:
                os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)
            if len(self.fisheye_idx) > 0:
                for fisheye_index in self.fisheye_idx:
                    os.makedirs(os.path.join(save_path, f'fisheye_{fisheye_index}'), exist_ok=True)
        if len(self.pose_idx) > 0:
            os.makedirs(os.path.join(save_path, 'pose'), exist_ok=True)
        if DEBUG:
            save_time = time.time()
        if len(self.color_idx) > 0:
            save_imgs(os.path.join(save_path, 'color'), streaming_data["color"], idx_bias=idx_bias)
        if len(self.depth_idx) > 0:
            save_imgs(os.path.join(save_path, 'depth'), streaming_data["depth"], idx_bias=idx_bias)
        if len(self.fisheye_idx) > 0:
            for fisheye_index in self.fisheye_idx:
                save_imgs(os.path.join(save_path, f'fisheye_{fisheye_index}'), streaming_data[f"fisheye_{fisheye_index}"], idx_bias=idx_bias)
        if len(self.pose_idx) > 0:
            np.save(os.path.join(save_path, 'pose', "xyz.npy"), np.array(streaming_data["xyz"], dtype=float))
            np.save(os.path.join(save_path, 'pose', "quat.npy"), np.array(streaming_data["quat"], dtype=float))
        if DEBUG:
            save_time = time.time() - save_time
            print(f"[DEBUG-r3kit] {self.name} save time: {save_time} seconds")
    
    def _write_image_streaming_data(self, save_path:str) -> None:
        os.makedirs(save_path, exist_ok=True)
        if len(self.color_idx) > 0:
            os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)
        if len(self.depth_idx) > 0:
            os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)
        if len(self.fisheye_idx) > 0:
            for fisheye_index in self.fisheye_idx:
                os.makedirs(os.path.join(save_path, f'fisheye_{fisheye_index}'), exist_ok=True)
        while True:
            if not self._write_flag:
                break

            to_write = False
            self.streaming_mutex.acquire()
            if len(self.color_idx) > 0 and len(self.streaming_data["color"]) > 0:
                to_write = True
                color_img = self.streaming_data["color"].pop(0)
            else:
                color_img = None
            if len(self.depth_idx) > 0 and len(self.streaming_data["depth"]) > 0:
                to_write = True
                depth_img = self.streaming_data["depth"].pop(0)
            else:
                depth_img = None
            if len(self.fisheye_idx) > 0:
                fisheye_imgs = {}
                for fisheye_index in self.fisheye_idx:
                    if len(self.streaming_data[f"fisheye_{fisheye_index}"]) > 0:
                        to_write = True
                        fisheye_imgs[fisheye_index] = self.streaming_data[f"fisheye_{fisheye_index}"].pop(0)
            else:
                fisheye_imgs = None
            self.streaming_mutex.release()
            if to_write:
                if color_img is not None:
                    save_img(self._write_idx, os.path.join(save_path, 'color'), color_img)
                if depth_img is not None:
                    save_img(self._write_idx, os.path.join(save_path, 'depth'), depth_img)
                if fisheye_imgs is not None:
                    for fisheye_index, fisheye_img in fisheye_imgs.items():
                        save_img(self._write_idx, os.path.join(save_path, f'fisheye_{fisheye_index}'), fisheye_img)
                self._write_idx += 1
                if DEBUG:
                    print(f"[DEBUG-r3kit] {self.name} writer {self._write_idx}")
    
    def collect_streaming(self, collect:bool=True) -> None:
        # NOTE: only valid for no-custom-callback
        self._collect_streaming_data = collect
    
    def shm_streaming(self, shm:Optional[str]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert (not self.in_streaming) or (not self._collect_streaming_data)
        self._shm = shm
    
    def get_streaming(self) -> Dict[str, List[Union[np.ndarray, float]]]:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            if hasattr(self, "_write_flag"):
                self._write_flag = False
                if DEBUG:
                    writer_time = time.time()
                self._image_streaming_data_writer.join()
                if DEBUG:
                    writer_time = time.time() - writer_time
                    print(f"[DEBUG-r3kit] {self.name} get_writer join time: {writer_time} seconds")

            streaming_data = self.streaming_data
            if INFO:
                print(f"[INFO-r3kit] {self.name} get_streaming data size: {len(streaming_data['timestamp_ms'])}")
        elif hasattr(self, "streaming_array"):
            streaming_data = {}
            with self.streaming_lock:
                for key in self.streaming_array.keys():
                    if key == "timestamp_ms":
                        streaming_data[key] = self.streaming_array[key].item()
                    else:
                        streaming_data[key] = np.copy(self.streaming_array[key])
        else:
            raise AttributeError
        return streaming_data
    
    def reset_streaming(self, streaming_save_path:Optional[str]=None, callback:Optional[callable]=None) -> None:
        # NOTE: only valid for non-custom-callback
        assert not self._collect_streaming_data
        if hasattr(self, "streaming_data"):
            if hasattr(self, "_write_flag"):
                self._write_flag = False
                if DEBUG:
                    writer_time = time.time()
                self._image_streaming_data_writer.join()
                if DEBUG:
                    writer_time = time.time() - writer_time
                    print(f"[DEBUG-r3kit] {self.name} reset_writer join time: {writer_time} seconds")

            for key in self.streaming_data.keys():
                self.streaming_data[key].clear()
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
        
        self._prepare_streaming(streaming_save_path=streaming_save_path, callback=callback)
    
    def _streaming_data(self, callback:Optional[callable]=None):
        while self.in_streaming:
            # get data
            if not self._collect_streaming_data:
                continue

            result = self._read()
            if callback is None:
                if hasattr(self, "streaming_data"):
                    self.streaming_mutex.acquire()
                    if len(self.color_idx) > 0:
                        self.streaming_data["color"].append(result["color"])
                    if len(self.depth_idx) > 0:
                        self.streaming_data["depth"].append(result["depth"])
                    if len(self.fisheye_idx) > 0:
                        for fisheye_index in self.fisheye_idx:
                            self.streaming_data[f"fisheye_{fisheye_index}"].append(result[f"fisheye_{fisheye_index}"])
                    if len(self.pose_idx) > 0:
                        self.streaming_data["xyz"].append(result["xyz"])
                        self.streaming_data["quat"].append(result["quat"])
                    self.streaming_data["timestamp_ms"].append(result["timestamp_ms"])
                    self.streaming_mutex.release()
                elif hasattr(self, "streaming_array"):
                    with self.streaming_lock:
                        if len(self.color_idx) > 0:
                            self.streaming_array["color"][:] = result["color"][:]
                        if len(self.depth_idx) > 0:
                            self.streaming_array["depth"][:] = result["depth"][:]
                        if len(self.fisheye_idx) > 0:
                            for fisheye_index in self.fisheye_idx:
                                self.streaming_array[f"fisheye_{fisheye_index}"][:] = result[f"fisheye_{fisheye_index}"][:]
                        if len(self.pose_idx) > 0:
                            self.streaming_array["xyz"][:] = result["xyz"][:]
                            self.streaming_array["quat"][:] = result["quat"][:]
                        self.streaming_array["timestamp_ms"][:] = result["timestamp_ms"]
                else:
                    raise AttributeError
            else:
                callback(deepcopy(result))
    
    @staticmethod
    def img2pc(depth_img:np.ndarray, intrinsics:np.ndarray, color_img:Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # depth_img: already scaled by depth_scale
        # intrinsics: [ppx, ppy, fx, fy]
        # color_img: already converted to rgb and scaled to [0, 1]
        height, weight = depth_img.shape
        [pixX, pixY] = np.meshgrid(np.arange(weight), np.arange(height))
        x = (pixX - intrinsics[0]) * depth_img / intrinsics[2]
        y = (pixY - intrinsics[1]) * depth_img / intrinsics[3]
        z = depth_img
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        if color_img is None:
            rgb = None
        else:
            rgb = color_img.reshape(-1, 3)
        return xyz, rgb
    
    @staticmethod
    def raw2pose(xyz:np.ndarray, quat:np.ndarray) -> np.ndarray:
        return xyzquat2mat(xyz, quat)
    
    @staticmethod
    def _resolve_stream(stream:Sequence) -> Tuple[rs.stream, int, int, int, rs.format, int]:
        if len(stream) == 0:
            raise ValueError("Stream spec cannot be empty")

        name_or_stream = stream[0]
        args = list(stream[1:])

        if isinstance(name_or_stream, str):
            key = name_or_stream.lower()
            if not hasattr(rs.stream, key):
                available = [k for k in dir(rs.stream) if not k.startswith("_")]
                raise ValueError(f"Unknown RealSense stream type name '{name_or_stream}'. Available examples: {available[:10]} ...")
            rs_stream = getattr(rs.stream, key)
        elif isinstance(name_or_stream, type(rs.stream.any)):
            rs_stream = name_or_stream
        else:
            raise TypeError(f"First element of stream spec must be str or rs.stream, got {type(name_or_stream)}")
        default_format = RealSenseCamera.STREAM_TYPE_FORMAT_DEFAULTS.get(rs_stream, rs.format.any)

        stream_index = -1
        width = 0
        height = 0
        rs_format = default_format
        framerate = 0

        have_format = False
        new_args_before, new_args_after = [], []
        for a in args:
            if isinstance(a, str):
                fmt_key = a.lower()
                if not hasattr(rs.format, fmt_key):
                    available = [k for k in dir(rs.format) if not k.startswith("_")]
                    raise ValueError(f"Unknown RealSense format '{a}'. Available examples: {available[:10]} ...")
                rs_format = getattr(rs.format, fmt_key)
                assert not have_format, "Cannot specify format more than once"
                have_format = True
            elif isinstance(a, type(rs.format.any)):
                rs_format = a
                assert not have_format, "Cannot specify format more than once"
                have_format = True
            else:
                if have_format:
                    new_args_after.append(a)
                else:
                    new_args_before.append(a)

        assert len(new_args_after) == 0 or len(new_args_after) == 1, f"Cannot mix format position in stream spec: {stream}"
        framerate = int(new_args_after[0]) if len(new_args_after) == 1 else 0
        args = new_args_before # index / width / height
        ints = [int(x) for x in args]

        if not have_format:
            # ('fisheye',)
            if len(ints) == 0:
                pass
            # ('fisheye', 2) -> index
            elif len(ints) == 1:
                stream_index = ints[0]
            # ('fisheye', 848, 800) -> width, height
            elif len(ints) == 2:
                width, height = ints
            # ('fisheye', 2, 848, 800) -> index, width, height
            elif len(ints) == 3:
                stream_index, width, height = ints
            # ('fisheye', 2, 848, 800, 30) -> index, width, height, fps
            elif len(ints) == 4:
                stream_index, width, height, framerate = ints
            else:
                raise ValueError(f"Cannot parse stream spec (no format): {stream}")
        else:
            # ('fisheye', 'y8')
            if len(ints) == 0:
                pass
            # ('fisheye', 848, 800, 'y8') -> width, height, format
            elif len(ints) == 2:
                width, height = ints
            # ('fisheye', 2, 848, 800, 'y8') -> index, width, height
            elif len(ints) == 3:
                stream_index, width, height = ints
            else:
                raise ValueError(f"Cannot parse stream spec (with format): {stream}")

        return rs_stream, stream_index, width, height, rs_format, framerate
    
    def __del__(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


if __name__ == "__main__":
    import cv2
    from r3kit.utils.vis import vis_pc

    camera = RealSenseCamera(id='319522062799', streams=[('depth', -1, 640, 480, 30), ('color', -1, 640, 480, 30)], name='D415')
    streaming = False
    shm = False

    if not streaming:
        i = 0
        while True:
            print(f"{i}th")
            result = camera.get()
            z = result['depth'] * camera.depth_scale
            rgb = cv2.cvtColor(result['color'], cv2.COLOR_BGR2RGB) / 255.0

            xyz, rgb = camera.img2pc(z, camera.color_intrinsics, rgb)
            # valid_mask = xyz[:, 2] <= 1.5
            # xyz = xyz[valid_mask, :]
            # rgb = rgb[valid_mask, :]
            print(np.mean(xyz[:, 2]))
            vis_pc(xyz, rgb)

            cv2.imshow('color', result['color'])
            while True:
                if cv2.getWindowProperty('color', cv2.WND_PROP_VISIBLE) <= 0:
                    break
                cv2.waitKey(1)
            cv2.destroyAllWindows()
            
            cmd = input("whether save? (y/n): ")
            if cmd == 'y':
                cv2.imwrite(f"rgb_{i}.png", result['color'])
                np.savez(f"xyzrgb_{i}.npz", xyz=xyz, rgb=rgb)
                i += 1
            elif cmd == 'n':
                cmd = input("whether quit? (y/n): ")
                if cmd == 'y':
                    break
                elif cmd == 'n':
                    pass
                else:
                    raise ValueError
            else:
                raise ValueError
    else:
        camera.collect_streaming(collect=True)
        camera.shm_streaming(shm='D415' if shm else None)
        camera.start_streaming()

        cmd = input("quit? (enter): ")
        streaming_data = camera.stop_streaming()
        print(len(streaming_data["timestamp_ms"]))
        color, depth = streaming_data["color"][-1], streaming_data["depth"][-1]
        z = depth * camera.depth_scale
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0

        xyz, rgb = camera.img2pc(z, camera.color_intrinsics, rgb)
        # valid_mask = xyz[:, 2] <= 1.5
        # xyz = xyz[valid_mask, :]
        # rgb = rgb[valid_mask, :]
        print(np.mean(xyz[:, 2]))
        vis_pc(xyz, rgb)

        cv2.imshow('color', color)
        while True:
            if cv2.getWindowProperty('color', cv2.WND_PROP_VISIBLE) <= 0:
                break
            cv2.waitKey(1)
        cv2.destroyAllWindows()

        cmd = input("whether save? (y/n): ")
        if cmd == 'y':
            camera.save_streaming('.', streaming_data)
        elif cmd == 'n':
            pass
        else:
            raise ValueError
