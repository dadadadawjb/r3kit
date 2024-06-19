from typing import Tuple, Optional
import time
import numpy as np
import cv2
from threading import Lock

try:
    import pyrealsense2 as rs
except ImportError:
    print("Camera RealSense L515 needs `pyrealsense2`")
    raise ImportError

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.utils import inpaint
from r3kit.devices.camera.realsense.config import *


class L515(CameraBase):
    def __init__(self, id:Optional[str]=L515_ID, name:str='L515') -> None:
        super().__init__(name=name)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if id is not None:
            self.config.enable_device(id)
        else:
            pass
        for stream_item in L515_STREAMS:
            self.config.enable_stream(*stream_item)
        # NOTE: hard code config
        self.align = rs.align(rs.stream.color)
        # self.hole_filling = rs.hole_filling_filter()
        self.hole_filling = None
        self.inpaint = False
        
        self.pipeline_profile = self.pipeline.start(self.config)
        depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame().as_video_frame()
        depth_frame = frames.get_depth_frame().as_depth_frame()
        self.depth2color = depth_frame.get_profile().get_extrinsics_to(color_frame.get_profile())
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = [color_intrinsics.ppx, color_intrinsics.ppy, color_intrinsics.fx, color_intrinsics.fy]
        
        self.in_streaming = False

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.in_streaming:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame().as_video_frame()
            depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
            if self.hole_filling is not None:
                depth_frame = self.hole_filling.process(depth_frame)
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            if self.inpaint:
                depth_image = inpaint(depth_image, missing_value=0)
            return (color_image, depth_image)
        else:
            if hasattr(self, "streaming_data"):
                self.streaming_mutex.acquire()
                color_image = self.streaming_data["color"][-1]
                depth_image = self.streaming_data["depth"][-1]
                self.streaming_mutex.release()
                return (color_image, depth_image)
            else:
                raise AttributeError
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        self.pipeline.stop()
        if callback is not None:
            self.pipeline_profile = self.pipeline.start(self.config, callback)
        else:
            self.streaming_mutex = Lock()
            self.streaming_data = {
                "depth": [], 
                "color": [], 
                "timestamp_ms": []
            }
            self.pipeline_profile = self.pipeline.start(self.config, self.callback)
        self.in_streaming = True

    def stop_streaming(self) -> None:
        self.pipeline.stop()
        self.streaming_mutex = None
        self.streaming_data.clear()
        self.pipeline_profile = self.pipeline.start(self.config)
        self.in_streaming = False
    
    def callback(self, frame):
        ts = time.time() * 1000
        if frame.is_frameset():
            frameset = frame.as_frameset()
            frameset = self.align.process(frameset)
            color_frame = frameset.get_color_frame().as_video_frame()
            depth_frame = frameset.get_depth_frame().as_depth_frame()
            if self.hole_filling is not None:
                depth_frame = self.hole_filling.process(depth_frame)
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            if self.inpaint:
                depth_image = inpaint(depth_image, missing_value=0)
            # ts = frameset.get_timestamp()
            self.streaming_mutex.acquire()
            if len(self.streaming_data["timestamp_ms"]) != 0 and ts == self.streaming_data["timestamp_ms"][-1]:
                pass
            else:
                self.streaming_data["depth"].append(depth_image.copy())
                self.streaming_data["color"].append(color_image.copy())
                self.streaming_data["timestamp_ms"].append(ts)
            self.streaming_mutex.release()

    def __del__(self) -> None:
        self.pipeline.stop()


if __name__ == "__main__":
    from r3kit.utils.data import get_point_cloud
    from r3kit.utils.vis import vis_pc

    camera = L515(id='f0172289', name='L515')

    i = 0
    while True:
        print(f"{i}th")
        color, depth = camera.get()
        z = depth * camera.depth_scale
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0

        xyz, rgb = get_point_cloud(z, camera.color_intrinsics, rgb)
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
            cv2.imwrite(f"rgb_{i}.png", color)
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
