from typing import Tuple, Optional
import time
import numpy as np
import cv2
from threading import Lock
import pyrealsense2 as rs

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.realsense.config import *


class T265(CameraBase):
    def __init__(self, id:Optional[str]=T265_ID, name:str='T265') -> None:
        super().__init__(name=name)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if id is not None:
            self.config.enable_device(id)
        else:
            pass
        for stream_item in T265_STREAMS:
            self.config.enable_stream(*stream_item)
        
        self.in_streaming = False

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.in_streaming:
            raise NotImplementedError
        else:
            if hasattr(self, "image_streaming_data"):
                self.image_streaming_mutex.acquire()
                self.pose_streaming_mutex.acquire()
                left_image = self.image_streaming_data["left"][-1]
                right_image = self.image_streaming_data["right"][-1]
                xyz = self.pose_streaming_data["xyz"][-1]
                quat = self.pose_streaming_data["quat"][-1]
                self.pose_streaming_mutex.release()
                self.image_streaming_mutex.release()
                return (left_image, right_image, xyz, quat)
            else:
                raise AttributeError
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        self.pipeline.stop()
        if callback is not None:
            self.pipeline_profile = self.pipeline.start(self.config, callback)
        else:
            self.image_streaming_mutex = Lock()
            self.image_streaming_data = {
                "left": [], 
                "right": [], 
                "timestamp_ms": [], 
            }
            self.pose_streaming_mutex = Lock()
            self.pose_streaming_data = {
                "xyz": [], 
                "quat": [], 
                "timestamp_ms": [], 
            }
            self.pipeline_profile = self.pipeline.start(self.config, self.callback)
        self.in_streaming = True

    def stop_streaming(self) -> None:
        self.pipeline.stop()
        self.image_streaming_mutex = None
        self.image_streaming_data.clear()
        self.pose_streaming_mutex = None
        self.pose_streaming_data.clear()
        self.pipeline_profile = self.pipeline.start(self.config)
        self.in_streaming = False
    
    def callback(self, frame):
        ts = time.time() * 1000
        if frame.is_frameset():
            frameset = frame.as_frameset()
            f1 = frameset.get_fisheye_frame(1).as_video_frame()
            f2 = frameset.get_fisheye_frame(2).as_video_frame()
            left_data = np.asanyarray(f1.get_data(), dtype=np.uint8)
            right_data = np.asanyarray(f2.get_data(), dtype=np.uint8)
            # ts = frameset.get_timestamp()
            self.image_streaming_mutex.acquire()
            if len(self.image_streaming_data["timestamp_ms"]) != 0 and ts == self.image_streaming_data["timestamp_ms"][-1]:
                pass
            else:
                self.image_streaming_data["left"].append(left_data.copy())
                self.image_streaming_data["right"].append(right_data.copy())
                self.image_streaming_data["timestamp_ms"].append(ts)
            self.image_streaming_mutex.release()
        
        if frame.is_pose_frame():
            pose_frame = frame.as_pose_frame()
            pose_data_ = pose_frame.get_pose_data()
            quat = np.array([pose_data_.rotation.x, pose_data_.rotation.y, pose_data_.rotation.z, pose_data_.rotation.w])
            xyz = np.array([pose_data_.translation.x, pose_data_.translation.y, pose_data_.translation.z])
            # ts = pose_frame.timestamp
            self.pose_streaming_mutex.acquire()
            if len(self.pose_streaming_data["timestamp_ms"]) != 0 and ts == self.pose_streaming_data["timestamp_ms"][-1]:
                pass
            else:
                self.pose_streaming_data["xyz"].append(xyz)
                self.pose_streaming_data["quat"].append(quat)
                self.pose_streaming_data["timestamp_ms"].append(ts)
            self.pose_streaming_mutex.release()

    def __del__(self) -> None:
        self.pipeline.stop()


if __name__ == "__main__":
    camera = T265(id='230222110234', name='T265')
    
    camera.start_streaming(callback=None)

    i = 0
    while True:
        print(f"{i}th")
        left, right, xyz, quat = camera.get()

        print(f"xyz: {xyz}")
        print(f"quat: {quat}")
        cv2.imshow('left', left)
        cv2.imshow('right', right)
        while True:
            if cv2.getWindowProperty('left', cv2.WND_PROP_VISIBLE) <= 0:
                break
            cv2.waitKey(1)
        while True:
            if cv2.getWindowProperty('right', cv2.WND_PROP_VISIBLE) <= 0:
                break
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        cmd = input("whether save? (y/n): ")
        if cmd == 'y':
            cv2.imwrite(f"left_{i}.png", left)
            cv2.imwrite(f"right_{i}.png", right)
            np.savetxt(f"xyz_{i}.txt", xyz)
            np.savetxt(f"quat_{i}.txt", quat)
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
