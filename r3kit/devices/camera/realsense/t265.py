import os
from typing import Tuple, Optional
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from threading import Lock
import pyrealsense2 as rs

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.realsense.config import *
from r3kit.utils.vis import draw_time, save_imgs


class T265(CameraBase):
    def __init__(self, id:Optional[str]=T265_ID, image:bool=True, name:str='T265') -> None:
        super().__init__(name=name)
        self._image = image

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if id is not None:
            self.config.enable_device(id)
        else:
            pass
        for stream_item in T265_STREAMS:
            if not image and stream_item[0] == rs.stream.fisheye:
                continue
            self.config.enable_stream(*stream_item)
        
        # NOTE: hard code to balance pose accuracy and smoothness
        self.pipeline.start(self.config)
        pose_sensor = self.pipeline.get_active_profile().get_device().first_pose_sensor()
        self.pipeline.stop()
        # pose_sensor.set_option(rs.option.enable_mapping, 0)
        pose_sensor.set_option(rs.option.enable_pose_jumping, 0)
        # pose_sensor.set_option(rs.option.enable_relocalization, 0)

        self.in_streaming = False

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        if not self.in_streaming:
            raise NotImplementedError
        else:
            if hasattr(self, "image_streaming_data"):
                self.image_streaming_mutex.acquire()
                self.pose_streaming_mutex.acquire()
                if self._image:
                    left_image = self.image_streaming_data["left"][-1]
                    right_image = self.image_streaming_data["right"][-1]
                else:
                    left_image = None
                    right_image = None
                xyz = self.pose_streaming_data["xyz"][-1]
                quat = self.pose_streaming_data["quat"][-1]
                self.pose_streaming_mutex.release()
                self.image_streaming_mutex.release()
                return (left_image, right_image, xyz, quat)
            else:
                raise AttributeError
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        # self.pipeline.stop()
        if not hasattr(self, "_collect_streaming_data"):
            self._collect_streaming_data = True
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

    def stop_streaming(self) -> Optional[dict]:
        streaming_data = None
        self.pipeline.stop()
        if hasattr(self, "image_streaming_mutex"):
            self.image_streaming_mutex = None
        if hasattr(self, "image_streaming_data"):
            streaming_data = {'image': self.image_streaming_data}
            self.image_streaming_data = {
                "left": [], 
                "right": [], 
                "timestamp_ms": [], 
            }
        if hasattr(self, "pose_streaming_mutex"):
            self.pose_streaming_mutex = None
        if hasattr(self, "pose_streaming_data"):
            streaming_data['pose'] = self.pose_streaming_data
            self.pose_streaming_data = {
                "xyz": [], 
                "quat": [], 
                "timestamp_ms": [], 
            }
        # self.pipeline_profile = self.pipeline.start(self.config)
        self.in_streaming = False
        return streaming_data
    
    def save_streaming(self, save_path:str, streaming_data:dict) -> None:
        assert len(streaming_data["image"]["left"]) == len(streaming_data["image"]["right"]) == len(streaming_data["image"]["timestamp_ms"])
        assert len(streaming_data["pose"]["xyz"]) == len(streaming_data["pose"]["quat"]) == len(streaming_data["pose"]["timestamp_ms"])
        if self._image:
            os.makedirs(os.path.join(save_path, 'image'), exist_ok=True)
            np.save(os.path.join(save_path, 'image', "timestamps.npy"), np.array(streaming_data["image"]["timestamp_ms"], dtype=float))
            freq = len(streaming_data["image"]["timestamp_ms"]) / (streaming_data["image"]["timestamp_ms"][-1] - streaming_data["image"]["timestamp_ms"][0])
            draw_time(streaming_data["image"]["timestamp_ms"], os.path.join(save_path, 'image', f"freq_{freq}.png"))
            os.makedirs(os.path.join(save_path, 'image', 'left'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'image', 'right'), exist_ok=True)
            save_imgs(os.path.join(save_path, 'image', 'left'), streaming_data["image"]["left"])
            save_imgs(os.path.join(save_path, 'image', 'right'), streaming_data["image"]["right"])
        os.makedirs(os.path.join(save_path, 'pose'), exist_ok=True)
        np.save(os.path.join(save_path, 'pose', "timestamps.npy"), np.array(streaming_data["pose"]["timestamp_ms"], dtype=float))
        freq = len(streaming_data["pose"]["timestamp_ms"]) / (streaming_data["pose"]["timestamp_ms"][-1] - streaming_data["pose"]["timestamp_ms"][0])
        draw_time(streaming_data["pose"]["timestamp_ms"], os.path.join(save_path, 'pose', f"freq_{freq}.png"))
        np.save(os.path.join(save_path, 'pose', "xyz.npy"), np.array(streaming_data["pose"]["xyz"], dtype=float))
        np.save(os.path.join(save_path, 'pose', "quat.npy"), np.array(streaming_data["pose"]["quat"], dtype=float))
    
    def collect_streaming(self, collect:bool=True) -> None:
        # NOTE: only valid for no-custom-callback
        self._collect_streaming_data = collect
    
    def callback(self, frame):
        ts = time.time() * 1000
        if not self._collect_streaming_data:
            return
        
        if frame.is_frameset() and self._image:
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
    
    @staticmethod
    def raw2pose(xyz:np.ndarray, quat:np.ndarray) -> np.ndarray:
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = Rot.from_quat(quat).as_matrix()
        pose_4x4[:3, 3] = xyz
        return pose_4x4

    def __del__(self) -> None:
        self.pipeline.stop()


if __name__ == "__main__":
    camera = T265(id='230222110234', image=True, name='T265')
    
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
