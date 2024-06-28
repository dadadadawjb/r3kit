"""
Modified from https://github.com/cmbruns/pyopenxr_examples/blob/main/xr_examples/vive_tracker.py
"""

from typing import List, Tuple, Optional
import ctypes
from ctypes import cast, byref, POINTER
import time
import numpy as np
from threading import Lock
import xr

from r3kit.devices.camera.base import CameraBase
from r3kit.devices.camera.vive.context_object import ContextObject
from r3kit.devices.camera.vive.config import *


class Ultimate(CameraBase):
    context:Optional[ContextObject] = None                                                  # all instances share the same context object
    roles:List[str] = []

    def __init__(self, role:str=ULTIMATE_ROLE, name:str='Ultimate') -> None:
        super().__init__(name=name)

        if Ultimate.context is None:
            # create shared context object
            Ultimate.context = ContextObject(
                instance_create_info=xr.InstanceCreateInfo(
                    enabled_extension_names=ULTIMATE_EXTENSION_NAMES,
                ),
            )
        if role not in Ultimate.roles:
            Ultimate.roles.append(role)
        else:
            raise ValueError(f"Role {role} already exists")
        self.instance = Ultimate.context.instance
        self.session = Ultimate.context.session
        # Save the function pointer
        self.enumerateViveTrackerPathsHTCX = cast(
            xr.get_instance_proc_addr(
                Ultimate.context.instance,
                "xrEnumerateViveTrackerPathsHTCX",
            ),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX
        )
        role_path_strings = [f"/user/vive_tracker_htcx/role/{role}"]
        role_paths = (xr.Path * 1)(
            *[xr.string_to_path(self.instance, role_path_string) for role_path_string in role_path_strings],
        )
        pose_action = xr.create_action(
            action_set=Ultimate.context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=1,
                subaction_paths=role_paths,
            ),
        )
        # Describe a suggested binding for that action and subaction path
        suggested_binding_paths = (xr.ActionSuggestedBinding * 1)(
            *[xr.ActionSuggestedBinding(
                pose_action,
                xr.string_to_path(self.instance, f"{role_path_string}/input/grip/pose"))
            for role_path_string in role_path_strings],
        )
        xr.suggest_interaction_profile_bindings(
            instance=self.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(self.instance, "/interaction_profiles/htc/vive_tracker_htcx"),
                count_suggested_bindings=1,
                suggested_bindings=suggested_binding_paths,
            )
        )
        # Create action spaces for locating trackers in each role
        self.tracker_action_spaces = (xr.Space * 1)(
            *[xr.create_action_space(
                session=self.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=pose_action,
                    subaction_path=role_path,
                )
            ) for role_path in role_paths],
        )
        # Warm up
        n_paths = ctypes.c_uint32(0)
        result = self.enumerateViveTrackerPathsHTCX(self.instance, 0, byref(n_paths), None)
        if xr.check_result(result).is_exception():
            raise result
        vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
        # print(xr.Result(result), n_paths.value)
        result = self.enumerateViveTrackerPathsHTCX(self.instance, n_paths, byref(n_paths), vive_tracker_paths)
        if xr.check_result(result).is_exception():
            raise result
        # print(xr.Result(result), n_paths.value)
        # print(*vive_tracker_paths)
        
        self.in_streaming = False

    def get(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # TODO: no support for streaming and images
        if not self.in_streaming:
            session_was_focused = False  # Check for a common problem
            for frame_index, frame_state in enumerate(Ultimate.context.frame_loop()):
                if Ultimate.context.session_state == xr.SessionState.FOCUSED:
                    session_was_focused = True
                    active_action_set = xr.ActiveActionSet(
                        action_set=Ultimate.context.default_action_set,
                        subaction_path=xr.NULL_PATH,
                    )
                    xr.sync_actions(
                        session=self.session,
                        sync_info=xr.ActionsSyncInfo(
                            count_active_action_sets=1,
                            active_action_sets=ctypes.pointer(active_action_set),
                        ),
                    )

                    n_paths = ctypes.c_uint32(0)
                    result = self.enumerateViveTrackerPathsHTCX(self.instance, 0, byref(n_paths), None)
                    if xr.check_result(result).is_exception():
                        raise result
                    vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(*([xr.ViveTrackerPathsHTCX()] * n_paths.value))
                    # print(xr.Result(result), n_paths.value)
                    result = self.enumerateViveTrackerPathsHTCX(self.instance, n_paths, byref(n_paths), vive_tracker_paths)
                    if xr.check_result(result).is_exception():
                        raise result
                    # print(xr.Result(result), n_paths.value)
                    # print(*vive_tracker_paths)

                    for index, space in enumerate(self.tracker_action_spaces):
                        space_location = xr.locate_space(
                            space=space,
                            base_space=Ultimate.context.space,
                            time=frame_state.predicted_display_time,
                        )
                        assert index == 0
                        if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                            return (np.array([space_location.pose.position.x, space_location.pose.position.y, space_location.pose.position.z]), 
                                    np.array([space_location.pose.orientation.x, space_location.pose.orientation.y, space_location.pose.orientation.z, space_location.pose.orientation.w]))
                if frame_index >= ULTIMATE_WAIT_TIMES - 1:
                    break
            if not session_was_focused:
                raise RuntimeError("This OpenXR session never entered the FOCUSED state. Did you modify the headless configuration?")
            return None
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
                # return (left_image, right_image, xyz, quat)
                return (xyz, quat)
            else:
                raise AttributeError
    
    def start_streaming(self, callback:Optional[callable]=None) -> None:
        # TODO: no support for streaming
        if callback is not None:
            raise NotImplementedError
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
            raise NotImplementedError
        self.in_streaming = True

    def stop_streaming(self) -> None:
        # TODO: no support for streaming
        raise NotImplementedError
        self.image_streaming_mutex = None
        self.image_streaming_data.clear()
        self.pose_streaming_mutex = None
        self.pose_streaming_data.clear()
        self.in_streaming = False
    
    def callback(self, frame):
        ts = time.time() * 1000
        raise NotImplementedError

    def __del__(self) -> None:
        Ultimate.roles.remove(self.role)
        if len(Ultimate.roles) == 0:
            del Ultimate.context
            Ultimate.context = None


if __name__ == "__main__":
    camera = Ultimate(role=ULTIMATE_ROLE, name='Ultimate')

    for i in range(10):
        data = camera.get()
        print(data)
        time.sleep(0.5)
