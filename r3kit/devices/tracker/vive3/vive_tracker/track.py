import time
import sys
import openvr
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class ViveTrackerUpdater:
    def __init__(self, names=["tracker_1", "tracker_2"]):
        self.vive_tracker_module = ViveTrackerModule()
        self.vive_tracker_module.print_discovered_objects()
        self.device_key = names
        self.tracking_devices = self.vive_tracker_module.return_selected_devices(self.device_key)

    def getpose(self):
        pose = [self.tracking_devices[key].get_T() for key in self.tracking_devices]
        return pose

    def read(self):
        poses = self.getpose()
        xyzs = [pose[:3, 3] for pose in poses]
        quats = [Rot.from_matrix(pose[:3, :3]).as_quat() for pose in poses]
        receive_time = time.time() * 1000
        return {'xyz': np.array(xyzs), 'quat': np.array(quats), 'timestamp_ms': receive_time}


class ViveTrackerModule:
    def __init__(self, configfile_path=None):
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()
        self.object_names = {"Tracking Reference": [], "HMD": [], "Controller": [], "Tracker": []}
        self.devices = {}
        self.device_index_map = {}
        poses = self.vr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bDeviceIsConnected:
                self.add_tracked_device(i)

    def __del__(self):
        openvr.shutdown()

    def return_selected_devices(self, device_keys: list):
        selected_devices = {}
        for key in device_keys:
            if key in self.devices:
                selected_devices[key] = self.devices[key]
        return selected_devices

    def get_pose(self):
        return get_pose(self.vr)

    def poll_vr_events(self):
        event = openvr.VREvent_t()
        while self.vrsystem.pollNextEvent(event):
            if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                self.add_tracked_device(event.trackedDeviceIndex)
            elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                if event.trackedDeviceIndex in self.device_index_map:
                    self.remove_tracked_device(event.trackedDeviceIndex)

    def add_tracked_device(self, tracked_device_index):
        i = tracked_device_index
        device_class = self.vr.getTrackedDeviceClass(i)
        if device_class == openvr.TrackedDeviceClass_Controller:
            device_name = "controller_" + str(len(self.object_names["Controller"]) + 1)
            self.object_names["Controller"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr, i, "Controller")
            self.device_index_map[i] = device_name
        elif device_class == openvr.TrackedDeviceClass_HMD:
            device_name = "hmd_" + str(len(self.object_names["HMD"]) + 1)
            self.object_names["HMD"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr, i, "HMD")
            self.device_index_map[i] = device_name
        elif device_class == openvr.TrackedDeviceClass_GenericTracker:
            device_name = "tracker_" + str(len(self.object_names["Tracker"]) + 1)
            self.object_names["Tracker"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr, i, "Tracker")
            self.device_index_map[i] = device_name
        elif device_class == openvr.TrackedDeviceClass_TrackingReference:
            device_name = "tracking_reference_" + str(len(self.object_names["Tracking Reference"]) + 1)
            self.object_names["Tracking Reference"].append(device_name)
            self.devices[device_name] = vr_tracking_reference(self.vr, i, "Tracking Reference")
            self.device_index_map[i] = device_name

    def remove_tracked_device(self, tracked_device_index):
        if tracked_device_index in self.device_index_map:
            device_name = self.device_index_map[tracked_device_index]
            self.object_names[self.devices[device_name].device_class].remove(device_name)
            del self.device_index_map[tracked_device_index]
            del self.devices[device_name]
        else:
            raise Exception("Tracked device index {} not valid. Not removing.".format(tracked_device_index))

    def rename_device(self, old_device_name, new_device_name):
        self.devices[new_device_name] = self.devices.pop(old_device_name)
        for i in range(len(self.object_names[self.devices[new_device_name].device_class])):
            if self.object_names[self.devices[new_device_name].device_class][i] == old_device_name:
                self.object_names[self.devices[new_device_name].device_class][i] = new_device_name

    def print_discovered_objects(self):
        for device_type in self.object_names:
            plural = device_type
            if len(self.object_names[device_type]) != 1:
                plural += "s"
            print("Found " + str(len(self.object_names[device_type])) + " " + plural)
            for device in self.object_names[device_type]:
                if device_type == "Tracking Reference":
                    print("  " + device + " (" + self.devices[device].get_serial() +
                          ", Mode " + self.devices[device].get_model() +
                          ", " + self.devices[device].get_model() + ")")
                else:
                    print("  " + device + " (" + self.devices[device].get_serial() +
                          ", " + self.devices[device].get_model() + ")")


def get_pose(vr_obj):
    return vr_obj.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)


class vr_tracked_device:
    def __init__(self, vr_obj, index, device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj
        self.T = np.eye(4)

    def get_serial(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_SerialNumber_String)

    def get_model(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_ModelNumber_String)

    def get_battery_percent(self):
        return self.vr.getFloatTrackedDeviceProperty(self.index, openvr.Prop_DeviceBatteryPercentage_Float)

    def is_charging(self):
        return self.vr.getBoolTrackedDeviceProperty(self.index, openvr.Prop_DeviceIsCharging_Bool)

    def get_T(self, pose=None):
        pose_mat = self.get_pose_matrix()
        if pose_mat:
            np_pose_mat = np.array(pose_mat)['m']
            self.T[:3, :] = np_pose_mat
        return self.T

    def get_pose_matrix(self, pose=None):
        if pose is None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].mDeviceToAbsoluteTracking
        else:
            return None


class vr_tracking_reference(vr_tracked_device):
    def get_mode(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_ModeLabel_String).decode('utf-8').upper()

