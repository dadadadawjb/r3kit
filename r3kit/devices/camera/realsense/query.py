from typing import Tuple, List, Dict, Union

import pyrealsense2 as rs


class RealSenseQuery(object):
    def __init__(self):
        self.ctx = rs.context()
    
    def devices(self) -> Dict[str, Dict[str, Union[str, List[Tuple[str, int, int, int, str, int]]]]]:
        """
        serial_number: {
            "name": device_name,
            "sensors": {
                sensor_name: [(type, idx, width, height, format, fps), ...]
            }
        """
        devices = self.ctx.query_devices()
        device_info = {}
        for dev in devices:
            device_sn = dev.get_info(rs.camera_info.serial_number)
            device_info[device_sn] = {
                "name": dev.get_info(rs.camera_info.name),
            }
            
            sensors = self.sensors(dev)
            device_info[device_sn]["sensors"] = sensors
        return device_info
    
    def sensors(self, device:Union[str, rs.device]) -> Dict[str, List[Tuple[str, int, int, int, str, int]]]:
        """
        name: [(type, idx, width, height, format, fps), ...]
        """
        if isinstance(device, str):
            devices = self.ctx.query_devices()
            device = next((d for d in devices if d.get_info(rs.camera_info.serial_number) == device), None)
            if device is None:
                raise ValueError(f"Device with serial number {device} not found.")
        
        sensors = device.query_sensors()
        sensor_info = {}
        for sensor in sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            sensor_info[sensor_name] = []
            for profile in sensor.get_stream_profiles():
                stream = profile.stream_type()
                idx = profile.stream_index()
                fmt = profile.format()
                fps = profile.fps()

                if profile.is_video_stream_profile():
                    vsp = profile.as_video_stream_profile()
                    width = vsp.width()
                    height = vsp.height()
                    sensor_info[sensor_name].append((stream.name, idx, width, height, fmt.name, fps))
                else:
                    sensor_info[sensor_name].append((stream.name, idx, 0, 0, fmt.name, fps))
        return sensor_info


if __name__ == "__main__":
    from rich import print
    
    rs_query = RealSenseQuery()
    devices = rs_query.devices()
    print(devices)
