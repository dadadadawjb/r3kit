import numpy as np

from r3kit.devices.base import DeviceBase
from r3kit.devices.robot.flexiv.rizon import Rizon
from r3kit.utils.vis import SequenceKeyboardListener

id = 'Rizon4s-063231'
name = 'angler'
cmd_vel = False


def mapping(angles:np.ndarray, middles:np.ndarray=np.array([77.08, 318.34, 285.21, 207.33, 238.27, 348.22, 312.89])) -> np.ndarray:
    # angles as degrees
    # output as radians
    angles[5] += 90.0
    angles = (angles - middles + 180.0) % 360.0 - 180.0
    angles[5] -= 90.0
    angles *= -1.0
    return angles * np.pi / 180.0


def main():
    # initialize robot
    robot = Rizon(id=id, gripper=False)
    robot.motion_mode('joint')
    robot.block(blocking=False)

    # read shm
    angler = DeviceBase()
    angler_streaming_array, angler_shared_memory, angler_streaming_lock = angler.get_streaming_memory(shm=name)

    # collect
    keyboard_listener = SequenceKeyboardListener(verbose=False)
    last_angles, last_timestamp_ms = None, None
    while not keyboard_listener.quit:
        with angler_streaming_lock:
            angles = np.copy(angler_streaming_array["angle"])
            timestamp_ms = angler_streaming_array["timestamp_ms"].item()
        angles = mapping(angles)
        if last_angles is None:
            last_angles = angles.copy()
            last_timestamp_ms = timestamp_ms
            velocities = np.zeros_like(angles)
        else:
            if timestamp_ms == last_timestamp_ms:
                continue
            elif timestamp_ms < last_timestamp_ms:
                print("Warning: timestamp went backwards!")
            else:
                velocities = (angles - last_angles) / ((timestamp_ms - last_timestamp_ms) / 1000.0)
                last_angles = angles.copy()
                last_timestamp_ms = timestamp_ms
        robot.joint_move(angles, velocities=velocities if cmd_vel else np.zeros_like(angles))
    keyboard_listener.stop()

    # disconnect
    angler_shared_memory.close()


if __name__ == '__main__':
    main()
