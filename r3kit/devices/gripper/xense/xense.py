from typing import Tuple
from xensegripper import XenseGripper

from r3kit.devices.gripper.base import GripperBase
from r3kit.devices.gripper.xense.config import *


class Xense(GripperBase):
    MAX_WIDTH:float = 0.085
    MAX_VELOCITY:float = 0.35
    MAX_FORCE:float = 60.0

    def __init__(self, id:str=XENSE_ID, name:str='Xense') -> None:
        super().__init__(name)

        self.gripper = XenseGripper.create(id)
        self.block(-1)
    
    def block(self, blocking:float) -> None:
        self.blocking = blocking
    
    def read(self) -> float:
        '''
        width: gripper width in meter
        '''
        status = self.gripper.get_gripper_status()
        return status['position'] / 1000.

    def move(self, width:float, vmax:float=0.08, fmax:float=27.0) -> bool:
        '''
        width: gripper width in meter
        reached: whether the gripper reaches the target width within timeout
        '''
        target_width = min(max(width, 0.0), self.MAX_WIDTH) * 1000.
        target_vmax = min(max(vmax, 0.0), self.MAX_VELOCITY) * 1000.
        target_fmax = min(max(fmax, 0.0), self.MAX_FORCE)
        if self.blocking <= 0:
            self.gripper.set_position(target_width, target_vmax, target_fmax)
            reached = True
        else:
            reached = self.gripper.set_position_sync(target_width, target_vmax, target_fmax, 
                                                    timeout=self.blocking, tolerance=XENSE_TOLERANCE)
        return reached
    
    def set_led_color(self, rgb:Tuple[int, int, int]) -> None:
        self.gripper.set_led_color(*rgb)


if __name__ == "__main__":
    gripper = Xense("5e77ff097831", "Xense")

    gripper.block(5)
    gripper.move(0.08)
    gripper.move(0.005)
    gripper.move(0.05)
    gripper_width = gripper.read()
    print("gripper width:", gripper_width)
