import os

from r3kit.devices.encoder.pdcd.angler import Angler
from r3kit.devices.camera.realsense.d415 import D415

encoder_id = '/dev/ttyUSB0'
encoder_index = (1, 2, 3, 4, 5, 6, 7, 8)
encoder_baudrate = 1000000
encoder_name = 'Angler'

camera_id: str = '104122061018'
camera_depth: bool = True
camera_name: str = 'D415'

save_path: str = "./data"


def main():
    encoder = Angler(id=encoder_id, index=encoder_index, fps=0, baudrate=encoder_baudrate, 
                     gap=-1, strict=True, name=encoder_name)
    camera = D415(id=camera_id, depth=camera_depth, name=camera_name)
    devices = [encoder, camera]

    input("Enter to start...")
    for device in devices:
        device.start_streaming()
    for device in devices:
        device.collect_streaming(collect=True)

    os.makedirs(save_path, exist_ok=True)
    idx = 0
    while True:
        input("Enter to stop...")

        for device in devices:
            device.collect_streaming(collect=False)
        print("Collection stopped, saving data...")

        save_path = os.path.join(save_path, f"demo_{idx:03d}")
        os.makedirs(save_path, exist_ok=True)
        for device in devices:
            data = device.get_streaming()
            device_save_path = os.path.join(save_path, device.name)
            os.makedirs(device_save_path, exist_ok=True)
            device.save_streaming(save_path=device_save_path, streaming_data=data)
            device.reset_streaming()
        
        continue_collect = input("Continue collecting? (y/n, default: y): ").strip().lower()
        if continue_collect == 'n':
            break
        
        for device in devices:
            device.collect_streaming(collect=True)
        idx += 1
    for device in devices:
        device.stop_streaming()


if __name__ == '__main__':
    main()
