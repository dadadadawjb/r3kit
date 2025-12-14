import os
from typing import Tuple, List
from tap import Tap

from r3kit.devices.encoder.pdcd.angler import Angler
from r3kit.devices.camera.realsense.general import RealSenseCamera

class ArgumentParser(Tap):
    encoder_id: str = '/dev/ttyUSB0'
    encoder_index: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    encoder_baudrate: int = 1000000
    encoder_name: str = 'Angler'

    camera_id: str = '319522062799'
    camera_streams: List[Tuple[str, int, int, int, int]] = [('depth', -1, 640, 480, 30), ('color', -1, 640, 480, 30)]
    camera_name: str = 'D415'

    save_path: str = "./data"


def main(args:ArgumentParser):
    # initialize devices
    encoder = Angler(id=args.encoder_id, index=args.encoder_index, fps=0, baudrate=args.encoder_baudrate, 
                     gap=-1, strict=True, name=args.encoder_name)
    camera = RealSenseCamera(id=args.camera_id, streams=args.camera_streams, name=args.camera_name)
    devices:Tuple[Angler, RealSenseCamera] = [encoder, camera]

    # prepare spaces
    input("Enter to start...")
    os.makedirs(args.save_path, exist_ok=True)
    idx = 0
    demo_save_path = os.path.join(args.save_path, f"demo_{idx:03d}")
    os.makedirs(demo_save_path, exist_ok=True)
    for device in devices:
        device_save_path =  os.path.join(demo_save_path, device.name)
        os.makedirs(device_save_path, exist_ok=True)
        device.device_save_path = device_save_path

    # enable streaming
    for device in devices:
        device.collect_streaming(collect=True)
        device.shm_streaming(shm=None)
    for device in devices:
        device.start_streaming(streaming_save_path=device.device_save_path)

    # loop
    while True:
        # collect
        input("Enter to stop...")

        # pause
        for device in devices:
            device.collect_streaming(collect=False)
        print("Collection stopped, saving data...")

        # save
        for device in devices:
            data = device.get_streaming()
            device.save_streaming(save_path=device.device_save_path, streaming_data=data)
        
        # ask to continue
        continue_collect = input("Continue collecting? (y/n, default: y): ").strip().lower()
        if continue_collect == 'n':
            break
        
        # prepare for next collection
        idx += 1
        demo_save_path = os.path.join(args.save_path, f"demo_{idx:03d}")
        os.makedirs(demo_save_path, exist_ok=True)
        for device in devices:
            device_save_path =  os.path.join(demo_save_path, device.name)
            os.makedirs(device_save_path, exist_ok=True)
            device.device_save_path = device_save_path
        
        # resume
        for device in devices:
            device.reset_streaming(streaming_save_path=device.device_save_path)
        for device in devices:
            device.collect_streaming(collect=True)
    
    # stop
    for device in devices:
        device.stop_streaming()


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
