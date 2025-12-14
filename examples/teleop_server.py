from typing import List
from tap import Tap

from r3kit.devices.encoder.pdcd.angler import Angler

class ArgumentParser(Tap):
    id: str = '/dev/ttyUSB0'
    index: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
    baudrate: int = 1000000
    shm_name: str = 'Angler'


def main(args:ArgumentParser):
    angler = Angler(id=args.id, index=args.index, fps=0, baudrate=args.baudrate, gap=-1, strict=True)

    angler.collect_streaming(collect=True)
    angler.shm_streaming(shm=args.shm_name)
    angler.start_streaming()

    input("Press Enter to stop...")

    angler.stop_streaming()


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
