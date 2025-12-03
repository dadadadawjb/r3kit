from r3kit.devices.encoder.pdcd.angler import Angler

id = '/dev/ttyUSB0'
index = [1,2,3,4,5,6,7]
baudrate = 1000000
name = 'angler'


def main():
    angler = Angler(id=id, index=index, fps=0, baudrate=baudrate, gap=-1, strict=True, name=name)

    angler.collect_streaming(collect=True)
    angler.shm_streaming(shm=name)
    angler.start_streaming()

    input("Press Enter to stop...")

    angler.stop_streaming()


if __name__ == '__main__':
    main()
