from setuptools import setup, find_packages

setup(
    name = 'r3kit', 
    version = '0.0.0', 
    license = 'MIT', 
    description = 'Research kits for real robots', 
    author = "Junbo Wang", 
    author_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    maintainer = "Junbo Wang", 
    maintainer_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    url = "https://github.com/dadadadawjb/r3kit", 
    packages = find_packages(), 
    include_package_data = True, 
    install_requires = [
        'numpy<2.0.0', 
        'opencv-python>=4.10.0', 
        'open3d>=0.18.0', 
        'matplotlib', 
        'scipy', 
        'transformations', 
        'psutil', 
        'pyrealsense2==2.53.1.4623', # needed by camera realsense
        'pyopenxr', # needed by camera vive ultimate
        'pyserial', # needed by encoder pdcd angle and ftsensor robotiq ft300
        'pymodbus', # needed by ftsensor robotiq ft300
        # 'frankx', # needed by robot franka
    ], 
    zip_safe = False
)
