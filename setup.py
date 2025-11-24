from setuptools import setup, find_packages

basics = [
    'numpy>=1.16.0,<2.0.0', 
    'opencv-python>=4.10.0', 
    'open3d>=0.18.0', 
    'matplotlib', 
    'scipy', 
    'fpsample', 
    'pycpd', 
    'psutil', 
    'yourdfpy', 
    'pynput', 
    'tqdm', 
]
extras = {
    'comm': [
        'pyserial', # needed by encoder pdcd angle and ftsensor robotiq ft300
        'pymodbus', # needed by ftsensor robotiq ft300
    ], 
    'rs': [
        'pyrealsense2==2.53.1.4623', # needed by camera realsense
    ], 
    'xr': [
        'pyopenxr', # needed by camera vive
    ], 
    'xense': [
        'xensesdk', # needed by camera xense
        'xensegripper', # needed by gripper xense
    ], 
    'vive3': [
        'openvr', # needed by tracker vive3
    ], 
    'flexiv': [
        'flexivrdk', 'spdlog' # needed by robot flexiv
    ],
    'franka': [
        'frankx', # needed by robot franka
    ], 
}
extras['all'] = list(set({pkg for pkgs in extras.values() for pkg in pkgs}))

setup(
    name = 'r3kit', 
    version = '0.0.2', 
    license = 'MIT', 
    description = 'Research kits for real robots', 
    author = "Junbo Wang", 
    author_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    maintainer = "Junbo Wang", 
    maintainer_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    url = "https://github.com/dadadadawjb/r3kit", 
    packages = find_packages(), 
    include_package_data = True, 
    install_requires = basics, 
    extras_require = extras, 
    zip_safe = False
)
