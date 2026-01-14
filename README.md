# r3kit
<img src="assets/logo.png" alt="r3kit" width="10%" >

Real Robot Research Kit

## Setup
```bash
conda create -n rrr python=3.10
conda activate rrr

git clone git@github.com:dadadadawjb/r3kit.git

cd r3kit
pip install -e . --config-settings editable_mode=compat # editable for easy modification, compat for pylance parse
```

Optional dependencies for different devices (see `pyproject.toml` for more options):
```bash
pip install -e .[rs,flexiv]
```

Additional manual dependencies (see `docs` to set up): 
* Franka Robot: `rt-linux`
* VIVE Camera: `SteamVR` and `VIVE_Hub`

## Usage
Nearly every class is accompanied by a demonstration in the main function.
Some more complex use cases can be found in `examples`.

> See more outputs by set environment variables `R3KIT_DEBUG=True` and/or `R3KIT_INFO=True`.

### Devices

#### Robot
```python
from r3kit.devices.robot.flexiv.rizon import Rizon # more similar alternatives can be found

robot = Rizon()
joints = robot.joint_read()
```

#### Camera
```python
from r3kit.devices.camera.realsense.general import RealSenseCamera

camera = RealSenseCamera()
data = camera.get()
```

#### Tracker
```python
from r3kit.devices.tracker.vive.ultimate import Ultimate

tracker = Ultimate()
data = tracker.get()
```

#### FTSensor
```python
from r3kit.devices.ftsensor.ati.pyati import PyATI # more similar alternatives can be found

ftsensor = PyATI()
ft = ftsensor.get()
```

#### Encoder
```python
from r3kit.devices.encoder.pdcd.angler import Angler # more similar alternatives can be found

encoder = Angler()
angle = encoder.get()
```

### Algorithms

#### Calibration
```python
from r3kit.algos.calib.chessboard import ChessboardExtCalibor # more similar alternatives can be found

calibor = ChessboardExtCalibor()
calibor.add_image(img)
result = calibor.run()
```
```python
from r3kit.algos.calib.handeye import HandEyeCalibor

calibor = HandEyeCalibor()
for img, pose in zip(imgs, poses):
    calibor.add_image_pose(img, pose)
result = calibor.run()
```
```python
from r3kit.algos.calib.tcp import LinearOffsetCalibor

calibor = LinearOffsetCalibor()
for pose in poses:
    calibor.add_data(pose)
offset = calibor.run()
```

#### Kinematics
```python
from r3kit.algos.kinematics.urdf import URDFKinematics

robot = URDFKinematics(urdf_path, end_link, base_link)
tcp_pose = robot.fk(joints)
joints, success, iters = robot.ik(tcp_pose)
```

#### Alignment
```python
from r3kit.algos.align.umeyama import umeyama_align # more similar alternatives can be found

align_transformation, aligned_sources = umeyama_align(sources, targets)
```
```python
from r3kit.algos.align.disp2metric import disp2metric

b, s, pred_metric = disp2metric(gt_metric, pred_disp)
```

#### Fit
```python
from r3kit.algos.fit.arc import fit_arc
from r3kit.algos.fit.line import fit_line

center, normal, r, error, _, _ = fit_arc(points)
pivot, direction, error, _, _ = fit_line(points)
```

#### Tare
```python
from r3kit.algos.tare.linear import LinearMFTarer # more similar alternatives can be found

tarer = LinearMFTarer()
tarer.add_data(f, pose)
tare = tarer.run()
```

### Utilities

#### Transformation
```python
from r3kit.utils.transformation import transform_pc # more similar alternatives can be found
from r3kit.utils.transformation import mean_xyz # more similar alternatives can be found
from r3kit.utils.transformation import xyzrot2mat # more similar alternatives can be found
from r3kit.utils.transformation import delta_xyz # more similar alternatives can be found

pc_world = transform_pc(pc_camera, c2w)
xyz = mean_xyz(xyzs)
mat = xyzrot2mat(xyz, rot)
delta = delta_xyz(xyz1, xyz2)
```

#### Visualization
```python
from r3kit.utils.vis import Sequence1DVisualizer, Sequence2DVisualizer, Sequence3DVisualizer
from r3kit.utils.vis import SequenceKeyboardListener

visualizer1, visualizer2, visualizer3 = Sequence1DVisualizer(), Sequence2DVisualizer(), Sequence3DVisualizer()
listener = SequenceKeyboardListener()
while not listener.quit:
    visualizer1.update_item('xyz', xyz)
    visualizer2.update_image('img', image, 'rgb')
    visualizer3.update_points('pc', pc_xyzs, pc_rgbs)
    visualizer3.update_view()
visualizer1.stop(); visualizer2.stop(); visualizer3.stop()
listener.stop()
```

#### Buffer
```python
from r3kit.utils.buffer import ObsBuffer, ActBuffer

obs_buffer = ObsBuffer()
act_buffer = ActBuffer()
while True:
    obs_buffer.add1(o)
    a = act_buffer.get1()
```
