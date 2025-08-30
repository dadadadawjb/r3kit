# r3kit
<img src="assets/logo.png" alt="r3kit" width="10%" >

Real Robot Research Kit

## Setup
```bash
conda create -n rrr python=3.10
conda activate rrr

git clone git@github.com:dadadadawjb/r3kit.git

cd r3kit
pip install -e .
```

Additional manual dependencies (see `docs` to set up): 
* Franka Robot: `rt-linux`
* Flexiv Robot: `flexivrdk`
* VIVE Camera: `SteamVR` and `VIVE_Hub`

## Usage
```python
from r3kit.devices.camera.realsense.l515 import L515

camera = L515()
image = camera.get()
```

```python
from r3kit.devices.ftsensor.ati.pyati import PyATI

ftsensor = PyATI()
ft = ftsensor.get()
```

```python
from r3kit.devices.encoder.pdcd.angler import Angler

encoder = Angler()
angle = encoder.get()
```

```python
from r3kit.devices.robot.flexiv.rizon import Rizon

robot = Rizon()
joints = robot.joint_read()
```

```python
from r3kit.algos.calib.chessboard import ChessboardExtCalibor

calibor = ChessboardExtCalibor()
calibor.add_image(img)
w2c = calibor.run()
```

```python
from r3kit.algos.tare.linear import LinearMFTarer

tarer = LinearMFTarer()
tarer.add_data(f, pose)
tare = tarer.run()
```

```python
from r3kit.algos.align.umeyama import umeyama_align

align_transformation, aligned_sources = umeyama_align(sources, targets)
```

```python
from r3kit.algos.align.disp2metric import disp2metric

b, s, pred_metric = disp2metric(gt_metric, pred_disp)
```

```python
from r3kit.algos.fit.arc import fit_arc

center, normal, r, error, _, _ = fit_arc(points)
```

```python
from r3kit.utils.vis import Sequence3DVisualizer

visualizer = Sequence3DVisualizer()
for pc_xyzs, pc_rgbs in pc_list:
    visualizer.update_points('pc', pc_xyzs, pc_rgbs)
    visualizer.update_view()
```

```python
from r3kit.utils.buffer import ObsBuffer, ActBuffer

obs_buffer = ObsBuffer()
act_buffer = ActBuffer()
while True:
    obs_buffer.add1(o)
    a = act_buffer.get1()
```
