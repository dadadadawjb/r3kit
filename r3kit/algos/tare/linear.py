from typing import Dict, Union
import numpy as np


class LinearTarer(object):
    def __init__(self) -> None:
        self.A_list = []
        self.b_list = []

    def add_A(self, A:np.ndarray) -> None:
        self.A_list.append(A)
    
    def add_b(self, b:np.ndarray) -> None:
        self.b_list.append(b)
    
    def run(self) -> np.ndarray:
        A = np.concatenate(self.A_list, axis=0)
        B = np.concatenate(self.b_list, axis=0)
        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        return X


class LinearMFTarer(LinearTarer):
    G_VALUE:float = -9.8
    G_VECTOR:np.ndarray = np.array([[0], [0], [G_VALUE]])

    def __init__(self) -> None:
        super().__init__()
    
    def add_data(self, f:np.ndarray, pose:np.ndarray) -> None:
        '''
        f: raw force data
        pose: 3x3 rotation matrix from ftsensor to base
        '''
        A = np.concatenate([self.G_VECTOR, pose], axis=1)
        b = pose @ f
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, Union[float, np.ndarray]]:
        X = super().run()
        return {
            'm': X[0], 
            'f0': X[1:4]
        }

class LinearFTarer(LinearTarer):
    G_VALUE:float = -9.8
    G_VECTOR:np.ndarray = np.array([[0], [0], [G_VALUE]])

    def __init__(self) -> None:
        super().__init__()
    
    def set_m(self, m:float) -> None:
        self.m = m
    
    def add_data(self, f:np.ndarray, pose:np.ndarray) -> None:
        '''
        f: raw force data
        pose: 3x3 rotation matrix from ftsensor to base
        '''
        A = pose
        b = pose @ f - (self.m * self.G_VECTOR).flatten()
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, np.ndarray]:
        X = super().run()
        return {
            'f0': X[0:3]
        }

class LinearCTTarer(LinearTarer):
    G_VALUE:float = -9.8
    G_VECTOR:np.ndarray = np.array([[0], [0], [G_VALUE]])

    def __init__(self) -> None:
        super().__init__()
    
    def set_m(self, m:float) -> None:
        self.m = m
    
    def add_data(self, t:np.ndarray, pose:np.ndarray) -> None:
        '''
        t: raw torque data
        pose: 3x3 rotation matrix from ftsensor to base
        '''
        Rmg = np.zeros_like(pose, dtype=pose.dtype)
        Rmg[0] = pose[1] * self.m * (-1) * self.G_VALUE
        Rmg[1] = pose[0] * self.m * self.G_VALUE
        A = np.concatenate([Rmg, pose], axis=1)
        b = pose @ t
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, np.ndarray]:
        X = super().run()
        return {
            'c': X[0:3], 
            't0': X[3]
        }


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--f_path', type=str, default='f.npy')
    parser.add_argument('--t_path', type=str, default='t.npy')
    parser.add_argument('--pose_path', type=str, default='pose.npy')
    args = parser.parse_args()

    f = np.load(args.f_path)
    t = np.load(args.t_path)
    pose = np.load(args.pose_path)

    mftarer = LinearMFTarer()
    for i in range(len(f)):
        mftarer.add_data(f[i], pose[i])
    result = mftarer.run()
    print(result)

    ftarer = LinearFTarer()
    ftarer.set_m(result['m'])
    for i in range(len(f)):
        ftarer.add_data(f[i], pose[i])
    result.update(ftarer.run())
    print(result)

    ctarer = LinearCTTarer()
    ctarer.set_m(result['m'])
    for i in range(len(t)):
        ctarer.add_data(t[i], pose[i])
    result.update(ctarer.run())
    print(result)
