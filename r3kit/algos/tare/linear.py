from typing import Dict, Union
import numpy as np

from r3kit.algos.lstsq import LinearSolver


class LinearMFTarer(LinearSolver):
    G_VALUE:float = -9.8
    G_VECTOR:np.ndarray = np.array([[0], [0], [G_VALUE]])

    def __init__(self) -> None:
        super().__init__()
    
    def add_data(self, f:np.ndarray, pose:np.ndarray) -> None:
        '''
        f: raw force data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to base
        '''
        A = np.concatenate([self.G_VECTOR, pose], axis=1)
        b = pose @ f
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, Union[float, np.ndarray]]:
        '''
        m: mass
        f0: force offset under ftsensor frame
        '''
        X = super().run()
        self.tare = {
            'm': X[0], 
            'f0': X[1:4]
        }
        return self.tare
    
    def raw2tare(self, raw_f:np.ndarray, pose:np.ndarray) -> np.ndarray:
        '''
        raw_f: raw force data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to base
        f: tared force data under ftsensor frame
        '''
        f = raw_f - self.tare['f0']
        f -= np.linalg.inv(pose) @ (self.tare['m'] * self.G_VECTOR.flatten())
        return f

class LinearFTarer(LinearSolver):
    G_VALUE:float = -9.8
    G_VECTOR:np.ndarray = np.array([[0], [0], [G_VALUE]])

    def __init__(self) -> None:
        super().__init__()
    
    def set_m(self, m:float) -> None:
        self.m = m
    
    def add_data(self, f:np.ndarray, pose:np.ndarray) -> None:
        '''
        f: raw force data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to base
        '''
        A = pose
        b = pose @ f - (self.m * self.G_VECTOR).flatten()
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, np.ndarray]:
        '''
        f0: force offset under ftsensor frame
        '''
        X = super().run()
        self.tare = {
            'f0': X[0:3]
        }
        return self.tare
    
    def raw2tare(self, raw_f:np.ndarray, pose:np.ndarray) -> np.ndarray:
        '''
        raw_f: raw force data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to base
        f: tared force data under ftsensor frame
        '''
        f = raw_f - self.tare['f0']
        f -= np.linalg.inv(pose) @ (self.m * self.G_VECTOR.flatten())
        return f

class LinearCTTarer(LinearSolver):
    G_VALUE:float = -9.8
    G_VECTOR:np.ndarray = np.array([[0], [0], [G_VALUE]])

    def __init__(self) -> None:
        super().__init__()
    
    def set_m(self, m:float) -> None:
        self.m = m
    
    def add_data(self, t:np.ndarray, pose:np.ndarray) -> None:
        '''
        t: raw torque data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to base
        '''
        Rmg = (-1) * np.array([[0, -self.m * self.G_VALUE, 0], [self.m * self.G_VALUE, 0, 0], [0, 0, 0]]) @ pose
        A = np.concatenate([Rmg, pose], axis=1)
        b = pose @ t
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, np.ndarray]:
        '''
        c: center of mass under ftsensor frame
        t0: torque offset under ftsensor frame
        '''
        X = super().run()
        self.tare = {
            'c': X[0:3], 
            't0': X[3:6]
        }
        return self.tare
    
    def raw2tare(self, raw_t:np.ndarray, pose:np.ndarray) -> np.ndarray:
        '''
        raw_t: raw torque data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to base
        t: tared torque data under ftsensor frame
        '''
        t = raw_t - self.tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(pose @ self.tare['c'], self.m * self.G_VECTOR.flatten())
        return t


class LinearMgFTarer(LinearSolver):
    G_VALUE:float = -9.8

    def __init__(self) -> None:
        super().__init__()
    
    def add_data(self, f:np.ndarray, pose:np.ndarray) -> None:
        '''
        f: raw force data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to fixed
        '''
        A = np.concatenate([np.eye(3), pose], axis=1)
        b = pose @ f
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, Union[float, np.ndarray]]:
        '''
        mg: gravity vector under fixed frame
        f0: force offset under ftsensor frame
        m: mass
        '''
        X = super().run()
        self.tare = {
            'mg': X[0:3], 
            'f0': X[3:6], 
            'm': np.linalg.norm(X[0:3]) / self.G_VALUE
        }
        return self.tare
    
    def raw2tare(self, raw_f:np.ndarray, pose:np.ndarray) -> np.ndarray:
        '''
        raw_f: raw force data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to fixed
        f: tared force data under ftsensor frame
        '''
        f = raw_f - self.tare['f0']
        f -= np.linalg.inv(pose) @ self.tare['mg']
        return f

class LinearCTTarer2(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
    
    def set_mg(self, mg:np.ndarray) -> None:
        '''
        mg: gravity vector under fixed frame
        '''
        self.mg = mg
    
    def add_data(self, t:np.ndarray, pose:np.ndarray) -> None:
        '''
        t: raw torque data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to fixed
        '''
        Rmg = (-1) * np.array([[0, -self.mg[2], self.mg[1]], [self.mg[2], 0, -self.mg[0]], [-self.mg[1], self.mg[0], 0]]) @ pose
        A = np.concatenate([Rmg, pose], axis=1)
        b = pose @ t
        self.add_A(A)
        self.add_b(b)
    
    def run(self) -> Dict[str, np.ndarray]:
        '''
        c: center of mass under ftsensor frame
        t0: torque offset under ftsensor frame
        '''
        X = super().run()
        self.tare = {
            'c': X[0:3], 
            't0': X[3:6]
        }
        return self.tare
    
    def raw2tare(self, raw_t:np.ndarray, pose:np.ndarray) -> np.ndarray:
        '''
        raw_t: raw torque data under ftsensor frame
        pose: 3x3 rotation matrix from ftsensor to base
        t: tared torque data under ftsensor frame
        '''
        t = raw_t - self.tare['t0']
        t -= np.linalg.inv(pose) @ np.cross(pose @ self.tare['c'], self.mg)
        return t


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
