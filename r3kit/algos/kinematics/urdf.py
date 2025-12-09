from typing import Tuple, Dict, Union, Optional
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from yourdfpy import URDF

from r3kit import DEBUG, INFO


class URDFKinematics(object):
    def __init__(self, urdf_path:str, end_link:str, base_link:Optional[str]=None) -> None:
        self.robot = URDF.load(urdf_path, build_scene_graph=True)

        self.end_link = end_link
        self.base_link = base_link
        if self.base_link is None and INFO:
            print(f"[INFO-r3kit] Base link: {self.robot.base_link}")

        self.all_joint_names = self.robot.joint_names
        self.actuated_joint_names = self.robot.actuated_joint_names
        self.link_names = list(self.robot.link_map.keys())
        self.num_dofs = self.robot.num_dofs
        assert self.num_dofs == len(self.actuated_joint_names)
        self.num_all_joints = len(self.all_joint_names)
        self.num_links = len(self.link_names)
        if INFO:
            print(f"[INFO-r3kit] Actuated joints: {self.actuated_joint_names}")
            print(f"[INFO-r3kit] All joints: {self.all_joint_names}")
            print(f"[INFO-r3kit] Links: {self.link_names}")
            print(f"[INFO-r3kit] Number of DOFs: {self.num_dofs}")
        
        self.lower_limits, self.upper_limits = [], []
        for name in self.actuated_joint_names:
            joint = self.robot.joint_map[name]
            if getattr(joint, "limit", None) is not None:
                lo = joint.limit.lower if joint.limit.lower is not None else -np.inf
                hi = joint.limit.upper if joint.limit.upper is not None else np.inf
            else:
                lo, hi = -np.inf, np.inf
            self.lower_limits.append(lo)
            self.upper_limits.append(hi)
        self.lower_limits = np.array(self.lower_limits)
        self.upper_limits = np.array(self.upper_limits)
    
    def to_dict(self, q:np.ndarray) -> Dict[str, float]:
        assert len(q) == self.num_dofs, f"Expected q of length {self.num_dofs}, got {len(q)}"
        return {name: float(val) for name, val in zip(self.actuated_joint_names, q)}
    
    def to_array(self, q_dict:Dict[str, float]) -> np.ndarray:
        return np.array([q_dict[name] for name in self.actuated_joint_names])

    def fk(self, q:Union[np.ndarray, Dict[str, float]]) -> np.ndarray:
        self.robot.update_cfg(q)

        T = self.robot.get_transform(
            frame_to=self.end_link,
            frame_from=self.base_link,
            collision_geometry=False
        )
        return T

    def ik(self, T_target:np.ndarray, q_init:Optional[Union[np.ndarray, Dict[str, float]]]=None,
           max_iters:int=300, tol:float=1e-4, damping:float=1e-1, w_pos:float=1.0, w_rot:float=1.0, max_step:float=0.1) -> Tuple[np.ndarray, bool, int]:
        """Simple damped least-squares IK.
        q: optimized joints
        success: whether IK converged
        iters: number of iterations taken
        """
        if q_init is None:
            q = self.robot.zero_cfg.copy()
        else:
            q = self.to_array(q_init) if isinstance(q_init, dict) else q_init.copy()

        for it in range(max_iters):
            T_cur = self.fk(q)
            e = self._pose_error(T_cur, T_target, w_pos=w_pos, w_rot=w_rot)
            if DEBUG:
                print(f"[DEBUG-r3kit] IK iter {it}: error norm = {np.linalg.norm(e)}")

            if np.linalg.norm(e) < tol:
                return (q, True, it)

            J = self._numerical_jacobian(q, T_target, w_pos=w_pos, w_rot=w_rot)

            # damped least-squares: (J^T J + λ^2 I) dq = - J^T e
            JT = J.T
            A = JT @ J + (damping ** 2) * np.eye(self.num_dofs)
            b = -JT @ e

            dq = np.linalg.solve(A, b)
            step_norm = np.max(np.abs(dq))
            if step_norm > max_step:
                dq *= (max_step / step_norm)
            q = q + dq
            q = np.clip(q, self.lower_limits, self.upper_limits)
        
        return (q, False, max_iters)
    
    @staticmethod
    def _pose_error(T_current:np.ndarray, T_target:np.ndarray, w_pos:float=1.0, w_rot:float=1.0) -> np.ndarray:
        p_cur = T_current[:3, 3]
        p_des = T_target[:3, 3]
        e_pos = p_des - p_cur

        R_cur = T_current[:3, :3]
        R_des = T_target[:3, :3]
        R_err = R_des @ R_cur.T
        rotvec = Rot.from_matrix(R_err).as_rotvec()

        e = np.hstack([w_pos * e_pos, w_rot * rotvec])
        return e

    def _numerical_jacobian(self, q:np.ndarray, T_target:np.ndarray, eps:float=1e-6, w_pos:float=1.0, w_rot:float=1.0) -> np.ndarray:
        n = self.num_dofs
        J = np.zeros((6, n))                # 6D pose error

        T0 = self.fk(q)
        e0 = self._pose_error(T0, T_target, w_pos=w_pos, w_rot=w_rot)
        for i in range(n):
            dq = np.zeros_like(q)
            dq[i] = eps

            T_eps = self.fk(q + dq)
            e_eps = self._pose_error(T_eps, T_target, w_pos=w_pos, w_rot=w_rot)

            J[:, i] = (e_eps - e0) / eps    # finite difference approximation: de/dq_i ≈ (e(q+eps) - e(q)) / eps
        return J
