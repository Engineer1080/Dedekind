# --- Differentiable Robotics & Kinematics ---

import torch

class KinematicChain:
    """
    Models a robotic arm (kinematic chain) via Denavit-Hartenberg parameters.
    Fully differentiable for inverse kinematics via gradient descent.
    """
    def __init__(self):
        self.joints = []

    def _get_val(self, v, dim):
        if isinstance(v, Quantity):
            return _convert_to_base(v.value, v.unit, dim)
        return float(v)

    def add_revolute_joint(self, d, a, alpha):
        """Adds a revolute joint. (d=offset, a=link length, alpha=twist)"""
        d_val = self._get_val(d, "length")
        a_val = self._get_val(a, "length")
        alpha_val = self._get_val(alpha, "angle")
        self.joints.append({'type': 'revolute', 'd': d_val, 'a': a_val, 'alpha': alpha_val})
        return self

    def add_prismatic_joint(self, theta, a, alpha):
        """Adds a prismatic joint. (theta=angle, a=link length, alpha=twist)"""
        t_val = self._get_val(theta, "angle")
        a_val = self._get_val(a, "length")
        alpha_val = self._get_val(alpha, "angle")
        self.joints.append({'type': 'prismatic', 'theta': t_val, 'a': a_val, 'alpha': alpha_val})
        return self

    def forward_kinematics(self, joint_vars):
        """
        Computes the 4x4 homogeneous transformation matrix of the end effector.
        `joint_vars` is a 1D tensor/list for a single configuration, or 2D (batched).
        """
        jv = _to_tensor(joint_vars).double()
        if jv.ndim == 1:
            jv = jv.unsqueeze(0)  # (1, num_joints)
            
        if jv.shape[1] != len(self.joints):
            raise ValueError(f"Expected {len(self.joints)} joint variables, but got {jv.shape[1]}")

        N = jv.shape[0]
        T = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(N, 1, 1)  # (N, 4, 4)
        
        for i, joint in enumerate(self.joints):
            var = jv[:, i]  # shape (N,)
            
            if joint['type'] == 'revolute':
                theta = var
                d = torch.tensor(joint['d'], dtype=torch.float64)
            else:
                theta = torch.tensor(joint['theta'], dtype=torch.float64)
                d = var
                
            a = torch.tensor(joint['a'], dtype=torch.float64)
            alpha = torch.tensor(joint['alpha'], dtype=torch.float64)
            
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(alpha) * torch.ones_like(theta)
            sa = torch.sin(alpha) * torch.ones_like(theta)
            zero = torch.zeros_like(theta)
            one = torch.ones_like(theta)
            
            d_t = d * one
            a_t = a * one
            
            # Build matrix A_i via torch.stack to preserve autograd graphs for theta/d!
            row0 = torch.stack([ct, -st*ca, st*sa, a_t*ct], dim=1)
            row1 = torch.stack([st, ct*ca, -ct*sa, a_t*st], dim=1)
            row2 = torch.stack([zero, sa, ca, d_t], dim=1)
            row3 = torch.stack([zero, zero, zero, one], dim=1)
            
            A_i = torch.stack([row0, row1, row2, row3], dim=1)  # (N, 4, 4)
            T = torch.bmm(T, A_i)
            
        if T.shape[0] == 1:
            return T.squeeze(0)  # (4, 4)
        return T  # (N, 4, 4)

def end_effector_pos_batched(T):
    if T.ndim == 2:
        return T[0:3, 3]
    return T[:, 0:3, 3]

def end_effector_rot_backend(T):
    if T.ndim == 2:
        return T[0:3, 0:3]
    return T[:, 0:3, 0:3]
