# --- Differentiable Robotics & Kinematics ---

import torch

class KinematicChain:
    """
    Modelliert einen Roboterarm (kinematische Kette) über Denavit-Hartenberg Parameter.
    Vollständig differenzierbar für Inverse Kinematik via Gradientenabstieg.
    """
    def __init__(self):
        self.joints = []

    def _get_val(self, v, dim):
        if isinstance(v, Quantity):
            return _convert_to_base(v.value, v.unit, dim)
        return float(v)

    def add_revolute_joint(self, d, a, alpha):
        """Fügt ein Drehgelenk hinzu. (d=offset, a=link length, alpha=twist)"""
        d_val = self._get_val(d, "length")
        a_val = self._get_val(a, "length")
        alpha_val = self._get_val(alpha, "angle")
        self.joints.append({'type': 'revolute', 'd': d_val, 'a': a_val, 'alpha': alpha_val})
        return self

    def add_prismatic_joint(self, theta, a, alpha):
        """Fügt ein Schubgelenk hinzu. (theta=angle, a=link length, alpha=twist)"""
        t_val = self._get_val(theta, "angle")
        a_val = self._get_val(a, "length")
        alpha_val = self._get_val(alpha, "angle")
        self.joints.append({'type': 'prismatic', 'theta': t_val, 'a': a_val, 'alpha': alpha_val})
        return self

    def forward_kinematics(self, joint_vars):
        """
        Berechnet die homogene 4x4 Transformationsmatrix des Endeffektors.
        `joint_vars` ist ein 1D Tensor/Liste für eine Konfiguration, oder 2D (Batched).
        """
        jv = _to_tensor(joint_vars).double()
        if jv.ndim == 1:
            jv = jv.unsqueeze(0)  # (1, num_joints)
            
        if jv.shape[1] != len(self.joints):
            raise ValueError(f"Erwartete {len(self.joints)} Gelenkvariablen, aber bekam {jv.shape[1]}")

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
            
            # Baue Matrix A_i über torch.stack auf, um die Autograd-Graphen für theta/d zu erhalten!
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
