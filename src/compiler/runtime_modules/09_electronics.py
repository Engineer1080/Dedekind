# --- Elektrotechnik und Regelungstechnik (Differentiable Engineering) ---

import torch

class Circuit:
    """
    Differentiable SPICE: Löst elektrische Schaltungen (DC) über Modified Nodal Analysis (MNA).
    Vollständig differenzierbar für Autograd (Optimierung von Bauteilwerten).
    """
    def __init__(self):
        self.nodes = set([0])
        self.resistors = []
        self.v_sources = []
        self.i_sources = []

    def _get_val(self, v, dim):
        if isinstance(v, Quantity):
            return _convert_to_base(v.value, v.unit, dim)
        return v
        
    def add_resistor(self, name, node1, node2, R):
        self.nodes.add(node1)
        self.nodes.add(node2)
        r_val = self._get_val(R, "resistance")
        self.resistors.append((node1, node2, r_val))
        return self

    def add_voltage_source(self, name, node1, node2, V):
        self.nodes.add(node1)
        self.nodes.add(node2)
        v_val = self._get_val(V, "electric_potential")
        self.v_sources.append((name, node1, node2, v_val))
        return self

    def add_current_source(self, name, node1, node2, I):
        self.nodes.add(node1)
        self.nodes.add(node2)
        i_val = self._get_val(I, "current")
        self.i_sources.append((node1, node2, i_val))
        return self

    def solve_dc(self):
        node_list = sorted(list(self.nodes))
        if 0 in node_list:
            node_list.remove(0)
            
        n_nodes = len(node_list)
        n_vs = len(self.v_sources)
        size = n_nodes + n_vs
        
        node_map = {node: i for i, node in enumerate(node_list)}
        node_map[0] = -1  # Ground
        
        # Build A and z as python lists of tensors to ensure autograd tracks gradient paths safely
        A = [[0.0 for _ in range(size)] for _ in range(size)]
        z = [0.0 for _ in range(size)]
        
        def add_to_A(i, j, val):
            if i >= 0 and j >= 0:
                A[i][j] = A[i][j] + val
                
        def add_to_z(i, val):
            if i >= 0:
                z[i] = z[i] + val

        # 1. Resistors (Conductance matrix G)
        for n1, n2, r_val in self.resistors:
            g = 1.0 / _to_tensor(r_val)
            idx1 = node_map[n1]
            idx2 = node_map[n2]
            add_to_A(idx1, idx1, g)
            add_to_A(idx2, idx2, g)
            add_to_A(idx1, idx2, -g)
            add_to_A(idx2, idx1, -g)
                
        # 2. Current sources
        for n1, n2, i_val in self.i_sources:
            # Leaves n1 (-), enters n2 (+)
            i_t = _to_tensor(i_val)
            idx1 = node_map[n1]
            idx2 = node_map[n2]
            add_to_z(idx1, -i_t)
            add_to_z(idx2, i_t)
                
        # 3. Voltage sources
        v_names = []
        for i, (name, n1, n2, v_val) in enumerate(self.v_sources):
            v_names.append(name)
            v_t = _to_tensor(v_val)
            idx1 = node_map[n1] # positive
            idx2 = node_map[n2] # negative
            v_idx = n_nodes + i
            
            add_to_z(v_idx, v_t)
            
            if idx1 >= 0:
                A[idx1][v_idx] = 1.0
                A[v_idx][idx1] = 1.0
            if idx2 >= 0:
                A[idx2][v_idx] = -1.0
                A[v_idx][idx2] = -1.0
                
        A_t = torch.stack([torch.stack([_to_tensor(e).float() for e in row]) for row in A])
        z_t = torch.stack([_to_tensor(e).float() for e in z])
        
        # Solve the system
        x = torch.linalg.solve(A_t, z_t)
        
        results = {}
        results["v_0"] = Quantity(0.0, "V")
        for node in node_list:
            idx = node_map[node]
            v_t = x[idx]
            if v_t.requires_grad:
                results[f"v_{node}"] = v_t
            else:
                results[f"v_{node}"] = Quantity(float(v_t.item()), "V")
        
        for i, name in enumerate(v_names):
            i_t = x[n_nodes + i]
            if i_t.requires_grad:
                results[f"i_{name}"] = i_t
            else:
                results[f"i_{name}"] = Quantity(float(i_t.item()), "A")
                
        return results


class Phasor:
    """Repräsentation einer komplexen Wechselstromgröße mit Einheit."""
    def __init__(self, mag, phase, unit=""):
        if isinstance(mag, Quantity):
            self.mag = float(mag.value)
            self.unit = mag.unit
        else:
            self.mag = float(mag)
            self.unit = unit
            
        if isinstance(phase, Quantity):
            self.phase = float(_convert_to_base(phase.value, phase.unit, "angle"))
        else:
            self.phase = float(phase)
            
    def complex_value(self):
        import cmath
        return cmath.rect(self.mag, self.phase)
        
    def __repr__(self):
        import math
        deg = math.degrees(self.phase)
        return f"{self.mag}[{self.unit}] \u2220 {deg:.2f}\xb0"


class StateSpace:
    """
    Linear Time-Invariant (LTI) System im Zustandsraum:
    x_dot = A*x + B*u
    y = C*x + D*u
    Voll differenzierbar für KI-Regler-Optimierung.
    """
    def __init__(self, A, B, C, D):
        def _ensure_2d(t):
            t = _to_tensor(t).float()
            if t.ndim == 0:
                return t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 1:
                return t.unsqueeze(1)
            return t
            
        self.A = _ensure_2d(A)
        self.B = _ensure_2d(B)
        self.C = _ensure_2d(C)
        self.D = _ensure_2d(D)
        
    def step_response(self, t):
        """Simuliert die Sprungantwort (u=1) über die Zeit t (Tensor) via ode_solve."""
        if 'ode_solve' not in globals():
            raise RuntimeError("ode_solve function not found. Ensure 03_solvers.py is loaded.")
        
        t_t = _to_tensor(t).float()
        
        def system_dynamics(t_scalar, x):
            # Step input u = [1, 1, ...]
            u = torch.ones((self.B.shape[1],), dtype=torch.float32)
            # x_dot = Ax + Bu
            return torch.matmul(self.A, x) + torch.matmul(self.B, u)
            
        x0 = torch.zeros((self.A.shape[0],), dtype=torch.float32)
        
        # ode_solve liefert x(t) als Tensor der Shape (len(t), states)
        x_out = globals()['ode_solve'](system_dynamics, x0, t_t, method="rk4")
        
        u_out = torch.ones((len(t_t), self.B.shape[1]), dtype=torch.float32)
        y_out = torch.matmul(x_out, self.C.t()) + torch.matmul(u_out, self.D.t())
        
        return t_t, y_out, x_out

    def frequency_response(self, omega):
        """Berechnet die komplexe Frequenzantwort H(jw) voll vektorisiert."""
        omega_t = _to_tensor(omega).float()
        
        # A, B, C, D in complex64
        A_c = self.A.to(torch.complex64)
        B_c = self.B.to(torch.complex64)
        C_c = self.C.to(torch.complex64)
        D_c = self.D.to(torch.complex64)
        
        n = self.A.shape[0]
        I = torch.eye(n, dtype=torch.complex64)
        
        # s = j*omega (shape: [N])
        s = 1j * omega_t
        
        # sI_A = s*I - A
        # Broadcasting: s is [N], I is [n, n] -> s[:, None, None] * I[None, :, :] -> [N, n, n]
        sI = s.unsqueeze(-1).unsqueeze(-1) * I.unsqueeze(0)
        sI_A = sI - A_c.unsqueeze(0)
        
        # Batched complex inversion: [N, n, n]
        inv = torch.linalg.inv(sI_A)
        
        # H = C * inv * B + D
        # C_c.unsqueeze(0) ist [1, p, n], B_c.unsqueeze(0) ist [1, n, q]
        H = torch.matmul(C_c.unsqueeze(0), torch.matmul(inv, B_c.unsqueeze(0))) + D_c.unsqueeze(0)
        
        # Berechne Magnitude in dB und Phase in Grad
        mag = torch.abs(H)
        mag_db = 20.0 * torch.log10(mag + 1e-12)
        
        phase = torch.angle(H)
        # Phase in Grad umrechnen
        import math
        phase_deg = phase * (180.0 / math.pi)
        
        return omega_t, mag_db, phase_deg


