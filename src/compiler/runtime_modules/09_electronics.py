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
        self.capacitors = []
        self.inductors = []
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

    def add_capacitor(self, name, node1, node2, C):
        self.nodes.add(node1)
        self.nodes.add(node2)
        c_val = self._get_val(C, "capacitance")
        self.capacitors.append((node1, node2, c_val))
        return self

    def add_inductor(self, name, node1, node2, L):
        self.nodes.add(node1)
        self.nodes.add(node2)
        l_val = self._get_val(L, "inductance")
        self.inductors.append((node1, node2, l_val))
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

    def solve_ac(self, omega):
        # We want double precision complex number computations
        omega_val = self._get_val(omega, "frequency")
        omega_t = _to_tensor(omega_val).double()
        
        node_list = sorted(list(self.nodes))
        if 0 in node_list:
            node_list.remove(0)
            
        n_nodes = len(node_list)
        n_vs = len(self.v_sources)
        size = n_nodes + n_vs
        
        node_map = {node: i for i, node in enumerate(node_list)}
        node_map[0] = -1  # Ground
        
        # Build complex A and z
        A = [[torch.tensor(0.0, dtype=torch.complex128, device=omega_t.device) for _ in range(size)] for _ in range(size)]
        z = [torch.tensor(0.0, dtype=torch.complex128, device=omega_t.device) for _ in range(size)]
        
        def add_to_A(i, j, val):
            if i >= 0 and j >= 0:
                A[i][j] = A[i][j] + val
                
        def add_to_z(i, val):
            if i >= 0:
                z[i] = z[i] + val

        # 1. Resistors (Conductance Y_R = 1/R)
        for n1, n2, r_val in self.resistors:
            r_t = _to_tensor(r_val).double()
            g = 1.0 / r_t
            g_c = torch.complex(g, torch.zeros_like(g))
            idx1 = node_map[n1]
            idx2 = node_map[n2]
            add_to_A(idx1, idx1, g_c)
            add_to_A(idx2, idx2, g_c)
            add_to_A(idx1, idx2, -g_c)
            add_to_A(idx2, idx1, -g_c)
            
        # 2. Capacitors (Admittance Y_C = j * omega * C)
        for n1, n2, c_val in self.capacitors:
            c_t = _to_tensor(c_val).double()
            y_c = torch.complex(torch.zeros_like(c_t), omega_t * c_t)
            idx1 = node_map[n1]
            idx2 = node_map[n2]
            add_to_A(idx1, idx1, y_c)
            add_to_A(idx2, idx2, y_c)
            add_to_A(idx1, idx2, -y_c)
            add_to_A(idx2, idx1, -y_c)
            
        # 3. Inductors (Admittance Y_L = -j / (omega * L))
        for n1, n2, l_val in self.inductors:
            l_t = _to_tensor(l_val).double()
            omega_safe = torch.where(omega_t == 0.0, torch.ones_like(omega_t), omega_t)
            l_safe = torch.where(l_t == 0.0, torch.ones_like(l_t), l_t)
            y_val = -1.0 / (omega_safe * l_safe)
            y_l = torch.complex(torch.zeros_like(l_t), y_val)
            idx1 = node_map[n1]
            idx2 = node_map[n2]
            add_to_A(idx1, idx1, y_l)
            add_to_A(idx2, idx2, y_l)
            add_to_A(idx1, idx2, -y_l)
            add_to_A(idx2, idx1, -y_l)
            
        # 4. Current sources (Complex Phasor)
        for n1, n2, i_val in self.i_sources:
            i_c = _to_complex_phasor(i_val)
            idx1 = node_map[n1]
            idx2 = node_map[n2]
            add_to_z(idx1, -i_c)
            add_to_z(idx2, i_c)
            
        # 5. Voltage sources (Complex Phasor)
        v_names = []
        for i, (name, n1, n2, v_val) in enumerate(self.v_sources):
            v_names.append(name)
            v_c = _to_complex_phasor(v_val)
            idx1 = node_map[n1]
            idx2 = node_map[n2]
            v_idx = n_nodes + i
            
            add_to_z(v_idx, v_c)
            
            one_c = torch.tensor(1.0, dtype=torch.complex128, device=omega_t.device)
            if idx1 >= 0:
                A[idx1][v_idx] = A[idx1][v_idx] + one_c
                A[v_idx][idx1] = A[v_idx][idx1] + one_c
            if idx2 >= 0:
                A[idx2][v_idx] = A[idx2][v_idx] - one_c
                A[v_idx][idx2] = A[v_idx][idx2] - one_c
                
        # Stack into complex tensors
        A_t = torch.stack([torch.stack(row) for row in A])
        z_t = torch.stack(z)
        
        # Solve the complex system
        x = torch.linalg.solve(A_t, z_t)
        
        results = {}
        # Node voltages
        results["v_0"] = Phasor(0.0, 0.0, "V")
        for node in node_list:
            idx = node_map[node]
            v_c = x[idx]
            results[f"v_{node}"] = Phasor(v_c, unit="V")
            
        # Voltage source currents
        for i, name in enumerate(v_names):
            i_c = x[n_nodes + i]
            results[f"i_{name}"] = Phasor(i_c, unit="A")
            
        return results


def _to_complex_phasor(v):
    if isinstance(v, Phasor):
        return v.complex_value()
    if isinstance(v, Quantity):
        val = _to_tensor(v.value).double()
        return torch.complex(val, torch.zeros_like(val))
    val = _to_tensor(v).double()
    if torch.is_complex(val):
        return val
    return torch.complex(val, torch.zeros_like(val))


class Phasor:
    """Repräsentation einer komplexen Wechselstromgröße mit Einheit."""
    def __init__(self, mag, phase=0.0, unit=""):
        if isinstance(mag, complex) or (isinstance(mag, torch.Tensor) and torch.is_complex(mag)):
            c_val = mag
            if isinstance(c_val, complex):
                import cmath
                mag_val, phase_val = cmath.polar(c_val)
                self.mag = torch.tensor(mag_val, dtype=torch.float64)
                self.phase = torch.tensor(phase_val, dtype=torch.float64)
            else:
                self.mag = torch.abs(c_val)
                self.phase = torch.angle(c_val)
            self.unit = unit
        else:
            if isinstance(mag, Quantity):
                self.mag = _to_tensor(mag.value).double()
                self.unit = mag.unit
            else:
                self.mag = _to_tensor(mag).double()
                self.unit = unit
                
            if isinstance(phase, Quantity):
                self.phase = _to_tensor(_convert_to_base(phase.value, phase.unit, "angle")).double()
            else:
                self.phase = _to_tensor(phase).double()
            
    def complex_value(self):
        mag = _to_tensor(self.mag).double()
        phase = _to_tensor(self.phase).double()
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        
    @property
    def value(self):
        return self.mag
        
    @property
    def magnitude(self):
        return self.mag
        
    @property
    def angle(self):
        return self.phase

    def __repr__(self):
        import math
        mag_f = float(self.mag.item()) if hasattr(self.mag, 'item') else float(self.mag)
        phase_f = float(self.phase.item()) if hasattr(self.phase, 'item') else float(self.phase)
        deg = math.degrees(phase_f)
        return f"{mag_f:.4f}[{self.unit}] \u2220 {deg:.2f}\xb0"


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


