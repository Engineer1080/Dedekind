# --- Signals & Systems (Electronics, DSP, Control) ---
import torch
import torch.nn.functional as F
import math
import builtins

def _get_val(v):
    if hasattr(v, "value"):
        return v.value
    return v

def _to_double_tensor(v):
    t = _to_tensor(v)
    return t.double()

# --- 1. Electronics (Differentiable SPICE & Phasors) ---

class Circuit:
    """
    Differentiable SPICE: Solves electrical circuits (DC) via Modified Nodal Analysis (MNA).
    Fully differentiable for autograd (optimization of component values).
    """
    def __init__(self):
        self.nodes = set([0])
        self.resistors = []
        self.capacitors = []
        self.inductors = []
        self.v_sources = []
        self.i_sources = []

    def _get_val_dim(self, v, dim):
        if isinstance(v, Quantity):
            return _convert_to_base(v.value, v.unit, dim)
        return v
        
    def add_resistor(self, name, node1, node2, R):
        self.nodes.add(node1)
        self.nodes.add(node2)
        r_val = self._get_val_dim(R, "resistance")
        self.resistors.append((node1, node2, r_val))
        return self

    def add_capacitor(self, name, node1, node2, C):
        self.nodes.add(node1)
        self.nodes.add(node2)
        c_val = self._get_val_dim(C, "capacitance")
        self.capacitors.append((node1, node2, c_val))
        return self

    def add_inductor(self, name, node1, node2, L):
        self.nodes.add(node1)
        self.nodes.add(node2)
        l_val = self._get_val_dim(L, "inductance")
        self.inductors.append((node1, node2, l_val))
        return self

    def add_voltage_source(self, name, node1, node2, V):
        self.nodes.add(node1)
        self.nodes.add(node2)
        v_val = self._get_val_dim(V, "electric_potential")
        self.v_sources.append((name, node1, node2, v_val))
        return self

    def add_current_source(self, name, node1, node2, I):
        self.nodes.add(node1)
        self.nodes.add(node2)
        i_val = self._get_val_dim(I, "current")
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
            results[f"v_{node}"] = Quantity(x[idx], "V")
        
        for i, name in enumerate(v_names):
            results[f"i_{name}"] = Quantity(x[n_nodes + i], "A")
                
        return results

    def solve_ac(self, omega):
        # We want double precision complex number computations
        omega_val = self._get_val_dim(omega, "frequency")
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
    """Representation of a complex AC quantity with unit."""
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

# --- 2. Digital Signal Processing (DSP) & Z-Transform ---

def fir_filter_impl(x, b):
    """
    Filters the input signal `x` with the FIR coefficients `b` (feedforward).
    Fully differentiable with respect to `b` and `x`.
    """
    x_t = _to_double_tensor(x)
    b_t = _to_double_tensor(b)
    
    is_1d = (x_t.ndim == 1)
    if is_1d:
        x_t = x_t.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
    elif x_t.ndim == 2:
        x_t = x_t.unsqueeze(1)  # (N, 1, L)
        
    M = b_t.shape[0]
    x_padded = F.pad(x_t, (M - 1, 0), mode='constant', value=0.0)
    
    # We flip b_t for correlation filtering (since PyTorch conv1d is correlation)
    b_flipped = torch.flip(b_t, dims=[0]).view(1, 1, M)
    
    y = F.conv1d(x_padded, b_flipped)
    if is_1d:
        return y.squeeze(0).squeeze(0)
    return y.squeeze(1)

def iir_filter_impl(x, b, a):
    """
    Implements a general IIR filter with feedforward coefficients `b`
    and feedback coefficients `a` (with standardized a[0] = 1.0).
    Fully differentiable with respect to `b`, `a`, and `x`.
    """
    x_t = _to_double_tensor(x)
    b_t = _to_double_tensor(b)
    a_t = _to_double_tensor(a)
    
    # Normalize by a[0] if it's not 1.0
    if a_t[0] != 1.0:
        norm = a_t[0]
        b_t = b_t / norm
        a_t = a_t / norm
        
    M = b_t.shape[0]
    N = a_t.shape[0]
    L = x_t.shape[0]
    
    y_list = []
    for n in range(L):
        val = b_t[0] * x_t[n]
        for k in range(1, M):
            if n - k >= 0:
                val = val + b_t[k] * x_t[n - k]
        for j in range(1, N):
            if n - j >= 0:
                val = val - a_t[j] * y_list[n - j]
        y_list.append(val)
        
    return torch.stack(y_list)

def _get_val_hz(v):
    if isinstance(v, Quantity):
        return _convert_to_base(v.value, v.unit, "frequency")
    return float(v)

def biquad_lowpass_impl(fc, Q, fs=1.0):
    """
    Computes biquad 2nd-order lowpass filter coefficients b, a.
    Cutoff frequency fc and quality factor Q can be tensors or numbers.
    Sampling rate fs can be a Quantity or float.
    Filter design is fully differentiable!
    """
    fs_val = _get_val_hz(fs)
    
    if isinstance(fc, Quantity):
        fc_t = _to_double_tensor(_convert_to_base(fc.value, fc.unit, "frequency"))
    else:
        fc_t = _to_double_tensor(fc)
        
    Q_t = _to_double_tensor(Q)
    
    w0 = 2.0 * 3.141592653589793 * fc_t / fs_val
    alpha = torch.sin(w0) / (2.0 * Q_t)
    cos_w0 = torch.cos(w0)
    
    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    
    b = torch.stack([b0, b1, b2]) / a0
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0])
    return b, a

def biquad_highpass_impl(fc, Q, fs=1.0):
    """
    Computes biquad 2nd-order highpass filter coefficients b, a.
    Cutoff frequency fc and quality factor Q can be tensors or numbers.
    Filter design is fully differentiable!
    """
    fs_val = _get_val_hz(fs)
    
    if isinstance(fc, Quantity):
        fc_t = _to_double_tensor(_convert_to_base(fc.value, fc.unit, "frequency"))
    else:
        fc_t = _to_double_tensor(fc)
    Q_t = _to_double_tensor(Q)
    
    w0 = 2.0 * 3.141592653589793 * fc_t / fs_val
    alpha = torch.sin(w0) / (2.0 * Q_t)
    cos_w0 = torch.cos(w0)
    
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    
    b = torch.stack([b0, b1, b2]) / a0
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0])
    return b, a

def biquad_bandpass_impl(fc, Q, fs=1.0):
    """
    Computes biquad 2nd-order bandpass filter coefficients b, a.
    Cutoff frequency fc and quality factor Q can be tensors or numbers.
    Filter design is fully differentiable!
    """
    fs_val = _get_val_hz(fs)
    
    if isinstance(fc, Quantity):
        fc_t = _to_double_tensor(_convert_to_base(fc.value, fc.unit, "frequency"))
    else:
        fc_t = _to_double_tensor(fc)
    Q_t = _to_double_tensor(Q)
    
    w0 = 2.0 * 3.141592653589793 * fc_t / fs_val
    alpha = torch.sin(w0) / (2.0 * Q_t)
    cos_w0 = torch.cos(w0)
    
    b0 = alpha
    b1 = torch.zeros_like(w0)
    b2 = -alpha
    
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    
    b = torch.stack([b0, b1, b2]) / a0
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0])
    return b, a

def freqz_impl(b, a, worN=512):
    """
    Computes the complex frequency response H(jw) fully vectorized at worN points.
    Fully differentiable with respect to b and a.
    """
    b_t = _to_double_tensor(b)
    a_t = _to_double_tensor(a)
    
    omega = torch.linspace(0.0, 3.141592653589793, int(worN), dtype=b_t.dtype, device=b_t.device)
    j_unit = torch.complex(torch.tensor(0.0, dtype=b_t.dtype), torch.tensor(1.0, dtype=b_t.dtype))
    
    num = torch.zeros(int(worN), dtype=torch.complex128, device=b_t.device)
    for k, bk in enumerate(b_t):
        num = num + bk * torch.exp(-j_unit * k * omega)
        
    den = torch.zeros(int(worN), dtype=torch.complex128, device=a_t.device)
    for k, ak in enumerate(a_t):
        den = den + ak * torch.exp(-j_unit * k * omega)
        
    h = num / den
    return omega, h

def butter_impl(order, Wn, btype='low', fs=None):
    """
    Wrapper for classic Butterworth filter design (SciPy).
    Returns the b and a coefficients as PyTorch float64 tensors.
    """
    import scipy.signal as sig  # type: ignore[import-untyped]
    
    if isinstance(Wn, Quantity):
        Wn_val = _convert_to_base(Wn.value, Wn.unit, "frequency")
    elif hasattr(Wn, "tolist"):
        Wn_val = Wn.tolist()
    else:
        Wn_val = Wn
        
    fs_val = _get_val_hz(fs) if fs is not None else None
    
    b, a = sig.butter(int(order), Wn_val, btype=btype, fs=fs_val)
    return torch.tensor(b, dtype=torch.float64), torch.tensor(a, dtype=torch.float64)

def cheby1_impl(order, rp, Wn, btype='low', fs=None):
    """
    Wrapper for classic Chebyshev Type I filter design (SciPy).
    Returns the b and a coefficients as PyTorch float64 tensors.
    """
    import scipy.signal as sig  # type: ignore[import-untyped]
    
    if isinstance(Wn, Quantity):
        Wn_val = _convert_to_base(Wn.value, Wn.unit, "frequency")
    elif hasattr(Wn, "tolist"):
        Wn_val = Wn.tolist()
    else:
        Wn_val = Wn
        
    fs_val = _get_val_hz(fs) if fs is not None else None
    
    b, a = sig.cheby1(int(order), float(rp), Wn_val, btype=btype, fs=fs_val)
    return torch.tensor(b, dtype=torch.float64), torch.tensor(a, dtype=torch.float64)

# --- 3. Control Theory & Block Diagram Simulation ---

class Block:
    """Base class for differentiable simulation blocks."""
    def __init__(self):
        self.inputs = []
        self._output = torch.tensor(0.0, dtype=torch.float64)
        self.is_stateful = False

    def set_input(self, inp):
        self.inputs = [inp]
        return self

    def set_inputs(self, inps):
        self.inputs = list(inps)
        return self

    def _eval_input(self, inp, memo, dt):
        if isinstance(inp, Block):
            return inp.eval(memo, dt)
        return _to_tensor(inp).double()

    def eval(self, memo, dt):
        if self in memo:
            return memo[self]
        if self.is_stateful:
            memo[self] = self._output
            return self._output

        out = self._compute_output(memo, dt)
        memo[self] = out
        return out

    def _compute_output(self, memo, dt):
        raise NotImplementedError("Subclasses must implement _compute_output")

    def reset(self):
        pass

class ConstantBlock(Block):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def _compute_output(self, memo, dt):
        return _to_tensor(self.value).double()

class GainBlock(Block):
    def __init__(self, inp=None, K=1.0):
        super().__init__()
        self.K = K
        if inp is not None:
            self.set_input(inp)

    def _compute_output(self, memo, dt):
        u = self._eval_input(self.inputs[0], memo, dt)
        return _to_tensor(self.K).double() * u

class SumBlock(Block):
    def __init__(self, inputs=None, signs=None):
        super().__init__()
        self.signs = signs
        if inputs is not None:
            self.set_inputs(inputs)

    def _compute_output(self, memo, dt):
        out = torch.tensor(0.0, dtype=torch.float64, device=self.inputs[0]._output.device if (self.inputs and isinstance(self.inputs[0], Block)) else None)
        for i, inp in enumerate(self.inputs):
            val = self._eval_input(inp, memo, dt)
            sign = 1.0
            if self.signs is not None and i < len(self.signs):
                sign = float(self.signs[i])
            out = out + sign * val
        return out

class ProductBlock(Block):
    def __init__(self, inputs=None):
        super().__init__()
        if inputs is not None:
            self.set_inputs(inputs)

    def _compute_output(self, memo, dt):
        out = torch.tensor(1.0, dtype=torch.float64)
        for inp in self.inputs:
            val = self._eval_input(inp, memo, dt)
            out = out * val
        return out

class SaturationBlock(Block):
    def __init__(self, inp=None, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        if inp is not None:
            self.set_input(inp)

    def _compute_output(self, memo, dt):
        u = self._eval_input(self.inputs[0], memo, dt)
        return torch.clamp(u, self.min_val, self.max_val)

class IntegratorBlock(Block):
    def __init__(self, inp=None, x0=0.0):
        super().__init__()
        self.is_stateful = True
        self.x0 = x0
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        self.state = _to_tensor(self.x0).double().clone()
        self._output = self.state

    def update_state(self, dt, memo):
        u = self._eval_input(self.inputs[0], memo, dt)
        self.state = self.state + u * dt
        self._output = self.state

class PIDBlock(Block):
    def __init__(self, inp=None, Kp=1.0, Ki=0.0, Kd=0.0, min_val=None, max_val=None):
        super().__init__()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min_val = min_val
        self.max_val = max_val
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        self.integral = torch.tensor(0.0, dtype=torch.float64)
        self.prev_err = torch.tensor(0.0, dtype=torch.float64)
        self.has_prev = False

    def _compute_output(self, memo, dt):
        err = self._eval_input(self.inputs[0], memo, dt)
        p_term = _to_tensor(self.Kp).double() * err
        i_term = _to_tensor(self.Ki).double() * self.integral
        if self.has_prev:
            d_term = _to_tensor(self.Kd).double() * (err - self.prev_err) / dt
        else:
            d_term = torch.tensor(0.0, dtype=torch.float64, device=err.device)
        u = p_term + i_term + d_term
        if self.min_val is not None and self.max_val is not None:
            u = torch.clamp(u, self.min_val, self.max_val)
        return u

    def update_state(self, dt, memo):
        err = self._eval_input(self.inputs[0], memo, dt)
        self.integral = self.integral + err * dt
        self.prev_err = err
        self.has_prev = True

class StateSpace:
    """
    Linear Time-Invariant (LTI) System in state space:
    x_dot = A*x + B*u
    y = C*x + D*u
    Fully differentiable for AI controller optimization.
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
        """Simulates the step response (u=1) over time t (tensor) via ode_solve."""
        if 'ode_solve' not in globals():
            raise RuntimeError("ode_solve function not found. Ensure 03_solvers.py is loaded.")
        
        t_t = _to_tensor(t).float()
        
        def system_dynamics(t_scalar, x):
            u = torch.ones((self.B.shape[1],), dtype=torch.float32)
            return torch.matmul(self.A, x) + torch.matmul(self.B, u)
            
        x0 = torch.zeros((self.A.shape[0],), dtype=torch.float32)
        x_out = globals()['ode_solve'](system_dynamics, x0, t_t, method="rk4")
        
        u_out = torch.ones((len(t_t), self.B.shape[1]), dtype=torch.float32)
        y_out = torch.matmul(x_out, self.C.t()) + torch.matmul(u_out, self.D.t())
        
        return t_t, y_out, x_out

    def frequency_response(self, omega):
        """Computes the complex frequency response H(jw) fully vectorized."""
        omega_t = _to_tensor(omega).float()
        A_c = self.A.to(torch.complex64)
        B_c = self.B.to(torch.complex64)
        C_c = self.C.to(torch.complex64)
        D_c = self.D.to(torch.complex64)
        
        n = self.A.shape[0]
        I = torch.eye(n, dtype=torch.complex64)
        s = 1j * omega_t
        sI = s.unsqueeze(-1).unsqueeze(-1) * I.unsqueeze(0)
        sI_A = sI - A_c.unsqueeze(0)
        
        inv = torch.linalg.inv(sI_A)
        H = torch.matmul(C_c.unsqueeze(0), torch.matmul(inv, B_c.unsqueeze(0))) + D_c.unsqueeze(0)
        
        mag = torch.abs(H)
        mag_db = 20.0 * torch.log10(mag + 1e-12)
        phase = torch.angle(H)
        phase_deg = phase * (180.0 / math.pi)
        
        return omega_t, mag_db, phase_deg

def tf2ss(num, den):
    den_coeffs = [_to_tensor(a).double() for a in den]
    num_coeffs = [_to_tensor(b).double() for b in num]
    
    a_lead = den_coeffs[0]
    den_coeffs = [a / a_lead for a in den_coeffs]
    num_coeffs = [b / a_lead for b in num_coeffs]
    
    n = len(den_coeffs) - 1
    if len(num_coeffs) < len(den_coeffs):
        num_coeffs = [torch.tensor(0.0, dtype=torch.float64, device=a_lead.device)] * (len(den_coeffs) - len(num_coeffs)) + num_coeffs
        
    if n == 0:
        A = torch.zeros((0, 0), dtype=torch.float64, device=a_lead.device)
        B = torch.zeros((0,), dtype=torch.float64, device=a_lead.device)
        C = torch.zeros((0,), dtype=torch.float64, device=a_lead.device)
        D = num_coeffs[0]
        return A, B, C, D

    A = [[torch.tensor(0.0, dtype=torch.float64, device=a_lead.device) for _ in range(n)] for _ in range(n)]
    for i in range(n - 1):
        A[i][i+1] = torch.tensor(1.0, dtype=torch.float64, device=a_lead.device)
    for j in range(n):
        A[n-1][j] = -den_coeffs[len(den_coeffs) - 1 - j]
        
    B = [torch.tensor(0.0, dtype=torch.float64, device=a_lead.device) for _ in range(n)]
    B[n-1] = torch.tensor(1.0, dtype=torch.float64, device=a_lead.device)
    
    b_n = num_coeffs[0]
    C = []
    for j in range(n):
        b_j = num_coeffs[len(num_coeffs) - 1 - j]
        a_j = den_coeffs[len(den_coeffs) - 1 - j]
        C.append(b_j - b_n * a_j)
        
    D = b_n
    
    A_t = torch.stack([torch.stack(row) for row in A])
    B_t = torch.stack(B)
    C_t = torch.stack(C)
    D_t = _to_tensor(D).double()
    return A_t, B_t, C_t, D_t

class StateSpaceBlock(Block):
    def __init__(self, inp=None, A=None, B=None, C=None, D=None):
        super().__init__()
        self.is_stateful = True
        self.A = _to_tensor(A).double()
        self.B = _to_tensor(B).double()
        self.C = _to_tensor(C).double()
        self.D = _to_tensor(D).double() if D is not None else torch.zeros((C.shape[0], B.shape[1] if B.ndim > 1 else 1), dtype=torch.float64)
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        n = self.A.shape[0]
        self.state = torch.zeros((n,), dtype=torch.float64, device=self.A.device)
        self._output = self.C @ self.state

    def update_state(self, dt, memo):
        u = self._eval_input(self.inputs[0], memo, dt)
        dx = self.A @ self.state + self.B * u
        self.state = self.state + dx * dt
        self._output = self.C @ self.state

class TransferFunctionBlock(Block):
    def __init__(self, inp=None, num=[1.0], den=[1.0, 1.0]):
        super().__init__()
        self.is_stateful = True
        self.num = num
        self.den = den
        if inp is not None:
            self.set_input(inp)
        self.reset()

    def reset(self):
        self.A, self.B, self.C, self.D = tf2ss(self.num, self.den)
        n = self.A.shape[0]
        self.state = torch.zeros((n,), dtype=torch.float64, device=self.A.device)
        self._output = self.C @ self.state

    def update_state(self, dt, memo):
        u = self._eval_input(self.inputs[0], memo, dt)
        dx = self.A @ self.state + self.B * u
        self.state = self.state + dx * dt
        self._output = self.C @ self.state

class BlockDiagram:
    def __init__(self):
        self.blocks = []

    def add_block(self, block):
        if block not in self.blocks:
            self.blocks.append(block)
        return block

    def simulate(self, t_max, dt):
        t_t = _to_tensor(t_max).double()
        dt_t = _to_tensor(dt).double()
        steps = int(torch.round(t_t / dt_t).item())

        for block in self.blocks:
            block.reset()

        history = {block: [] for block in self.blocks}
        time_history = []

        t = 0.0
        for _ in range(steps):
            memo = {}
            for block in self.blocks:
                block.eval(memo, dt_t)

            for block in self.blocks:
                history[block].append(memo[block])
            time_history.append(t)

            for block in self.blocks:
                if hasattr(block, 'update_state'):
                    block.update_state(dt_t, memo)

            t += dt_t.item()

        res = {}
        for block in self.blocks:
            res[block] = torch.stack(history[block])
        res["t"] = torch.tensor(time_history, dtype=torch.float64)
        return res

# Helper factories for DDK binding
def block_diagram_impl():
    return BlockDiagram()

def constant_block_impl(value):
    return ConstantBlock(value)

def gain_block_impl(inp, K):
    return GainBlock(inp, K)

def sum_block_impl(inputs, signs=None):
    return SumBlock(inputs, signs)

def product_block_impl(inputs):
    return ProductBlock(inputs)

def saturation_block_impl(inp, min_val, max_val):
    return SaturationBlock(inp, min_val, max_val)

def integrator_block_impl(inp, x0=0.0):
    return IntegratorBlock(inp, x0)

def pid_block_impl(inp, Kp, Ki, Kd):
    return PIDBlock(inp, Kp, Ki, Kd)

def pid_block_saturated_impl(inp, Kp, Ki, Kd, min_val, max_val):
    return PIDBlock(inp, Kp, Ki, Kd, min_val, max_val)

def state_space_block_impl(inp, A, B, C, D=None):
    return StateSpaceBlock(inp, A, B, C, D)

def transfer_function_block_impl(inp, num, den):
    return TransferFunctionBlock(inp, num, den)
