import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]

class Quantity:
    """Physikalische Größe mit Einheit (z. B. 10[m], 5[m/s]). Rechenregeln: gleiche Einheit für +/-, Einheiten multiplizieren/dividieren."""
    def __init__(self, value, unit=""):
        self.value = float(value)
        self.unit = str(unit) if unit else ""

    def _same_unit(self, other):
        if not isinstance(other, Quantity):
            return False
        return self.unit == other.unit

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.unit: raise ValueError(f"Kann Zahl nicht zu Größe mit Einheit [{self.unit}] addieren.")
            return Quantity(self.value + other, "")
        if isinstance(other, Quantity):
            if not self._same_unit(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}]")
            return Quantity(self.value + other.value, self.unit)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if self.unit: raise ValueError(f"Kann Zahl nicht von Größe mit Einheit [{self.unit}] subtrahieren.")
            return Quantity(self.value - other, "")
        if isinstance(other, Quantity):
            if not self._same_unit(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}]")
            return Quantity(self.value - other.value, self.unit)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)) and not self.unit:
            return Quantity(other - self.value, "")
        return NotImplemented

    def __neg__(self):
        return Quantity(-self.value, self.unit)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit)
        if isinstance(other, Quantity):
            v = self.value * other.value
            u = _unit_mul(self.unit, other.unit)
            return Quantity(v, u)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(other * self.value, self.unit)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit)
        if isinstance(other, Quantity):
            v = self.value / other.value
            u = _unit_div(self.unit, other.unit)
            return Quantity(v, u)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(other / self.value, _unit_inv(self.unit))
        return NotImplemented

    def __pow__(self, exp):
        if isinstance(exp, (int, float)):
            return Quantity(self.value ** exp, _unit_pow(self.unit, exp))
        return NotImplemented

    def __repr__(self):
        if not self.unit:
            return str(self.value)
        display_unit = _unit_simplify(self.unit)
        return f"{self.value}[{display_unit}]"

def _unit_mul(u1, u2):
    if not u1: return u2
    if not u2: return u1
    return f"{u1}*{u2}" if "*" not in u1 and "/" not in u1 and "*" not in u2 and "/" not in u2 else f"({u1})*({u2})"

def _unit_div(u1, u2):
    if not u2: return u1
    if not u1: return f"1/{u2}" if "*" not in u2 and "/" not in u2 else f"1/({u2})"
    return f"{u1}/{u2}" if "*" not in u1 and "/" not in u1 and "*" not in u2 and "/" not in u2 else f"({u1})/({u2})"

def _unit_inv(u):
    if not u: return ""
    return f"1/{u}" if "*" not in u and "/" not in u else f"1/({u})"

def _unit_pow(u, exp):
    """Einheit hoch exp (z. B. m^2, (m/s)^2)."""
    if not u: return ""
    e = float(exp)
    if abs(e - 1.0) < 1e-12: return u
    if abs(e + 1.0) < 1e-12: return _unit_inv(u)
    base = u if ("*" not in u and "/" not in u) else f"({u})"
    if abs(e - round(e)) < 1e-12:
        return f"{base}^{int(round(e))}"
    return f"{base}^{e}"

def _unit_simplify(u):
    """Vereinfacht Einheiten-String für Anzeige (z. B. (kg)*((m/s)^2) -> J)."""
    if not u: return u
    # Joule: kg*m^2/s^2 bzw. (kg)*((m/s)^2)
    if "(kg)*((m/s)^2)" in u or u.strip() == "kg*m^2/s^2":
        return "J"
    if "(J*s)*(Hz)" in u or "J*s*Hz" in u or ("J*s" in u and "Hz" in u):
        return "J"
    # Newton: Gravitation G*m1*m2/r^2 oder Coulomb k_e*q1*q2/r^2
    if "m^3/(kg*s^2)" in u and "m^2" in u:
        return "N"
    if "N*m^2/C^2" in u and "m^2" in u:
        return "N"
    if "kg*m/s^2" in u or u.strip() == "N":
        return "N"
    return u

# --- Mathematical Constants (dimensionless) ---
pi  = Quantity(3.14159265358979323846, "")
e   = Quantity(2.71828182845904523536, "")  # Euler's number

# --- Fundamental Physical Constants (CODATA 2018/2022) as Quantity with SI units ---
c       = Quantity(299792458.0, "m/s")
G       = Quantity(6.67430e-11, "m^3/(kg*s^2)")
h       = Quantity(6.62607015e-34, "J*s")
hbar    = Quantity(1.054571817e-34, "J*s")   # h / (2*pi)
k_B     = Quantity(1.380649e-23, "J/K")
k_e     = Quantity(8.9875517923e9, "N*m^2/C^2")
e_charge = Quantity(1.602176634e-19, "C")    # elementary charge
epsilon_0 = Quantity(8.8541878128e-12, "F/m")  # vacuum permittivity
mu_0    = Quantity(1.25663706212e-6, "N/A^2") # vacuum permeability
m_e     = Quantity(9.1093837015e-31, "kg")   # electron mass
m_p     = Quantity(1.67262192369e-27, "kg")  # proton mass
N_A     = Quantity(6.02214076e23, "1/mol")   # Avogadro constant
R_gas   = Quantity(8.314462618, "J/(K*mol)")  # universal gas constant R = N_A * k_B
alpha   = Quantity(7.2973525693e-3, "")      # fine-structure constant (dimensionless)
sigma_SB = Quantity(5.670374419e-8, "W/(m^2*K^4)")  # Stefan-Boltzmann
F_faraday = Quantity(96485.33212, "C/mol")   # Faraday constant F = e_charge * N_A
# ----------------------------------------------------------------------------------

class Quaternion:
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w + other, self.x, self.y, self.z)
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w - other, self.x, self.y, self.z)
        if isinstance(other, Quaternion):
            return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(other - self.w, -self.x, -self.y, -self.z)
        return NotImplemented

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        if isinstance(other, Quaternion):
            # Hamilton Product
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return (self.w**2 + self.x**2 + self.y**2 + self.z**2)**0.5

    def normalize(self):
        n = self.norm()
        if n == 0: return self
        return self * (1.0 / n)

    def rotate(self, v):
        """Rotates a 3D vector v (list or tensor) using this quaternion."""
        if isinstance(v, (list, tuple, torch.Tensor)):
            v_q = Quaternion(0, v[0], v[1], v[2])
            res_q = self * v_q * self.conjugate()
            return [res_q.x, res_q.y, res_q.z]
        return NotImplemented

    def __repr__(self):
        return f"({self.w} + {self.x}i + {self.y}j + {self.z}k)"

class FourierDense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None
    
    @property
    def shape(self):
        return self.linear.weight.shape

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FourierSequential(nn.Module):
    def __init__(self, layers_data):
        super().__init__()
        self.raw_layers = layers_data
        self.built_layers = nn.ModuleList()
        self.initialized = False

    def __len__(self):
        return len(self.built_layers) if self.initialized else len(self.raw_layers)

    @property
    def shape(self):
        # For a model, we return the internal weights or informative string
        # if not built yet.
        if not self.initialized: return "uninitialized"
        # Just return a summary for now
        return [l.shape if hasattr(l, 'shape') else 'unknown' for l in self.built_layers]

    def _build(self, input_data):
        # Convert to tensor if not already
        input_data = _to_tensor(input_data)
        
        # Ensure 2D (batch, features)
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
            
        current_size = input_data.shape[-1]
        for layer_item in self.raw_layers:
            if callable(layer_item) and not isinstance(layer_item, nn.Module):
                built_layer = layer_item(current_size)
            else:
                built_layer = layer_item
            
            self.built_layers.append(built_layer)
            if hasattr(built_layer, 'linear'):
                current_size = built_layer.linear.out_features
            elif isinstance(built_layer, nn.Linear):
                current_size = built_layer.out_features
            elif hasattr(built_layer, 'out_features'):
                current_size = built_layer.out_features
        self.initialized = True

    def _recursive_to_tensor(self, data):
        if isinstance(data, torch.Tensor): return data
        if isinstance(data, (list, tuple)):
            try:
                converted = [self._recursive_to_tensor(x) for x in data]
                if any(isinstance(x, torch.Tensor) for x in converted):
                    return torch.stack(converted)
            except: pass
        # Use dynamic dtype inference
        try: return torch.as_tensor(data)
        except: return data

    def forward(self, x):
        # Robust tensor conversion
        if not isinstance(x, torch.Tensor):
            x = self._recursive_to_tensor(x)
            
        # Fallback for nested lists that _to_tensor might have missed
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.as_tensor(x, dtype=torch.float32)
            except:
                pass
        
        # Automatic batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Device management
        device = "cpu"
        params = list(self.parameters())
        if params:
            device = params[0].device
        x = x.to(device)

        if not self.initialized:
            self._build(x)
            
        for layer in self.built_layers:
            x = layer(x)
        return x

def Dense(out_features, activation=None, in_features=None):
    if in_features is None:
        return lambda in_feat: FourierDense(in_feat, out_features, activation)
    return FourierDense(in_features, out_features, activation)

def Sequential(layers):
    return FourierSequential(layers)

class FourierCompiledModel:
    """
    Robust wrapper for torch.compile. Builds the model on first forward (if lazy),
    then compiles so Inductor sees a static graph. Falls back to interpreted mode on failure.
    """
    def __init__(self, original_model):
        self.original_model = original_model
        self.failed = False
        self._compiled = None  # compile after first forward (once model is built)

    def __call__(self, *args, **kwargs):
        # First call: ensure lazy Sequential is built, then compile once
        if self._compiled is None and not self.failed:
            if isinstance(self.original_model, FourierSequential) and not self.original_model.initialized:
                self.original_model(*args, **kwargs)  # trigger _build
            try:
                self._compiled = torch.compile(
                    self.original_model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                self.failed = True
                self._compiled = False
        if self._compiled and not self.failed:
            try:
                return self._compiled(*args, **kwargs)
            except Exception as e:
                print(f"Fourier: Compilation failed ({type(e).__name__}); using interpreted mode. Result is correct.")
                self.failed = True
        return self.original_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original_model, name)

def _to_grad(data):
    """Marks a tensor to require gradients."""
    tensor = _to_tensor(data)
    if tensor.is_floating_point():
        tensor.requires_grad = True
    return tensor

def _to_tensor(data):
    """Internal helper to convert nested lists/NumPy to PyTorch tensors.
    Lists that contain non-tensor items (e.g. functions, nn.Modules) are returned unchanged
    so that e.g. Sequential([Dense(...), ...]) receives the list of layers as-is.
    Quantity is converted to tensor of its numeric value (for pi, e, scalars)."""
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, Quantity):
        return torch.tensor(data.value, dtype=torch.float32)
    if isinstance(data, UncertainQuantity):
        return torch.tensor(data.value, dtype=torch.float32)
    if isinstance(data, (list, tuple)):
        if not data:
            return torch.tensor([], dtype=torch.float32)
        try:
            return torch.as_tensor(data)
        except (TypeError, ValueError, RuntimeError):
            pass
        # Recursively convert; if any element is not a tensor (e.g. function, module), return list unchanged
        converted = []
        for x in data:
            try:
                t = _to_tensor(x)
                if isinstance(t, torch.Tensor):
                    converted.append(t)
                else:
                    return data
            except (TypeError, ValueError, RuntimeError):
                return data
        if converted and len(converted) == len(data):
            return torch.stack(converted)
        return data
    try:
        return torch.as_tensor(data, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError):
        return data

def _to_sparse(data):
    """Converts a dense tensor to a sparse representation (COO)."""
    tensor = _to_tensor(data)
    if not tensor.is_sparse:
        return tensor.to_sparse()
    return tensor

def compile_model(model):
    """
    Fourier Native Compilation Hook.
    Returns a robust wrapper that manages the transition to native code.
    """
    if hasattr(torch, 'compile'):
        return FourierCompiledModel(model)
    return model

def random_vector(size):
    return torch.randn(size)

def random_matrix(rows, cols):
    return torch.randn(rows, cols)

# --- Standard Library: Matrix Operations ---

def transpose(data):
    data = _to_tensor(data)
    return data.t()

def inverse(data):
    data = _to_tensor(data)
    return torch.inverse(data)

def dot_product(a, b):
    a = _to_tensor(a)
    b = _to_tensor(b)
    return torch.dot(a.flatten(), b.flatten())

# --- Standard Library: Machine Learning ---

def relu(data):
    data = _to_tensor(data)
    return torch.relu(data)

def softmax(data, dim=-1):
    data = _to_tensor(data)
    return torch.softmax(data, dim=dim)

def convolution(input, kernel, padding=0, stride=1):
    input = _to_tensor(input)
    kernel = _to_tensor(kernel)
    # Basic 2D convolution assumption for now
    if input.dim() == 2: input = input.unsqueeze(0).unsqueeze(0)
    if kernel.dim() == 2: kernel = kernel.unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.conv2d(input, kernel, padding=padding, stride=stride)

def pooling(input, kernel_size=2):
    input = _to_tensor(input)
    if input.dim() == 2: input = input.unsqueeze(0).unsqueeze(0)
    return torch.nn.functional.max_pool2d(input, kernel_size=kernel_size)

# --- Standard Library: Signal Processing ---

def _to_complex_tensor(data):
    """Convert list of Quaternions (w+xi) or numbers to 1D complex tensor for FFT."""
    if isinstance(data, torch.Tensor):
        return data if data.is_complex() else data.to(torch.complex64)
    if isinstance(data, (list, tuple)) and data:
        first = data[0]
        if isinstance(first, Quaternion):
            return torch.tensor([complex(q.w, q.x) for q in data], dtype=torch.complex64)
        if isinstance(first, (int, float)):
            return torch.tensor(data, dtype=torch.complex64)
    data = _to_tensor(data)
    return data if data.is_complex() else data.to(torch.complex64)

def fft(data):
    data = _to_complex_tensor(data)
    return torch.fft.fft(data)

def ifft(data):
    data = _to_complex_tensor(data)
    return torch.fft.ifft(data)

# --- Standard Library: Differentiable ODE Solvers ---

def linspace(start, stop, steps):
    """Erzeugt einen 1D-Tensor mit `steps` äquidistanten Werten von start bis stop."""
    s = _to_tensor(start).float().squeeze()
    e = _to_tensor(stop).float().squeeze()
    n = int(steps)
    return torch.linspace(float(s.item()) if s.numel() == 1 else float(s.item()),
                          float(e.item()) if e.numel() == 1 else float(e.item()),
                          n)

def ode_solve(fun, y0, t, method="rk4"):
    """
    Differenzierbarer ODE-Löser: dy/dt = fun(t, y).
    fun(t, y) muss dy/dt zurückgeben (t Skalar, y Tensor); y0 Anfangsbedingung, t 1D-Zeitgitter.
    Rückgabe: Tensor der Form (len(t), *y0.shape) mit Lösung y(t).
    Gradients fließen durch y0 und durch in fun verwendete Parameter (z.B. für Physics-Informed ML).
    """
    y0 = _to_tensor(y0).float()
    t = _to_tensor(t).float().flatten()
    if t.dim() != 1 or t.numel() < 2:
        raise ValueError("ode_solve: t muss ein 1D-Vektor mit mindestens 2 Zeitpunkten sein.")
    t = t.to(y0.device)
    out = [y0]
    y = y0.clone()
    for i in range(t.numel() - 1):
        t_cur = t[i]
        h = t[i + 1] - t[i]
        if method == "euler":
            dy = fun(t_cur, y)
            y = y + h * dy
        else:
            # RK4 (default)
            k1 = fun(t_cur, y)
            k2 = fun(t_cur + h * 0.5, y + h * 0.5 * k1)
            k3 = fun(t_cur + h * 0.5, y + h * 0.5 * k2)
            k4 = fun(t_cur + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y)
    return torch.stack(out, dim=0)

# --- Standard Library: Differentiable PDE Solvers ---
# 1D/2D Heat equation u_t = k * Laplacian(u); Finite-Differenzen + ode_solve (differenzierbar in u0, k).

def pde_heat_1d(u0, x, t, k, bc="dirichlet"):
    """
    Differenzierbarer 1D-Heat-Solver: u_t = k * u_xx.
    u0: Anfangsbedingung (1D, Länge = len(x)); x: Ortsgitter (1D); t: Zeitgitter (1D); k: Diffusivität.
    bc: 'dirichlet' = Randwerte u[0], u[-1] aus u0 fest. Rückgabe: (len(t), len(x)).
    """
    u0 = _to_tensor(u0).float().flatten()
    x = _to_tensor(x).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    k = _to_tensor(k).float()
    n = u0.numel()
    if x.numel() != n:
        raise ValueError("pde_heat_1d: len(u0) muss len(x) entsprechen.")
    if n < 3:
        raise ValueError("pde_heat_1d: mindestens 3 Gitterpunkte.")
    dx = (x[-1] - x[0]) / (n - 1.0)
    dx2 = dx * dx

    def rhs(t_cur, u):
        u = u.flatten()
        lap = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx2
        dudt = torch.zeros_like(u)
        dudt[1:-1] = k * lap if k.dim() == 0 or k.numel() == 1 else k.expand_as(lap).clone() * lap
        return dudt

    return ode_solve(rhs, u0, t)

def pde_heat_2d(u0, x, y, t, k, bc="dirichlet"):
    """
    Differenzierbarer 2D-Heat-Solver: u_t = k * (u_xx + u_yy).
    u0: Anfangsbedingung 2D (shape nx*ny); x, y: 1D-Gitter; t: Zeitgitter; k: Diffusivität.
    bc: 'dirichlet' = Rand aus u0 fest. Rückgabe: (len(t), nx, ny).
    """
    u0 = _to_tensor(u0).float()
    if u0.dim() == 1:
        raise ValueError("pde_heat_2d: u0 muss 2D-Gitter (nx, ny) sein.")
    nx, ny = u0.shape[0], u0.shape[1]
    x = _to_tensor(x).float().flatten().to(u0.device)
    y = _to_tensor(y).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    k = _to_tensor(k).float()
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_heat_2d: x/y Länge muss zu u0 passen.")
    dx = float((x[-1] - x[0]).item()) / max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / max(ny - 1, 1)
    dx2, dy2 = dx * dx, dy * dy

    def rhs(t_cur, u_flat):
        u = u_flat.reshape(nx, ny)
        lap = torch.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx2
            + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy2
        )
        dudt = k * lap
        return dudt.flatten()

    return ode_solve(rhs, u0.flatten(), t).reshape(t.numel(), nx, ny)

# --- Standard Library: Probabilistic Programming ---
# Verteilungen und Bayesian Inference (torch.distributions)

def _to_loc_scale(mu, sigma):
    """Scalar or tensor -> loc, scale for distributions."""
    loc = _to_tensor(mu).float()
    scale = _to_tensor(sigma).float()
    return loc, scale

def Normal(mu, sigma):
    """Normal(mean, std). Returns a distribution; use sample(d) or d.sample()."""
    loc, scale = _to_loc_scale(mu, sigma)
    return torch.distributions.Normal(loc=loc, scale=scale.clamp(min=1e-6))

def Uniform(low, high):
    """Uniform(low, high). Returns a distribution."""
    low = _to_tensor(low).float()
    high = _to_tensor(high).float()
    return torch.distributions.Uniform(low=low, high=high)

def Bernoulli(p):
    """Bernoulli(prob). Returns a distribution."""
    p = _to_tensor(p).float().clamp(0.0, 1.0)
    return torch.distributions.Bernoulli(probs=p)

def Exponential(rate):
    """Exponential(rate). Returns a distribution; use sample(d) or sample(d, n)."""
    rate = _to_tensor(rate).float().clamp(min=1e-6)
    return torch.distributions.Exponential(rate=rate)

def Gamma(concentration, rate):
    """Gamma(concentration, rate). Returns a distribution."""
    concentration = _to_tensor(concentration).float().clamp(min=1e-6)
    rate = _to_tensor(rate).float().clamp(min=1e-6)
    return torch.distributions.Gamma(concentration=concentration, rate=rate)

def Beta(alpha, beta):
    """Beta(alpha, beta). Returns a distribution."""
    alpha = _to_tensor(alpha).float().clamp(min=1e-6)
    beta = _to_tensor(beta).float().clamp(min=1e-6)
    return torch.distributions.Beta(concentration1=alpha, concentration0=beta)

def Poisson(rate):
    """Poisson(rate). Returns a distribution; samples are integer-valued."""
    rate = _to_tensor(rate).float().clamp(min=1e-6)
    return torch.distributions.Poisson(rate=rate)

def sample(dist, sample_shape=None):
    """Draw sample(s) from distribution. sample_shape: e.g. [] for one sample, [n] for n samples."""
    if sample_shape is None:
        sample_shape = []
    if isinstance(sample_shape, (int, float)):
        sample_shape = [int(sample_shape)]
    s = dist.sample(sample_shape if sample_shape else ())
    return _to_tensor(s) if not isinstance(s, torch.Tensor) else s

def log_prob(dist, value):
    """Log-probability of value under distribution (for Bayesian inference)."""
    value = _to_tensor(value)
    return dist.log_prob(value).sum()

def _log_scalar(x):
    """Convert log-probability result to float (for MCMC)."""
    t = _to_tensor(x).detach()
    return float(t.sum().item()) if t.numel() > 0 else float(t.item() if t.dim() == 0 else 0.0)

def metropolis(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size=0.1):
    """
    Simple Metropolis-Hastings MCMC. log_prior_fn(theta), log_likelihood_fn(data, theta) return log-probs (tensor or scalar).
    init_theta: initial parameter (tensor or list); proposal is Normal(current, step_size).
    Returns tensor of shape (num_steps, *theta_shape) with posterior samples.
    """
    theta = _to_tensor(init_theta).float().clone().detach()
    samples = [theta.clone()]
    data = _to_tensor(data)
    for _ in range(num_steps - 1):
        proposal = theta + step_size * torch.randn_like(theta)
        try:
            lp_cur = _log_scalar(log_prior_fn(theta))
            lp_prop = _log_scalar(log_prior_fn(proposal))
            ll_cur = _log_scalar(log_likelihood_fn(data, theta))
            ll_prop = _log_scalar(log_likelihood_fn(data, proposal))
        except Exception:
            lp_cur = lp_prop = ll_cur = ll_prop = float("-inf")
        log_accept = (lp_prop + ll_prop) - (lp_cur + ll_cur)
        accept_prob = 1.0 if log_accept >= 0 else min(1.0, torch.exp(torch.tensor(log_accept)).item())
        if torch.rand(1).item() < accept_prob:
            theta = proposal.clone()
        samples.append(theta.clone())
    return torch.stack(samples, dim=0)

# --- Standard Library: Math (for integration and expressions) ---
def sin(x):
    """sin(x); x Tensor or scalar."""
    return torch.sin(_to_tensor(x).float())
def cos(x):
    """cos(x); x Tensor or scalar."""
    return torch.cos(_to_tensor(x).float())

# --- Standard Library: Numerical Integration ---
# Differenzierbar, wenn f Tensor-Argument akzeptiert und differenzierbar ist.

def integrate(f, a, b, n=100):
    """
    Numerische Integration: int_a^b f(x) dx mit Trapezregel.
    f: Callable mit einem Argument (Tensor oder Skalar); soll Tensor zurückgeben für Gradienten.
    a, b: Integrationsgrenzen (Skalare); n: Anzahl Stützstellen (default 100).
    Rückgabe: Skalar-Tensor; differenzierbar bzgl. in f verwendeter Parameter.
    """
    a_val = float(_to_tensor(a).float().squeeze().item())
    b_val = float(_to_tensor(b).float().squeeze().item())
    n_int = max(2, int(n))
    x = torch.linspace(a_val, b_val, n_int)
    y = f(x)
    y = _to_tensor(y).float().flatten()
    if y.numel() != n_int:
        raise ValueError("integrate: f(x) muss gleiche Länge wie x haben.")
    dx = (b_val - a_val) / (n_int - 1.0)
    result = (dx / 2.0) * (y[0] + 2.0 * y[1:-1].sum() + y[-1])
    return result.squeeze()

# --- Standard Library: Uncertainty Propagation (Gaussian) ---
# Fehlerfortpflanzung für Wissenschaftler: value ± std; Gauß'sche Näherung für +, -, *, /, ^.

class UncertainQuantity:
    """Größe mit Unsicherheit: value ± std. Gauß'sche Fehlerfortpflanzung für +, -, *, /, ^."""
    def __init__(self, value, std=0.0, unit=""):
        self.value = float(value)
        self.std = max(0.0, float(std))
        self.unit = str(unit) if unit else ""

    def _same_unit(self, other):
        if not isinstance(other, UncertainQuantity):
            return False
        return self.unit == other.unit

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError("UncertainQuantity: Kann Zahl nicht zu Größe mit Einheit addieren.")
            return UncertainQuantity(self.value + other, self.std, "")
        if isinstance(other, UncertainQuantity):
            if not self._same_unit(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}]")
            v = self.value + other.value
            s = (self.std ** 2 + other.std ** 2) ** 0.5
            return UncertainQuantity(v, s, self.unit)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError("UncertainQuantity: Kann Zahl nicht subtrahieren.")
            return UncertainQuantity(self.value - other, self.std, "")
        if isinstance(other, UncertainQuantity):
            if not self._same_unit(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}]")
            v = self.value - other.value
            s = (self.std ** 2 + other.std ** 2) ** 0.5
            return UncertainQuantity(v, s, self.unit)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)) and not self.unit:
            return UncertainQuantity(other - self.value, self.std, "")
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return UncertainQuantity(self.value * other, abs(other) * self.std, self.unit)
        if isinstance(other, UncertainQuantity):
            v = self.value * other.value
            if abs(v) < 1e-300:
                s = 0.0
            else:
                r1 = (self.std / self.value) ** 2 if self.value != 0 else 0.0
                r2 = (other.std / other.value) ** 2 if other.value != 0 else 0.0
                s = abs(v) * (r1 + r2) ** 0.5
            u = _unit_mul(self.unit, other.unit)
            return UncertainQuantity(v, s, u)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return UncertainQuantity(other * self.value, abs(other) * self.std, self.unit)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("UncertainQuantity: Division durch null.")
            return UncertainQuantity(self.value / other, self.std / abs(other), self.unit)
        if isinstance(other, UncertainQuantity):
            v = self.value / other.value
            if other.value == 0:
                raise ValueError("UncertainQuantity: Division durch null.")
            r1 = (self.std / self.value) ** 2 if self.value != 0 else 0.0
            r2 = (other.std / other.value) ** 2
            s = abs(v) * (r1 + r2) ** 0.5
            u = _unit_div(self.unit, other.unit)
            return UncertainQuantity(v, s, u)
        return NotImplemented

    def __pow__(self, exp):
        if isinstance(exp, (int, float)):
            v = self.value ** exp
            if self.value == 0 and exp != 0:
                s = 0.0
            else:
                s = abs(v) * abs(exp) * (self.std / abs(self.value)) if self.value != 0 else 0.0
            u = _unit_pow(self.unit, exp)
            return UncertainQuantity(v, s, u)
        return NotImplemented

    def __repr__(self):
        u = f" [{self.unit}]" if self.unit else ""
        return f"{self.value} ± {self.std}{u}"

def uncertain(value, std, unit=""):
    """Größe mit Unsicherheit: value ± std. Nutze für Fehlerfortpflanzung (Gauß'sche Näherung)."""
    return UncertainQuantity(value, std, unit)

# --- Standard Library: Fitting / Regression ---
# Minimiert loss_fn(params, data) via Gradient Descent (oder MCMC).

def fit(loss_fn, params_init, data, method="gd", lr=0.01, steps=500):
    """
    Kurvenanpassung: minimiert loss_fn(params, data).
    params_init: Startparameter (Tensor oder Liste); werden kopiert und mit Gradienten versehen.
    data: beliebige Daten (an loss_fn übergeben).
    method: 'gd' (Gradient Descent, default) oder 'mcmc' (Metropolis-Hastings).
    lr: Lernrate (nur bei gd). steps: Anzahl Schritte.
    Rückgabe: Tensor der optimalen Parameter (bei gd); bei mcmc Tensor (steps, *params_shape).
    """
    params = _to_tensor(params_init).float().clone().detach()
    params = params.requires_grad_(True)
    data_t = _to_tensor(data)

    if method == "mcmc":
        def log_prior(p):
            return torch.tensor(0.0)
        def log_likelihood(d, p):
            return -loss_fn(p, d)
        return metropolis(log_prior, log_likelihood, data_t, params.detach(), steps, step_size=lr)

    # Gradient Descent
    for _ in range(steps):
        loss = loss_fn(params, data_t)
        if params.grad is not None:
            params.grad.zero_()
        loss.backward()
        with torch.no_grad():
            params.sub_(lr * params.grad)
    return params.detach()

# --- Standard Library: Sorting ---

def sort(data, descending=False):
    data = _to_tensor(data)
    sorted_values, _ = torch.sort(data, descending=descending)
    return sorted_values

def quicksort(data):
    return sort(data)

# --- Standard Library: Visualization ---
# Plots werden in _fourier_plots gesammelt und vom Server an die IDE zurückgegeben.

_fourier_plots = []

def _plot_ndarray(x, y=None, title=None, xlabel=None, ylabel=None):
    """Intern: Erzeugt einen Plot und hängt ihn als Base64-PNG an _fourier_plots."""
    try:
        import base64
        import io
        import matplotlib  # type: ignore[import-untyped]
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    except ImportError:
        print("plot(): matplotlib nicht installiert. pip install matplotlib")
        return
    if y is None:
        y = x
        x = list(range(len(y)))
    if x is None:
        x = list(range(len(y) if hasattr(y, '__len__') else 0))
    if hasattr(x, 'cpu'): x = x.detach().cpu().numpy()
    elif not isinstance(x, (list, tuple)): x = list(x)
    if hasattr(y, 'cpu'): y = y.detach().cpu().numpy()
    elif not isinstance(y, (list, tuple)): y = list(y)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    _fourier_plots.append(base64.b64encode(buf.getvalue()).decode('ascii'))

def plot(x=None, y=None, title=None, xlabel=None, ylabel=None):
    """
    Zeichnet Daten und zeigt sie in Fourier Studio an.
    plot(y)           – y über Index
    plot(x, y)        – y über x
    plot(y, title="…") – mit Titel
    """
    if x is None and y is None:
        return
    if y is None:
        y = _to_tensor(x) if isinstance(x, (list, tuple)) else x
        x = None
    elif not isinstance(x, (list, tuple)) and not hasattr(x, 'shape'):
        x = _to_tensor(x) if hasattr(x, '__iter__') else x
    if not isinstance(y, (list, tuple)) and not hasattr(y, 'shape'):
        y = _to_tensor(y) if hasattr(y, '__iter__') else y
    _plot_ndarray(x, y, title=title, xlabel=xlabel, ylabel=ylabel)
