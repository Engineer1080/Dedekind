    factor = tab[u_from] / tab[u_to]
    return float(value) * factor, float(std) * abs(factor)


class Quantity:
    """Physikalische Größe mit Einheit (z. B. 10[m], 5[m/s], 0.1[M], 50[ppm]). Rechenregeln: gleiche Einheit für +/-, Einheiten multiplizieren/dividieren. Chemie: mol, L, M (= mol/L), ppm, bar, atm, g. Radioaktivität: Bq (1/s), Gy (J/kg), Sv (J/kg, Äquivalentdosis)."""
    def __init__(self, value, unit=""):
        self.value = float(value)
        self.unit = str(unit) if unit else ""

    def _same_unit(self, other):
        if not isinstance(other, Quantity):
            return False
        return _normalize_unit_for_compare(self.unit) == _normalize_unit_for_compare(other.unit)

    def _add_sub_quantity(self, other, is_add):
        """Addition/Subtraktion mit automatischer Umrechnung bei gleicher Dimension (Länge, Masse, Zeit, Druck)."""
        dim_self = _get_dimension(self.unit)
        dim_other = _get_dimension(other.unit)
        if dim_self is not None and dim_self == dim_other:
            v_self_base = _convert_to_base(self.value, self.unit, dim_self)
            v_other_base = _convert_to_base(other.value, other.unit, dim_other)
            result_base = (v_self_base + v_other_base) if is_add else (v_self_base - v_other_base)
            result_value = _convert_from_base(result_base, self.unit, dim_self)
            return Quantity(result_value, self.unit)
        if self._same_unit(other):
            v = (self.value + other.value) if is_add else (self.value - other.value)
            return Quantity(v, self.unit)
        raise ValueError(
            f"Einheitenfehler bei {'Addition' if is_add else 'Subtraktion'}: [{self.unit}] vs [{other.unit}]. "
            "Gleiche Einheit oder kompatible Einheiten derselben Dimension (z. B. Länge, Masse, Zeit, Druck, Strom, Temperatur, mol, cd, Volumen, Energie, Spannung, Frequenz, Ladung, Widerstand, Leistung, Winkel rad/deg) erforderlich."
        )

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError(
                    f"Einheitenfehler: Kann reine Zahl nicht zu Größe mit Einheit [{self.unit}] addieren. "
                    "Für Addition brauchen beide Seiten die gleiche Einheit (oder dimensionslos)."
                )
            return Quantity(self.value + other, "")
        if isinstance(other, Quantity):
            return self._add_sub_quantity(other, is_add=True)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError(
                    f"Einheitenfehler: Kann reine Zahl nicht von Größe mit Einheit [{self.unit}] subtrahieren. "
                    "Für Subtraktion brauchen beide Seiten die gleiche Einheit (oder dimensionslos)."
                )
            return Quantity(self.value - other, "")
        if isinstance(other, Quantity):
            return self._add_sub_quantity(other, is_add=False)
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
        if isinstance(other, torch.Tensor):
            if self.unit:
                return NotImplemented
            return other * self.value
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(other * self.value, self.unit)
        if isinstance(other, torch.Tensor):
            if self.unit:
                return NotImplemented
            return other * self.value
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


def _split_product_top_level(u):
    """Splittet Einheiten-String an '*' nur auf oberster Ebene (Klammern respektierend)."""
    parts = []
    current = []
    depth = 0
    for c in u:
        if c == "(":
            depth += 1
            current.append(c)
        elif c == ")":
            depth -= 1
            current.append(c)
        elif c == "*" and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(c)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_base_exp(part):
    """Liefert (base, exp) für einen Faktor wie 'm', 'm^2', '(m/s)^2'. exp als Zahl."""
    part = part.strip()
    if "^" in part:
        # Rechts vom letzten ^ (nicht in Klammern) steht der Exponent
        idx = part.rfind("^")
        base = part[:idx].strip()
        exp_str = part[idx + 1:].strip()
        try:
            exp = int(exp_str) if "." not in exp_str else float(exp_str)
        except ValueError:
            return part, 1
        # Klammern um einfache Basis entfernen für einheitliche Darstellung
        if base.startswith("(") and base.endswith(")") and base.count("(") == 1 and "/" not in base[1:-1]:
            base = base[1:-1].strip()
        return base, exp
    return part, 1


def _collapse_product(u):
    """Fasst gleiche Faktoren zusammen: m*m -> m^2, m*m*m -> m^3, m^2*m -> m^3, m*m*kg -> m^2*kg."""
    if not u or "/" in u:
        return u
    parts = _split_product_top_level(u)
    if len(parts) <= 1:
        return u
    # Teile, die selbst Produkte sind (z. B. (m*m)), zuerst rekursiv zusammenfassen
    normalized = []
    for p in parts:
        p = p.strip()
        # Äußere Klammern abziehen, dann ggf. Produkt zusammenfassen
        if p.startswith("(") and p.endswith(")") and p.count("(") == 1:
            p = p[1:-1].strip()
        if "*" in p and "/" not in p:
            p = _collapse_product(p)
        normalized.append(p)
    # Jeden Faktor als (base, exp) und zusammenfassen
    merged = {}
    for p in normalized:
        base, exp = _parse_base_exp(p)
        merged[base] = merged.get(base, 0) + exp
    # Sortiert ausgeben: einheitliche Reihenfolge
    out = []
    for base in sorted(merged.keys()):
        e = merged[base]
        if e == 1:
            out.append(base)
        else:
            out.append(f"{base}^{int(e)}" if isinstance(e, float) and e == int(e) else f"{base}^{e}")
    return "*".join(out)


def _unit_simplify(u):
    """Vereinfacht Einheiten-String für Anzeige. SI-Basis: m, kg, s, A, K, mol, cd. Abgeleitet: J, N, Pa, W, Hz, V, F, ohm, S, Wb, T, H, lm, lx, Gy, kat, M. Chemie/Druck: bar, atm. Masse: g. Radioaktivität: Bq (1/s), Sv (J/kg); Anzeige 1/s→Hz, J/kg→Gy."""
    if not u:
        return u
    u = u.strip()
    # Chemie: Konzentration mol/L -> M; ppm bleibt ppm
    if u in ("mol/L", "mol*L^-1", "mol*L^(-1)", "mol/L"):
        return "M"
    if u.replace(" ", "") in ("mol/L", "mol*L^-1"):
        return "M"
    # 1/s -> Hz (Frequenz); Bq gleiche Dimension, Anzeige Hz
    if u in ("1/s", "s^-1", "s^(-1)") or u.replace(" ", "") == "s^-1":
        return "Hz"
    # mol/s -> kat (katal, katalytische Aktivität)
    if u in ("mol/s", "mol*s^-1", "mol*s^(-1)"):
        return "kat"
    # J/kg, m^2/s^2 -> Gy (Gray, absorbierte Dosis); Sv gleiche Dimension
    if u in ("J/kg", "m^2/s^2", "(m^2)/(s^2)") or "(m/s)^2" in u:
        return "Gy"
    # Joule: kg*m^2/s^2 bzw. (kg)*((m/s)^2); nicht J/C (V) oder J/s (W)
    if "(kg)*((m/s)^2)" in u or u == "kg*m^2/s^2":
        return "J"
    if "kg" in u and "m^2" in u and "s^2" in u and "A" not in u and "mol" not in u and "s^3" not in u and "s^(-3)" not in u and "/" in u:
        return "J"
    if "(J*s)*(Hz)" in u or "J*s*Hz" in u or ("J*s" in u and "Hz" in u):
        return "J"
    # Newton: kg*m/s^2
    if "m^3/(kg*s^2)" in u and "m^2" in u:
        return "N"
    if "N*m^2/C^2" in u and "m^2" in u:
        return "N"
    if "kg*m/s^2" in u or u == "N":
        return "N"
    # Pa (Pascal): N/m^2, kg/(m*s^2) — nicht N*m^2/C^2
    if u in ("N/m^2", "N*m^-2", "N*m^(-2)"):
        return "Pa"
    if "kg/(m*s^2)" in u or u in ("kg*m^-1*s^-2", "kg*m^(-1)*s^(-2)"):
        return "Pa"
    # W (Watt): J/s, kg*m^2/s^3
    if u in ("J/s", "J*s^-1", "J*s^(-1)") or "kg*m^2/s^3" in u or u == "kg*m^2/s^3":
        return "W"
    # V (Volt): J/C, W/A, kg*m^2/(s^3*A)
    if u in ("J/C", "W/A") or "kg*m^2/(s^3*A)" in u:
        return "V"
    # S (Siemens): 1/ohm, s^3*A^2/(kg*m^2) — vor ohm prüfen
    if "s^3*A^2/(kg*m^2)" in u:
        return "S"
    # ohm (Ω): V/A, kg*m^2/(s^3*A^2)
    if u in ("V/A", "V*A^-1", "V*A^(-1)") or "kg*m^2/(s^3*A^2)" in u:
        return "ohm"
    # F (Farad): C/V, s^4*A^2/(kg*m^2)
    if u in ("C/V", "C*V^-1", "C*V^(-1)") or ("s^4*A^2" in u and "kg*m^2" in u):
        return "F"
    # H (Henry): Wb/A, kg*m^2/(s^2*A^2) — vor Wb prüfen
    if u in ("Wb/A", "Wb*A^-1") or "kg*m^2/(s^2*A^2)" in u:
        return "H"
    # Wb (Weber): V*s, kg*m^2/(s^2*A) (nicht A^2)
    if u == "V*s" or ("kg*m^2" in u and "/(s^2*A)" in u and "A^2" not in u):
        return "Wb"
    # T (Tesla): Wb/m^2, kg/(s^2*A)
    if u in ("Wb/m^2", "Wb*m^-2", "kg*s^-2*A^-1") or "kg/(s^2*A)" in u:
        return "T"
    # lm (Lumen): cd*sr (Candela * Steradiant)
    if "cd*sr" in u or (u.replace(" ", "") == "cd*sr"):
        return "lm"
    # lx (Lux): lm/m^2
    if u in ("lm/m^2", "lm*m^-2"):
        return "lx"
    # Gleiche Faktoren zusammenfassen: m*m -> m^2, m*m*m -> m^3, m^2*m -> m^3, m*m*kg -> m^2*kg
    if "*" in u:
        u = _collapse_product(u)
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

class DedekindDense(nn.Module):
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

class DedekindSequential(nn.Module):
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
        return lambda in_feat: DedekindDense(in_feat, out_features, activation)
    return DedekindDense(in_features, out_features, activation)

def Sequential(layers):
    return DedekindSequential(layers)

class DedekindCompiledModel:
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
            if isinstance(self.original_model, DedekindSequential) and not self.original_model.initialized:
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
                print(f"Dedekind: Compilation failed ({type(e).__name__}); using interpreted mode. Result is correct.")
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
    Dedekind Native Compilation Hook.
    Returns a robust wrapper that manages the transition to native code.
    """
    if hasattr(torch, 'compile'):
        return DedekindCompiledModel(model)
    return model

