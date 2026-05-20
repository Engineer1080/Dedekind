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

    def _cmp_value(self, other):
        """Liefert (self_val, other_val) für Vergleich. Konvertiert Einheiten, wenn möglich."""
        if isinstance(other, Quantity):
            dim_s = _get_dimension(self.unit)
            dim_o = _get_dimension(other.unit)
            if dim_s is not None and dim_s == dim_o:
                return _convert_to_base(self.value, self.unit, dim_s), \
                       _convert_to_base(other.value, other.unit, dim_o)
            if _normalize_unit_for_compare(self.unit) == _normalize_unit_for_compare(other.unit):
                return self.value, other.value
            raise ValueError(
                f"Vergleich nicht möglich: [{self.unit}] vs [{other.unit}] (unterschiedliche Dimension)."
            )
        if isinstance(other, (int, float)):
            return self.value, float(other)
        return self.value, other

    def __lt__(self, other):
        a, b = self._cmp_value(other)
        return a < b

    def __le__(self, other):
        a, b = self._cmp_value(other)
        return a <= b

    def __gt__(self, other):
        a, b = self._cmp_value(other)
        return a > b

    def __ge__(self, other):
        a, b = self._cmp_value(other)
        return a >= b

    def __eq__(self, other):
        if isinstance(other, Quantity):
            try:
                a, b = self._cmp_value(other)
                return a == b
            except ValueError:
                return False
        if isinstance(other, (int, float)):
            return self.value == float(other)
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __hash__(self):
        return hash((self.value, _normalize_unit_for_compare(self.unit)))


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

def _has_tensor_like(x):
    if isinstance(x, (torch.Tensor, Quantity, UncertainQuantity)):
        return True
    if isinstance(x, (list, tuple)):
        return any(_has_tensor_like(item) for item in x)
    return False

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
        if not _has_tensor_like(data):
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

def _to_gpu(data):
    data = _to_tensor(data)
    if torch.cuda.is_available():
        return data.to('cuda')
    return data

def _to_cpu(data):
    data = _to_tensor(data)
    return data.to('cpu')


def compile_model(model):
    """
    Dedekind Native Compilation Hook.
    Returns a robust wrapper that manages the transition to native code.
    """
    if hasattr(torch, 'compile'):
        return DedekindCompiledModel(model)
    return model



def _register_user_unit(name, factor, base_unit):
    """Registriert eine benutzerdefinierte Einheit: 1 NAME = factor * base_unit.

    Beispiele:
        _register_user_unit("Foot", 0.3048, "m")     # 1 Foot = 0.3048 m  (length)
        _register_user_unit("Mile", 1609.34, "m")    # 1 Mile = 1609.34 m  (length)
        _register_user_unit("Darcy", 9.869233e-13, "m^2")  # permeability
        _register_user_unit("MilliFoot", 0.001, "Foot")    # chaining wird aufgel├Âst

    base_unit muss bereits in irgendeiner Dimensionstabelle vorkommen (Basis oder Alias).
    """
    name = str(name).strip()
    base_unit = str(base_unit).strip()
    factor = float(factor)
    if not name:
        raise ValueError("unit-Name darf nicht leer sein.")
    # Dimension der base_unit suchen und Kettenaufl├Âsung durchf├╝hren
    target_dim = None
    base_factor = 1.0
    for dim, (_b, tab) in DIMENSION_TO_BASE.items():
        if base_unit in tab:
            target_dim = dim
            base_factor = tab[base_unit]
            break
    if target_dim is None:
        raise ValueError(
            f"`unit {name} = ...[{base_unit}]`: Basiseinheit '{base_unit}' ist keiner bekannten Dimension "
            f"zugeordnet (L├ñnge, Masse, Zeit, Druck, Energie, ...). Bitte alias zu einer bestehenden Einheit."
        )
    DIMENSION_TO_BASE[target_dim][1][name] = factor * base_factor
    _rebuild_additive_unit_sets()
    return name

def _unit_of_value(value):
    """Liefert die Einheit eines Wertes als String. '' fuer unitless / Zahlen / Tensoren."""
    if isinstance(value, Quantity):
        return value.unit or ""
    if isinstance(value, UncertainQuantity):
        return value.unit or ""
    return ""

def _check_param_unit(value, param_name, fn_name, arg_name, unit_env):
    """Bindet eine polymorphe Einheits-Variable konsistent ueber alle Argumente eines Calls.

    Erstes Auftreten: bindet param_name -> Einheit von value.
    Folgende Auftritte: muss die gleiche Dimension haben; bei Bedarf wird value
    in die gebundene Einheit umgerechnet.

    Beispiel:
        fn add<U>(a: [U], b: [U]) -> [U] { return a + b }
        add(2[m], 3[m])       # U bindet auf 'm'
        add(2[m], 100[cm])    # U bindet auf 'm', 100[cm] wird zu 1[m]
        add(2[m], 3[kg])      # ValueError (Dimensions-Mismatch)
    """
    actual = _unit_of_value(value)
    if param_name not in unit_env:
        unit_env[param_name] = actual
        return value
    bound = unit_env[param_name]
    if bound == actual:
        return value
    # Beide non-empty + gleiche Einheit (durch String-Vergleich)?
    if _normalize_unit_for_compare(bound) == _normalize_unit_for_compare(actual):
        return value
    # Beide non-empty + gleiche Dimension? -> in gebundene Einheit umrechnen.
    if isinstance(value, Quantity) and bound:
        try:
            return _coerce_to_expected_unit(value, bound, f"{fn_name}({arg_name}) [Typ-Param {param_name}]")
        except ValueError:
            pass
    # Caller gab plain number, Bindung erwartet Einheit
    if bound and not actual and isinstance(value, (int, float)):
        return Quantity(float(value), bound)
    raise ValueError(
        f"Typ-Param-Einheit '{param_name}' in {fn_name}({arg_name}): "
        f"bereits an [{bound or 'unitless'}] gebunden, hier [{actual or 'unitless'}]."
    )

def _check_return_param_unit(value, param_name, fn_name, unit_env):
    """Pruefe / binde polymorphe Einheits-Variable am Return-Punkt."""
    actual = _unit_of_value(value)
    if param_name not in unit_env:
        # Return ist die erste Stelle, die param_name sieht ÔÇö binden, ohne Pruefung
        unit_env[param_name] = actual
        return value
    bound = unit_env[param_name]
    if bound == actual:
        return value
    if _normalize_unit_for_compare(bound) == _normalize_unit_for_compare(actual):
        return value
    if isinstance(value, Quantity) and bound:
        try:
            return _coerce_to_expected_unit(value, bound, f"return von {fn_name} [Typ-Param {param_name}]")
        except ValueError:
            pass
    if bound and not actual and isinstance(value, (int, float)):
        return Quantity(float(value), bound)
    raise ValueError(
        f"Typ-Param-Einheit '{param_name}' im Return von {fn_name}: "
        f"erwartet [{bound or 'unitless'}], erhalten [{actual or 'unitless'}]."
    )

def _shape_of(value):
    """Liefert die Shape eines Wertes als Tupel von ints. Skalar -> (), unbekannt -> None."""
    if isinstance(value, (int, float, bool)):
        return ()
    if isinstance(value, Quantity):
        return ()
    if hasattr(value, "shape"):
        try:
            sh = tuple(int(d) for d in value.shape)
            return sh
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return (0,)
        first_sub = _shape_of(value[0])
        if first_sub is None:
            return (len(value),)
        # Konsistenz aller Elemente (sonst unregelm├ñ├ƒig -> nur erste Dimension)
        for item in value[1:]:
            if _shape_of(item) != first_sub:
                return (len(value),)
        return (len(value),) + first_sub
    return None

def _format_shape(dims):
    """Liefert eine kompakte String-Darstellung einer Shape-Liste/Tupel."""
    parts = [str(d) for d in dims]
    return "[" + ", ".join(parts) + "]"

def _check_shape(value, expected_dims, fn_name, arg_name, shape_env):
    """Prueft, dass `value` die deklarierte Shape `expected_dims` hat.
    Symbolische Dimensionen (Strings) werden in `shape_env` gebunden bzw. verglichen.
    Wirft ValueError bei Mismatch. Liefert `value` unveraendert zurueck (passthrough)."""
    actual = _shape_of(value)
    if actual is None:
        # Shape nicht ermittelbar (z. B. generischer Iterator) - skip
        return value
    if len(actual) != len(expected_dims):
        raise ValueError(
            f"Shape-Mismatch in {fn_name}({arg_name}): erwartet {_format_shape(expected_dims)} "
            f"({len(expected_dims)}-D), erhalten {_format_shape(actual)} ({len(actual)}-D)."
        )
    for i, (want, got) in enumerate(zip(expected_dims, actual)):
        if isinstance(want, int):
            if want != got:
                raise ValueError(
                    f"Shape-Mismatch in {fn_name}({arg_name}): Dimension {i} erwartet {want}, "
                    f"erhalten {got}. Volle Shape: erwartet {_format_shape(expected_dims)}, "
                    f"erhalten {_format_shape(actual)}."
                )
        else:
            # Symbolische Dimension: erste Begegnung bindet, danach Konsistenz pruefen
            bound = shape_env.get(want)
            if bound is None:
                shape_env[want] = got
            elif bound != got:
                raise ValueError(
                    f"Symbolische Shape-Dimension '{want}' in {fn_name}({arg_name}): bereits "
                    f"als {bound} gebunden, hier {got}. Volle Shape: erwartet "
                    f"{_format_shape(expected_dims)}, erhalten {_format_shape(actual)}."
                )
    return value

def _check_return_shape(value, expected_dims, fn_name, shape_env):
    actual = _shape_of(value)
    if actual is None:
        return value
    if len(actual) != len(expected_dims):
        raise ValueError(
            f"Return-Shape-Mismatch in {fn_name}: erwartet {_format_shape(expected_dims)}, "
            f"erhalten {_format_shape(actual)}."
        )
    for i, (want, got) in enumerate(zip(expected_dims, actual)):
        if isinstance(want, int):
            if want != got:
                raise ValueError(
                    f"Return-Shape-Mismatch in {fn_name}: Dim {i} erwartet {want}, erhalten {got}."
                )
        else:
            bound = shape_env.get(want)
            if bound is None:
                shape_env[want] = got
            elif bound != got:
                raise ValueError(
                    f"Symbolische Return-Dimension '{want}' in {fn_name}: bereits {bound}, "
                    f"hier {got}."
                )
    return value

def _graph_dims(value):
    """Liefert (num_nodes, num_edges) fuer torch_geometric.data.Data oder
    aehnliche Objekte. None wenn nicht ermittelbar."""
    if hasattr(value, "num_nodes") and hasattr(value, "num_edges"):
        try:
            return int(value.num_nodes), int(value.num_edges)
        except Exception:
            pass
    # Fallback: x.shape[0] / edge_index.shape[1]
    n_nodes = None
    n_edges = None
    if hasattr(value, "x") and value.x is not None and hasattr(value.x, "shape"):
        try:
            n_nodes = int(value.x.shape[0])
        except Exception:
            pass
    if hasattr(value, "edge_index") and value.edge_index is not None and hasattr(value.edge_index, "shape"):
        try:
            n_edges = int(value.edge_index.shape[1])
        except Exception:
            pass
    if n_nodes is not None and n_edges is not None:
        return n_nodes, n_edges
    return None

def _check_graph_dims_against_env(actual, expected_dims, fn_name, arg_name, shape_env, where):
    """Pruefe (N_nodes, N_edges)-Tupel gegen erwartete Dimensionen mit Symbol-Bindung."""
    labels = ("Knoten", "Kanten")
    for i, (want, got, label) in enumerate(zip(expected_dims, actual, labels)):
        if isinstance(want, int):
            if want != got:
                raise ValueError(
                    f"Graph-Shape-Mismatch in {where} {fn_name}({arg_name}): "
                    f"{label}-Anzahl erwartet {want}, erhalten {got}."
                )
        else:
            bound = shape_env.get(want)
            if bound is None:
                shape_env[want] = got
            elif bound != got:
                raise ValueError(
                    f"Symbolische Graph-Dimension '{want}' in {where} {fn_name}({arg_name}): "
                    f"bereits als {bound} gebunden, hier {got} ({label})."
                )

def _check_graph_shape(value, expected_dims, fn_name, arg_name, shape_env):
    """Validiert, dass `value` ein torch_geometric.data.Data-aehnliches Objekt mit den
    deklarierten (N_nodes, N_edges)-Dimensionen ist. Symbolische Dimensionen werden
    in `shape_env` gebunden bzw. konsistent gehalten."""
    actual = _graph_dims(value)
    if actual is None:
        raise ValueError(
            f"Graph-Shape-Check in {fn_name}({arg_name}): erwarte ein Graph-Objekt "
            f"mit num_nodes/num_edges-Attributen (z. B. torch_geometric.data.Data), "
            f"erhalten {type(value).__name__}."
        )
    _check_graph_dims_against_env(actual, expected_dims, fn_name, arg_name, shape_env, "Argument")
    return value

def _check_return_graph_shape(value, expected_dims, fn_name, shape_env):
    actual = _graph_dims(value)
    if actual is None:
        raise ValueError(
            f"Graph-Return-Shape-Check in {fn_name}: erwarte ein Graph-Objekt, "
            f"erhalten {type(value).__name__}."
        )
    _check_graph_dims_against_env(actual, expected_dims, fn_name, "return", shape_env, "Return")
    return value

def _labeled_dims(value):
    """Liefert tuple der dim-Namen fuer xarray.DataArray oder DataArray-aehnliche
    Objekte. None wenn nicht ermittelbar."""
    if hasattr(value, "dims"):
        try:
            return tuple(str(d) for d in value.dims)
        except Exception:
            pass
    return None

def _check_labeled_shape(value, expected_dims, fn_name, arg_name, shape_env):
    """Validiert, dass `value` ein xarray.DataArray-aehnliches Objekt mit GENAU der
    deklarierten Menge von dim-Namen ist (Reihenfolge irrelevant ÔÇö xarray operiert
    namens-basiert)."""
    actual = _labeled_dims(value)
    if actual is None:
        raise ValueError(
            f"LabeledTensor-Shape-Check in {fn_name}({arg_name}): erwarte ein "
            f"DataArray-aehnliches Objekt mit .dims (z. B. xarray.DataArray), "
            f"erhalten {type(value).__name__}."
        )
    expected_set = set(str(d) for d in expected_dims)
    actual_set = set(actual)
    if expected_set != actual_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        msgs = []
        if missing:
            msgs.append(f"fehlende Dimensionen: {sorted(missing)}")
        if extra:
            msgs.append(f"zusaetzliche Dimensionen: {sorted(extra)}")
        raise ValueError(
            f"LabeledTensor-Shape-Mismatch in {fn_name}({arg_name}): "
            f"erwartet {{ {', '.join(sorted(expected_set))} }}, "
            f"erhalten {{ {', '.join(sorted(actual_set))} }} ({'; '.join(msgs)})."
        )
    return value

def _check_return_labeled_shape(value, expected_dims, fn_name, shape_env):
    actual = _labeled_dims(value)
    if actual is None:
        raise ValueError(
            f"LabeledTensor-Return-Check in {fn_name}: erwarte DataArray-aehnliches "
            f"Objekt, erhalten {type(value).__name__}."
        )
    expected_set = set(str(d) for d in expected_dims)
    actual_set = set(actual)
    if expected_set != actual_set:
        raise ValueError(
            f"LabeledTensor-Return-Mismatch in {fn_name}: erwartet "
            f"{{ {', '.join(sorted(expected_set))} }}, "
            f"erhalten {{ {', '.join(sorted(actual_set))} }}."
        )
    return value

def _validate_sequence_string(value, kind, where):
    """Validiert: value ist ein String und enthaelt nur Zeichen aus dem Alphabet."""
    if not isinstance(value, str):
        raise TypeError(
            f"Sequence[{kind}]-Check in {where}: erwarte String, erhalten {type(value).__name__}."
        )
    alphabet = _SEQ_ALPHABETS.get(kind.upper())
    if alphabet is None:
        raise ValueError(f"Sequence-Kind {kind!r} unbekannt (erlaubt: DNA, RNA, Protein).")
    upper_val = value.upper()
    bad = [c for c in upper_val if c not in alphabet]
    if bad:
        # Erstes Vorkommen mit Position
        idx = next(i for i, c in enumerate(upper_val) if c not in alphabet)
        raise ValueError(
            f"Sequence[{kind}]-Check in {where}: ungueltiges Zeichen "
            f"{value[idx]!r} an Position {idx} (erlaubt: {''.join(sorted(alphabet))})."
        )

def _check_sequence_shape(value, expected_dims, fn_name, arg_name, shape_env):
    kind = str(expected_dims[0])
    _validate_sequence_string(value, kind, f"{fn_name}({arg_name})")
    return value

def _check_return_sequence_shape(value, expected_dims, fn_name, shape_env):
    kind = str(expected_dims[0])
    _validate_sequence_string(value, kind, f"return von {fn_name}")
    return value

def _check_qubit_shape(val, dims, fn_name, arg_name, shape_env):
    """Prueft Qubit[N] ÔÇö erwartet QuantumCircuit oder int."""
    if isinstance(val, QuantumCircuit):
        expected = dims[0] if dims else None
        if expected is not None:
            if isinstance(expected, str):
                if expected in shape_env:
                    if shape_env[expected] != val.n_qubits:
                        raise TypeError(
                            f"{fn_name}(): Argument '{arg_name}' Qubit[{expected}]={shape_env[expected]} "
                            f"aber QuantumCircuit hat {val.n_qubits} Qubits."
                        )
                else:
                    shape_env[expected] = val.n_qubits
            elif val.n_qubits != int(expected):
                raise TypeError(
                    f"{fn_name}(): Argument '{arg_name}' erwartet Qubit[{expected}], "
                    f"QuantumCircuit hat {val.n_qubits} Qubits."
                )
    return val

def _check_statevec_shape(val, dims, fn_name, arg_name, shape_env):
    """Prueft StateVec[N] ÔÇö erwartet Liste der Laenge 2^N."""
    expected_n = dims[0] if dims else None
    if isinstance(val, list):
        actual_len = len(val)
        if expected_n is not None:
            if isinstance(expected_n, str):
                if expected_n in shape_env:
                    exp_len = 1 << shape_env[expected_n]
                    if actual_len != exp_len:
                        raise TypeError(
                            f"{fn_name}(): '{arg_name}' StateVec[{expected_n}]: "
                            f"erwartet Laenge 2^{shape_env[expected_n]}={exp_len}, bekam {actual_len}."
                        )
                else:
                    import math
                    if actual_len == 0 or (actual_len & (actual_len - 1)) != 0:
                        raise TypeError(
                            f"{fn_name}(): '{arg_name}' StateVec[{expected_n}]: "
                            f"Laenge muss Potenz von 2 sein, bekam {actual_len}."
                        )
                    shape_env[expected_n] = int(math.log2(actual_len))
            else:
                exp_len = 1 << int(expected_n)
                if actual_len != exp_len:
                    raise TypeError(
                        f"{fn_name}(): '{arg_name}' erwartet StateVec[{expected_n}] "
                        f"(Laenge {exp_len}), bekam {actual_len}."
                    )
    return val

def _check_return_qubit_shape(val, dims, fn_name, shape_env):
    """Prueft Rueckgabe Qubit[N]/Circuit[N,G]."""
    _check_qubit_shape(val, dims, fn_name, "return", shape_env)
    return val

def _check_return_statevec_shape(val, dims, fn_name, shape_env):
    """Prueft Rueckgabe StateVec[N]."""
    _check_statevec_shape(val, dims, fn_name, "return", shape_env)
    return val

def _pauli(name):
    m = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z, "H": PAULI_H}
    if name not in m:
        raise ValueError(f"Unbekannte Pauli-Matrix '{name}'. Erlaubt: I, X, Y, Z, H.")
    return m[name]

class QuantumCircuit:
    """Leichtgewichtiger Dedekind-Schaltkreis (kein Qiskit n├Âtig).

    Speichert Gatter als Liste von (name, qubits, params). Kann via
    `statevec_sim()` simuliert werden oder per `to_qiskit()` exportiert werden.
    """

    def __init__(self, n_qubits: int):
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError(f"QuantumCircuit: n_qubits muss >= 1 sein, bekam {n_qubits!r}.")
        self.n_qubits = n_qubits
        self._gates = []   # [(name, qubit_list, params)]
        self._measurements = []  # [(qubit, clbit)]

    def h(self, qubit: int):
        """Hadamard-Gatter."""
        self._validate_qubit(qubit)
        self._gates.append(("H", [qubit], []))
        return self

    def x(self, qubit: int):
        """Pauli-X (NOT) Gatter."""
        self._validate_qubit(qubit)
        self._gates.append(("X", [qubit], []))
        return self

    def y(self, qubit: int):
        """Pauli-Y Gatter."""
        self._validate_qubit(qubit)
        self._gates.append(("Y", [qubit], []))
        return self

    def z(self, qubit: int):
        """Pauli-Z Gatter."""
        self._validate_qubit(qubit)
        self._gates.append(("Z", [qubit], []))
        return self

    def cx(self, control: int, target: int):
        """CNOT-Gatter."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("CNOT: control und target d├╝rfen nicht gleich sein.")
        self._gates.append(("CX", [control, target], []))
        return self

    def cz(self, control: int, target: int):
        """CZ-Gatter."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self._gates.append(("CZ", [control, target], []))
        return self

    def rx(self, theta, qubit: int):
        """Rx-Rotation um Winkel theta (Bogenmass)."""
        self._validate_qubit(qubit)
        th = float(_unwrap_q(theta))
        self._gates.append(("RX", [qubit], [th]))
        return self

    def ry(self, theta, qubit: int):
        """Ry-Rotation."""
        self._validate_qubit(qubit)
        th = float(_unwrap_q(theta))
        self._gates.append(("RY", [qubit], [th]))
        return self

    def rz(self, theta, qubit: int):
        """Rz-Rotation."""
        self._validate_qubit(qubit)
        th = float(_unwrap_q(theta))
        self._gates.append(("RZ", [qubit], [th]))
        return self

    def swap(self, q0: int, q1: int):
        """SWAP-Gatter."""
        self._validate_qubit(q0)
        self._validate_qubit(q1)
        self._gates.append(("SWAP", [q0, q1], []))
        return self

    def t(self, qubit: int):
        """T-Gatter (pi/4-Phase)."""
        self._validate_qubit(qubit)
        self._gates.append(("T", [qubit], []))
        return self

    def s(self, qubit: int):
        """S-Gatter (pi/2-Phase)."""
        self._validate_qubit(qubit)
        self._gates.append(("S", [qubit], []))
        return self

    def measure(self, qubit: int, clbit: int = None):
        """F├╝gt Messung hinzu."""
        self._validate_qubit(qubit)
        self._measurements.append((qubit, clbit if clbit is not None else qubit))
        return self

    def measure_all(self):
        """Misst alle Qubits."""
        for i in range(self.n_qubits):
            self.measure(i, i)
        return self

    def _validate_qubit(self, q):
        if not isinstance(q, int) or q < 0 or q >= self.n_qubits:
            raise ValueError(
                f"Qubit-Index {q} ausserhalb des Bereichs [0, {self.n_qubits - 1}]."
            )

    def depth(self) -> int:
        """Schaltkreistiefe (naiv: Anzahl Gatter-Schichten ohne Parallelisierung)."""
        layers = [0] * self.n_qubits
        for name, qubits, _ in self._gates:
            d = _builtin_max(layers[q] for q in qubits) + 1
            for q in qubits:
                layers[q] = d
        return _builtin_max(layers) if layers else 0

    def n_gates(self) -> int:
        """Gesamtanzahl der Gatter."""
        return len(self._gates)

    def __repr__(self):
        lines = [f"QuantumCircuit({self.n_qubits} Qubits, {self.n_gates()} Gatter, Tiefe={self.depth()})"]
        for name, qubits, params in self._gates:
            pstr = f"({', '.join(f'{p:.4f}' for p in params)})" if params else ""
            qstr = ", ".join(str(q) for q in qubits)
            lines.append(f"  {name}{pstr}  q[{qstr}]")
        if self._measurements:
            lines.append(f"  MEASURE: {self._measurements}")
        return "\n".join(lines)

    def to_qiskit(self):
        """Konvertiert zu echtem Qiskit QuantumCircuit (erfordert `pyimport qiskit`)."""
        try:
            import qiskit
        except ImportError:
            raise ImportError(
                "Qiskit nicht gefunden. Installiere es: pip install qiskit\n"
                "Alternativ: statevec_sim(circuit) fuer reine Simulation."
            )
        from qiskit import QuantumCircuit as _QC
        qc = _QC(self.n_qubits, self.n_qubits if self._measurements else 0)
        _QISKIT_MAP = {
            "H": qc.h, "X": qc.x, "Y": qc.y, "Z": qc.z,
            "CX": qc.cx, "CZ": qc.cz, "SWAP": qc.swap,
            "T": qc.t, "S": qc.s,
        }
        _QISKIT_ROT = {"RX": qc.rx, "RY": qc.ry, "RZ": qc.rz}
        for name, qubits, params in self._gates:
            if name in _QISKIT_ROT:
                _QISKIT_ROT[name](params[0], qubits[0])
            elif name in _QISKIT_MAP:
                _QISKIT_MAP[name](*qubits)
        for qubit, clbit in self._measurements:
            qc.measure(qubit, clbit)
        return qc
