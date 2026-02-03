import builtins
import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]

_builtin_min = builtins.min
_builtin_max = builtins.max

def _normalize_unit_for_compare(unit):
    """Normalisiert Einheiten für Vergleich: M, mol/L, mol*L^-1 gelten als gleich (chemische Konzentration)."""
    if not unit:
        return unit
    u = str(unit).strip()
    if u in ("M", "mol/L", "mol*L^-1", "mol*L^(-1)"):
        return "M"
    return u


# Einheiten mit automatischer Umrechnung bei +/-: (Basiseinheit, {Einheit: Faktor zur Basis})
# Wert in Einheit * Faktor = Wert in Basiseinheit
DIMENSION_TO_BASE = {
    # SI-Basis
    "length": ("m", {"m": 1.0, "cm": 0.01, "km": 1000.0, "mm": 0.001, "dm": 0.1}),
    "mass": ("kg", {"kg": 1.0, "g": 0.001, "t": 1000.0, "mg": 1e-6}),
    "time": ("s", {"s": 1.0, "min": 60.0, "h": 3600.0, "ms": 0.001}),
    "current": ("A", {"A": 1.0, "mA": 0.001, "kA": 1000.0, "uA": 1e-6, "muA": 1e-6}),
    "temperature": ("K", {"K": 1.0, "mK": 0.001}),
    "amount_of_substance": ("mol", {"mol": 1.0, "mmol": 0.001, "kmol": 1000.0}),
    "luminous_intensity": ("cd", {"cd": 1.0, "mcd": 0.001}),
    # Abgeleitet / häufig
    "pressure": ("Pa", {"Pa": 1.0, "bar": 1e5, "atm": 101325.0}),
    "volume": ("L", {"L": 1.0, "mL": 0.001, "dm^3": 1.0, "m^3": 1000.0}),  # dm³ = 1 L, m³ = 1000 L
    "energy": ("J", {"J": 1.0, "kJ": 1000.0, "MJ": 1e6, "Wh": 3600.0, "kWh": 3.6e6}),
    "electric_potential": ("V", {"V": 1.0, "mV": 0.001, "kV": 1000.0}),
    "frequency": ("Hz", {"Hz": 1.0, "kHz": 1000.0, "MHz": 1e6, "GHz": 1e9}),
    "charge": ("C", {"C": 1.0, "mC": 0.001, "uC": 1e-6}),
    "resistance": ("ohm", {"ohm": 1.0, "kohm": 1000.0, "Mohm": 1e6}),
    "power": ("W", {"W": 1.0, "kW": 1000.0, "MW": 1e6}),
    # Chemie/Biologie: Massenkonzentration (% w/v = g/100mL)
    "mass_concentration": ("g/L", {"g/L": 1.0, "mg/mL": 1.0, "percent_wv": 10.0}),  # 1% w/v = 10 g/L
}
# Für Units-Checker: Liste der Einheitenmengen pro Dimension
ADDITIVE_DIMENSION_UNIT_SETS = [frozenset(tab.keys()) for _b, tab in DIMENSION_TO_BASE.values()]


def _get_dimension(unit):
    """Liefert Dimensionsname (length, mass, time, pressure) oder None."""
    u = str(unit).strip() if unit else ""
    for dim, (_base, tab) in DIMENSION_TO_BASE.items():
        if u in tab:
            return dim
    return None


def _convert_to_base(value, unit, dimension):
    """Wert in gegebener Einheit in Basiseinheit umrechnen."""
    _, tab = DIMENSION_TO_BASE[dimension]
    u = str(unit).strip()
    return float(value) * tab.get(u, 1.0)


def _convert_from_base(value_base, unit, dimension):
    """Wert in Basiseinheit in gegebene Einheit umrechnen."""
    _, tab = DIMENSION_TO_BASE[dimension]
    u = str(unit).strip()
    if u not in tab:
        return float(value_base)
    return float(value_base) / tab[u]


def _convert_between_units(value, std, from_unit, to_unit, dimension):
    """Wert und Std von from_unit in to_unit umrechnen (für gleiche Dimension). Rückgabe (value, std)."""
    _, tab = DIMENSION_TO_BASE[dimension]
    u_from = str(from_unit).strip()
    u_to = str(to_unit).strip()
    if u_from not in tab or u_to not in tab:
        return float(value), float(std)
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
            "Gleiche Einheit oder kompatible Einheiten derselben Dimension (z. B. Länge, Masse, Zeit, Druck, Strom, Temperatur, mol, cd, Volumen, Energie, Spannung, Frequenz, Ladung, Widerstand, Leistung) erforderlich."
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

def random_vector(size):
    return torch.randn(size)

def random_matrix(rows, cols):
    return torch.randn(rows, cols)

def shuffle(x, dim=0):
    """
    Zufälliges Mischen entlang Achse dim. x: Tensor. dim: Achse (default 0).
    Rückgabe: Tensor gleicher Form (Permutation entlang dim). Nutzt aktuellen Zufallsstand.
    """
    t = _to_tensor(x).clone()
    idx = torch.randperm(t.shape[dim], device=t.device)
    return t.index_select(dim, idx)

def permutation(n):
    """
    Zufällige Permutation der Indizes 0 .. n-1. n: ganze Zahl. Rückgabe: 1D Long-Tensor.
    """
    n_int = int(n)
    if n_int < 0:
        raise ValueError("permutation: n muss nichtnegativ sein.")
    return torch.randperm(n_int)

def choice(a, size=1, replace=True):
    """
    Zufällige Stichprobe aus a. a: 1D-Tensor oder Liste. size: Anzahl Ziehungen (default 1).
    replace: True = mit Zurücklegen, False = ohne. Rückgabe: Tensor der Länge size.
    """
    a_t = _to_tensor(a).float().flatten()
    n = a_t.numel()
    if n == 0:
        raise ValueError("choice: a darf nicht leer sein.")
    size_int = int(size)
    if not replace and size_int > n:
        raise ValueError("choice: size darf bei replace=False nicht größer als len(a) sein.")
    idx = torch.randint(0, n, (size_int,), device=a_t.device) if replace else torch.randperm(n, device=a_t.device)[:size_int]
    return a_t[idx]

def autocorr(x, max_lag=None):
    """
    Autokorrelation (normiert, Lag 0 = 1). x: 1D-Tensor. max_lag: optional (default len(x)-1).
    Rückgabe: 1D-Tensor der Länge max_lag+1.
    """
    x_t = _to_tensor(x).float().flatten()
    n = x_t.numel()
    if n < 2:
        return torch.ones(1, device=x_t.device, dtype=x_t.dtype)
    x_centered = x_t - x_t.mean()
    c0 = (x_centered * x_centered).sum()
    if c0 < 1e-14:
        return torch.ones(_builtin_min(n, max_lag or n) if max_lag is not None else n, device=x_t.device, dtype=x_t.dtype)
    max_lag = _builtin_min(max_lag if max_lag is not None else n - 1, n - 1)
    out = []
    for k in range(max_lag + 1):
        c = (x_centered[:-k] * x_centered[k:]).sum() if k > 0 else c0
        out.append((c / c0).item())
    return torch.tensor(out, device=x_t.device, dtype=x_t.dtype)

def moving_mean(x, window):
    """
    Gleitender Mittelwert. x: 1D-Tensor. window: Fensterbrechte (ungerade empfohlen).
    Rückgabe: 1D-Tensor (Länge len(x)-window+1); keine Randbehandlung (reduzierte Länge).
    """
    x_t = _to_tensor(x).float().flatten()
    w = _builtin_max(1, int(window))
    if w > x_t.numel():
        return x_t.mean().unsqueeze(0)
    kernel = torch.ones(w, device=x_t.device, dtype=x_t.dtype) / w
    return convolve1d(x_t, kernel, mode="valid")

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

def cross(a, b):
    """
    Kreuzprodukt (3D). a, b: 1D-Tensoren der Länge 3. Rückgabe: 1D-Tensor der Länge 3.
    """
    a_t = _to_tensor(a).float().flatten()
    b_t = _to_tensor(b).float().flatten()
    if a_t.numel() != 3 or b_t.numel() != 3:
        raise ValueError("cross: a und b müssen Länge 3 haben.")
    return torch.linalg.cross(a_t, b_t)

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

def fftfreq(n, d=1.0):
    """
    Frequenzachsen für FFT. n: Anzahl Punkte (int). d: Abtastabstand (Skalar, default 1).
    Rückgabe: 1D-Tensor der Länge n mit Frequenzen (Einheit 1/d); für Interpretation von fft(x).
    """
    n_int = int(n)
    d_val = float(_to_tensor(d).float().squeeze().item()) if d != 1.0 else 1.0
    return torch.fft.fftfreq(n_int, d=d_val)

def diff(x, n=1, dim=-1):
    """
    Diskrete Ableitung (Differenzen): diff(x) = x[1:] - x[:-1].
    x: Tensor. n: Ordnung (default 1). dim: Achse (default -1). Rückgabe: Tensor (Länge um n kürzer entlang dim).
    """
    t = _to_tensor(x).float()
    for _ in range(n):
        t = torch.diff(t, dim=dim)
    return t

def cumsum(x, dim=None):
    """
    Kumulative Summe entlang Achse. x: Tensor. dim: Achse (None = über alle, dann flach).
    Rückgabe: Tensor gleicher Form wie x (bzw. 1D wenn dim None).
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.flatten().cumsum(0)
    return t.cumsum(dim=dim)

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
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
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

# --- Advection: u_t + v·∇u = 0 (Upwind-Schema, periodische Randbedingungen) ---

def pde_advection_1d(u0, x, t, v, bc="periodic"):
    """
    Differenzierbarer 1D-Advektionssolver: u_t + v*u_x = 0.
    Upwind-Schema (stabil für CFL |v|*dt/dx <= 1).
    u0: Anfangsbedingung (1D); x: Ortsgitter; t: Zeitgitter; v: Geschwindigkeit (Skalar).
    bc: 'periodic' = periodische Ränder (default); 'zero_gradient' = du/dx=0 am Rand.
    Rückgabe: (len(t), len(x)).
    """
    u0 = _to_tensor(u0).float().flatten()
    x = _to_tensor(x).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    v_val = float(_to_tensor(v).float().item())
    n = u0.numel()
    if x.numel() != n:
        raise ValueError("pde_advection_1d: len(u0) muss len(x) entsprechen.")
    if n < 2:
        raise ValueError("pde_advection_1d: mindestens 2 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(n - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0

    def rhs(t_cur, u):
        u = u.flatten()
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
        else:
            u_left = torch.cat([u[:1], u[:-1]])
            u_right = torch.cat([u[1:], u[-1:]])
        if v_val > 0:
            dudt = -v_val * (u - u_left) / dx
        else:
            dudt = -v_val * (u_right - u) / dx
        return dudt

    return ode_solve(rhs, u0, t)

def pde_advection_2d(u0, x, y, t, vx, vy, bc="periodic"):
    """
    Differenzierbarer 2D-Advektionssolver: u_t + vx*u_x + vy*u_y = 0.
    Upwind-Schema (stabil für CFL).
    u0: Anfangsbedingung 2D (nx, ny); x, y: Ortsgitter; t: Zeitgitter; vx, vy: Geschwindigkeitskomponenten.
    bc: 'periodic' (default) oder 'zero_gradient'. Rückgabe: (len(t), nx, ny).
    """
    u0 = _to_tensor(u0).float()
    if u0.dim() == 1:
        raise ValueError("pde_advection_2d: u0 muss 2D-Gitter (nx, ny) sein.")
    nx, ny = u0.shape[0], u0.shape[1]
    x = _to_tensor(x).float().flatten().to(u0.device)
    y = _to_tensor(y).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    vx_val = float(_to_tensor(vx).float().item())
    vy_val = float(_to_tensor(vy).float().item())
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_advection_2d: x/y Länge muss zu u0 passen.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    if abs(dy) < 1e-14:
        dy = 1.0

    def rhs(t_cur, u_flat):
        u = u_flat.reshape(nx, ny)
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
            u_bottom = torch.roll(u, 1, dims=1)
            u_top = torch.roll(u, -1, dims=1)
        else:
            u_left = torch.cat([u[:1, :], u[:-1, :]], dim=0)
            u_right = torch.cat([u[1:, :], u[-1:, :]], dim=0)
            u_bottom = torch.cat([u[:, :1], u[:, :-1]], dim=1)
            u_top = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
        dudt = torch.zeros_like(u)
        if vx_val > 0:
            dudt = dudt - vx_val * (u - u_left) / dx
        else:
            dudt = dudt - vx_val * (u_right - u) / dx
        if vy_val > 0:
            dudt = dudt - vy_val * (u - u_bottom) / dy
        else:
            dudt = dudt - vy_val * (u_top - u) / dy
        return dudt.flatten()

    return ode_solve(rhs, u0.flatten(), t).reshape(t.numel(), nx, ny)

# --- Wellengleichung: u_tt = c² ∇²u (Reduktion auf System 1. Ordnung: u_t=v, v_t=c²∇²u) ---

def pde_wave_1d(u0, x, t, c, v0=None, bc="periodic"):
    """
    Differenzierbarer 1D-Wellensolver: u_tt = c² u_xx.
    Reduktion auf u_t=v, v_t=c² u_xx; zentrale Differenzen für u_xx.
    u0: Anfangsauslenkung u(0,x); x: Ortsgitter; t: Zeitgitter; c: Ausbreitungsgeschwindigkeit.
    v0: Anfangsgeschwindigkeit u_t(0,x) (optional, default 0).
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (len(t), len(x)).
    """
    u0 = _to_tensor(u0).float().flatten()
    v0 = _to_tensor(v0).float().flatten().to(u0.device) if v0 is not None else torch.zeros_like(u0)
    x = _to_tensor(x).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    c_val = float(_to_tensor(c).float().item())
    n = u0.numel()
    if x.numel() != n or v0.numel() != n:
        raise ValueError("pde_wave_1d: len(u0), len(v0), len(x) müssen übereinstimmen.")
    if n < 3:
        raise ValueError("pde_wave_1d: mindestens 3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(n - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    dx2 = dx * dx
    y0 = torch.cat([u0, v0], dim=0)

    def rhs(t_cur, y):
        u = y[:n]
        v = y[n:]
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
        else:
            u_left = torch.cat([u[:1], u[:-1]])
            u_right = torch.cat([u[1:], u[-1:]])
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        dudt = v
        dvdt = c_val * c_val * u_xx
        return torch.cat([dudt, dvdt], dim=0)

    sol = ode_solve(rhs, y0, t)
    return sol[:, :n]

def pde_wave_2d(u0, x, y, t, c, v0=None, bc="periodic"):
    """
    Differenzierbarer 2D-Wellensolver: u_tt = c² (u_xx + u_yy).
    u0: Anfangsauslenkung 2D; x, y: Ortsgitter; t: Zeitgitter; c: Ausbreitungsgeschwindigkeit.
    v0: Anfangsgeschwindigkeit 2D (optional, default 0).
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (len(t), nx, ny).
    """
    u0 = _to_tensor(u0).float()
    if u0.dim() == 1:
        raise ValueError("pde_wave_2d: u0 muss 2D-Gitter (nx, ny) sein.")
    nx, ny = u0.shape[0], u0.shape[1]
    v0 = _to_tensor(v0).float().to(u0.device) if v0 is not None else torch.zeros_like(u0)
    if v0.shape != u0.shape:
        v0 = torch.zeros_like(u0)
    x = _to_tensor(x).float().flatten().to(u0.device)
    y = _to_tensor(y).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    c_val = float(_to_tensor(c).float().item())
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_wave_2d: x/y Länge muss zu u0 passen.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    if abs(dy) < 1e-14:
        dy = 1.0
    dx2, dy2 = dx * dx, dy * dy
    n = nx * ny
    y0 = torch.cat([u0.flatten(), v0.flatten()], dim=0)

    def rhs(t_cur, y_flat):
        u = y_flat[:n].reshape(nx, ny)
        v = y_flat[n:].reshape(nx, ny)
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
            u_bottom = torch.roll(u, 1, dims=1)
            u_top = torch.roll(u, -1, dims=1)
        else:
            u_left = torch.cat([u[:1, :], u[:-1, :]], dim=0)
            u_right = torch.cat([u[1:, :], u[-1:, :]], dim=0)
            u_bottom = torch.cat([u[:, :1], u[:, :-1]], dim=1)
            u_top = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        u_yy = (u_bottom - 2.0 * u + u_top) / dy2
        lap_u = u_xx + u_yy
        dudt = v
        dvdt = c_val * c_val * lap_u
        return torch.cat([dudt.flatten(), dvdt.flatten()], dim=0)

    sol = ode_solve(rhs, y0, t)
    return sol[:, :n].reshape(t.numel(), nx, ny)

# --- Burgers-Gleichung: u_t + u*u_x = ν*u_xx (Upwind für Advektion, zentral für Diffusion) ---

def pde_burgers_1d(u0, x, t, nu, bc="periodic"):
    """
    Differenzierbarer 1D-Burgers-Solver: u_t + u*u_x = ν*u_xx.
    Upwind für nichtlinearen Advektionsterm u*u_x; zentrale Differenzen für ν*u_xx.
    u0: Anfangsbedingung (1D); x: Ortsgitter; t: Zeitgitter; nu: Viskosität (ν≥0).
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (len(t), len(x)).
    """
    u0 = _to_tensor(u0).float().flatten()
    x = _to_tensor(x).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    nu_val = float(_to_tensor(nu).float().item())
    n = u0.numel()
    if x.numel() != n:
        raise ValueError("pde_burgers_1d: len(u0) muss len(x) entsprechen.")
    if n < 3:
        raise ValueError("pde_burgers_1d: mindestens 3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(n - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    dx2 = dx * dx

    def rhs(t_cur, u):
        u = u.flatten()
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
        else:
            u_left = torch.cat([u[:1], u[:-1]])
            u_right = torch.cat([u[1:], u[-1:]])
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        adv = torch.where(u > 0, -u * (u - u_left) / dx, -u * (u_right - u) / dx)
        dudt = adv + nu_val * u_xx
        return dudt

    return ode_solve(rhs, u0, t)

def pde_burgers_2d(u0, x, y, t, nu, bc="periodic"):
    """
    Differenzierbarer 2D-Burgers-Solver: u_t + u*u_x + u*u_y = ν*(u_xx + u_yy).
    Upwind für nichtlinearen Advektionsterm u·∇u; zentrale Differenzen für ν∇²u.
    u0: Anfangsbedingung 2D (nx, ny); x, y: Ortsgitter; t: Zeitgitter; nu: Viskosität (ν≥0).
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (len(t), nx, ny).
    """
    u0 = _to_tensor(u0).float()
    if u0.dim() == 1:
        raise ValueError("pde_burgers_2d: u0 muss 2D-Gitter (nx, ny) sein.")
    nx, ny = u0.shape[0], u0.shape[1]
    x = _to_tensor(x).float().flatten().to(u0.device)
    y = _to_tensor(y).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    nu_val = float(_to_tensor(nu).float().item())
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_burgers_2d: x/y Länge muss zu u0 passen.")
    if nx < 3 or ny < 3:
        raise ValueError("pde_burgers_2d: mindestens 3×3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    if abs(dy) < 1e-14:
        dy = 1.0
    dx2, dy2 = dx * dx, dy * dy

    def rhs(t_cur, u_flat):
        u = u_flat.reshape(nx, ny)
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
            u_bottom = torch.roll(u, 1, dims=1)
            u_top = torch.roll(u, -1, dims=1)
        else:
            u_left = torch.cat([u[:1, :], u[:-1, :]], dim=0)
            u_right = torch.cat([u[1:, :], u[-1:, :]], dim=0)
            u_bottom = torch.cat([u[:, :1], u[:, :-1]], dim=1)
            u_top = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        u_yy = (u_bottom - 2.0 * u + u_top) / dy2
        lap_u = u_xx + u_yy
        adv_x = torch.where(u > 0, -u * (u - u_left) / dx, -u * (u_right - u) / dx)
        adv_y = torch.where(u > 0, -u * (u - u_bottom) / dy, -u * (u_top - u) / dy)
        dudt = adv_x + adv_y + nu_val * lap_u
        return dudt.flatten()

    return ode_solve(rhs, u0.flatten(), t).reshape(t.numel(), nx, ny)

# --- Reaktions-Diffusion: u_t = D∇²u + f(u) bzw. Gray-Scott (u,v) ---

def pde_reaction_diffusion_1d(u0, x, t, D, r, reaction="fisher", bc="periodic"):
    """
    Differenzierbarer 1D-Reaktions-Diffusionssolver: u_t = D*u_xx + f(u).
    reaction="fisher": Fisher-KPP f(u)=r*u*(1-u); zentrale Differenzen für Diffusion.
    u0: Anfangsbedingung (1D); x: Ortsgitter; t: Zeitgitter; D: Diffusionskoeffizient; r: Reaktionsrate.
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (len(t), len(x)).
    """
    u0 = _to_tensor(u0).float().flatten()
    x = _to_tensor(x).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    D_val = float(_to_tensor(D).float().item())
    r_val = float(_to_tensor(r).float().item())
    n = u0.numel()
    if x.numel() != n:
        raise ValueError("pde_reaction_diffusion_1d: len(u0) muss len(x) entsprechen.")
    if n < 3:
        raise ValueError("pde_reaction_diffusion_1d: mindestens 3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(n - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    dx2 = dx * dx

    def rhs(t_cur, u):
        u = u.flatten()
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
        else:
            u_left = torch.cat([u[:1], u[:-1]])
            u_right = torch.cat([u[1:], u[-1:]])
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        if reaction == "fisher":
            react = r_val * u * (1.0 - u)
        else:
            react = r_val * u
        dudt = D_val * u_xx + react
        return dudt

    return ode_solve(rhs, u0, t)

def pde_reaction_diffusion_2d(u0, v0, x, y, t, Du, Dv, a, b, bc="periodic"):
    """
    Differenzierbarer 2D-Gray-Scott-Solver: u_t = Du∇²u - u*v² + a*(1-u), v_t = Dv∇²v + u*v² - b*v.
    Turing-Muster, zentrale Differenzen. u0, v0: Anfangsbedingungen 2D (nx, ny); x, y: Ortsgitter.
    t: Zeitgitter; Du, Dv: Diffusionskoeffizienten; a, b: Gray-Scott-Parameter.
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (u_sol, v_sol) je (len(t), nx, ny).
    """
    u0 = _to_tensor(u0).float()
    v0 = _to_tensor(v0).float().to(u0.device)
    if u0.dim() == 1 or v0.dim() == 1:
        raise ValueError("pde_reaction_diffusion_2d: u0 und v0 müssen 2D-Gitter (nx, ny) sein.")
    nx, ny = u0.shape[0], u0.shape[1]
    if v0.shape != u0.shape:
        raise ValueError("pde_reaction_diffusion_2d: u0 und v0 müssen gleiche Form haben.")
    x = _to_tensor(x).float().flatten().to(u0.device)
    y = _to_tensor(y).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    Du_val = float(_to_tensor(Du).float().item())
    Dv_val = float(_to_tensor(Dv).float().item())
    a_val = float(_to_tensor(a).float().item())
    b_val = float(_to_tensor(b).float().item())
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_reaction_diffusion_2d: x/y Länge muss zu u0 passen.")
    if nx < 3 or ny < 3:
        raise ValueError("pde_reaction_diffusion_2d: mindestens 3×3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    if abs(dy) < 1e-14:
        dy = 1.0
    dx2, dy2 = dx * dx, dy * dy
    n = nx * ny
    y0 = torch.cat([u0.flatten(), v0.flatten()], dim=0)

    def _lap(u):
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
            u_bottom = torch.roll(u, 1, dims=1)
            u_top = torch.roll(u, -1, dims=1)
        else:
            u_left = torch.cat([u[:1, :], u[:-1, :]], dim=0)
            u_right = torch.cat([u[1:, :], u[-1:, :]], dim=0)
            u_bottom = torch.cat([u[:, :1], u[:, :-1]], dim=1)
            u_top = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        u_yy = (u_bottom - 2.0 * u + u_top) / dy2
        return u_xx + u_yy

    def rhs(t_cur, y_flat):
        u = y_flat[:n].reshape(nx, ny)
        v = y_flat[n:].reshape(nx, ny)
        uv2 = u * v * v
        lap_u = _lap(u)
        lap_v = _lap(v)
        dudt = Du_val * lap_u - uv2 + a_val * (1.0 - u)
        dvdt = Dv_val * lap_v + uv2 - b_val * v
        return torch.cat([dudt.flatten(), dvdt.flatten()], dim=0)

    sol = ode_solve(rhs, y0, t)
    u_sol = sol[:, :n].reshape(t.numel(), nx, ny)
    v_sol = sol[:, n:].reshape(t.numel(), nx, ny)
    return u_sol, v_sol

# --- Advektions-Diffusion: u_t + v·∇u = D∇²u ---

def pde_advection_diffusion_1d(u0, x, t, v, D, bc="periodic"):
    """
    Differenzierbarer 1D-Advektions-Diffusionssolver: u_t + v*u_x = D*u_xx.
    Upwind für Advektion, zentrale Differenzen für Diffusion.
    u0: Anfangsbedingung (1D); x: Ortsgitter; t: Zeitgitter; v: Geschwindigkeit; D: Diffusionskoeffizient.
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (len(t), len(x)).
    """
    u0 = _to_tensor(u0).float().flatten()
    x = _to_tensor(x).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    v_val = float(_to_tensor(v).float().item())
    D_val = float(_to_tensor(D).float().item())
    n = u0.numel()
    if x.numel() != n:
        raise ValueError("pde_advection_diffusion_1d: len(u0) muss len(x) entsprechen.")
    if n < 3:
        raise ValueError("pde_advection_diffusion_1d: mindestens 3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(n - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    dx2 = dx * dx

    def rhs(t_cur, u):
        u = u.flatten()
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
        else:
            u_left = torch.cat([u[:1], u[:-1]])
            u_right = torch.cat([u[1:], u[-1:]])
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        if v_val > 0:
            adv = -v_val * (u - u_left) / dx
        else:
            adv = -v_val * (u_right - u) / dx
        dudt = adv + D_val * u_xx
        return dudt

    return ode_solve(rhs, u0, t)

def pde_advection_diffusion_2d(u0, x, y, t, vx, vy, D, bc="periodic"):
    """
    Differenzierbarer 2D-Advektions-Diffusionssolver: u_t + vx*u_x + vy*u_y = D*(u_xx + u_yy).
    Upwind für Advektion, zentrale Differenzen für Diffusion.
    u0: Anfangsbedingung 2D (nx, ny); x, y: Ortsgitter; t: Zeitgitter; vx, vy: Geschwindigkeit; D: Diffusionskoeffizient.
    bc: 'periodic' (default) oder 'dirichlet'. Rückgabe: (len(t), nx, ny).
    """
    u0 = _to_tensor(u0).float()
    if u0.dim() == 1:
        raise ValueError("pde_advection_diffusion_2d: u0 muss 2D-Gitter (nx, ny) sein.")
    nx, ny = u0.shape[0], u0.shape[1]
    x = _to_tensor(x).float().flatten().to(u0.device)
    y = _to_tensor(y).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    vx_val = float(_to_tensor(vx).float().item())
    vy_val = float(_to_tensor(vy).float().item())
    D_val = float(_to_tensor(D).float().item())
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_advection_diffusion_2d: x/y Länge muss zu u0 passen.")
    if nx < 3 or ny < 3:
        raise ValueError("pde_advection_diffusion_2d: mindestens 3×3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    if abs(dy) < 1e-14:
        dy = 1.0
    dx2, dy2 = dx * dx, dy * dy

    def rhs(t_cur, u_flat):
        u = u_flat.reshape(nx, ny)
        if bc == "periodic":
            u_left = torch.roll(u, 1, dims=0)
            u_right = torch.roll(u, -1, dims=0)
            u_bottom = torch.roll(u, 1, dims=1)
            u_top = torch.roll(u, -1, dims=1)
        else:
            u_left = torch.cat([u[:1, :], u[:-1, :]], dim=0)
            u_right = torch.cat([u[1:, :], u[-1:, :]], dim=0)
            u_bottom = torch.cat([u[:, :1], u[:, :-1]], dim=1)
            u_top = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
        u_xx = (u_left - 2.0 * u + u_right) / dx2
        u_yy = (u_bottom - 2.0 * u + u_top) / dy2
        lap_u = u_xx + u_yy
        dudt = torch.zeros_like(u)
        if vx_val > 0:
            dudt = dudt - vx_val * (u - u_left) / dx
        else:
            dudt = dudt - vx_val * (u_right - u) / dx
        if vy_val > 0:
            dudt = dudt - vy_val * (u - u_bottom) / dy
        else:
            dudt = dudt - vy_val * (u_top - u) / dy
        dudt = dudt + D_val * lap_u
        return dudt.flatten()

    return ode_solve(rhs, u0.flatten(), t).reshape(t.numel(), nx, ny)

# --- Maxwell-Gleichungen: FDTD (∇×E=-∂B/∂t, ∇×B=μ₀ε₀∂E/∂t) ---

def pde_maxwell_1d(E0, B0, x, t, c_light=1.0, bc="periodic"):
    """
    Differenzierbarer 1D-Maxwell-FDTD: Ebene Welle E_y(x,t), B_z(x,t).
    ∂E_y/∂t = -c² ∂B_z/∂x, ∂B_z/∂t = -∂E_y/∂x; zentrale Differenzen.
    E0, B0: Anfangsbedingungen 1D (E_y, B_z); x: Ortsgitter; t: Zeitgitter.
    c_light: Lichtgeschwindigkeit (default 1). bc: 'periodic' (default) oder 'dirichlet'.
    Rückgabe: (E_sol, B_sol) je (len(t), len(x)). CFL: dt <= dx/c empfohlen.
    """
    E0 = _to_tensor(E0).float().flatten()
    B0 = _to_tensor(B0).float().flatten().to(E0.device)
    x = _to_tensor(x).float().flatten().to(E0.device)
    t = _to_tensor(t).float().flatten().to(E0.device)
    c_val = float(_to_tensor(c_light).float().item())
    n = E0.numel()
    if B0.numel() != n or x.numel() != n:
        raise ValueError("pde_maxwell_1d: len(E0), len(B0), len(x) müssen übereinstimmen.")
    if n < 3:
        raise ValueError("pde_maxwell_1d: mindestens 3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(n - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    c2 = c_val * c_val
    y0 = torch.cat([E0, B0], dim=0)

    def rhs(t_cur, y):
        E = y[:n]
        B = y[n:]
        if bc == "periodic":
            E_left = torch.roll(E, 1, dims=0)
            E_right = torch.roll(E, -1, dims=0)
            B_left = torch.roll(B, 1, dims=0)
            B_right = torch.roll(B, -1, dims=0)
        else:
            E_left = torch.cat([E[:1], E[:-1]])
            E_right = torch.cat([E[1:], E[-1:]])
            B_left = torch.cat([B[:1], B[:-1]])
            B_right = torch.cat([B[1:], B[-1:]])
        dB_dx = (B_right - B_left) / (2.0 * dx)
        dE_dx = (E_right - E_left) / (2.0 * dx)
        dE_dt = -c2 * dB_dx
        dB_dt = -dE_dx
        return torch.cat([dE_dt, dB_dt], dim=0)

    sol = ode_solve(rhs, y0, t)
    E_sol = sol[:, :n]
    B_sol = sol[:, n:]
    return E_sol, B_sol

def pde_maxwell_2d(Ez0, Hx0, Hy0, x, y, t, c_light=1.0, bc="periodic"):
    """
    Differenzierbarer 2D-Maxwell-FDTD (TM-Mode): E_z, H_x, H_y.
    ∂E_z/∂t = c²(∂H_y/∂x - ∂H_x/∂y), ∂H_x/∂t = -∂E_z/∂y, ∂H_y/∂t = ∂E_z/∂x.
    Ez0, Hx0, Hy0: Anfangsbedingungen 2D (nx, ny); x, y: Ortsgitter; t: Zeitgitter.
    c_light: Lichtgeschwindigkeit. bc: 'periodic' (default) oder 'dirichlet'.
    Rückgabe: (Ez_sol, Hx_sol, Hy_sol) je (len(t), nx, ny). CFL: dt <= dx/(c√2) empfohlen.
    """
    Ez0 = _to_tensor(Ez0).float()
    Hx0 = _to_tensor(Hx0).float().to(Ez0.device)
    Hy0 = _to_tensor(Hy0).float().to(Ez0.device)
    if Ez0.dim() == 1 or Hx0.dim() == 1 or Hy0.dim() == 1:
        raise ValueError("pde_maxwell_2d: Ez0, Hx0, Hy0 müssen 2D-Gitter (nx, ny) sein.")
    nx, ny = Ez0.shape[0], Ez0.shape[1]
    if Hx0.shape != (nx, ny) or Hy0.shape != (nx, ny):
        raise ValueError("pde_maxwell_2d: Ez0, Hx0, Hy0 müssen gleiche Form haben.")
    x = _to_tensor(x).float().flatten().to(Ez0.device)
    y = _to_tensor(y).float().flatten().to(Ez0.device)
    t = _to_tensor(t).float().flatten().to(Ez0.device)
    c_val = float(_to_tensor(c_light).float().item())
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_maxwell_2d: x/y Länge muss zu Ez0 passen.")
    if nx < 3 or ny < 3:
        raise ValueError("pde_maxwell_2d: mindestens 3×3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    if abs(dy) < 1e-14:
        dy = 1.0
    c2 = c_val * c_val
    n = nx * ny
    y0 = torch.cat([Ez0.flatten(), Hx0.flatten(), Hy0.flatten()], dim=0)

    def rhs(t_cur, y_flat):
        Ez = y_flat[:n].reshape(nx, ny)
        Hx = y_flat[n:2 * n].reshape(nx, ny)
        Hy = y_flat[2 * n:].reshape(nx, ny)
        if bc == "periodic":
            Ez_left = torch.roll(Ez, 1, dims=0)
            Ez_right = torch.roll(Ez, -1, dims=0)
            Ez_bottom = torch.roll(Ez, 1, dims=1)
            Ez_top = torch.roll(Ez, -1, dims=1)
            Hy_left = torch.roll(Hy, 1, dims=0)
            Hy_right = torch.roll(Hy, -1, dims=0)
            Hx_bottom = torch.roll(Hx, 1, dims=1)
            Hx_top = torch.roll(Hx, -1, dims=1)
        else:
            Ez_left = torch.cat([Ez[:1, :], Ez[:-1, :]], dim=0)
            Ez_right = torch.cat([Ez[1:, :], Ez[-1:, :]], dim=0)
            Ez_bottom = torch.cat([Ez[:, :1], Ez[:, :-1]], dim=1)
            Ez_top = torch.cat([Ez[:, 1:], Ez[:, -1:]], dim=1)
            Hy_left = torch.cat([Hy[:1, :], Hy[:-1, :]], dim=0)
            Hy_right = torch.cat([Hy[1:, :], Hy[-1:, :]], dim=0)
            Hx_bottom = torch.cat([Hx[:, :1], Hx[:, :-1]], dim=1)
            Hx_top = torch.cat([Hx[:, 1:], Hx[:, -1:]], dim=1)
        dHy_dx = (Hy_right - Hy_left) / (2.0 * dx)
        dHx_dy = (Hx_top - Hx_bottom) / (2.0 * dy)
        dEz_dx = (Ez_right - Ez_left) / (2.0 * dx)
        dEz_dy = (Ez_top - Ez_bottom) / (2.0 * dy)
        dEz_dt = c2 * (dHy_dx - dHx_dy)
        dHx_dt = -dEz_dy
        dHy_dt = dEz_dx
        return torch.cat([dEz_dt.flatten(), dHx_dt.flatten(), dHy_dt.flatten()], dim=0)

    sol = ode_solve(rhs, y0, t)
    Ez_sol = sol[:, :n].reshape(t.numel(), nx, ny)
    Hx_sol = sol[:, n:2 * n].reshape(t.numel(), nx, ny)
    Hy_sol = sol[:, 2 * n:].reshape(t.numel(), nx, ny)
    return Ez_sol, Hx_sol, Hy_sol

# --- Sparse PDE: 2D Laplacian und Diffusion ---

def sparse_laplacian_2d(N, dx=None):
    """
    Baut den 2D-Laplacian als sparse COO-Matrix (5-Punkt-Stencil) für ein N×N-Gitter.
    Dirichlet-Rand: Randzeilen sind 0 (dT/dt=0 am Rand).
    N: Gittergröße; dx: Gitterabstand (default 1/(N-1) für Einheitsintervall).
    Rückgabe: Sparse-Tensor (N², N²).
    """
    N = int(N)
    if N < 2:
        raise ValueError("sparse_laplacian_2d: N muss >= 2 sein.")
    if dx is None:
        dx = 1.0 / _builtin_max(N - 1, 1)
    dx2 = dx * dx
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            row = i * N + j
            # Dirichlet BC: Randpunkte haben dT/dt=0 → Laplacian-Zeile = 0
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                rows.append(row)
                cols.append(row)
                vals.append(0.0)  # Zeile = 0, Rand bleibt unverändert (wird nach Step auf 0 gesetzt)
                continue
            # Innen: 5-Punkt-Stencil (∇²u ≈ (Nachbarn - 4*Mitte)/dx²)
            n_neighbors = 0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                col = ni * N + nj
                rows.append(row)
                cols.append(col)
                vals.append(1.0 / dx2)
                n_neighbors += 1
            rows.append(row)
            cols.append(row)
            vals.append(-n_neighbors / dx2)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    L = torch.sparse_coo_tensor(indices, values, (N * N, N * N)).coalesce()
    return L

def sparse_diffusion_step(T, L, dt, alpha):
    """
    Ein expliziter Euler-Schritt für ∂T/∂t = α ∇²T.
    T: 2D-Tensor (N,N); L: sparse Laplacian von sparse_laplacian_2d(N); dt, alpha: Skalare.
    Dirichlet BC: Rand wird nach dem Schritt auf 0 gesetzt.
    Rückgabe: T neu (2D).
    """
    T = _to_tensor(T).float()
    if T.dim() != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("sparse_diffusion_step: T muss quadratische 2D-Matrix sein.")
    N = T.shape[0]
    if L.shape != (N * N, N * N):
        raise ValueError("sparse_diffusion_step: L muss zu T passen (N²×N²).")
    dt = float(dt)
    alpha = float(alpha)
    T_flat = T.flatten().unsqueeze(1)
    lap_T = torch.sparse.mm(L, T_flat).squeeze(1)
    T_new_flat = T_flat.squeeze(1) + dt * alpha * lap_T
    T_new = T_new_flat.reshape(N, N)
    # Dirichlet BC: Rand = 0
    T_new[0, :] = 0.0
    T_new[-1, :] = 0.0
    T_new[:, 0] = 0.0
    T_new[:, -1] = 0.0
    return T_new

def sparse_diffusion_simulate(T0, n_steps, dt, alpha):
    """
    Vollständige sparse 2D-Diffusionssimulation: ∂T/∂t = α ∇²T.
    T0: Anfangsbedingung 2D (N,N); n_steps: Zeitschritte; dt, alpha: Skalare.
    Dirichlet BC: Rand = 0. Rückgabe: Liste [T0, T1, ..., T_n_steps].
    """
    T0 = _to_tensor(T0).float()
    if T0.dim() != 2 or T0.shape[0] != T0.shape[1]:
        raise ValueError("sparse_diffusion_simulate: T0 muss quadratische 2D-Matrix sein.")
    N = T0.shape[0]
    L = sparse_laplacian_2d(N)
    result = [T0.clone()]
    T = T0.clone()
    for _ in range(int(n_steps) - 1):
        T = sparse_diffusion_step(T, L, dt, alpha)
        result.append(T.clone())
    return result

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
        accept_prob = 1.0 if log_accept >= 0 else _builtin_min(1.0, torch.exp(torch.tensor(log_accept)).item())
        if torch.rand(1).item() < accept_prob:
            theta = proposal.clone()
        samples.append(theta.clone())
    return torch.stack(samples, dim=0)

def hmc(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size=0.1, num_leapfrog=10):
    """
    Hamiltonian Monte Carlo (HMC). Gleiche API wie metropolis; nutzt Gradienten für bessere Proposals.
    log_prior_fn(theta), log_likelihood_fn(data, theta) liefern Log-Wahrscheinlichkeiten (Tensor/Skalar).
    init_theta: Startparameter (Tensor/Liste); step_size: Schrittweite; num_leapfrog: Leapfrog-Schritte.
    Rückgabe: Tensor (num_steps, *theta_shape) mit Posterior-Samples.
    """
    theta = _to_tensor(init_theta).float().clone().detach()
    data_t = _to_tensor(data)
    samples = [theta.clone()]

    def neg_log_post(th):
        th = th.detach().clone().requires_grad_(True)
        lp = log_prior_fn(th)
        ll = log_likelihood_fn(data_t, th)
        lp_t = _to_tensor(lp).sum()
        ll_t = _to_tensor(ll).sum()
        nlp = -(lp_t + ll_t)
        if th.grad is not None:
            th.grad.zero_()
        nlp.backward()
        grad = th.grad.clone() if th.grad is not None else torch.zeros_like(th)
        return nlp.detach().item(), grad

    for _ in range(num_steps - 1):
        p = torch.randn_like(theta)
        q = theta.clone()
        eps = float(step_size)
        L = _builtin_max(1, int(num_leapfrog))
        U_cur = neg_log_post(theta)[0]
        K_cur = 0.5 * (p ** 2).sum().item()
        H_cur = U_cur + K_cur
        _, g = neg_log_post(q)
        p = p + 0.5 * eps * g
        for _ in range(L - 1):
            q = q + eps * p
            _, g = neg_log_post(q)
            p = p + eps * g
        q = q + eps * p
        _, g = neg_log_post(q)
        p = p + 0.5 * eps * g
        U_prop = neg_log_post(q)[0]
        K_prop = 0.5 * (p ** 2).sum().item()
        H_prop = U_prop + K_prop
        log_accept = H_cur - H_prop
        accept_prob = _builtin_min(1.0, float(torch.exp(torch.tensor(log_accept)).item()))
        if torch.rand(1).item() < accept_prob:
            theta = q.clone()
        samples.append(theta.clone())
    return torch.stack(samples, dim=0)

# --- Standard Library: Math (for integration and expressions) ---
# Alle Funktionen: Tensor oder Skalar (via _to_tensor), elementweise, differenzierbar.

def sin(x):
    """sin(x); x Tensor oder Skalar."""
    return torch.sin(_to_tensor(x).float())

def cos(x):
    """cos(x); x Tensor oder Skalar."""
    return torch.cos(_to_tensor(x).float())

def tan(x):
    """tan(x); x Tensor oder Skalar."""
    return torch.tan(_to_tensor(x).float())

def exp(x):
    """exp(x); x Tensor oder Skalar."""
    return torch.exp(_to_tensor(x).float())

def log(x):
    """Natürlicher Logarithmus log(x) = ln(x); x Tensor oder Skalar."""
    return torch.log(_to_tensor(x).float())

def log10(x):
    """Zehnerlogarithmus log10(x); x Tensor oder Skalar."""
    return torch.log10(_to_tensor(x).float())

def sqrt(x):
    """Quadratwurzel sqrt(x); x Tensor oder Skalar (nicht negativ)."""
    return torch.sqrt(_to_tensor(x).float())

def abs(x):
    """Betrag |x|; x Tensor oder Skalar."""
    return torch.abs(_to_tensor(x).float())

def asin(x):
    """Arkussinus asin(x); x im Intervall [-1, 1]."""
    return torch.asin(_to_tensor(x).float())

def acos(x):
    """Arkuskosinus acos(x); x im Intervall [-1, 1]."""
    return torch.acos(_to_tensor(x).float())

def atan(x):
    """Arkustangens atan(x)."""
    return torch.atan(_to_tensor(x).float())

def atan2(y, x):
    """Arkustangens atan2(y, x); Winkel der (x,y)-Richtung, Wertebereich (-pi, pi]."""
    return torch.atan2(_to_tensor(y).float(), _to_tensor(x).float())

def sinh(x):
    """Sinus hyperbolicus sinh(x)."""
    return torch.sinh(_to_tensor(x).float())

def cosh(x):
    """Kosinus hyperbolicus cosh(x)."""
    return torch.cosh(_to_tensor(x).float())

def tanh(x):
    """Tangens hyperbolicus tanh(x)."""
    return torch.tanh(_to_tensor(x).float())

def erf(x):
    """Fehlerfunktion erf(x); elementweise. Für Normalverteilung, Diffusion."""
    return torch.erf(_to_tensor(x).float())

def erfc(x):
    """Komplementäre Fehlerfunktion erfc(x) = 1 - erf(x)."""
    return torch.erfc(_to_tensor(x).float())

def lgamma(x):
    """Log-Gamma ln(Gamma(x)); elementweise. Für Verteilungen, Faktorielle."""
    return torch.lgamma(_to_tensor(x).float())

def gamma(x):
    """Gamma-Funktion Gamma(x); elementweise. Für x > 0."""
    return torch.exp(torch.lgamma(_to_tensor(x).float()))

def bessel_j0(x):
    """Bessel-Funktion erster Art, Ordnung 0: J_0(x); elementweise."""
    return torch.special.bessel_j0(_to_tensor(x).float())

def bessel_j1(x):
    """Bessel-Funktion erster Art, Ordnung 1: J_1(x); elementweise."""
    return torch.special.bessel_j1(_to_tensor(x).float())

def bessel_j(n, x):
    """
    Bessel-Funktion erster Art, Ordnung n: J_n(x).
    n: Ordnung (Skalar oder Tensor). x: Argument (Tensor oder Skalar).
    Erfordert scipy.
    """
    try:
        import scipy.special as sc  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("bessel_j(n, x) erfordert scipy. Bitte installieren: pip install scipy")
    x_t = _to_tensor(x).float()
    n_val = int(n) if hasattr(n, "__int__") else float(_to_tensor(n).float().item())
    out = sc.jv(n_val, x_t.detach().cpu().numpy())
    return torch.tensor(out, dtype=torch.float32, device=x_t.device)

def legendre(n, x):
    """
    Legendre-Polynom P_n(x). n: Grad (int). x: Argument (Tensor oder Skalar).
    Erfordert scipy.
    """
    try:
        import scipy.special as sc  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("legendre(n, x) erfordert scipy. Bitte installieren: pip install scipy")
    x_t = _to_tensor(x).float()
    n_val = int(n)
    out = sc.eval_legendre(n_val, x_t.detach().cpu().numpy())
    return torch.tensor(out, dtype=torch.float32, device=x_t.device)

def hypergeom(a, b, c, z):
    """
    Hypergeometrische Funktion 2F1(a, b; c; z).
    a, b, c: Parameter (Skalare). z: Argument (Tensor oder Skalar).
    Erfordert scipy.
    """
    try:
        import scipy.special as sc  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("hypergeom(a, b, c, z) erfordert scipy. Bitte installieren: pip install scipy")
    z_t = _to_tensor(z).float()
    a_val = float(a)
    b_val = float(b)
    c_val = float(c)
    out = sc.hyp2f1(a_val, b_val, c_val, z_t.detach().cpu().numpy())
    return torch.tensor(out, dtype=torch.float32, device=z_t.device)

# --- Standard Library: Zahlentheorie (gcd, is_prime, mod, mod_inv, mod_pow) ---
def gcd(a, b):
    """Größter gemeinsamer Teiler von a und b (ganze Zahlen)."""
    a_val = int(_to_tensor(a).float().squeeze().item()) if hasattr(a, "__float__") else int(a)
    b_val = int(_to_tensor(b).float().squeeze().item()) if hasattr(b, "__float__") else int(b)
    a_val, b_val = abs(a_val), abs(b_val)
    while b_val:
        a_val, b_val = b_val, a_val % b_val
    return a_val

def is_prime(n):
    """True wenn n eine Primzahl ist (n >= 2, ganzzahlig)."""
    n_val = int(_to_tensor(n).float().squeeze().item()) if hasattr(n, "__float__") else int(n)
    if n_val < 2:
        return False
    if n_val == 2:
        return True
    if n_val % 2 == 0:
        return False
    d = 3
    while d * d <= n_val:
        if n_val % d == 0:
            return False
        d += 2
    return True

def mod(a, m):
    """a mod m (positiver Rest). a, m: ganze Zahlen; m > 0."""
    a_val = int(_to_tensor(a).float().squeeze().item()) if hasattr(a, "__float__") else int(a)
    m_val = int(_to_tensor(m).float().squeeze().item()) if hasattr(m, "__float__") else int(m)
    if m_val <= 0:
        raise ValueError("mod: m muss positiv sein.")
    r = a_val % m_val
    return r if r >= 0 else r + m_val

def mod_inv(a, m):
    """Modulares Inverses: x mit (a*x) mod m == 1. Erfordert gcd(a, m) == 1."""
    a_val = int(_to_tensor(a).float().squeeze().item()) if hasattr(a, "__float__") else int(a)
    m_val = int(_to_tensor(m).float().squeeze().item()) if hasattr(m, "__float__") else int(m)
    if m_val <= 0:
        raise ValueError("mod_inv: m muss positiv sein.")
    a_val = a_val % m_val
    if gcd(a_val, m_val) != 1:
        raise ValueError("mod_inv: a und m müssen teilerfremd sein.")
    t, t_new = 0, 1
    r, r_new = m_val, a_val
    while r_new:
        q = r // r_new
        t, t_new = t_new, t - q * t_new
        r, r_new = r_new, r - q * r_new
    return t % m_val if t < 0 else t

def mod_pow(base, exp, m):
    """Modulare Potenz: (base^exp) mod m. Effizient (Square-and-Multiply)."""
    base_val = int(_to_tensor(base).float().squeeze().item()) if hasattr(base, "__float__") else int(base)
    exp_val = int(_to_tensor(exp).float().squeeze().item()) if hasattr(exp, "__float__") else int(exp)
    m_val = int(_to_tensor(m).float().squeeze().item()) if hasattr(m, "__float__") else int(m)
    if m_val <= 0:
        raise ValueError("mod_pow: m muss positiv sein.")
    base_val = base_val % m_val
    result = 1
    while exp_val:
        if exp_val % 2:
            result = (result * base_val) % m_val
        exp_val //= 2
        base_val = (base_val * base_val) % m_val
    return result

def factorial(n):
    """
    Fakultät: n! = n * (n-1) * ... * 2 * 1.
    n: Integer oder Tensor (elementweise).
    Nutzt Gamma-Funktion: n! = gamma(n+1).
    Für negative Zahlen: ValueError.
    """
    t = _to_tensor(n)
    # Konvertiere zu Float für Gamma-Funktion
    t_float = t.float()
    # Prüfe auf negative Werte
    if torch.any(t_float < 0):
        raise ValueError("factorial: Argument muss nicht-negativ sein.")
    # n! = gamma(n+1)
    result = gamma(t_float + 1.0)
    # Wenn Eingabe Integer war, runde Ergebnis (Gamma gibt Float zurück)
    if isinstance(n, (int, float)) and not isinstance(n, bool):
        return float(result.item()) if result.numel() == 1 else result
    return result

# --- Standard Library: Reduktionen (min, max, argmin, argmax) ---
def min(x, dim=None):
    """
    Minimum der Elemente. x: Tensor oder Skalar.
    dim: optional, Achse entlang der reduziert wird (None = über alle).
    Rückgabe: Skalar wenn dim=None; sonst (value, index) bei dim gesetzt — hier einheitlich Tensor.
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.min()
    return t.min(dim=dim)[0]

def max(x, dim=None):
    """
    Maximum der Elemente. x: Tensor oder Skalar.
    dim: optional, Achse (None = über alle). Rückgabe: Skalar oder Tensor.
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.max()
    return t.max(dim=dim)[0]

def argmin(x, dim=None):
    """
    Index des Minimums. x: Tensor. dim: optional (None = abgeflacht).
    Rückgabe: Long-Tensor (Skalar wenn dim=None).
    """
    t = _to_tensor(x).float()
    if dim is None:
        return torch.argmin(t)
    return torch.argmin(t, dim=dim)

def argmax(x, dim=None):
    """
    Index des Maximums. x: Tensor. dim: optional (None = abgeflacht).
    Rückgabe: Long-Tensor (Skalar wenn dim=None).
    """
    t = _to_tensor(x).float()
    if dim is None:
        return torch.argmax(t)
    return torch.argmax(t, dim=dim)

# --- Standard Library: Statistik (mean, std, var, median, quantile, percentile) ---
def mean(x, dim=None):
    """
    Arithmetischer Mittelwert. x: Tensor oder Skalar.
    dim: optional, Achse (None = über alle Elemente). Rückgabe: Skalar oder Tensor.
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.mean()
    return t.mean(dim=dim)

def std(x, dim=None, unbiased=True):
    """
    Standardabweichung. x: Tensor. dim: optional (None = über alle).
    unbiased: True = Stichprobe (N-1), False = Population (N). Rückgabe: Skalar oder Tensor.
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.std(unbiased=unbiased)
    return t.std(dim=dim, unbiased=unbiased)

def var(x, dim=None, unbiased=True):
    """
    Varianz. x: Tensor. dim: optional (None = über alle).
    unbiased: True = Stichprobe (N-1), False = Population (N). Rückgabe: Skalar oder Tensor.
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.var(unbiased=unbiased)
    return t.var(dim=dim, unbiased=unbiased)

def median(x, dim=None):
    """
    Median der Elemente. x: Tensor. dim: optional (None = über alle).
    Bei dim gesetzt: Rückgabe wie min/max nur die Werte (Median pro Achse).
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.median()
    return t.median(dim=dim).values

def quantile(x, q, dim=None):
    """
    Quantil(e). x: Tensor. q: Skalar oder Tensor von Quantilen (0..1).
    dim: optional (None = über alle). Rückgabe: Tensor mit Quantilswerten.
    """
    t = _to_tensor(x).float()
    q_t = _to_tensor(q).float() if not isinstance(q, (int, float)) else torch.tensor(float(q), device=t.device, dtype=t.dtype)
    return torch.quantile(t, q_t, dim=dim)

def percentile(x, p, dim=None):
    """
    Perzentil(e). x: Tensor. p: Skalar oder Tensor von Perzentilen (0..100).
    dim: optional (None = über alle). Rückgabe: Tensor mit Perzentilwerten.
    """
    t = _to_tensor(x).float()
    if isinstance(p, (int, float)):
        return torch.quantile(t, float(p) / 100.0, dim=dim)
    q_t = _to_tensor(p).float() / 100.0
    return torch.quantile(t, q_t, dim=dim)

def cov(x, y=None, unbiased=True):
    """
    Kovarianz. x: 1D-Vektor oder 2D-Matrix (Zeilen = Beobachtungen).
    Wenn y fehlt und x 2D: Kovarianzmatrix (Variablen x Variablen).
    Wenn x, y 1D: Skalar-Kovarianz cov(x, y). unbiased: True = (N-1), False = N.
    """
    t = _to_tensor(x).float().flatten() if _to_tensor(x).dim() == 1 else _to_tensor(x).float()
    if y is None:
        if t.dim() == 1:
            raise ValueError("cov: Bei einem Argument muss x 2D (Zeilen = Beobachtungen) sein.")
        n = t.shape[0]
        c = (n - 1) if unbiased and n > 1 else n
        centered = t - t.mean(dim=0)
        return (centered.T @ centered) / c
    # zwei 1D-Vektoren
    tx, ty = _to_tensor(x).float().flatten(), _to_tensor(y).float().flatten()
    if tx.numel() != ty.numel():
        raise ValueError("cov: x und y müssen gleiche Länge haben.")
    n = tx.numel()
    c = (n - 1) if unbiased and n > 1 else n
    return ((tx - tx.mean()) * (ty - ty.mean())).sum() / c

def corrcoef(x, y=None):
    """
    Korrelationskoeffizient(en). x: 1D oder 2D (Zeilen = Beobachtungen).
    Wenn y fehlt und x 2D: Korrelationsmatrix. Wenn x, y 1D: Skalar r_xy.
    """
    t = _to_tensor(x).float()
    if y is None:
        if t.dim() == 1:
            raise ValueError("corrcoef: Bei einem Argument muss x 2D sein.")
        c = cov(t, unbiased=True)
        if c.dim() == 0:
            return torch.tensor(1.0, device=c.device, dtype=c.dtype)
        std = torch.sqrt(torch.diag(c)).clamp(min=1e-12)
        return c / (std.unsqueeze(1) * std.unsqueeze(0))
    tx, ty = _to_tensor(x).float().flatten(), _to_tensor(y).float().flatten()
    if tx.numel() != ty.numel():
        raise ValueError("corrcoef: x und y müssen gleiche Länge haben.")
    c = cov(tx, ty, unbiased=True)
    sx, sy = tx.std(unbiased=True), ty.std(unbiased=True)
    if sx < 1e-12 or sy < 1e-12:
        return torch.tensor(0.0 if c.item() == 0 else float("nan"), device=c.device, dtype=c.dtype)
    return c / (sx * sy)

def skew(x, dim=None, unbiased=True):
    """
    Schiefe (third standardized moment). x: Tensor. dim: optional (None = über alle).
    unbiased: True = Stichproben-Schiefe (Anpassung für kleine N). Rückgabe: Skalar oder Tensor.
    """
    t = _to_tensor(x).float()
    m = t.mean(dim=dim, keepdim=dim is not None)
    if dim is not None:
        m = m.squeeze(dim)
    c = t - m if dim is None else t - m.unsqueeze(dim)
    n = c.numel() if dim is None else t.shape[dim]
    s = c.std(unbiased=unbiased, dim=dim)
    s = s.clamp(min=1e-12)
    m3 = (c ** 3).mean(dim=dim)
    return m3 / (s ** 3)

def kurtosis(x, dim=None, unbiased=True, excess=True):
    """
    Kurtosis (fourth standardized moment). x: Tensor. dim: optional (None = über alle).
    unbiased: Stichproben-Anpassung. excess: True = Überschuss-Kurtosis (Normalverteilung = 0). Rückgabe: Skalar oder Tensor.
    """
    t = _to_tensor(x).float()
    m = t.mean(dim=dim, keepdim=dim is not None)
    if dim is not None:
        m = m.squeeze(dim)
    c = t - m if dim is None else t - m.unsqueeze(dim)
    s = c.std(unbiased=unbiased, dim=dim).clamp(min=1e-12)
    m4 = (c ** 4).mean(dim=dim)
    k = m4 / (s ** 4)
    return k - 3.0 if excess else k

def histogram(x, bins=10, range_lim=None):
    """
    Histogramm: Zählt Werte in Klassen. x: 1D-Tensor (oder flach).
    bins: Anzahl Klassen (int) oder 1D-Tensor mit Klassengrenzen (aufsteigend).
    range_lim: (min, max) nur wenn bins int; sonst ignoriert. (Nicht 'range' wg. Built-in.)
    Rückgabe: (counts, bin_edges); counts 1D, bin_edges Länge = len(counts)+1.
    """
    t = _to_tensor(x).float().flatten()
    if t.numel() == 0:
        raise ValueError("histogram: x darf nicht leer sein.")
    if isinstance(bins, (int, float)):
        n_bins = _builtin_max(1, int(bins))
        if range_lim is not None:
            low, high = float(range_lim[0]), float(range_lim[1])
        else:
            low, high = t.min().item(), t.max().item()
            if low == high:
                low, high = low - 0.5, high + 0.5
        edges = torch.linspace(low, high, n_bins + 1, device=t.device, dtype=t.dtype)
    else:
        edges = _to_tensor(bins).float().flatten()
        if edges.numel() < 2:
            raise ValueError("histogram: bins als Kanten müssen mindestens 2 Werte haben.")
        n_bins = edges.numel() - 1
    # Bin i = [edges[i], edges[i+1]). searchsorted(edges, t, side='right') - 1 = Bin-Index; clamp für Randwerte.
    idx = torch.searchsorted(edges, t, side="right") - 1
    idx = idx.clamp(0, n_bins - 1)
    counts = torch.zeros(n_bins, device=t.device, dtype=torch.long)
    for i in range(n_bins):
        counts[i] = (idx == i).sum()
    return counts.float(), edges

# --- Standard Library: Runden ---
def round(x):
    """Rundet auf nächste ganze Zahl; x Tensor oder Skalar."""
    return torch.round(_to_tensor(x).float())

def floor(x):
    """Ganzzahlig nach unten; x Tensor oder Skalar."""
    return torch.floor(_to_tensor(x).float())

def ceil(x):
    """Ganzzahlig nach oben; x Tensor oder Skalar."""
    return torch.ceil(_to_tensor(x).float())

def clip(x, min_val=None, max_val=None):
    """
    Begrenzt Werte auf [min_val, max_val]. x: Tensor. min_val/max_val: Skalar oder None (keine Grenze).
    Mindestens eine Grenze angeben. Rückgabe: Tensor gleicher Form.
    """
    t = _to_tensor(x).float()
    if min_val is None and max_val is None:
        return t
    min_t = _to_tensor(min_val).float().squeeze() if min_val is not None else None
    max_t = _to_tensor(max_val).float().squeeze() if max_val is not None else None
    return torch.clamp(t, min=min_t, max=max_t)

# --- Standard Library: Lineare Algebra (Norm, Det, Spur) ---
def norm(x, p=None, dim=None):
    """
    Vektor- oder Matrixnorm. x: Tensor.
    p: optional, Art der Norm (2 = L2/Frobenius default; "fro" für Frobenius; 1, inf möglich).
    dim: optional, Achse(n) für Norm (None = über alle).
    """
    t = _to_tensor(x).float()
    if p is None:
        p = 2
    if dim is not None:
        return torch.linalg.norm(t, ord=p, dim=dim)
    return torch.linalg.norm(t, ord=p)

def det(A):
    """Determinante der Matrix A (2D-Tensor)."""
    t = _to_tensor(A).float()
    if t.dim() != 2:
        raise ValueError("det: Erwartet 2D-Matrix.")
    return torch.linalg.det(t)

def trace(A):
    """Spur der Matrix A (Summe der Diagonalelemente)."""
    t = _to_tensor(A).float()
    if t.dim() != 2:
        raise ValueError("trace: Erwartet 2D-Matrix.")
    return torch.trace(t)

def solve(A, b):
    """
    Lineares Gleichungssystem: löst A x = b. A: (n,n)-Matrix, b: (n,) oder (n,k).
    Rückgabe: x mit gleicher Form wie b.
    """
    A_t = _to_tensor(A).float()
    b_t = _to_tensor(b).float()
    if A_t.dim() != 2 or b_t.dim() not in (1, 2):
        raise ValueError("solve: A muss 2D-Matrix sein, b Vektor oder Matrix.")
    if b_t.dim() == 1:
        b_t = b_t.unsqueeze(1)
    x = torch.linalg.solve(A_t, b_t)
    return x.squeeze(-1) if b_t.shape[-1] == 1 else x

def eigh(A):
    """
    Eigenwerte und -vektoren einer symmetrischen (oder hermiteschen) Matrix A.
    Rückgabe: (eigenvalues, eigenvectors); eigenvalues 1D, eigenvectors Spalten = Eigenvektoren.
    """
    A_t = _to_tensor(A).float()
    if A_t.dim() != 2:
        raise ValueError("eigh: Erwartet 2D-Matrix.")
    evals, evecs = torch.linalg.eigh(A_t)
    return evals, evecs

def eig(A):
    """
    Eigenwerte und -vektoren einer allgemeinen Matrix A.
    Rückgabe: (eigenvalues, eigenvectors); kann komplex sein bei reeller Matrix.
    """
    A_t = _to_tensor(A).float()
    if A_t.dim() != 2:
        raise ValueError("eig: Erwartet 2D-Matrix.")
    evals, evecs = torch.linalg.eig(A_t)
    return evals, evecs

def svd(A, full_matrices=True):
    """
    Singulärwertzerlegung: A = U @ diag(S) @ Vh.
    Rückgabe: (U, S, Vh). S: Singulärwerte (1D); U, Vh: unitäre Matrizen.
    full_matrices: True = volle U/Vh, False = reduzierte Form.
    """
    A_t = _to_tensor(A).float()
    if A_t.dim() != 2:
        raise ValueError("svd: Erwartet 2D-Matrix.")
    U, S, Vh = torch.linalg.svd(A_t, full_matrices=full_matrices)
    return U, S, Vh

def lstsq(A, y, rcond=None):
    """
    Least Squares: minimiert ||A x - y||. A: (m,n), y: (m,) oder (m,k).
    Rückgabe: Lösung x, Form (n,) oder (n,k).
    """
    A_t = _to_tensor(A).float()
    y_t = _to_tensor(y).float()
    if A_t.dim() != 2 or y_t.dim() not in (1, 2):
        raise ValueError("lstsq: A muss 2D-Matrix sein, y Vektor oder Matrix.")
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    result = torch.linalg.lstsq(
        A_t, y_t, rcond=rcond if rcond is not None else 1e-15
    )
    x = result.solution
    return x.squeeze(-1) if x.shape[-1] == 1 else x

def cond(A, p=None):
    """
    Konditionszahl der Matrix A (bezüglich Norm p).
    p: optional, "fro" oder 2 (default), "nuc", inf, -inf, 1, -1.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2:
        raise ValueError("cond: Erwartet 2D-Matrix.")
    p_val = p if p is not None else 2
    return torch.linalg.cond(t, p=p_val)

def rank(A, tol=None):
    """
    Numerischer Rang der Matrix A (Anzahl Singulärwerte > tol).
    tol: optional; wenn None, wird ein vernünftiger Default verwendet.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2:
        raise ValueError("rank: Erwartet 2D-Matrix.")
    if tol is not None:
        return torch.linalg.matrix_rank(t, tol=float(tol))
    return torch.linalg.matrix_rank(t)

def pinv(A, rcond=None):
    """
    Moore-Penrose-Pseudo-Inverse von A. A: (m,n); Rückgabe (n,m).
    rcond: optional; Singulärwerte unter rcond*max(S) werden weggelassen.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2:
        raise ValueError("pinv: Erwartet 2D-Matrix.")
    r = rcond if rcond is not None else 1e-15
    return torch.linalg.pinv(t, rcond=r)

def expm(A):
    """
    Matrix-Exponential: exp(A) = I + A + A^2/2! + ... .
    A: quadratische Matrix; Rückgabe: gleiche Form.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("expm: Erwartet quadratische 2D-Matrix.")
    return torch.linalg.matrix_exp(t)

def logm(A):
    """
    Matrix-Logarithmus: logm(A) so dass expm(logm(A)) = A.
    A: quadratische Matrix; Rückgabe kann komplex sein bei nicht-positiven Eigenwerten.
    Implementierung über Eigenwertzerlegung.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("logm: Erwartet quadratische 2D-Matrix.")
    evals, evecs = torch.linalg.eig(t)
    log_evals = torch.log(evals)
    # evecs: Spalten = Eigenvektoren; A = evecs @ diag(evals) @ inv(evecs)
    evecs_inv = torch.linalg.inv(evecs)
    diag_log = torch.diag(log_evals)
    return evecs @ diag_log @ evecs_inv

def interp(x, xp, fp):
    """
    1D-lineare Interpolation: Werte x anhand Stützstellen (xp, fp).
    x: Stellen, an denen interpoliert wird (Tensor oder Liste).
    xp, fp: Stützstellen (monoton steigend xp). Rückgabe: Tensor gleicher Form wie x.
    """
    import numpy as np  # type: ignore[reportMissingImports]
    x_t = _to_tensor(x).float().flatten()
    xp_t = _to_tensor(xp).float().flatten()
    fp_t = _to_tensor(fp).float().flatten()
    if xp_t.numel() != fp_t.numel():
        raise ValueError("interp: xp und fp müssen gleiche Länge haben.")
    x_np = x_t.detach().cpu().numpy()
    xp_np = xp_t.detach().cpu().numpy()
    fp_np = fp_t.detach().cpu().numpy()
    out_np = np.interp(x_np, xp_np, fp_np)
    return torch.tensor(out_np, dtype=torch.float32, device=x_t.device)

def trapz(y, x=None):
    """
    Trapez-Integration für diskrete Daten: int y dx.
    y: Ordinaten (Tensor 1D); x: optional, Abszissen (1D, gleiche Länge wie y).
    Wenn x fehlt, äquidistante Abstände 1. Rückgabe: Skalar.
    """
    import numpy as np  # type: ignore[reportMissingImports]
    y_t = _to_tensor(y).float().flatten()
    y_np = y_t.detach().cpu().numpy()
    if x is not None:
        x_t = _to_tensor(x).float().flatten()
        if x_t.numel() != y_t.numel():
            raise ValueError("trapz: x und y müssen gleiche Länge haben.")
        x_np = x_t.detach().cpu().numpy()
        result = np.trapz(y_np, x_np)
    else:
        result = np.trapz(y_np)
    return torch.tensor(float(result), dtype=torch.float32)

def root_bisect(f, a, b, tol=1e-8, max_iter=200):
    """
    Nullstelle von f im Intervall [a, b] (Bisektion). f(a) und f(b) müssen unterschiedliche Vorzeichen haben.
    f: Callable mit einem Skalar; Rückgabe Skalar.
    tol: Abbruch wenn |b-a| < tol. max_iter: maximale Schrittzahl.
    Rückgabe: Näherung der Nullstelle (Python float).
    """
    a_val = float(_to_tensor(a).float().squeeze().item())
    b_val = float(_to_tensor(b).float().squeeze().item())
    fa, fb = f(a_val), f(b_val)
    if fa * fb > 0:
        raise ValueError("root_bisect: f(a) und f(b) müssen unterschiedliche Vorzeichen haben.")
    for _ in range(max_iter):
        c = (a_val + b_val) / 2.0
        if (b_val - a_val) / 2.0 < tol:
            return c
        fc = f(c)
        if fc == 0:
            return c
        if fa * fc < 0:
            b_val, fb = c, fc
        else:
            a_val, fa = c, fc
    return (a_val + b_val) / 2.0

def qr(A):
    """
    QR-Zerlegung: A = Q @ R. A: (m,n)-Matrix.
    Rückgabe: (Q, R). Q: (m,m) orthogonal, R: (m,n) obere Dreiecksmatrix.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2:
        raise ValueError("qr: Erwartet 2D-Matrix.")
    Q, R = torch.linalg.qr(t)
    return Q, R

def cholesky(A):
    """
    Cholesky-Zerlegung: A = L @ L.T für symmetrische positiv definite A.
    Rückgabe: L (untere Dreiecksmatrix).
    """
    t = _to_tensor(A).float()
    if t.dim() != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("cholesky: Erwartet quadratische 2D-Matrix.")
    return torch.linalg.cholesky(t)

def lu(A):
    """
    LU-Zerlegung (mit Zeilenpivot): P @ A = L @ U.
    Rückgabe: (P, L, U). P: Permutationsmatrix, L: untere, U: obere Dreiecksmatrix.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("lu: Erwartet quadratische 2D-Matrix.")
    P, L, U = torch.linalg.lu(t)
    return P, L, U

def matrix_power(A, n):
    """
    Matrix-Potenz A^n (n ganzzahlig). A: quadratische Matrix.
    """
    t = _to_tensor(A).float()
    n_int = int(n)
    if t.dim() != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("matrix_power: Erwartet quadratische 2D-Matrix.")
    return torch.linalg.matrix_power(t, n_int)

def kron(A, B):
    """
    Kronecker-Produkt A ⊗ B. A: (m,n), B: (p,q) → (m*p, n*q).
    """
    a_t = _to_tensor(A).float()
    b_t = _to_tensor(B).float()
    return torch.kron(a_t, b_t)

def outer(a, b):
    """
    Äußeres Produkt a ⊗ b. a, b: 1D-Vektoren → Matrix (len(a), len(b)).
    """
    a_t = _to_tensor(a).float().flatten()
    b_t = _to_tensor(b).float().flatten()
    return torch.outer(a_t, b_t)

def vander(x, n=None):
    """
    Vandermonde-Matrix: Zeile i = [x_i^0, x_i^1, ..., x_i^(n-1)].
    x: 1D-Tensor. n: Spaltenanzahl (Default: len(x)); n=None → len(x).
    Rückgabe: (len(x), n) Matrix.
    """
    x_t = _to_tensor(x).float().flatten()
    if n is None:
        n = x_t.numel()
    n_int = int(n)
    return torch.linalg.vander(x_t, N=n_int)

def matrix_sqrt(A):
    """
    Matrix-Quadratwurzel: B mit B @ B = A. A: quadratische positiv semidefinite Matrix.
    Implementierung über Eigenwertzerlegung.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("matrix_sqrt: Erwartet quadratische 2D-Matrix.")
    evals, evecs = torch.linalg.eigh(t)
    if (evals < -1e-10).any():
        raise ValueError("matrix_sqrt: Matrix muss positiv semidefinit sein.")
    evals_sqrt = torch.clamp(evals, min=0.0).sqrt()
    return evecs @ torch.diag(evals_sqrt) @ evecs.T

def matrix_norm(A, ord=None):
    """
    Matrix-Norm. A: 2D-Tensor.
    ord: "fro" (Frobenius), "nuc" (nuklear), 2 (Spektralnorm), inf, -inf, 1, -1.
    Default: Frobenius.
    """
    t = _to_tensor(A).float()
    if t.dim() != 2:
        raise ValueError("matrix_norm: Erwartet 2D-Matrix.")
    return torch.linalg.norm(t, ord=ord if ord is not None else "fro")

def cdist(X, Y, p=2):
    """
    Paarweise Abstände. X: (n, d), Y: (m, d). p: Norm (2 = euklidisch, 1 = Manhattan).
    Rückgabe: (n, m) Tensor mit Abständen.
    """
    X_t = _to_tensor(X).float()
    Y_t = _to_tensor(Y).float()
    if X_t.dim() != 2 or Y_t.dim() != 2 or X_t.shape[1] != Y_t.shape[1]:
        raise ValueError("cdist: X und Y müssen 2D mit gleicher Spaltenanzahl sein.")
    return torch.cdist(X_t, Y_t, p=p)

def polyfit(x, y, deg):
    """
    Polynom-Anpassung: p(x) = p[0] + p[1]*x + ... + p[deg]*x^deg.
    x, y: 1D-Tensoren gleicher Länge. deg: Polynomgrad.
    Rückgabe: Koeffizienten-Tensor (Länge deg+1), niedrigster Grad zuerst.
    """
    x_t = _to_tensor(x).float().flatten()
    y_t = _to_tensor(y).float().flatten()
    if x_t.numel() != y_t.numel():
        raise ValueError("polyfit: x und y müssen gleiche Länge haben.")
    d = _builtin_max(0, int(deg))
    n = x_t.numel()
    # Vandermonde: Zeile i = [1, x_i, x_i^2, ...]
    rows = []
    for i in range(n):
        row = [x_t[i].pow(k).item() for k in range(d + 1)]
        rows.append(row)
    V = torch.tensor(rows, dtype=torch.float32, device=x_t.device)
    result = torch.linalg.lstsq(
        V, y_t.unsqueeze(1), rcond=None
    )
    return result.solution.squeeze()

def polyval(p, x):
    """
    Polynom auswerten: p[0] + p[1]*x + p[2]*x^2 + ...
    p: 1D-Koeffizienten (niedrigster Grad zuerst). x: Tensor oder Skalar.
    """
    p_t = _to_tensor(p).float().flatten()
    x_t = _to_tensor(x).float()
    out = torch.zeros_like(x_t, dtype=torch.float32)
    for k in range(p_t.numel()):
        out = out + p_t[k] * x_t.pow(k)
    return out

def unique(x, sorted=True):
    """
    Eindeutige Werte. x: Tensor (wird flach gemacht).
    sorted: True = aufsteigend sortiert (Default). Rückgabe: 1D-Tensor.
    """
    t = _to_tensor(x).float().flatten()
    u = torch.unique(t, sorted=sorted)
    return u

def argsort(x, dim=-1, descending=False):
    """
    Indizes, die x sortieren. x: Tensor. dim: Achse (default -1).
    descending: True = absteigend. Rückgabe: Long-Tensor gleicher Form.
    """
    t = _to_tensor(x).float()
    return torch.argsort(t, dim=dim, descending=descending)

def convolve1d(a, v, mode="full"):
    """
    1D-Faltung. a, v: 1D-Tensoren. mode: "full" (default), "same" oder "valid".
    full: Ausgabe Länge len(a)+len(v)-1; same: len(a); valid: len(a)-len(v)+1 (ohne Padding).
    """
    a_t = _to_tensor(a).float().flatten()
    v_t = _to_tensor(v).float().flatten()
    na, nv = a_t.numel(), v_t.numel()
    if nv == 0:
        raise ValueError("convolve1d: Kernel v darf nicht leer sein.")
    # Faltung als conv1d: a als Signal (1, 1, L), v als Kernel (1, 1, K); groups=1.
    a_2 = a_t.unsqueeze(0).unsqueeze(0)  # (1, 1, na)
    v_2 = v_t.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, nv) für Korrelation = Faltung
    padding = nv - 1 if mode == "full" else (nv // 2) if mode == "same" else 0
    out = torch.nn.functional.conv1d(a_2, v_2, padding=padding)
    out = out.squeeze()
    if mode == "valid" and na >= nv:
        pass  # out hat schon Länge na - nv + 1
    return out

def minimize_scalar(f, bounds, tol=1e-6, max_iter=100):
    """
    1D-Minimierung von f im Intervall bounds = (a, b). Golden-Section-Search.
    f: Callable mit einem Skalar; Rückgabe Skalar (oder Tensor mit einem Element).
    Rückgabe: (x_min, f_min) als Python floats.
    """
    a_val = float(_to_tensor(bounds[0]).float().squeeze().item())
    b_val = float(_to_tensor(bounds[1]).float().squeeze().item())
    if a_val >= b_val:
        raise ValueError("minimize_scalar: bounds muss (a, b) mit a < b sein.")
    phi = (1.0 + 5.0 ** 0.5) / 2.0  # golden ratio
    c = b_val - (b_val - a_val) / phi
    d = a_val + (b_val - a_val) / phi
    fc = float(_to_tensor(f(c)).float().squeeze().item())
    fd = float(_to_tensor(f(d)).float().squeeze().item())
    for _ in range(max_iter):
        if (b_val - a_val) < tol:
            x_min = (a_val + b_val) / 2.0
            return x_min, float(_to_tensor(f(x_min)).float().squeeze().item())
        if fc < fd:
            b_val, d, fd = d, c, fc
            c = b_val - (b_val - a_val) / phi
            fc = float(_to_tensor(f(c)).float().squeeze().item())
        else:
            a_val, c, fc = c, d, fd
            d = a_val + (b_val - a_val) / phi
            fd = float(_to_tensor(f(d)).float().squeeze().item())
    x_min = (a_val + b_val) / 2.0
    return x_min, float(_to_tensor(f(x_min)).float().squeeze().item())

def newton(f, x0, tol=1e-8, max_iter=50, h=1e-6):
    """
    Nullstelle per Newton-Verfahren (1D). f: Callable mit einem Skalar.
    x0: Startwert. Numerische Ableitung mit Schrittweite h.
    Rückgabe: Näherung der Nullstelle (Python float).
    """
    x = float(_to_tensor(x0).float().squeeze().item())
    for _ in range(max_iter):
        fx = f(x)
        fx = float(_to_tensor(fx).float().squeeze().item())
        if abs(fx) < tol:
            return x
        df = (f(x + h) - f(x - h)) / (2.0 * h)
        df = float(_to_tensor(df).float().squeeze().item())
        if abs(df) < 1e-14:
            break
        x = x - fx / df
    return x

def minimize(f, x0, method="gd", lr=0.01, steps=500):
    """
    Mehrdimensionale Minimierung von f(x). x0: Startvektor (1D-Tensor oder Liste).
    method: "gd" (Gradient Descent) oder "lbfgs". Rückgabe: (x_opt, f_opt) als Tensor und Skalar.
    """
    x = _to_tensor(x0).float().clone().detach().requires_grad_(True)
    if x.dim() != 1:
        x = x.flatten()
    n_params = x.numel()
    if method == "lbfgs":
        optimizer = torch.optim.LBFGS([x], lr=1.0)
        def closure():
            optimizer.zero_grad()
            out = f(x)
            out = _to_tensor(out).float()
            loss = out.sum() if out.numel() > 1 else out
            loss.backward()
            return loss
        for _ in range(_builtin_min(steps, 20)):
            optimizer.step(closure)
    else:
        optimizer = torch.optim.SGD([x], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            out = f(x)
            out = _to_tensor(out).float()
            loss = out.sum() if out.numel() > 1 else out
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        f_opt = f(x)
        f_opt = _to_tensor(f_opt).float()
        f_val = f_opt.sum().item() if f_opt.numel() > 1 else f_opt.item()
    return x.detach(), f_val

def fsolve(f, x0, tol=1e-8, max_iter=50):
    """
    Nullstelle für Vektor-Funktion f: R^n -> R^n (Newton für Systeme). x0: Startvektor.
    Rückgabe: 1D-Tensor (Näherung der Nullstelle).
    """
    x = _to_tensor(x0).float().clone()
    if x.dim() != 1:
        x = x.flatten()
    for _ in range(max_iter):
        fx = f(x)
        fx = _to_tensor(fx).float().flatten()
        if fx.numel() != x.numel():
            raise ValueError("fsolve: f(x) muss gleiche Länge wie x haben.")
        if torch.linalg.norm(fx).item() < tol:
            return x
        J = jacobian(f, x)
        dx = solve(J, fx.unsqueeze(1)).squeeze()
        x = x - dx
    return x

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
    n_int = _builtin_max(2, int(n))
    x = torch.linspace(a_val, b_val, n_int)
    y = f(x)
    y = _to_tensor(y).float().flatten()
    if y.numel() != n_int:
        raise ValueError("integrate: f(x) muss gleiche Länge wie x haben.")
    dx = (b_val - a_val) / (n_int - 1.0)
    result = (dx / 2.0) * (y[0] + 2.0 * y[1:-1].sum() + y[-1])
    return result.squeeze()

def simpson(y, x=None):
    """
    Simpson-Regel für diskrete Daten: int y dx. y: Ordinaten (1D); x: optional, Abszissen (gleiche Länge).
    Bei ungerader Punktanzahl: volle Simpson 1/3; bei gerader: letztes Intervall Trapez. Rückgabe: Skalar.
    """
    y_t = _to_tensor(y).float().flatten()
    n = y_t.numel()
    if n < 2:
        return y_t.sum()
    if n == 2:
        return trapz(y_t, x)
    if x is not None:
        x_t = _to_tensor(x).float().flatten()
        if x_t.numel() != n:
            raise ValueError("simpson: x und y müssen gleiche Länge haben.")
        h = (x_t[-1] - x_t[0]).item() / (n - 1.0)
    else:
        h = 1.0
    # Simpson 1/3: (h/3)*(y0 + 4*y1 + 2*y2 + 4*y3 + ... + y_n)
    if n % 2 == 1:
        # ungerade n = gerade Anzahl Intervalle
        coeff = torch.ones(n, device=y_t.device, dtype=y_t.dtype)
        coeff[1:-1:2] = 4.0
        coeff[2:-1:2] = 2.0
        s = (h / 3.0) * (y_t * coeff).sum()
    else:
        # gerade n: Simpson für erste n-1 Punkte, Trapez für letztes Intervall
        coeff = torch.ones(n - 1, device=y_t.device, dtype=y_t.dtype)
        coeff[1:-1:2] = 4.0
        coeff[2:-1:2] = 2.0
        h_seg = h * (n - 2) / (n - 1) if x is not None else h
        s = (h_seg / 3.0) * (y_t[:-1] * coeff).sum()
        s = s + (h / (n - 1.0) / 2.0) * (y_t[-2] + y_t[-1])
    return s.squeeze()

# --- Standard Library: Uncertainty Propagation (Gaussian) ---
# Fehlerfortpflanzung für Wissenschaftler: value ± std; Gauß'sche Näherung für +, -, *, /, ^.

class UncertainQuantity:
    """Größe mit Unsicherheit: value ± std. Gauß'sche Fehlerfortpflanzung für +, -, *, /, ^. Längen (m, cm, km, mm, dm) werden automatisch umgerechnet."""
    def __init__(self, value, std=0.0, unit=""):
        self.value = float(value)
        self.std = _builtin_max(0.0, float(std))
        self.unit = str(unit) if unit else ""

    def _compatible_add_sub(self, other):
        if not isinstance(other, UncertainQuantity):
            return False
        if self.unit == other.unit:
            return True
        d1 = _get_dimension(self.unit)
        d2 = _get_dimension(other.unit)
        return d1 is not None and d1 == d2

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError("UncertainQuantity: Kann Zahl nicht zu Größe mit Einheit addieren.")
            return UncertainQuantity(self.value + other, self.std, "")
        if isinstance(other, UncertainQuantity):
            if not self._compatible_add_sub(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}] (gleiche Einheit oder kompatible Dimension).")
            dim = _get_dimension(self.unit)
            if dim is not None and dim == _get_dimension(other.unit):
                ov, os_ = _convert_between_units(other.value, other.std, other.unit, self.unit, dim)
                v = self.value + ov
                s = (self.std ** 2 + os_ ** 2) ** 0.5
                return UncertainQuantity(v, s, self.unit)
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
            if not self._compatible_add_sub(other):
                raise ValueError(f"Einheiten passen nicht: [{self.unit}] vs [{other.unit}] (gleiche Einheit oder kompatible Dimension).")
            dim = _get_dimension(self.unit)
            if dim is not None and dim == _get_dimension(other.unit):
                ov, os_ = _convert_between_units(other.value, other.std, other.unit, self.unit, dim)
                v = self.value - ov
                s = (self.std ** 2 + os_ ** 2) ** 0.5
                return UncertainQuantity(v, s, self.unit)
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
        display_unit = _unit_simplify(self.unit) if self.unit else ""
        u = f" [{display_unit}]" if display_unit else ""
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
    method: 'gd' (Gradient Descent), 'mcmc' (Metropolis-Hastings) oder 'hmc' (Hamiltonian Monte Carlo).
    lr: Lernrate (gd) bzw. Schrittweite (mcmc/hmc). steps: Anzahl Schritte.
    Rückgabe: Tensor der optimalen Parameter (bei gd); bei mcmc/hmc Tensor (steps, *params_shape).
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
    if method == "hmc":
        def log_prior(p):
            return torch.tensor(0.0)
        def log_likelihood(d, p):
            return -loss_fn(p, d)
        return hmc(log_prior, log_likelihood, data_t, params.detach(), steps, step_size=lr, num_leapfrog=10)

    # Gradient Descent
    for _ in range(steps):
        loss = loss_fn(params, data_t)
        if params.grad is not None:
            params.grad.zero_()
        loss.backward()
        with torch.no_grad():
            params.sub_(lr * params.grad)
    return params.detach()

# --- Standard Library: Convenience (Chemie/Biologie) ---
# Quick-Wins: Michaelis-Menten, logistisches Wachstum, Arrhenius, lineare Regression.

def michaelis_menten(S, Vmax, Km):
    """
    Michaelis-Menten: v = Vmax * S / (Km + S).
    S, Vmax, Km: Tensor oder Skalar; differenzierbar.
    """
    S = _to_tensor(S).float()
    Vmax = _to_tensor(Vmax).float()
    Km = _to_tensor(Km).float()
    return Vmax * S / (Km + S)


def logistic(t, r, K, N0):
    """
    Analytische Lösung logistisches Wachstum: N(t) = K / (1 + (K/N0 - 1) * exp(-r*t)).
    t, r, K, N0: Tensor oder Skalar; differenzierbar.
    """
    t = _to_tensor(t).float()
    r = _to_tensor(r).float()
    K = _to_tensor(K).float()
    N0 = _to_tensor(N0).float()
    ratio = K / N0 - 1.0
    return K / (1.0 + ratio * torch.exp(-r * t))


def logistic_growth_dt(N, r, K):
    """
    RHS für ode_solve: dN/dt = r * N * (1 - N/K).
    Nutzung: fn rhs(t, y) { return [logistic_growth_dt(y[0], r, K)] }; ode_solve(rhs, [N0], t).
    """
    N = _to_tensor(N).float()
    r = _to_tensor(r).float()
    K = _to_tensor(K).float()
    return r * N * (1.0 - N / K)


def arrhenius(T, A, Ea):
    """
    Arrhenius-Gleichung: k = A * exp(-Ea / (R*T)).
    T: Temperatur (K), A: Präexponent (1/s), Ea: Aktivierungsenergie (J/mol).
    R = R_gas (universal gas constant). Rückgabe: k (differenzierbar).
    """
    T = _to_tensor(T).float()
    A = _to_tensor(A).float()
    Ea = _to_tensor(Ea).float()
    R_val = float(R_gas.value)  # J/(K*mol)
    return A * torch.exp(-Ea / (R_val * T))


def linear_regression(x, y):
    """
    Lineare Regression: y = slope * x + intercept.
    x, y: 1D-Tensoren oder Listen gleicher Länge.
    Rückgabe: [slope, intercept] (Tensor mit 2 Elementen).
    """
    x_t = _to_tensor(x).float().flatten()
    y_t = _to_tensor(y).float().flatten()
    if x_t.numel() != y_t.numel():
        raise ValueError("linear_regression: x und y müssen gleiche Länge haben.")
    n = x_t.numel()
    if n < 2:
        raise ValueError("linear_regression: mindestens 2 Punkte nötig.")
    params_init = torch.tensor([0.0, 0.0], dtype=torch.float32)
    data = [x_t, y_t]

    def loss(params, data):
        xx, yy = data[0], data[1]
        slope, intercept = params[0], params[1]
        pred = slope * xx + intercept
        return ((pred - yy) ** 2).sum()

    params_opt = fit(loss, params_init, data, method="gd", lr=0.01, steps=500)
    return params_opt  # [slope, intercept]

# --- Standard Library: Chemische Elemente (Atommassen, Ordnungszahlen) ---
# IUPAC-nah (g/mol); häufigste Elemente für Chemie/Biologie.

ATOMIC_MASSES = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.065, "Cl": 35.45, "Ar": 39.948, "K": 39.098,
    "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.63, "As": 74.922, "Se": 78.96, "Br": 79.904, "Kr": 83.798, "Rb": 85.468,
    "Sr": 87.62, "Ag": 107.87, "Cd": 112.41, "I": 126.90, "Ba": 137.33, "Pt": 195.08,
    "Au": 196.97, "Hg": 200.59, "Pb": 207.2, "U": 238.03,
}

ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Ag": 47, "Cd": 48, "I": 53,
    "Ba": 56, "Pt": 78, "Au": 79, "Hg": 80, "Pb": 82, "U": 92,
}


def atomic_mass(symbol):
    """
    Atommasse des Elements (IUPAC-nah) in g/mol.
    symbol: String, z. B. \"H\", \"C\", \"O\", \"Na\".
    Rückgabe: Quantity(value, \"g/mol\").
    """
    s = str(symbol).strip()
    if s not in ATOMIC_MASSES:
        raise ValueError(f"atomic_mass: unbekanntes Element '{s}'. Bekannt: H, C, N, O, S, P, Cl, Na, K, Fe, ...")
    return Quantity(ATOMIC_MASSES[s], "g/mol")


def atomic_number(symbol):
    """
    Ordnungszahl des Elements.
    symbol: String, z. B. \"C\", \"Na\".
    Rückgabe: int (dimensionslos).
    """
    s = str(symbol).strip()
    if s not in ATOMIC_NUMBERS:
        raise ValueError(f"atomic_number: unbekanntes Element '{s}'.")
    return ATOMIC_NUMBERS[s]

def concentration_to_pH(c_M):
    """Konzentration [H+] in mol/L -> pH = -log10(c). c: Skalar oder Tensor."""
    t = _to_tensor(c_M).float()
    return -torch.log10(torch.clamp(t, min=1e-30))

def pH_to_concentration(pH):
    """pH -> [H+] in mol/L = 10^(-pH). pH: Skalar oder Tensor."""
    t = _to_tensor(pH).float()
    return torch.pow(10.0, -t)

# --- Standard Library: Differentialgeometrie (Christoffel, Riemann) ---
def christoffel_symbols(g_func, x, h=1e-5):
    """
    Christoffel-Symbole Gamma^k_ij aus Metrik g_ij(x).
    g_func: Callable(x) -> 2D-Tensor (n,n) Metrik an der Stelle x.
    x: 1D-Tensor oder Liste der Koordinaten.
    h: Schrittweite für numerische Ableitung.
    Rückgabe: 3D-Tensor (n,n,n) mit Gamma[k,i,j] = Gamma^k_ij.
    """
    x_t = _to_tensor(x).float().flatten()
    n = x_t.shape[0]
    g0 = _to_tensor(g_func(x_t)).float()
    if g0.dim() != 2 or g0.shape[0] != g0.shape[1] or g0.shape[0] != n:
        raise ValueError("christoffel_symbols: g_func(x) muss (n,n)-Matrix liefern, n = len(x).")
    g_inv = torch.linalg.inv(g0)
    Gamma = torch.zeros(n, n, n, device=g0.device, dtype=g0.dtype)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                s = 0.0
                for m in range(n):
                    x_plus = x_t.clone()
                    x_plus[j] = x_plus[j] + h
                    g_plus = _to_tensor(g_func(x_plus)).float()
                    dg_im_j = (g_plus[i, m].item() - g0[i, m].item()) / h
                    x_plus2 = x_t.clone()
                    x_plus2[i] = x_plus2[i] + h
                    g_plus2 = _to_tensor(g_func(x_plus2)).float()
                    dg_jm_i = (g_plus2[j, m].item() - g0[j, m].item()) / h
                    x_plus3 = x_t.clone()
                    x_plus3[m] = x_plus3[m] + h
                    g_plus3 = _to_tensor(g_func(x_plus3)).float()
                    dg_ij_m = (g_plus3[i, j].item() - g0[i, j].item()) / h
                    s += 0.5 * g_inv[k, m].item() * (dg_im_j + dg_jm_i - dg_ij_m)
                Gamma[k, i, j] = s
    return Gamma

def riemann_tensor(g_func, x, h=1e-5):
    """
    Riemann-Tensor R^a_bcd aus Metrik g(x).
    g_func: Callable(x) -> 2D-Tensor Metrik. x: Koordinaten.
    Rückgabe: 4D-Tensor (n,n,n,n) mit R[a,b,c,d].
    """
    Gamma = christoffel_symbols(g_func, x, h)
    n = Gamma.shape[0]
    x_t = _to_tensor(x).float().flatten()
    R = torch.zeros(n, n, n, n, device=Gamma.device, dtype=Gamma.dtype)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    dG_ac_d = 0.0
                    dG_ad_c = 0.0
                    if d < n:
                        x_plus = x_t.clone()
                        x_plus[d] = x_plus[d] + h
                        G_plus = christoffel_symbols(g_func, x_plus, h)
                        dG_ac_d = (G_plus[a, c, b].item() - Gamma[a, c, b].item()) / h
                    if c < n:
                        x_plus = x_t.clone()
                        x_plus[c] = x_plus[c] + h
                        G_plus = christoffel_symbols(g_func, x_plus, h)
                        dG_ad_c = (G_plus[a, d, b].item() - Gamma[a, d, b].item()) / h
                    term1 = sum(Gamma[a, e, c].item() * Gamma[e, b, d].item() for e in range(n))
                    term2 = sum(Gamma[a, e, d].item() * Gamma[e, b, c].item() for e in range(n))
                    R[a, b, c, d] = dG_ac_d - dG_ad_c + term1 - term2
    return R

def covariant_derivative(T, g_func, x, h=1e-5):
    """
    Kovariante Ableitung eines Tensorfelds T in Richtung der Koordinaten.
    T: Callable(x) -> Tensor (beliebiger Rang). g_func: Metrik. x: Koordinaten.
    Rückgabe: Kovariante Ableitung (gleicher Rang wie T).
    Vereinfacht für Skalarfeld: Nabla_i f = d_i f (partielle Ableitung).
    """
    x_t = _to_tensor(x).float().flatten()
    T0 = _to_tensor(T(x_t)).float()
    n = x_t.shape[0]
    if T0.dim() == 0:
        grad = torch.zeros(n, device=T0.device, dtype=T0.dtype)
        for i in range(n):
            x_plus = x_t.clone()
            x_plus[i] = x_plus[i] + h
            grad[i] = (_to_tensor(T(x_plus)).float().item() - T0.item()) / h
        return grad
    raise NotImplementedError("covariant_derivative: nur für Skalarfelder implementiert.")

# --- Standard Library: Stöchiometrie ---
def _parse_formula(s):
    """Parst Summenformel wie H2O, CaCO3 -> Dict Element -> Anzahl."""
    import re
    d = {}
    for m in re.finditer(r"([A-Z][a-z]?)(\d*)", s):
        elem, num = m.group(1), m.group(2) or "1"
        d[elem] = d.get(elem, 0) + int(num)
    return d

def balance_equation(reactants_str, products_str):
    """
    Stöchiometrische Koeffizienten für ausgeglichene Reaktionsgleichung.
    reactants_str: String wie \"H2 + O2\" (Edukte, + getrennt).
    products_str: String wie \"H2O\" (Produkte).
    Rückgabe: (coeff_reactants, coeff_products) als Listen; z. B. ([2, 1], [2]) für 2*H2 + O2 -> 2*H2O.
    """
    r_parts = [p.strip() for p in str(reactants_str).split("+") if p.strip()]
    p_parts = [p.strip() for p in str(products_str).split("+") if p.strip()]
    r_formulas = [_parse_formula(p) for p in r_parts]
    p_formulas = [_parse_formula(p) for p in p_parts]
    all_elems = set()
    for d in r_formulas + p_formulas:
        all_elems.update(d.keys())
    all_elems = sorted(all_elems)
    n_r, n_p = len(r_formulas), len(p_formulas)
    A = [[d.get(e, 0) for d in r_formulas] + [-d.get(e, 0) for d in p_formulas] for e in all_elems]
    import numpy as np  # type: ignore[reportMissingImports]
    A_np = np.array(A, dtype=float)
    _, _, vh = np.linalg.svd(A_np)
    null_vec = vh[-1]
    if np.allclose(np.abs(null_vec), 0):
        raise ValueError("balance_equation: Reaktion nicht ausbalancierbar.")
    if np.min(null_vec) <= 0:
        null_vec = -null_vec
    min_pos = np.min(null_vec[null_vec > 0.001]) if np.any(null_vec > 0.001) else 1.0
    scaled = null_vec / min_pos
    scaled_int = np.round(scaled * 1000).astype(int)
    g = scaled_int[0]
    for x in scaled_int:
        g = gcd(g, abs(x))
    coeffs = [int(_builtin_max(1, abs(x // g))) for x in scaled_int]
    return list(coeffs[:n_r]), list(coeffs[n_r:])

# --- Standard Library: File I/O, Network, JSON ---

def read_file(path):
    """
    Liest eine Datei als Text (UTF-8).
    path: String (Dateipfad).
    Rückgabe: String (Inhalt).
    """
    p = str(path)
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(path, content):
    """
    Schreibt Text in eine Datei (UTF-8); überschreibt bei Existenz.
    path: String (Dateipfad). content: String (Inhalt).
    """
    p = str(path)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(str(content))


def file_exists(path):
    """
    Prüft, ob eine Datei existiert.
    path: String (Dateipfad). Rückgabe: bool.
    """
    import os
    return os.path.isfile(str(path))


def http_get(url):
    """
    HTTP GET-Anfrage; gibt Antworttext (UTF-8) zurück.
    url: String (z. B. \"https://example.com\").
    """
    import urllib.request
    req = urllib.request.Request(str(url), method='GET')
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8')


def http_post(url, data):
    """
    HTTP POST-Anfrage. data: String (Body) oder Dict/List (wird als JSON gesendet).
    url: String. Rückgabe: Antworttext (UTF-8).
    """
    import urllib.request
    import json
    u = str(url)
    if isinstance(data, dict) or isinstance(data, list):
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(u, data=body, method='POST', headers={'Content-Type': 'application/json'})
    else:
        body = str(data).encode('utf-8')
        req = urllib.request.Request(u, data=body, method='POST')
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8')


def json_parse(s):
    """
    Parst einen JSON-String zu einem Objekt (Dict/List); für Zugriff z. B. obj[\"key\"].
    s: String (gültiger JSON).
    """
    import json
    return json.loads(str(s))


def json_stringify(obj):
    """
    Wandelt ein Objekt (Dict, List, Zahl, String) in einen JSON-String um.
    """
    import json
    if hasattr(obj, 'tolist'):
        obj = obj.tolist()
    elif hasattr(obj, 'item'):
        obj = obj.item()
    return json.dumps(obj, ensure_ascii=False)

# --- Standard Library: Assert (Testing) ---

def _dedekind_assert(condition, message=None):
    """
    Prüft eine Bedingung; löst AssertionError aus, wenn condition falsch ist.
    assert(condition) oder assert(condition, "Fehlermeldung").
    """
    val = condition
    if hasattr(val, "item"):
        val = val.item() if val.numel() == 1 else bool(val.all().item())
    elif hasattr(val, "__len__") and len(val) == 1:
        val = val[0]
    if not bool(val):
        raise AssertionError(message if message is not None else "Assertion failed")

# --- Standard Library: Symbolische Ableitung (eingebettet, kein Import) ---
# Einfacher AST für Ausdrücke
class _SymExpr:
    pass

class _SymVar(_SymExpr):
    def __init__(self, name: str):
        self.name = name

class _SymConst(_SymExpr):
    def __init__(self, value: float):
        self.value = value

class _SymAdd(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymSub(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymMul(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymDiv(_SymExpr):
    def __init__(self, left: _SymExpr, right: _SymExpr):
        self.left = left
        self.right = right

class _SymPow(_SymExpr):
    def __init__(self, base: _SymExpr, exp: _SymExpr):
        self.base = base
        self.exp = exp

class _SymNeg(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymSin(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymCos(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymTan(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymExp(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymLog(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

class _SymSqrt(_SymExpr):
    def __init__(self, arg: _SymExpr):
        self.arg = arg

def _sym_tokenize(s):
    """Tokenizer ohne re: Zeichen für Zeichen."""
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in " \t":
            i += 1
            continue
        if c in "+":
            tokens.append(("PLUS", "+"))
            i += 1
            continue
        if c in "-":
            tokens.append(("MINUS", "-"))
            i += 1
            continue
        if c in "*":
            tokens.append(("MUL", "*"))
            i += 1
            continue
        if c in "/":
            tokens.append(("DIV", "/"))
            i += 1
            continue
        if c in "^":
            tokens.append(("POW", "^"))
            i += 1
            continue
        if c in "(":
            tokens.append(("LPAREN", "("))
            i += 1
            continue
        if c in ")":
            tokens.append(("RPAREN", ")"))
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            start = i
            if s[i] == ".":
                i += 1
            while i < n and s[i].isdigit():
                i += 1
            if i < n and s[i] == ".":
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
            if i < n and s[i] in "eE":
                i += 1
                if i < n and s[i] in "+-":
                    i += 1
                while i < n and s[i].isdigit():
                    i += 1
            try:
                value = float(s[start:i])
            except ValueError:
                raise ValueError("Ungültige Zahl: " + s[start:i])
            tokens.append(("NUMBER", value))
            continue
        if c.isalpha() or c == "_":
            start = i
            while i < n and (s[i].isalnum() or s[i] == "_"):
                i += 1
            tokens.append(("ID", s[start:i]))
            continue
        raise ValueError("Unerwartetes Zeichen: " + repr(c))
    return tokens

class _SymParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _advance(self):
        if self.pos >= len(self.tokens):
            return None
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _parse_expr(self) -> _SymExpr:
        left = self._parse_term()
        while True:
            p = self._peek()
            if p is None:
                break
            if p[0] == "PLUS":
                self._advance()
                left = _SymAdd(left, self._parse_term())
            elif p[0] == "MINUS":
                self._advance()
                left = _SymSub(left, self._parse_term())
            else:
                break
        return left

    def _parse_term(self) -> _SymExpr:
        left = self._parse_factor()
        while True:
            p = self._peek()
            if p is None:
                break
            if p[0] == "MUL":
                self._advance()
                left = _SymMul(left, self._parse_factor())
            elif p[0] == "DIV":
                self._advance()
                left = _SymDiv(left, self._parse_factor())
            else:
                break
        return left

    def _parse_factor(self) -> _SymExpr:
        base = self._parse_unary()
        p = self._peek()
        if p is not None and p[0] == "POW":
            self._advance()
            exp = self._parse_factor()
            return _SymPow(base, exp)
        return base

    def _parse_unary(self) -> _SymExpr:
        p = self._peek()
        if p is None:
            raise ValueError("Unerwartetes Ende des Ausdrucks")
        if p[0] == "PLUS":
            self._advance()
            return self._parse_unary()
        if p[0] == "MINUS":
            self._advance()
            return _SymNeg(self._parse_unary())
        return self._parse_atom()

    def _parse_atom(self) -> _SymExpr:
        p = self._advance()
        if p is None:
            raise ValueError("Unerwartetes Ende des Ausdrucks")
        kind, value = p
        if kind == "NUMBER":
            return _SymConst(float(value))
        if kind == "ID":
            name = str(value)
            if name in ("sin", "cos", "tan", "exp", "log", "sqrt"):
                lp = self._peek()
                if lp is not None and lp[0] == "LPAREN":
                    self._advance()
                    arg = self._parse_expr()
                    rp = self._advance()
                    if rp is None or rp[0] != "RPAREN":
                        raise ValueError("Fehlende ')' nach " + name)
                    if name == "sin":
                        return _SymSin(arg)
                    if name == "cos":
                        return _SymCos(arg)
                    if name == "tan":
                        return _SymTan(arg)
                    if name == "exp":
                        return _SymExp(arg)
                    if name == "log":
                        return _SymLog(arg)
                    if name == "sqrt":
                        return _SymSqrt(arg)
            return _SymVar(name)
        if kind == "LPAREN":
            e = self._parse_expr()
            rp = self._advance()
            if rp is None or rp[0] != "RPAREN":
                raise ValueError("Fehlende ')'")
            return e
        raise ValueError(f"Unerwartetes Token: {p}")

def _sym_parse(expr_str):
    s = expr_str.strip().replace(" ", "")
    if not s:
        raise ValueError("Leerer Ausdruck")
    tokens = _sym_tokenize(s)
    if not tokens:
        raise ValueError("Keine gültigen Tokens")
    p = _SymParser(tokens)
    e = p._parse_expr()
    if p.pos < len(tokens):
        raise ValueError("Rest nach Ausdruck")
    return e

def _sym_diff(e, var):
    if isinstance(e, _SymVar):
        return _SymConst(1.0) if e.name == var else _SymConst(0.0)
    if isinstance(e, _SymConst):
        return _SymConst(0.0)
    if isinstance(e, _SymAdd):
        return _SymAdd(_sym_diff(e.left, var), _sym_diff(e.right, var))
    if isinstance(e, _SymSub):
        return _SymSub(_sym_diff(e.left, var), _sym_diff(e.right, var))
    if isinstance(e, _SymMul):
        return _SymAdd(
            _SymMul(_sym_diff(e.left, var), e.right),
            _SymMul(e.left, _sym_diff(e.right, var))
        )
    if isinstance(e, _SymDiv):
        return _SymDiv(
            _SymSub(_SymMul(_sym_diff(e.left, var), e.right), _SymMul(e.left, _sym_diff(e.right, var))),
            _SymPow(e.right, _SymConst(2.0))
        )
    if isinstance(e, _SymPow):
        if isinstance(e.exp, _SymConst):
            c = e.exp.value
            if c == 0:
                return _SymConst(0.0)
            if c == 1:
                return _sym_diff(e.base, var)
            return _SymMul(_SymConst(c), _SymMul(_SymPow(e.base, _SymConst(c - 1)), _sym_diff(e.base, var)))
        return _SymMul(
            _SymPow(e.base, e.exp),
            _SymAdd(
                _SymMul(_sym_diff(e.exp, var), _SymLog(e.base)),
                _SymMul(e.exp, _SymDiv(_sym_diff(e.base, var), e.base))
            )
        )
    if isinstance(e, _SymNeg):
        return _SymNeg(_sym_diff(e.arg, var))
    if isinstance(e, _SymSin):
        return _SymMul(_SymCos(e.arg), _sym_diff(e.arg, var))
    if isinstance(e, _SymCos):
        return _SymMul(_SymNeg(_SymSin(e.arg)), _sym_diff(e.arg, var))
    if isinstance(e, _SymTan):
        return _SymMul(_SymAdd(_SymConst(1.0), _SymPow(_SymTan(e.arg), _SymConst(2.0))), _sym_diff(e.arg, var))
    if isinstance(e, _SymExp):
        return _SymMul(_SymExp(e.arg), _sym_diff(e.arg, var))
    if isinstance(e, _SymLog):
        return _SymMul(_SymDiv(_SymConst(1.0), e.arg), _sym_diff(e.arg, var))
    if isinstance(e, _SymSqrt):
        return _SymMul(_SymDiv(_SymConst(1.0), _SymMul(_SymConst(2.0), _SymSqrt(e.arg))), _sym_diff(e.arg, var))
    raise TypeError(f"Unbekannter Ausdruckstyp: {type(e)}")

def _sym_to_string(e, parent_prec=0):
    if isinstance(e, _SymVar):
        return e.name
    if isinstance(e, _SymConst):
        v = e.value
        if v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(e, _SymAdd):
        s = _sym_to_string(e.left, 1) + " + " + _sym_to_string(e.right, 1)
        return f"({s})" if parent_prec > 0 else s
    if isinstance(e, _SymSub):
        s = _sym_to_string(e.left, 1) + " - " + _sym_to_string(e.right, 2)
        return f"({s})" if parent_prec > 0 else s
    if isinstance(e, _SymMul):
        left = _sym_to_string(e.left, 2)
        right = _sym_to_string(e.right, 2)
        if isinstance(e.left, (_SymAdd, _SymSub)):
            left = f"({left})"
        if isinstance(e.right, (_SymAdd, _SymSub)):
            right = f"({right})"
        s = f"{left}*{right}"
        return f"({s})" if parent_prec > 1 else s
    if isinstance(e, _SymDiv):
        num = _sym_to_string(e.left, 2)
        den = _sym_to_string(e.right, 2)
        if isinstance(e.left, (_SymAdd, _SymSub)):
            num = f"({num})"
        if isinstance(e.right, (_SymAdd, _SymSub, _SymMul)):
            den = f"({den})"
        s = f"{num}/{den}"
        return f"({s})" if parent_prec > 1 else s
    if isinstance(e, _SymPow):
        base = _sym_to_string(e.base, 3)
        exp = _sym_to_string(e.exp, 3)
        if isinstance(e.base, (_SymAdd, _SymSub, _SymMul, _SymDiv)):
            base = f"({base})"
        s = f"{base}^{exp}"
        return f"({s})" if parent_prec > 2 else s
    if isinstance(e, _SymNeg):
        return "-" + _sym_to_string(e.arg, 3)
    if isinstance(e, _SymSin):
        return "sin(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymCos):
        return "cos(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymTan):
        return "tan(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymExp):
        return "exp(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymLog):
        return "log(" + _sym_to_string(e.arg, 0) + ")"
    if isinstance(e, _SymSqrt):
        return "sqrt(" + _sym_to_string(e.arg, 0) + ")"
    return "?"

def _sym_simplify(e):
    if isinstance(e, _SymConst):
        return e
    if isinstance(e, _SymVar):
        return e
    if isinstance(e, _SymAdd):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(l, _SymConst) and l.value == 0:
            return r
        if isinstance(r, _SymConst) and r.value == 0:
            return l
        return _SymAdd(l, r)
    if isinstance(e, _SymSub):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(r, _SymConst) and r.value == 0:
            return l
        return _SymSub(l, r)
    if isinstance(e, _SymMul):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(l, _SymConst):
            if l.value == 0:
                return _SymConst(0.0)
            if l.value == 1:
                return r
        if isinstance(r, _SymConst):
            if r.value == 0:
                return _SymConst(0.0)
            if r.value == 1:
                return l
        return _SymMul(l, r)
    if isinstance(e, _SymDiv):
        l, r = _sym_simplify(e.left), _sym_simplify(e.right)
        if isinstance(l, _SymConst) and l.value == 0:
            return _SymConst(0.0)
        return _SymDiv(l, r)
    if isinstance(e, _SymPow):
        b, x = _sym_simplify(e.base), _sym_simplify(e.exp)
        if isinstance(x, _SymConst):
            if x.value == 0:
                return _SymConst(1.0)
            if x.value == 1:
                return b
        return _SymPow(b, x)
    if isinstance(e, _SymNeg):
        a = _sym_simplify(e.arg)
        if isinstance(a, _SymConst):
            return _SymConst(-a.value)
        return _SymNeg(a)
    if isinstance(e, (_SymSin, _SymCos, _SymTan, _SymExp, _SymLog, _SymSqrt)):
        a = _sym_simplify(e.arg)
        return type(e)(a)
    return e

def diff_sym(expr, var):
    """
    Symbolische Ableitung: leitet den Ausdruck expr nach der Variable var ab.
    expr: String, z.B. 'x^2 + sin(x)', 'exp(x)*log(x)'.
    var: Name der Variable, z.B. 'x'.
    Rückgabe: Ableitung als String.
    """
    expr = str(expr).strip()
    var = str(var).strip()
    if not var:
        raise ValueError("Variable darf nicht leer sein")
    ast = _sym_parse(expr)
    d = _sym_diff(ast, var)
    d = _sym_simplify(d)
    return _sym_to_string(d)

# --- Standard Library: Jacobian / Hessian (Autograd) ---

def jacobian(f, x):
    """
    Jacobi-Matrix von f an der Stelle x. f: R^n -> R^m (Callable, Tensor -> Tensor).
    x: 1D-Tensor der Länge n. Rückgabe: Tensor (m, n); Zeile i = Gradient von f_i.
    """
    x_t = _to_tensor(x).float().clone().detach().requires_grad_(True)
    out = f(x_t)
    out = _to_tensor(out).float()
    if out.dim() == 0:
        out = out.unsqueeze(0)
    out_flat = out.flatten()
    m, n = out_flat.numel(), x_t.numel()
    J = torch.zeros(m, n, device=x_t.device, dtype=x_t.dtype)
    for i in range(m):
        grad_out = torch.zeros_like(out_flat)
        grad_out[i] = 1.0
        g, = torch.autograd.grad(out_flat, x_t, grad_outputs=grad_out, retain_graph=True)
        J[i] = g.flatten()
    return J

def hessian(f, x):
    """
    Hesse-Matrix von f an der Stelle x. f: R^n -> R (skalar).
    x: 1D-Tensor der Länge n. Rückgabe: Tensor (n, n).
    """
    x_t = _to_tensor(x).float().clone().detach().requires_grad_(True)
    out = f(x_t)
    out = _to_tensor(out).float()
    if out.dim() != 0:
        out = out.sum()
    grad_x, = torch.autograd.grad(out, x_t, create_graph=True)
    n = x_t.numel()
    H = torch.zeros(n, n, device=x_t.device, dtype=x_t.dtype)
    for i in range(n):
        g, = torch.autograd.grad(grad_x[i], x_t, retain_graph=True)
        H[i] = g.flatten()
    return H

# --- Standard Library: Sorting ---

def sort(data, descending=False):
    data = _to_tensor(data)
    sorted_values, _ = torch.sort(data, descending=descending)
    return sorted_values

def quicksort(data):
    return sort(data)

# --- Standard Library: Visualization ---
# Plots werden in _dedekind_plots gesammelt und vom Server an die IDE zurückgegeben.

_dedekind_plots = []

def _plot_ndarray(x, y=None, title=None, xlabel=None, ylabel=None, kind="line", xscale="linear", yscale="linear"):
    """Intern: Erzeugt einen Plot und hängt ihn als Base64-PNG an _dedekind_plots. kind: line, scatter."""
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
    import numpy as np  # type: ignore[reportMissingImports]
    if getattr(x, 'dtype', None) is not None and np.issubdtype(x.dtype, np.complexfloating):
        x = np.real(x)
    if getattr(y, 'dtype', None) is not None and np.issubdtype(y.dtype, np.complexfloating):
        y = np.real(y)
    fig, ax = plt.subplots()
    if kind == "scatter":
        ax.scatter(x, y)
    else:
        ax.plot(x, y)
    if xscale == "log": ax.set_xscale("log")
    if yscale == "log": ax.set_yscale("log")
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    _dedekind_plots.append(base64.b64encode(png_bytes).decode('ascii'))
    try:
        import sys
        for i in range(1, 8):
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            g = f.f_globals
            if '_dedekind_display_image' in g:
                g['_dedekind_display_image'](png_bytes, 'image/png')
                break
        else:
            from IPython import get_ipython  # type: ignore[import-untyped]
            ip = get_ipython()
            if ip is not None:
                from IPython.display import display, Image  # type: ignore[import-untyped]
                display(Image(data=png_bytes))
    except Exception:
        pass

def _plot_contour_inner(X, Y, Z, title=None, xlabel=None, ylabel=None, levels=10):
    """Intern: Contour-Plot; X, Y, Z als NumPy-Arrays (Z 2D)."""
    try:
        import base64
        import io
        import matplotlib  # type: ignore[import-untyped]
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[reportMissingImports]
    except ImportError:
        print("contour(): matplotlib nicht installiert. pip install matplotlib")
        return
    fig, ax = plt.subplots()
    nlev = int(levels) if levels is not None else 10
    ax.contour(X, Y, Z, levels=nlev)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    png_bytes = buf.getvalue()
    _dedekind_plots.append(base64.b64encode(png_bytes).decode('ascii'))
    try:
        import sys
        for i in range(1, 8):
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            g = f.f_globals
            if '_dedekind_display_image' in g:
                g['_dedekind_display_image'](png_bytes, 'image/png')
                break
        else:
            from IPython import get_ipython  # type: ignore[import-untyped]
            ip = get_ipython()
            if ip is not None:
                from IPython.display import display, Image  # type: ignore[import-untyped]
                display(Image(data=png_bytes))
    except Exception:
        pass

def print_latex(s):
    """
    Gibt LaTeX-String in der Konsole als Unicode-Formel aus (α, Δ, ∫, ½ etc.); keine Bilder in Plots.
    Bsp.: print_latex(r"\\frac{1}{2}"), print_latex(r"\\alpha + \\Delta x").
    Wenn kein Kernel vorhanden, Fallback: print(s).
    """
    s = str(s)
    try:
        import sys
        for i in range(1, 8):
            try:
                f = sys._getframe(i)
            except ValueError:
                break
            g = f.f_globals
            if '_dedekind_display_latex' in g:
                g['_dedekind_display_latex'](s)
                return
        print(s)
    except Exception:
        print(s)


def plot(x=None, y=None, title=None, xlabel=None, ylabel=None, xscale="linear", yscale="linear"):
    """
    Zeichnet Daten (Linie) und zeigt sie in Dedekind Studio an.
    plot(y) – y über Index; plot(x, y) – y über x.
    xscale, yscale: "linear" (default) oder "log".
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
    _plot_ndarray(x, y, title=title, xlabel=xlabel, ylabel=ylabel, kind="line", xscale=xscale, yscale=yscale)

def scatter(x=None, y=None, title=None, xlabel=None, ylabel=None):
    """
    Streudiagramm: Punkte (x, y). scatter(y) – y über Index; scatter(x, y) – Punkte.
    Zeigt in Dedekind Studio an.
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
    _plot_ndarray(x, y, title=title, xlabel=xlabel, ylabel=ylabel, kind="scatter")

def contour(X, Y, Z, title=None, xlabel=None, ylabel=None, levels=10):
    """
    Höhenlinien-Plot. X, Y: 1D- oder 2D-Koordinaten (z. B. linspace); Z: 2D-Matrix (Werte).
    levels: Anzahl Konturlinien (default 10). Zeigt in Dedekind Studio an.
    """
    import numpy as np  # type: ignore[reportMissingImports]
    X_t = _to_tensor(X).float().detach().cpu().numpy()
    Y_t = _to_tensor(Y).float().detach().cpu().numpy()
    Z_t = _to_tensor(Z).float().detach().cpu().numpy()
    if Z_t.ndim != 2:
        raise ValueError("contour: Z muss 2D-Matrix sein.")
    if X_t.ndim == 1 and Y_t.ndim == 1:
        X_t, Y_t = np.meshgrid(X_t, Y_t)
    _plot_contour_inner(X_t, Y_t, Z_t, title=title, xlabel=xlabel, ylabel=ylabel, levels=levels)
