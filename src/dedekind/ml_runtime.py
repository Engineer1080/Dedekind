import builtins
import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]

_builtin_min = builtins.min
_builtin_max = builtins.max

def _normalize_unit_for_compare(unit):
    """Normalizes units for comparison: M, mol/L, mol*L^-1 are considered equal (chemical concentration)."""
    if not unit:
        return unit
    u = str(unit).strip()
    if u in ("M", "mol/L", "mol*L^-1", "mol*L^(-1)"):
        return "M"
    return u


# Units with automatic conversion on +/-: (Base unit, {Unit: Factor to base})
# Value in unit * factor = Value in base unit
DIMENSION_TO_BASE = {
    # SI base
    "length": ("m", {"m": 1.0, "cm": 0.01, "km": 1000.0, "mm": 0.001, "um": 1e-6, "nm": 1e-9, "angstrom": 1e-10, "Angstrom": 1e-10, "pm": 1e-12, "fm": 1e-15, "dm": 0.1, "AU": 149597870700.0, "ly": 9460730472580800.0, "pc": 3.085677581e16}),
    "mass": ("kg", {"kg": 1.0, "g": 0.001, "t": 1000.0, "mg": 1e-6, "ug": 1e-9, "Da": 1.66053906660e-27, "amu": 1.66053906660e-27, "M_sun": 1.98847e30}),
    "time": ("s", {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "yr": 31557600.0, "a": 31557600.0, "ms": 0.001, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15}),
    "current": ("A", {"A": 1.0, "mA": 0.001, "kA": 1000.0, "uA": 1e-6, "muA": 1e-6}),
    "temperature": ("K", {"K": 1.0, "mK": 0.001}),
    "amount_of_substance": ("mol", {"mol": 1.0, "mmol": 0.001, "umol": 1e-6, "nmol": 1e-9, "kmol": 1000.0}),
    "luminous_intensity": ("cd", {"cd": 1.0, "mcd": 0.001}),
    # Derived / common
    "pressure": ("Pa", {"Pa": 1.0, "hPa": 100.0, "kPa": 1000.0, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "atm": 101325.0}),
    "force": ("N", {"N": 1.0, "kN": 1000.0, "MN": 1e6}),
    "permeability": ("m^2", {"m^2": 1.0, "D": 9.869233e-13, "mD": 9.869233e-16}),  # Darcy: 1 D ≈ 9.87e-13 m²
    "volume": ("L", {"L": 1.0, "mL": 0.001, "dm^3": 1.0, "m^3": 1000.0}),  # dm³ = 1 L, m³ = 1000 L
    "energy": ("J", {"J": 1.0, "kJ": 1000.0, "MJ": 1e6, "Wh": 3600.0, "kWh": 3.6e6, "eV": 1.602176634e-19, "meV": 1.602176634e-22, "keV": 1.602176634e-16, "MeV": 1.602176634e-13, "GeV": 1.602176634e-10, "cal": 4.184, "kcal": 4184.0}),
    "electric_potential": ("V", {"V": 1.0, "mV": 0.001, "kV": 1000.0}),
    "frequency": ("Hz", {"Hz": 1.0, "kHz": 1000.0, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}),
    "charge": ("C", {"C": 1.0, "mC": 0.001, "uC": 1e-6}),
    "resistance": ("ohm", {"ohm": 1.0, "kohm": 1000.0, "Mohm": 1e6}),
    "power": ("W", {"W": 1.0, "kW": 1000.0, "MW": 1e6, "L_sun": 3.828e26}),
    "magnetic_flux_density": ("T", {"T": 1.0, "G": 1e-4}),
    "magnetic_flux": ("Wb", {"Wb": 1.0}),
    "inductance": ("H", {"H": 1.0, "mH": 0.001, "uH": 1e-6}),
    "capacitance": ("F", {"F": 1.0, "mF": 0.001, "uF": 1e-6, "muF": 1e-6, "nF": 1e-9, "pF": 1e-12}),
    "amount_concentration": ("M", {"M": 1.0, "mM": 0.001, "uM": 1e-6, "nM": 1e-9}),
    "absorbed_dose": ("Gy", {"Gy": 1.0, "mGy": 0.001}),
    "equivalent_dose": ("Sv", {"Sv": 1.0, "mSv": 0.001}),
    # Chemistry/Biology: mass concentration (% w/v = g/100mL)
    "mass_concentration": ("g/L", {"g/L": 1.0, "mg/mL": 1.0, "percent_wv": 10.0}),  # 1% w/v = 10 g/L
    # Angle (SI supplement: rad; deg = pi/180 rad)
    "angle": ("rad", {"rad": 1.0, "deg": 0.017453292519943295}),  # 1 deg = pi/180 rad
    # Molar energy (MD/Chemistry): Base kJ/mol (1 kcal/mol = 4.184 kJ/mol)
    "molar_energy": ("kJ/mol", {"kJ/mol": 1.0, "kcal/mol": 4.184, "J/mol": 0.001, "eV/atom": 96.485, "Hartree/mol": 2625.5}),
}
# For Units checker: List of unit sets per dimension
ADDITIVE_DIMENSION_UNIT_SETS = [frozenset(tab.keys()) for _b, tab in DIMENSION_TO_BASE.values()]


def _get_dimension(unit):
    """Returns dimension name (length, mass, time, pressure) or None."""
    u = str(unit).strip() if unit else ""
    for dim, (_base, tab) in DIMENSION_TO_BASE.items():
        if u in tab:
            return dim
    return None


def _convert_to_base(value, unit, dimension):
    """Convert value in given unit to base unit."""
    _, tab = DIMENSION_TO_BASE[dimension]
    u = str(unit).strip()
    return float(value) * tab.get(u, 1.0)


def _convert_from_base(value_base, unit, dimension):
    """Convert value in base unit to given unit."""
    _, tab = DIMENSION_TO_BASE[dimension]
    u = str(unit).strip()
    if u not in tab:
        return float(value_base)
    return float(value_base) / tab[u]


def _convert_between_units(value, std, from_unit, to_unit, dimension):
    """Convert value and std from from_unit to to_unit (for same dimension). Return (value, std)."""
    _, tab = DIMENSION_TO_BASE[dimension]
    u_from = str(from_unit).strip()
    u_to = str(to_unit).strip()
    if u_from not in tab or u_to not in tab:
        return float(value), float(std)
    factor = tab[u_from] / tab[u_to]
    return float(value) * factor, float(std) * abs(factor)

class Quantity:
    """Physical quantity with unit (e.g. 10[m], 5[m/s], 0.1[M], 50[ppm]). Calculation rules: same unit for +/-, multiply/divide units. Chemistry: mol, L, M (= mol/L), ppm, bar, atm, g. Radioactivity: Bq (1/s), Gy (J/kg), Sv (J/kg, equivalent dose)."""
    def __init__(self, value, unit=""):
        if type(value).__name__ == "Tensor" or hasattr(value, "requires_grad"):
            self.value = value
        else:
            try:
                self.value = float(value)
            except Exception:
                self.value = value
        self.unit = str(unit) if unit else ""


    def _same_unit(self, other):
        if not isinstance(other, Quantity):
            return False
        return _normalize_unit_for_compare(self.unit) == _normalize_unit_for_compare(other.unit)

    def _add_sub_quantity(self, other, is_add):
        """Addition/subtraction with automatic conversion for the same dimension (length, mass, time, pressure)."""
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
            f"Unit error during {'addition' if is_add else 'subtraction'}: [{self.unit}] vs [{other.unit}]. "
            "Same unit or compatible units of the same dimension (e.g., length, mass, time, pressure, current, temperature, mol, cd, volume, energy, voltage, frequency, charge, resistance, power, angle rad/deg) required."
        )

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.unit:
                raise ValueError(
                    f"Unit error: Cannot add a pure number to a quantity with unit [{self.unit}]. "
                    "For addition, both sides must have the same unit (or be dimensionless)."
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
                    f"Unit error: Cannot subtract a pure number from a quantity with unit [{self.unit}]. "
                    "For subtraction, both sides must have the same unit (or be dimensionless)."
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
        """Returns (self_val, other_val) for comparison. Converts units if possible."""
        if isinstance(other, Quantity):
            dim_s = _get_dimension(self.unit)
            dim_o = _get_dimension(other.unit)
            if dim_s is not None and dim_s == dim_o:
                return _convert_to_base(self.value, self.unit, dim_s), \
                       _convert_to_base(other.value, other.unit, dim_o)
            if _normalize_unit_for_compare(self.unit) == _normalize_unit_for_compare(other.unit):
                return self.value, other.value
            raise ValueError(
                f"Comparison not possible: [{self.unit}] vs [{other.unit}] (different dimension)."
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

    def __getitem__(self, key):
        """Index/slice of a tensor- or list-valued Quantity; unit is preserved.
        Raises TypeError if the value is a scalar (no meaningful indexing)."""
        v = self.value
        try:
            sub = v[key]
        except TypeError:
            raise TypeError(
                f"Quantity value of type {type(v).__name__} cannot be indexed (scalar?)."
            )
        if isinstance(sub, Quantity):
            return sub
        return Quantity(sub, self.unit)


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
    """Unit to the power of exp (e.g. m^2, (m/s)^2)."""
    if not u: return ""
    e = float(exp)
    if abs(e - 1.0) < 1e-12: return u
    if abs(e + 1.0) < 1e-12: return _unit_inv(u)
    base = u if ("*" not in u and "/" not in u) else f"({u})"
    if abs(e - round(e)) < 1e-12:
        return f"{base}^{int(round(e))}"
    return f"{base}^{e}"


def _split_product_top_level(u):
    """Splits unit string at '*' only on the top level (respecting parentheses)."""
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
    """Returns (base, exp) for a factor like 'm', 'm^2', '(m/s)^2'. exp as a number."""
    part = part.strip()
    if "^" in part:
        # Right of the last ^ (not in parentheses) is the exponent
        idx = part.rfind("^")
        base = part[:idx].strip()
        exp_str = part[idx + 1:].strip()
        try:
            exp = int(exp_str) if "." not in exp_str else float(exp_str)
        except ValueError:
            return part, 1
        # Remove parentheses around simple base for uniform representation
        if base.startswith("(") and base.endswith(")") and base.count("(") == 1 and "/" not in base[1:-1]:
            base = base[1:-1].strip()
        return base, exp
    return part, 1


def _collapse_product(u):
    """Collapses identical factors: m*m -> m^2, m*m*m -> m^3, m^2*m -> m^3, m*m*kg -> m^2*kg."""
    if not u or "/" in u:
        return u
    parts = _split_product_top_level(u)
    if len(parts) <= 1:
        return u
    # Parts that are themselves products (e.g. (m*m)), first collapse recursively
    normalized = []
    for p in parts:
        p = p.strip()
        # Strip outer parentheses, then collapse product if applicable
        if p.startswith("(") and p.endswith(")") and p.count("(") == 1:
            p = p[1:-1].strip()
        if "*" in p and "/" not in p:
            p = _collapse_product(p)
        normalized.append(p)
    # Parse each factor as (base, exp) and merge
    merged = {}
    for p in normalized:
        base, exp = _parse_base_exp(p)
        merged[base] = merged.get(base, 0) + exp
    # Output sorted: uniform order
    out = []
    for base in sorted(merged.keys()):
        e = merged[base]
        if e == 1:
            out.append(base)
        else:
            out.append(f"{base}^{int(e)}" if isinstance(e, float) and e == int(e) else f"{base}^{e}")
    return "*".join(out)


def _unit_simplify(u):
    """Simplifies unit string for display. SI base: m, kg, s, A, K, mol, cd. Derived: J, N, Pa, W, Hz, V, F, ohm, S, Wb, T, H, lm, lx, Gy, kat, M. Chemistry/Pressure: bar, atm. Mass: g. Radioactivity: Bq (1/s), Sv (J/kg); Display 1/s→Hz, J/kg→Gy."""
    if not u:
        return u
    u = u.strip()
    # Chemistry: concentration mol/L -> M; ppm stays ppm
    if u in ("mol/L", "mol*L^-1", "mol*L^(-1)", "mol/L"):
        return "M"
    if u.replace(" ", "") in ("mol/L", "mol*L^-1"):
        return "M"
    # 1/s -> Hz (frequency); Bq same dimension, display Hz
    if u in ("1/s", "s^-1", "s^(-1)") or u.replace(" ", "") == "s^-1":
        return "Hz"
    # mol/s -> kat (katal, catalytic activity)
    if u in ("mol/s", "mol*s^-1", "mol*s^(-1)"):
        return "kat"
    # J/kg, m^2/s^2 -> Gy (Gray, absorbed dose); Sv same dimension
    if u in ("J/kg", "m^2/s^2", "(m^2)/(s^2)") or "(m/s)^2" in u:
        return "Gy"
    # Joule: kg*m^2/s^2 or (kg)*((m/s)^2); not J/C (V) or J/s (W)
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
    # Pa (Pascal): N/m^2, kg/(m*s^2) — not N*m^2/C^2
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
    # S (Siemens): 1/ohm, s^3*A^2/(kg*m^2) — check before ohm
    if "s^3*A^2/(kg*m^2)" in u:
        return "S"
    # ohm (Ω): V/A, kg*m^2/(s^3*A^2)
    if u in ("V/A", "V*A^-1", "V*A^(-1)") or "kg*m^2/(s^3*A^2)" in u:
        return "ohm"
    # F (Farad): C/V, s^4*A^2/(kg*m^2)
    if u in ("C/V", "C*V^-1", "C*V^(-1)") or ("s^4*A^2" in u and "kg*m^2" in u):
        return "F"
    # H (Henry): Wb/A, kg*m^2/(s^2*A^2) — check before Wb
    if u in ("Wb/A", "Wb*A^-1") or "kg*m^2/(s^2*A^2)" in u:
        return "H"
    # Wb (Weber): V*s, kg*m^2/(s^2*A) (not A^2)
    if u == "V*s" or ("kg*m^2" in u and "/(s^2*A)" in u and "A^2" not in u):
        return "Wb"
    # T (Tesla): Wb/m^2, kg/(s^2*A)
    if u in ("Wb/m^2", "Wb*m^-2", "kg*s^-2*A^-1") or "kg/(s^2*A)" in u:
        return "T"
    # lm (Lumen): cd*sr (Candela * Steradian)
    if "cd*sr" in u or (u.replace(" ", "") == "cd*sr"):
        return "lm"
    # lx (Lux): lm/m^2
    if u in ("lm/m^2", "lm*m^-2"):
        return "lx"
    # Collapse identical factors: m*m -> m^2, m*m*m -> m^3, m^2*m -> m^3, m*m*kg -> m^2*kg
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
        if isinstance(data.value, torch.Tensor):
            return data.value.float()
        return torch.tensor(data.value, dtype=torch.float32)
    if isinstance(data, UncertainQuantity):
        if isinstance(data.value, torch.Tensor):
            return data.value.float()
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
            try:
                return torch.stack(converted)
            except (TypeError, ValueError, RuntimeError):
                return converted
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
    """Registers a user-defined unit: 1 NAME = factor * base_unit.

    Examples:
        _register_user_unit("Foot", 0.3048, "m")     # 1 Foot = 0.3048 m  (length)
        _register_user_unit("Mile", 1609.34, "m")    # 1 Mile = 1609.34 m  (length)
        _register_user_unit("Darcy", 9.869233e-13, "m^2")  # permeability
        _register_user_unit("MilliFoot", 0.001, "Foot")    # chaining is resolved

    base_unit must already exist in some dimension table (base or alias).
    """
    name = str(name).strip()
    base_unit = str(base_unit).strip()
    factor = float(factor)
    if not name:
        raise ValueError("unit name must not be empty.")
    # Find the dimension of base_unit and perform chain resolution
    target_dim = None
    base_factor = 1.0
    for dim, (_b, tab) in DIMENSION_TO_BASE.items():
        if base_unit in tab:
            target_dim = dim
            base_factor = tab[base_unit]
            break
    if target_dim is None:
        raise ValueError(
            f"`unit {name} = ...[{base_unit}]`: Base unit '{base_unit}' is not assigned to any known dimension "
            f"(length, mass, time, pressure, energy, ...). Please alias to an existing unit."
        )
    DIMENSION_TO_BASE[target_dim][1][name] = factor * base_factor
    _rebuild_additive_unit_sets()
    return name

def _unit_of_value(value):
    """Returns the unit of a value as a string. '' for unitless / numbers / tensors."""
    if isinstance(value, Quantity):
        return value.unit or ""
    if isinstance(value, UncertainQuantity):
        return value.unit or ""
    return ""

def _check_param_unit(value, param_name, fn_name, arg_name, unit_env):
    """Binds a polymorphic unit variable consistently across all arguments of a call.

    First occurrence: binds param_name -> unit of value.
    Subsequent occurrences: must have the same dimension; if needed, value
    is converted to the bound unit.

    Example:
        fn add<U>(a: [U], b: [U]) -> [U] { return a + b }
        add(2[m], 3[m])       # U binds to 'm'
        add(2[m], 100[cm])    # U binds to 'm', 100[cm] is converted to 1[m]
        add(2[m], 3[kg])      # ValueError (dimension mismatch)
    """
    actual = _unit_of_value(value)
    if param_name not in unit_env:
        unit_env[param_name] = actual
        return value
    bound = unit_env[param_name]
    if bound == actual:
        return value
    # Both non-empty + same unit (by string comparison)?
    if _normalize_unit_for_compare(bound) == _normalize_unit_for_compare(actual):
        return value
    # Both non-empty + same dimension? -> convert to bound unit.
    if isinstance(value, Quantity) and bound:
        try:
            return _coerce_to_expected_unit(value, bound, f"{fn_name}({arg_name}) [Typ-Param {param_name}]")
        except ValueError:
            pass
    # Caller gave plain number, binding expects unit
    if bound and not actual and isinstance(value, (int, float)):
        return Quantity(float(value), bound)
    raise ValueError(
        f"Type param unit '{param_name}' in {fn_name}({arg_name}): "
        f"already bound to [{bound or 'unitless'}], got [{actual or 'unitless'}]."
    )

def _check_return_param_unit(value, param_name, fn_name, unit_env):
    """Check / bind polymorphic unit variable at return point."""
    actual = _unit_of_value(value)
    if param_name not in unit_env:
        # Return is the first place that sees param_name — bind without check
        unit_env[param_name] = actual
        return value
    bound = unit_env[param_name]
    if bound == actual:
        return value
    if _normalize_unit_for_compare(bound) == _normalize_unit_for_compare(actual):
        return value
    if isinstance(value, Quantity) and bound:
        try:
            return _coerce_to_expected_unit(value, bound, f"return of {fn_name} [type param {param_name}]")
        except ValueError:
            pass
    if bound and not actual and isinstance(value, (int, float)):
        return Quantity(float(value), bound)
    raise ValueError(
        f"Type param unit '{param_name}' in return of {fn_name}: "
        f"expected [{bound or 'unitless'}], got [{actual or 'unitless'}]."
    )

def _shape_of(value):
    """Returns the shape of a value as a tuple of ints. Scalar -> (), unknown -> None."""
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
        # Consistency of all elements (otherwise irregular -> only first dimension)
        for item in value[1:]:
            if _shape_of(item) != first_sub:
                return (len(value),)
        return (len(value),) + first_sub
    return None

def _format_shape(dims):
    """Returns a compact string representation of a shape list/tuple."""
    parts = [str(d) for d in dims]
    return "[" + ", ".join(parts) + "]"

def _check_shape(value, expected_dims, fn_name, arg_name, shape_env):
    """Checks that `value` has the declared shape `expected_dims`.
    Symbolic dimensions (strings) are bound or compared in `shape_env`.
    Raises ValueError on mismatch. Returns `value` unchanged (passthrough)."""
    actual = _shape_of(value)
    if actual is None:
        # Shape not determinable (e.g. generic iterator) - skip
        return value
    if len(actual) != len(expected_dims):
        raise ValueError(
            f"Shape mismatch in {fn_name}({arg_name}): expected {_format_shape(expected_dims)} "
            f"({len(expected_dims)}-D), got {_format_shape(actual)} ({len(actual)}-D)."
        )
    for i, (want, got) in enumerate(zip(expected_dims, actual)):
        if isinstance(want, int):
            if want != got:
                raise ValueError(
                    f"Shape mismatch in {fn_name}({arg_name}): dimension {i} expected {want}, "
                    f"got {got}. Full shape: expected {_format_shape(expected_dims)}, "
                    f"got {_format_shape(actual)}."
                )
        else:
            # Symbolic dimension: first encounter binds, then check consistency
            bound = shape_env.get(want)
            if bound is None:
                shape_env[want] = got
            elif bound != got:
                raise ValueError(
                    f"Symbolic shape dimension '{want}' in {fn_name}({arg_name}): already "
                    f"bound as {bound}, got {got}. Full shape: expected "
                    f"{_format_shape(expected_dims)}, got {_format_shape(actual)}."
                )
    return value

def _check_return_shape(value, expected_dims, fn_name, shape_env):
    actual = _shape_of(value)
    if actual is None:
        return value
    if len(actual) != len(expected_dims):
        raise ValueError(
            f"Return shape mismatch in {fn_name}: expected {_format_shape(expected_dims)}, "
            f"got {_format_shape(actual)}."
        )
    for i, (want, got) in enumerate(zip(expected_dims, actual)):
        if isinstance(want, int):
            if want != got:
                raise ValueError(
                    f"Return shape mismatch in {fn_name}: dim {i} expected {want}, got {got}."
                )
        else:
            bound = shape_env.get(want)
            if bound is None:
                shape_env[want] = got
            elif bound != got:
                raise ValueError(
                    f"Symbolic return dimension '{want}' in {fn_name}: already {bound}, "
                    f"got {got}."
                )
    return value

def _graph_dims(value):
    """Returns (num_nodes, num_edges) for torch_geometric.data.Data or
    similar objects. None if not determinable."""
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
    """Check (N_nodes, N_edges) tuple against expected dimensions with symbol binding."""
    labels = ("nodes", "edges")
    for i, (want, got, label) in enumerate(zip(expected_dims, actual, labels)):
        if isinstance(want, int):
            if want != got:
                raise ValueError(
                    f"Graph shape mismatch in {where} {fn_name}({arg_name}): "
                    f"{label} count expected {want}, got {got}."
                )
        else:
            bound = shape_env.get(want)
            if bound is None:
                shape_env[want] = got
            elif bound != got:
                raise ValueError(
                    f"Symbolic graph dimension '{want}' in {where} {fn_name}({arg_name}): "
                    f"already bound as {bound}, got {got} ({label})."
                )

def _check_graph_shape(value, expected_dims, fn_name, arg_name, shape_env):
    """Validates that `value` is a torch_geometric.data.Data-like object with the
    declared (N_nodes, N_edges) dimensions. Symbolic dimensions are
    bound or kept consistent in `shape_env`."""
    actual = _graph_dims(value)
    if actual is None:
        raise ValueError(
            f"Graph shape check in {fn_name}({arg_name}): expected a graph object "
            f"with num_nodes/num_edges attributes (e.g. torch_geometric.data.Data), "
            f"got {type(value).__name__}."
        )
    _check_graph_dims_against_env(actual, expected_dims, fn_name, arg_name, shape_env, "Argument")
    return value

def _check_return_graph_shape(value, expected_dims, fn_name, shape_env):
    actual = _graph_dims(value)
    if actual is None:
        raise ValueError(
            f"Graph return shape check in {fn_name}: expected a graph object, "
            f"got {type(value).__name__}."
        )
    _check_graph_dims_against_env(actual, expected_dims, fn_name, "return", shape_env, "Return")
    return value

def _labeled_dims(value):
    """Returns tuple of dim names for xarray.DataArray or DataArray-like
    objects. None if not determinable."""
    if hasattr(value, "dims"):
        try:
            return tuple(str(d) for d in value.dims)
        except Exception:
            pass
    return None

def _check_labeled_shape(value, expected_dims, fn_name, arg_name, shape_env):
    """Validates that `value` is an xarray.DataArray-like object with EXACTLY the
    declared set of dim names (order is irrelevant — xarray operates
    name-based)."""
    actual = _labeled_dims(value)
    if actual is None:
        raise ValueError(
            f"LabeledTensor shape check in {fn_name}({arg_name}): expected a "
            f"DataArray-like object with .dims (e.g. xarray.DataArray), "
            f"got {type(value).__name__}."
        )
    expected_set = set(str(d) for d in expected_dims)
    actual_set = set(actual)
    if expected_set != actual_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        msgs = []
        if missing:
            msgs.append(f"missing dimensions: {sorted(missing)}")
        if extra:
            msgs.append(f"extra dimensions: {sorted(extra)}")
        raise ValueError(
            f"LabeledTensor shape mismatch in {fn_name}({arg_name}): "
            f"expected {{ {', '.join(sorted(expected_set))} }}, "
            f"got {{ {', '.join(sorted(actual_set))} }} ({'; '.join(msgs)})."
        )
    return value

def _check_return_labeled_shape(value, expected_dims, fn_name, shape_env):
    actual = _labeled_dims(value)
    if actual is None:
        raise ValueError(
            f"LabeledTensor return check in {fn_name}: expected DataArray-like "
            f"object, got {type(value).__name__}."
        )
    expected_set = set(str(d) for d in expected_dims)
    actual_set = set(actual)
    if expected_set != actual_set:
        raise ValueError(
            f"LabeledTensor return mismatch in {fn_name}: expected "
            f"{{ {', '.join(sorted(expected_set))} }}, "
            f"got {{ {', '.join(sorted(actual_set))} }}."
        )
    return value

def _validate_sequence_string(value, kind, where):
    """Validates: value is a string and contains only characters from the alphabet."""
    if not isinstance(value, str):
        raise TypeError(
            f"Sequence[{kind}] check in {where}: expected string, got {type(value).__name__}."
        )
    alphabet = _SEQ_ALPHABETS.get(kind.upper())
    if alphabet is None:
        raise ValueError(f"Sequence kind {kind!r} unknown (allowed: DNA, RNA, Protein).")
    upper_val = value.upper()
    bad = [c for c in upper_val if c not in alphabet]
    if bad:
        # First occurrence with position
        idx = next(i for i, c in enumerate(upper_val) if c not in alphabet)
        raise ValueError(
            f"Sequence[{kind}] check in {where}: invalid character "
            f"{value[idx]!r} at position {idx} (allowed: {''.join(sorted(alphabet))})."
        )

def _check_sequence_shape(value, expected_dims, fn_name, arg_name, shape_env):
    kind = str(expected_dims[0])
    _validate_sequence_string(value, kind, f"{fn_name}({arg_name})")
    return value

def _check_return_sequence_shape(value, expected_dims, fn_name, shape_env):
    kind = str(expected_dims[0])
    _validate_sequence_string(value, kind, f"return of {fn_name}")
    return value

def _check_qubit_shape(val, dims, fn_name, arg_name, shape_env):
    """Checks Qubit[N] — expects QuantumCircuit or int."""
    if isinstance(val, QuantumCircuit):
        expected = dims[0] if dims else None
        if expected is not None:
            if isinstance(expected, str):
                if expected in shape_env:
                    if shape_env[expected] != val.n_qubits:
                        raise TypeError(
                            f"{fn_name}(): Argument '{arg_name}' Qubit[{expected}]={shape_env[expected]} "
                            f"but QuantumCircuit has {val.n_qubits} qubits."
                        )
                else:
                    shape_env[expected] = val.n_qubits
            elif val.n_qubits != int(expected):
                raise TypeError(
                    f"{fn_name}(): Argument '{arg_name}' expected Qubit[{expected}], "
                    f"QuantumCircuit has {val.n_qubits} qubits."
                )
    return val

def _check_statevec_shape(val, dims, fn_name, arg_name, shape_env):
    """Checks StateVec[N] — expects list of length 2^N."""
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
                            f"expected length 2^{shape_env[expected_n]}={exp_len}, got {actual_len}."
                        )
                else:
                    import math
                    if actual_len == 0 or (actual_len & (actual_len - 1)) != 0:
                        raise TypeError(
                            f"{fn_name}(): '{arg_name}' StateVec[{expected_n}]: "
                            f"length must be a power of 2, got {actual_len}."
                        )
                    shape_env[expected_n] = int(math.log2(actual_len))
            else:
                exp_len = 1 << int(expected_n)
                if actual_len != exp_len:
                    raise TypeError(
                        f"{fn_name}(): '{arg_name}' expected StateVec[{expected_n}] "
                        f"(length {exp_len}), got {actual_len}."
                    )
    return val

def _check_return_qubit_shape(val, dims, fn_name, shape_env):
    """Checks return Qubit[N]/Circuit[N,G]."""
    _check_qubit_shape(val, dims, fn_name, "return", shape_env)
    return val

def _check_return_statevec_shape(val, dims, fn_name, shape_env):
    """Checks return StateVec[N]."""
    _check_statevec_shape(val, dims, fn_name, "return", shape_env)
    return val

def _pauli(name):
    m = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z, "H": PAULI_H}
    if name not in m:
        raise ValueError(f"Unknown Pauli matrix '{name}'. Allowed: I, X, Y, Z, H.")
    return m[name]

class QuantumCircuit:
    """Lightweight Dedekind quantum circuit (no Qiskit required).

    Stores gates as a list of (name, qubits, params). Can be simulated via
    `statevec_sim()` or exported via `to_qiskit()`.
    """

    def __init__(self, n_qubits: int):
        if not isinstance(n_qubits, int) or n_qubits < 1:
            raise ValueError(f"QuantumCircuit: n_qubits must be >= 1, got {n_qubits!r}.")
        self.n_qubits = n_qubits
        self._gates = []   # [(name, qubit_list, params)]
        self._measurements = []  # [(qubit, clbit)]

    def h(self, qubit: int):
        """Hadamard gate."""
        self._validate_qubit(qubit)
        self._gates.append(("H", [qubit], []))
        return self

    def x(self, qubit: int):
        """Pauli-X (NOT) gate."""
        self._validate_qubit(qubit)
        self._gates.append(("X", [qubit], []))
        return self

    def y(self, qubit: int):
        """Pauli-Y gate."""
        self._validate_qubit(qubit)
        self._gates.append(("Y", [qubit], []))
        return self

    def z(self, qubit: int):
        """Pauli-Z gate."""
        self._validate_qubit(qubit)
        self._gates.append(("Z", [qubit], []))
        return self

    def cx(self, control: int, target: int):
        """CNOT gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError("CNOT: control and target must not be the same.")
        self._gates.append(("CX", [control, target], []))
        return self

    def cz(self, control: int, target: int):
        """CZ gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        self._gates.append(("CZ", [control, target], []))
        return self

    def rx(self, theta, qubit: int):
        """Rx rotation by angle theta (radians)."""
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
        """SWAP gate."""
        self._validate_qubit(q0)
        self._validate_qubit(q1)
        self._gates.append(("SWAP", [q0, q1], []))
        return self

    def t(self, qubit: int):
        """T gate (pi/4 phase)."""
        self._validate_qubit(qubit)
        self._gates.append(("T", [qubit], []))
        return self

    def s(self, qubit: int):
        """S gate (pi/2 phase)."""
        self._validate_qubit(qubit)
        self._gates.append(("S", [qubit], []))
        return self

    def measure(self, qubit: int, clbit: int = None):
        """Adds a measurement."""
        self._validate_qubit(qubit)
        self._measurements.append((qubit, clbit if clbit is not None else qubit))
        return self

    def measure_all(self):
        """Measures all qubits."""
        for i in range(self.n_qubits):
            self.measure(i, i)
        return self

    def _validate_qubit(self, q):
        if not isinstance(q, int) or q < 0 or q >= self.n_qubits:
            raise ValueError(
                f"Qubit index {q} out of range [0, {self.n_qubits - 1}]."
            )

    def depth(self) -> int:
        """Circuit depth (naive: number of gate layers without parallelization)."""
        layers = [0] * self.n_qubits
        for name, qubits, _ in self._gates:
            d = _builtin_max(layers[q] for q in qubits) + 1
            for q in qubits:
                layers[q] = d
        return _builtin_max(layers) if layers else 0

    def n_gates(self) -> int:
        """Total number of gates."""
        return len(self._gates)

    def __repr__(self):
        lines = [f"QuantumCircuit({self.n_qubits} qubits, {self.n_gates()} gates, depth={self.depth()})"]
        for name, qubits, params in self._gates:
            pstr = f"({', '.join(f'{p:.4f}' for p in params)})" if params else ""
            qstr = ", ".join(str(q) for q in qubits)
            lines.append(f"  {name}{pstr}  q[{qstr}]")
        if self._measurements:
            lines.append(f"  MEASURE: {self._measurements}")
        return "\n".join(lines)

    def to_qiskit(self):
        """Converts to a real Qiskit QuantumCircuit (requires `pyimport qiskit`)."""
        try:
            import qiskit
        except ImportError:
            raise ImportError(
                "Qiskit not found. Install it: pip install qiskit\n"
                "Alternatively: statevec_sim(circuit) for pure simulation."
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
def random_vector(size):
    return torch.randn(size)

def random_matrix(rows, cols):
    return torch.randn(rows, cols)

def shuffle(x, dim=0):
    """
    Random shuffle along axis dim. x: Tensor. dim: axis (default 0).
    Returns: Tensor of same shape (permutation along dim). Uses current random state.
    """
    t = _to_tensor(x).clone()
    idx = torch.randperm(t.shape[dim], device=t.device)
    return t.index_select(dim, idx)

def permutation(n):
    """
    Random permutation of indices 0 .. n-1. n: integer. Returns: 1D Long tensor.
    """
    n_int = int(n)
    if n_int < 0:
        raise ValueError("permutation: n must be non-negative.")
    return torch.randperm(n_int)

def choice(a, size=1, replace=True):
    """
    Random sample from a. a: 1D tensor or list. size: number of draws (default 1).
    replace: True = with replacement, False = without. Returns: Tensor of length size.
    """
    a_t = _to_tensor(a).float().flatten()
    n = a_t.numel()
    if n == 0:
        raise ValueError("choice: a must not be empty.")
    size_int = int(size)
    if not replace and size_int > n:
        raise ValueError("choice: size must not exceed len(a) when replace=False.")
    idx = torch.randint(0, n, (size_int,), device=a_t.device) if replace else torch.randperm(n, device=a_t.device)[:size_int]
    return a_t[idx]

def autocorr(x, max_lag=None):
    """
    Autocorrelation (normalized, lag 0 = 1). x: 1D tensor. max_lag: optional (default len(x)-1).
    Returns: 1D tensor of length max_lag+1.
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
    Moving average. x: 1D tensor. window: window width (odd recommended).
    Returns: 1D tensor (length len(x)-window+1); no boundary handling (reduced length).
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
    Cross product (3D). a, b: 1D tensors of length 3. Returns: 1D tensor of length 3.
    """
    a_t = _to_tensor(a).float().flatten()
    b_t = _to_tensor(b).float().flatten()
    if a_t.numel() != 3 or b_t.numel() != 3:
        raise ValueError("cross: a and b must have length 3.")
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
    Frequency bins for FFT. n: number of points (int). d: sample spacing (scalar, default 1).
    Returns: 1D tensor of length n with frequencies (unit 1/d); for interpretation of fft(x).
    """
    n_int = int(n)
    d_val = float(_to_tensor(d).float().squeeze().item()) if d != 1.0 else 1.0
    return torch.fft.fftfreq(n_int, d=d_val)

def diff(x, n=1, dim=-1):
    """
    Discrete derivative (differences): diff(x) = x[1:] - x[:-1].
    x: Tensor. n: order (default 1). dim: axis (default -1). Returns: Tensor (length reduced by n along dim).
    """
    t = _to_tensor(x).float()
    for _ in range(n):
        t = torch.diff(t, dim=dim)
    return t

def cumsum(x, dim=None):
    """
    Cumulative sum along axis. x: Tensor. dim: axis (None = over all, then flat).
    Returns: Tensor of same shape as x (or 1D if dim is None).
    """
    t = _to_tensor(x).float()
    if dim is None:
        return t.flatten().cumsum(0)
    return t.cumsum(dim=dim)

# --- Standard Library: Differentiable ODE Solvers ---

def linspace(start, stop, steps):
    """Creates a 1D tensor with `steps` equally spaced values from start to stop."""
    s = _to_tensor(start).float().squeeze()
    e = _to_tensor(stop).float().squeeze()
    n = int(steps)
    return torch.linspace(float(s.item()) if s.numel() == 1 else float(s.item()),
                          float(e.item()) if e.numel() == 1 else float(e.item()),
                          n)

def logspace(start, stop, steps, base=10.0):
    """Creates a 1D tensor with `steps` logarithmically spaced values from base^start to base^stop."""
    s = _to_tensor(start).float().squeeze()
    e = _to_tensor(stop).float().squeeze()
    b = float(_to_tensor(base).float().squeeze().item())
    n = int(steps)
    return torch.logspace(float(s.item()) if s.numel() == 1 else float(s.item()),
                          float(e.item()) if e.numel() == 1 else float(e.item()),
                          n, base=b)


# --- Mathematical Sequences (arithmetic, geometric, general) ---

def arange(start_or_stop, stop=None, step=None):
    """
    Integer sequence like numpy.arange.
    arange(n) -> [0, 1, 2, ..., n-1]
    arange(start, stop) -> [start, start+1, ..., stop-1]
    arange(start, stop, step) -> [start, start+step, ...] (stop exclusive)

    Return dtype: int64 for pure integer calls (suitable for indexing),
    float32 as soon as an explicit `step` (even integer) is given (consistency
    with `linspace`). Tensor arithmetic promotes automatically when mixed with floats.
    """
    if stop is None and step is None:
        return torch.arange(int(start_or_stop), dtype=torch.int64)
    if step is None:
        return torch.arange(int(start_or_stop), int(stop), dtype=torch.int64)
    return torch.arange(float(start_or_stop), float(stop), float(step))


def arithmetic(a0, d, n):
    """
    Arithmetic sequence: a_n = a0 + n*d for n = 0, 1, ..., n-1.
    a0: start value, d: common difference, n: number of terms.
    Returns: 1D tensor [a0, a0+d, a0+2d, ..., a0+(n-1)d]; differentiable in a0, d.
    """
    a0_t = _to_tensor(a0).float().squeeze()
    d_t = _to_tensor(d).float().squeeze()
    n_val = int(n)
    k = torch.arange(n_val, dtype=torch.float32, device=a0_t.device if a0_t.dim() > 0 else None)
    return a0_t + d_t * k


def geometric(a0, r, n):
    """
    Geometric sequence: a_n = a0 * r^n for n = 0, 1, ..., n-1.
    a0: start value, r: common ratio, n: number of terms.
    Returns: 1D tensor [a0, a0*r, a0*r^2, ..., a0*r^(n-1)]; differentiable in a0, r.
    """
    a0_t = _to_tensor(a0).float().squeeze()
    r_t = _to_tensor(r).float().squeeze()
    n_val = int(n)
    k = torch.arange(n_val, dtype=torch.float32, device=a0_t.device if a0_t.dim() > 0 else None)
    return a0_t * torch.pow(r_t, k)


def sequence(f, n):
    """
    General sequence: [f(0), f(1), ..., f(n-1)].
    f: function with one argument (index n); n: number of terms.
    Usage: fn term(k) { return k * k }; seq = sequence(term, 10)
    Returns: 1D tensor; f must return a scalar or 0D tensor.
    """
    n_val = int(n)
    out = []
    for k in range(n_val):
        val = f(k)
        out.append(_to_tensor(val).float().squeeze())
    return torch.stack(out)




def labeled_tensor(data, dims, coords=None, name=None, attrs=None):
    """Creates an xarray.DataArray for climate/geo/earth science workflows.

    Unlike bare tensors, labeled tensors carry the *names* of their
    axes ('lat', 'lon', 'time', ...), and xarray operations can
    work name-based (instead of index-based). This prevents the classic
    confusion 'mean over lat vs. mean over time'.

    Parameters:
      data:   Tensor/numpy array/list — the values.
      dims:   List/tuple of strings — axis names, same order as data.shape.
      coords: Optional dict {dim_name: 1D values} — coordinate axes.
      name:   Optional name for the DataArray.
      attrs:  Optional dict — metadata (e.g. {"units": "K", "crs": "EPSG:4326"}).

    Returns: xarray.DataArray.
    """
    try:
        import xarray as _xr  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("labeled_tensor requires xarray. Installation: pip install xarray")
    import numpy as _np  # type: ignore[reportMissingImports]
    if hasattr(data, "detach"):
        arr = data.detach().cpu().numpy()
    else:
        arr = _np.asarray(data)
    dims_t = tuple(str(d) for d in dims)
    if len(dims_t) != arr.ndim:
        raise ValueError(
            f"labeled_tensor: data has {arr.ndim} axes, dims has {len(dims_t)} "
            f"({dims_t!r}). Counts must match."
        )
    coords_dict = dict(coords) if coords else {}
    return _xr.DataArray(arr, dims=dims_t, coords=coords_dict if coords_dict else None,
                          name=name, attrs=attrs or {})
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


def lagrange_ode_rhs(L_func):
    """
    Erzeugt die rechte Seite für ode_solve aus einer Lagrangefunktion L(q, v).
    Euler-Lagrange: d/dt(∂L/∂v) = ∂L/∂q  =>  q̈ = (∂L/∂q - ∂²L/∂q∂v·v) / ∂²L/∂v².
    L_func(q, v): Callable; q, v werden als 1D-Tensoren übergeben (Länge 1 für 1D).
    Rückgabe: fun(t, y) mit y = [q, v]; dy/dt = [v, q̈].
    """
    def rhs(t, y):
        y_t = _to_tensor(y).float().flatten()
        if y_t.numel() < 2:
            raise ValueError("lagrange_ode_rhs: y muss mindestens [q, v] (Länge 2) haben.")
        n = y_t.numel() // 2
        q = y_t[:n].clone().detach().requires_grad_(True)
        v = y_t[n:].clone().detach().requires_grad_(True)
        L_val = L_func(q, v)
        L_val = _to_tensor(L_val).float().squeeze()
        if L_val.dim() > 0:
            L_val = L_val.sum()
        dL_dq, = torch.autograd.grad(L_val, q, create_graph=True, allow_unused=True)
        dL_dv, = torch.autograd.grad(L_val, v, create_graph=True, allow_unused=True)
        dL_dq = dL_dq if dL_dq is not None else torch.zeros_like(q)
        dL_dv = dL_dv if dL_dv is not None else torch.zeros_like(v)
        d2L_dv2 = torch.autograd.grad(dL_dv.sum(), v, retain_graph=True, allow_unused=True)[0]
        d2L_dqdv = torch.autograd.grad(dL_dv.sum(), q, retain_graph=True, allow_unused=True)[0]
        d2L_dv2 = d2L_dv2 if d2L_dv2 is not None else torch.ones_like(v)
        d2L_dqdv = d2L_dqdv if d2L_dqdv is not None else torch.zeros_like(q)
        denom = d2L_dv2
        if (denom.abs() < 1e-12).any():
            raise ValueError("lagrange_ode_rhs: ∂²L/∂v² zu klein (singulär). Typisch bei L = T - V mit T = ½mv².")
        a = (dL_dq - d2L_dqdv * v) / denom
        return torch.cat([v.detach(), a.detach()])
    return rhs


def hamilton_ode_rhs(H_func):
    """
    Erzeugt die rechte Seite für ode_solve aus einer Hamiltonfunktion H(q, p).
    Hamilton-Gleichungen: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.
    H_func(q, p): Callable; q, p als 1D-Tensoren.
    Rückgabe: fun(t, y) mit y = [q, p]; dy/dt = [∂H/∂p, -∂H/∂q].
    """
    def rhs(t, y):
        y_t = _to_tensor(y).float().flatten()
        if y_t.numel() < 2:
            raise ValueError("hamilton_ode_rhs: y muss mindestens [q, p] (Länge 2) haben.")
        n = y_t.numel() // 2
        q = y_t[:n].clone().detach().requires_grad_(True)
        p = y_t[n:].clone().detach().requires_grad_(True)
        H_val = H_func(q, p)
        H_val = _to_tensor(H_val).float().squeeze()
        if H_val.dim() > 0:
            H_val = H_val.sum()
        dH_dq, = torch.autograd.grad(H_val, q)
        dH_dp, = torch.autograd.grad(H_val, p)
        return torch.cat([dH_dp.detach(), -dH_dq.detach()])
    return rhs


def lotka_volterra(x0, y0, a, b, c, d, t):
    """
    Lotka-Volterra Räuber-Beute-Modell: dx/dt = a*x - b*x*y, dy/dt = -c*y + d*x*y.
    x0, y0: Anfangsbestände (Beute, Räuber). a, b, c, d: Parameter.
    t: Zeitgitter (1D). Rückgabe: (len(t), 2) – erste Spalte Beute, zweite Räuber.
    """
    def rhs(_, y):
        x, y_pred = y[0], y[1]
        dx = a * x - b * x * y_pred
        dy = -c * y_pred + d * x * y_pred
        return torch.stack([dx, dy])
    y0_t = torch.tensor([float(x0), float(y0)], dtype=torch.float32)
    return ode_solve(rhs, y0_t, t)


def chemical_equilibrium(K, n_A, n_B, n_C, n_D, A0, B0, C0, D0, tol=1e-10):
    """
    Chemisches Gleichgewicht für aA + bB <-> cC + dD (Massenwirkungsgesetz).
    K: Gleichgewichtskonstante; n_A, n_B, n_C, n_D: Stöchiometriekoeffizienten (a, b, c, d).
    A0, B0, C0, D0: Anfangskonzentrationen [mol/L].
    Reaktionslaufzahl ξ: [A]=A0-n_A*ξ, [B]=B0-n_B*ξ, [C]=C0+n_C*ξ, [D]=D0+n_D*ξ.
    K = ([C]^n_C * [D]^n_D) / ([A]^n_A * [B]^n_B). Rückgabe: (A_eq, B_eq, C_eq, D_eq).
    """
    K = float(K)
    n_A, n_B, n_C, n_D = float(n_A), float(n_B), float(n_C), float(n_D)
    A0, B0, C0, D0 = float(A0), float(B0), float(C0), float(D0)
    xi_max = _builtin_min(A0 / n_A if n_A > 0 else 1e10, B0 / n_B if n_B > 0 else 1e10)
    xi_max = _builtin_max(0.0, _builtin_min(xi_max, 1e10))

    def residual(xi):
        A = A0 - n_A * xi
        B = B0 - n_B * xi
        C = C0 + n_C * xi
        D = D0 + n_D * xi
        if A <= 0 or B <= 0:
            return 1e10
        lhs = (C ** n_C) * (D ** n_D)
        rhs = K * (A ** n_A) * (B ** n_B)
        return lhs - rhs

    xi_lo, xi_hi = 0.0, xi_max
    if residual(xi_lo) * residual(xi_hi) > 0:
        xi_eq = 0.0
    else:
        for _ in range(200):
            xi_mid = (xi_lo + xi_hi) / 2.0
            if (xi_hi - xi_lo) / 2.0 < tol:
                xi_eq = xi_mid
                break
            if residual(xi_mid) * residual(xi_lo) < 0:
                xi_hi = xi_mid
            else:
                xi_lo = xi_mid
        else:
            xi_eq = (xi_lo + xi_hi) / 2.0
    A_eq = _builtin_max(0.0, A0 - n_A * xi_eq)
    B_eq = _builtin_max(0.0, B0 - n_B * xi_eq)
    C_eq = C0 + n_C * xi_eq
    D_eq = D0 + n_D * xi_eq
    return (A_eq, B_eq, C_eq, D_eq)


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

# --- Navier-Stokes 2D (inkompressibel): Chorin-Projektionsmethode ---

def pde_navier_stokes_2d(u0, v0, x, y, t, nu, bc="periodic",
                          obstacle_mask=None, eta=None, rho=1.0,
                          body_force_x=0.0, body_force_y=0.0):
    """
    Differenzierbarer 2D-Navier-Stokes-Solver (inkompressibel): u_t + (u·∇)u = -∇p + ν∇²u, ∇·u = 0.
    Chorin-Projektionsmethode: 1) Prädiktor (Konvektion + Diffusion), 2) Druck-Poisson (FFT), 3) Projektion.
    u0, v0: Anfangsgeschwindigkeiten 2D (nx, ny); x, y: Ortsgitter; t: Zeitgitter; nu: kinematische Viskosität.
    bc: 'periodic' (default; FFT für Druck-Poisson). 'dirichlet' experimentell (Jacobi-Iteration).

    Immersed-Boundary-Methode (Brinkman-Penalisierung): wenn `obstacle_mask`
    (Tensor (nx, ny) in [0, 1]) übergeben wird, wird im Prädiktor ein
    implizites Penalty-Forcing `-η·M·u` eingebracht, das die Geschwindigkeit
    im Solid (M=1) zu Null treibt. Der Penalty-Schritt ist unbedingt stabil
    (implizit), η = `eta` Parameter (default: 1/dt → Solid-Zeitskala = Schritt).
    Nach der Projektion wird im Solid u = 0 erzwungen (direkte IBM-Erzwingung),
    um Druckkorrektur-Drift zu verhindern.
    Drag/Lift auf das Hindernis per Newtons drittem Gesetz aus der
    Penalty-Reaktion: F = ρ·η·Σ M·u_star·damping · dx·dy (pro Tiefeneinheit, 2D).
    `rho` ist die physikalische Dichte für die Force-Skalierung (default 1.0).

    Rückgabe ohne Hindernis: (u_sol, v_sol, p_sol).
    Rückgabe mit Hindernis:  (u_sol, v_sol, p_sol, F_history) wobei
    F_history Shape (len(t), 2) hat — [F_x, F_y] pro Zeitschritt.
    body_force_x/y: konstante Körperkraft (m/s²) für Strömungsantrieb in periodischen Setups.
    CFL: dt·max(|u|)/dx < 1 empfohlen.
    """
    u0 = _to_tensor(u0).float()
    v0 = _to_tensor(v0).float().to(u0.device)
    if u0.dim() == 1 or v0.dim() == 1:
        raise ValueError("pde_navier_stokes_2d: u0 und v0 müssen 2D-Gitter (nx, ny) sein.")
    nx, ny = u0.shape[0], u0.shape[1]
    if v0.shape != (nx, ny):
        raise ValueError("pde_navier_stokes_2d: u0 und v0 müssen gleiche Form haben.")
    x = _to_tensor(x).float().flatten().to(u0.device)
    y = _to_tensor(y).float().flatten().to(u0.device)
    t = _to_tensor(t).float().flatten().to(u0.device)
    nu_val = float(_to_tensor(nu).float().item())
    if x.numel() != nx or y.numel() != ny:
        raise ValueError("pde_navier_stokes_2d: x/y Länge muss zu u0 passen.")
    if nx < 3 or ny < 3:
        raise ValueError("pde_navier_stokes_2d: mindestens 3×3 Gitterpunkte.")
    dx = float((x[-1] - x[0]).item()) / _builtin_max(nx - 1, 1)
    dy = float((y[-1] - y[0]).item()) / _builtin_max(ny - 1, 1)
    if abs(dx) < 1e-14:
        dx = 1.0
    if abs(dy) < 1e-14:
        dy = 1.0
    dx2, dy2 = dx * dx, dy * dy

    # FFT-Wellenvektoren für periodische Druck-Poisson-Lösung.
    # fftfreq gibt Frequenzen f in Zyklen/Länge; der Laplace-Operator im
    # Fourierraum braucht Winkelfrequenzen ω = 2πf → ω² = 4π²f².
    import math as _math
    _twopi = 2.0 * _math.pi
    kx = _twopi * torch.fft.fftfreq(nx, d=dx).to(u0.device)
    ky = _twopi * torch.fft.fftfreq(ny, d=dy).to(u0.device)
    KX = kx.reshape(-1, 1).expand(nx, ny)
    KY = ky.reshape(1, -1).expand(nx, ny)
    k_sq = KX * KX + KY * KY
    k_sq[0, 0] = 1.0  # Vermeide Division durch null; p̂(0,0)=0 wird separat gesetzt

    def _roll(u, dim, shift):
        if bc == "periodic":
            return torch.roll(u, shift, dims=dim)
        if dim == 0:
            if shift == 1:
                return torch.cat([u[:1, :], u[:-1, :]], dim=0)
            return torch.cat([u[1:, :], u[-1:, :]], dim=0)
        if shift == 1:
            return torch.cat([u[:, :1], u[:, :-1]], dim=1)
        return torch.cat([u[:, 1:], u[:, -1:]], dim=1)

    def _convective_upwind(u, v, fld):
        """(u·∇)fld mit Upwind: u*fld_x + v*fld_y"""
        fld_left = _roll(fld, 0, 1)
        fld_right = _roll(fld, 0, -1)
        fld_bottom = _roll(fld, 1, 1)
        fld_top = _roll(fld, 1, -1)
        adv_x = torch.where(u > 0, -u * (fld - fld_left) / dx, -u * (fld_right - fld) / dx)
        adv_y = torch.where(v > 0, -v * (fld - fld_bottom) / dy, -v * (fld_top - fld) / dy)
        return adv_x + adv_y

    def _laplacian(u):
        u_left = _roll(u, 0, 1)
        u_right = _roll(u, 0, -1)
        u_bottom = _roll(u, 1, 1)
        u_top = _roll(u, 1, -1)
        return (u_left - 2.0 * u + u_right) / dx2 + (u_bottom - 2.0 * u + u_top) / dy2

    def _divergence(u, v):
        u_left = _roll(u, 0, 1)
        u_right = _roll(u, 0, -1)
        v_bottom = _roll(v, 1, 1)
        v_top = _roll(v, 1, -1)
        return (u_right - u_left) / (2.0 * dx) + (v_top - v_bottom) / (2.0 * dy)

    def _gradient(p):
        p_left = _roll(p, 0, 1)
        p_right = _roll(p, 0, -1)
        p_bottom = _roll(p, 1, 1)
        p_top = _roll(p, 1, -1)
        px = (p_right - p_left) / (2.0 * dx)
        py = (p_top - p_bottom) / (2.0 * dy)
        return px, py

    def _solve_pressure_fft(div_u_star, dt_val):
        """Löst ∇²p = div_u_star/dt per FFT (periodisch)."""
        rhs = div_u_star / dt_val
        rhs_hat = torch.fft.fft2(rhs)
        p_hat = -rhs_hat / k_sq
        p_hat[0, 0] = 0.0
        return torch.fft.ifft2(p_hat).real

    # IBM: Obstacle-Maske verarbeiten
    has_obstacle = obstacle_mask is not None
    M = None
    if has_obstacle:
        M = _to_tensor(obstacle_mask).float().to(u0.device)
        if M.shape != (nx, ny):
            raise ValueError(
                f"pde_navier_stokes_2d: obstacle_mask muss Form (nx={nx}, ny={ny}) haben, "
                f"bekam {tuple(M.shape)}."
            )
        M = M.clamp(0.0, 1.0)
    rho_val = float(_to_tensor(rho).float().item())

    nt = t.numel()
    u_sol = torch.zeros(nt, nx, ny, device=u0.device, dtype=u0.dtype)
    v_sol = torch.zeros(nt, nx, ny, device=u0.device, dtype=u0.dtype)
    p_sol = torch.zeros(nt, nx, ny, device=u0.device, dtype=u0.dtype)
    F_history = torch.zeros(nt, 2, device=u0.device, dtype=u0.dtype) if has_obstacle else None
    u_sol[0] = u0
    v_sol[0] = v0
    # p_sol[0] bleibt 0 (Druck ist bis auf Konstante bestimmt; t=0 ohne Vorgabe)
    u_cur = u0.clone()
    v_cur = v0.clone()

    for n in range(nt - 1):
        dt = float((t[n + 1] - t[n]).item())
        if dt <= 0:
            continue
        # 1) Prädiktor: u* = u - dt*(u·∇)u + dt*ν∇²u + dt*f_body
        conv_u = _convective_upwind(u_cur, v_cur, u_cur)
        conv_v = _convective_upwind(u_cur, v_cur, v_cur)
        lap_u = _laplacian(u_cur)
        lap_v = _laplacian(v_cur)
        u_star = u_cur + dt * (-conv_u + nu_val * lap_u + float(body_force_x))
        v_star = v_cur + dt * (-conv_v + nu_val * lap_v + float(body_force_y))
        # 1b) Brinkman-Penalisierung (implizit, unbedingt stabil)
        if has_obstacle:
            eta_val = float(eta) if eta is not None else 1.0 / dt
            damping = 1.0 / (1.0 + dt * eta_val * M)
            # Kraft = Newton III: Reaktion der Penalisierung auf das Fluid
            Fx = float((rho_val * eta_val * M * u_star * damping).sum().item()) * dx * dy
            Fy = float((rho_val * eta_val * M * v_star * damping).sum().item()) * dx * dy
            F_history[n + 1, 0] = Fx
            F_history[n + 1, 1] = Fy
            u_star = u_star * damping
            v_star = v_star * damping
        # 2) Druck-Poisson: ∇²p = ∇·u*/dt
        if bc == "periodic":
            div_star = _divergence(u_star, v_star)
            p = _solve_pressure_fft(div_star, dt)
        else:
            # Dirichlet: vereinfachte Jacobi-Iteration (langsamer, aber differenzierbar)
            div_star = _divergence(u_star, v_star)
            p = torch.zeros_like(u_cur)
            for _ in range(50):
                p_left = _roll(p, 0, 1)
                p_right = _roll(p, 0, -1)
                p_bottom = _roll(p, 1, 1)
                p_top = _roll(p, 1, -1)
                p_new = 0.25 * (p_left + p_right + p_bottom + p_top - dx2 * div_star / dt)
                p = p_new
        # 3) Projektion: u^{n+1} = u* - dt*∇p
        px, py = _gradient(p)
        u_cur = u_star - dt * px
        v_cur = v_star - dt * py
        # 3b) Re-enforce solid no-slip: Druck-Projektion würde sonst
        #     Geschwindigkeit im Solid aufbauen (direkte IBM-Erzwingung).
        if has_obstacle:
            u_cur = u_cur * (1.0 - M)
            v_cur = v_cur * (1.0 - M)
        u_sol[n + 1] = u_cur
        v_sol[n + 1] = v_cur
        p_sol[n + 1] = p

    if has_obstacle:
        return u_sol, v_sol, p_sol, F_history
    return u_sol, v_sol, p_sol

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

def Dirichlet(alpha):
    """
    Dirichlet(alpha). Multivariate Verteilung auf dem Simplex; Verallgemeinerung von Beta.
    alpha: Konzentrationsparameter (1D-Tensor oder Liste), z. B. [1,1,1] für uniform auf 3-Simplex.
    Stichproben haben Summe 1; für kategoriale Anteile, Topic-Modeling, Bayesian Mixtures.
    Rückgabe: Verteilung; sample(d) oder sample(d, n); log_prob(d, x).
    """
    alpha = _to_tensor(alpha).float().flatten()
    if alpha.numel() < 2:
        raise ValueError("Dirichlet: alpha muss mindestens 2 Komponenten haben.")
    alpha = alpha.clamp(min=1e-6)
    return torch.distributions.Dirichlet(concentration=alpha)

def dirichlet_function(x):
    """
    Dirichlet-Funktion D(x): 1 wenn x rational (mit Nenner ≤ 10000), sonst 0.
    Theoretisch: D(x)=1 für x∈Q, D(x)=0 für x∉Q; überall unstetig.
    Numerisch: Heuristik über fractions.Fraction; x gilt als rational, wenn als p/q mit q≤10000 darstellbar (Toleranz 1e-6).
    x: Skalar oder Tensor (elementweise).
    """
    from fractions import Fraction
    x_t = _to_tensor(x).float()
    if x_t.dim() == 0:
        try:
            f = Fraction(float(x_t.item())).limit_denominator(10000)
            return 1.0 if abs(float(x_t.item()) - float(f)) < 1e-6 else 0.0
        except (OverflowError, ValueError):
            return 0.0
    out = torch.zeros_like(x_t)
    flat = x_t.flatten()
    for i in range(flat.numel()):
        v = float(flat[i].item())
        try:
            f = Fraction(v).limit_denominator(10000)
            out.flatten()[i] = 1.0 if abs(v - float(f)) < 1e-6 else 0.0
        except (OverflowError, ValueError):
            out.flatten()[i] = 0.0
    return out

# --- Dedekind-Schnitte (Dedekind Cuts): Konstruktion der reellen Zahlen aus Q ---
class DedekindCut:
    """
    Dedekind-Schnitt: Repräsentation einer reellen Zahl als untere Menge A ⊆ Q mit:
    - A ≠ ∅, A ≠ Q; A abwärts abgeschlossen; A hat kein Maximum.
    - Untere Menge A = {q ∈ Q : q < x} für die reelle Zahl x.
    Implementierung: Speichert x als float; lower_set_contains(q) prüft q < x.
    """
    def __init__(self, value):
        """Erzeuge Schnitt für reelle Zahl value. value: float, int oder Tensor-Skalar."""
        try:
            t = _to_tensor(value)
            self._value = float(t.squeeze().item())
        except Exception:
            self._value = float(value)

    def lower_set_contains(self, q):
        """Prüft, ob rationale Zahl q in der unteren Menge liegt: q < x."""
        q_val = float(_to_tensor(q).squeeze().item()) if hasattr(q, "item") or hasattr(q, "__float__") else float(q)
        return q_val < self._value

    def to_float(self):
        """Reelle Zahl als float (approximativ)."""
        return self._value

    def __add__(self, other):
        if isinstance(other, DedekindCut):
            return DedekindCut(self._value + other._value)
        return DedekindCut(self._value + float(other))

    def __radd__(self, other):
        return DedekindCut(float(other) + self._value)

    def __sub__(self, other):
        if isinstance(other, DedekindCut):
            return DedekindCut(self._value - other._value)
        return DedekindCut(self._value - float(other))

    def __rsub__(self, other):
        return DedekindCut(float(other) - self._value)

    def __mul__(self, other):
        if isinstance(other, DedekindCut):
            return DedekindCut(self._value * other._value)
        return DedekindCut(self._value * float(other))

    def __rmul__(self, other):
        return DedekindCut(float(other) * self._value)

    def __neg__(self):
        return DedekindCut(-self._value)

    def __lt__(self, other):
        if isinstance(other, DedekindCut):
            return self._value < other._value
        return self._value < float(other)

    def __le__(self, other):
        if isinstance(other, DedekindCut):
            return self._value <= other._value
        return self._value <= float(other)

    def __gt__(self, other):
        if isinstance(other, DedekindCut):
            return self._value > other._value
        return self._value > float(other)

    def __ge__(self, other):
        if isinstance(other, DedekindCut):
            return self._value >= other._value
        return self._value >= float(other)

    def __repr__(self):
        return f"DedekindCut({self._value})"


def dedekind_cut_from_rational(p, q):
    """Erzeuge Dedekind-Schnitt für rationale Zahl p/q (q ≠ 0)."""
    p_val = int(_to_tensor(p).squeeze().item())
    q_val = int(_to_tensor(q).squeeze().item())
    if q_val == 0:
        raise ValueError("dedekind_cut_from_rational: Nenner darf nicht 0 sein.")
    return DedekindCut(p_val / q_val)


def dedekind_cut_sqrt2():
    """Dedekind-Schnitt für √2: Untere Menge = {q ∈ Q : q < 0 oder q² < 2}."""
    return DedekindCut(1.4142135623730951)  # sqrt(2)


def ideal(n):
    """Erzeuge Hauptideal (n) im Ring Z. n: ganze Zahl."""
    return DedekindIdeal(n, "Z")


def ideal_factor(i):
    """Zerlege Ideal in Primideale. Für Z: (n) = ∏ (p_i)^(e_i). Rückgabe: [(p, e), ...]."""
    if isinstance(i, DedekindIdeal):
        return i.factor()
    raise TypeError("ideal_factor: Argument muss DedekindIdeal sein.")


# --- Dedekind-Ringe: Integritätsbereiche mit eindeutiger Ideal-Faktorisierung ---
class DedekindIdeal:
    """
    Ideal in einem Dedekind-Ring. Für Z: Hauptideal (n) = nZ.
    Jedes Ideal faktorisiert eindeutig in Primideale.
    """
    def __init__(self, n, ring="Z"):
        """n: Erzeuger für Z (nZ = (n)). ring: 'Z' für ganze Zahlen."""
        self.n = abs(int(n)) if n != 0 else 0
        self.ring = ring

    def factor(self):
        """Zerlege Ideal in Primideale. Für Z: (n) = ∏ (p_i)^(e_i) mit n = ∏ p_i^e_i."""
        if self.n == 0:
            return []
        n = self.n
        factors = []
        d = 2
        while d * d <= n:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            if e > 0:
                factors.append((d, e))
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append((n, 1))
        return factors

    def norm(self):
        """Norm des Ideals. Für Z: |(n)| = |n|."""
        return self.n

    def __mul__(self, other):
        """Ideal-Multiplikation. Für Z: (a)(b) = (ab)."""
        if isinstance(other, DedekindIdeal) and self.ring == other.ring == "Z":
            return DedekindIdeal(self.n * other.n, self.ring)
        return NotImplemented

    def __repr__(self):
        return f"DedekindIdeal({self.n})"


class DedekindRingZ:
    """
    Ring der ganzen Zahlen Z als Dedekind-Ring.
    Jedes Ideal (n) faktorisiert eindeutig in Primideale.
    """
    def __init__(self):
        pass

    def ideal(self, n):
        """Erzeuge Hauptideal (n) = nZ."""
        return DedekindIdeal(n, "Z")

    def __repr__(self):
        return "DedekindRingZ()"


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
    if isinstance(x, (int, float, complex)):
        return builtins.abs(x)
    t = _to_tensor(x)
    if isinstance(t, torch.Tensor):
        return torch.abs(t)
    return builtins.abs(x)

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

def deg_to_rad(x):
    """Winkel von Grad in Radiant. x: Skalar, Tensor oder Quantity [deg]/[rad]. Bei [deg] → [rad]; bei [rad] unverändert; Skalar als Grad angenommen."""
    u = (getattr(x, "unit", None) or "").strip()
    if isinstance(x, Quantity):
        if _get_dimension(u) == "angle" and u == "rad":
            return Quantity(x.value, "rad")
        return Quantity(x.value * 0.017453292519943295, "rad")  # deg oder dimensionslos
    if isinstance(x, UncertainQuantity):
        if _get_dimension(u) == "angle" and u == "rad":
            return UncertainQuantity(x.value, x.std, "rad")
        return UncertainQuantity(x.value * 0.017453292519943295, x.std * 0.017453292519943295, "rad")
    val = float(_to_tensor(x).item()) if hasattr(_to_tensor(x), 'item') else float(_to_tensor(x))
    return val * 0.017453292519943295  # pi/180

def rad_to_deg(x):
    """Winkel von Radiant in Grad. x: Skalar, Tensor oder Quantity [rad]/[deg]. Bei [rad] → [deg]; bei [deg] unverändert; Skalar als Radiant angenommen."""
    u = (getattr(x, "unit", None) or "").strip()
    if isinstance(x, Quantity):
        if _get_dimension(u) == "angle" and u == "deg":
            return Quantity(x.value, "deg")
        return Quantity(x.value * 57.29577951308232, "deg")  # rad oder dimensionslos
    if isinstance(x, UncertainQuantity):
        if _get_dimension(u) == "angle" and u == "deg":
            return UncertainQuantity(x.value, x.std, "deg")
        return UncertainQuantity(x.value * 57.29577951308232, x.std * 57.29577951308232, "deg")
    val = float(_to_tensor(x).item()) if hasattr(_to_tensor(x), 'item') else float(_to_tensor(x))
    return val * 57.29577951308232  # 180/pi

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


def binom(n, k):
    """
    Binomialkoeffizient: n über k = C(n,k) = n! / (k! * (n-k)!).
    n, k: nichtnegative ganze Zahlen. Nutzt multiplikative Formel für Stabilität.
    """
    n_val = int(n)
    k_val = int(k)
    if n_val < 0 or k_val < 0:
        raise ValueError("binom: n und k müssen nichtnegativ sein.")
    if k_val > n_val:
        return 0
    if k_val == 0 or k_val == n_val:
        return 1
    k_val = _builtin_min(k_val, n_val - k_val)
    result = 1.0
    for i in range(k_val):
        result = result * (n_val - i) / (i + 1)
    return int(result) if abs(result - round(result)) < 1e-9 else result


def ttest_one_sample(x, mu0):
    """
    Ein-Stichproben-t-Test: Prüft, ob Mittelwert der Stichprobe x von mu0 abweicht.
    x: 1D-Tensor oder Liste. mu0: hypothetischer Mittelwert.
    Rückgabe: (t_statistik, p_value) — zweiseitiger Test.
    """
    import scipy.stats as scistats  # type: ignore[import-untyped]
    t = _to_tensor(x).float().flatten()
    n = t.numel()
    if n < 2:
        raise ValueError("ttest_one_sample: mindestens 2 Beobachtungen nötig.")
    m = t.mean().item()
    s = t.std(unbiased=True).item()
    if s < 1e-14:
        return float(m - mu0), 0.0 if abs(m - mu0) > 1e-14 else 1.0
    t_stat = (m - mu0) / (s / (n ** 0.5))
    df = n - 1
    p_val = 2.0 * (1.0 - scistats.t.cdf(abs(t_stat), df))
    return (float(t_stat), float(p_val))


def ttest_two_sample(x, y):
    """
    Zwei-Stichproben-t-Test (Welch): Prüft, ob die Mittelwerte von x und y sich unterscheiden.
    Annahme: ungleiche Varianzen (Welch-t-Test).
    x, y: 1D-Tensoren oder Listen.
    Rückgabe: (t_statistik, p_value) — zweiseitiger Test.
    """
    import scipy.stats as scistats  # type: ignore[import-untyped]
    tx = _to_tensor(x).float().flatten()
    ty = _to_tensor(y).float().flatten()
    n1, n2 = tx.numel(), ty.numel()
    if n1 < 2 or n2 < 2:
        raise ValueError("ttest_two_sample: mindestens 2 Beobachtungen pro Stichprobe nötig.")
    m1, m2 = tx.mean().item(), ty.mean().item()
    v1, v2 = tx.var(unbiased=True).item(), ty.var(unbiased=True).item()
    se = (v1 / n1 + v2 / n2) ** 0.5
    if se < 1e-14:
        return float(m1 - m2), 0.0 if abs(m1 - m2) > 1e-14 else 1.0
    t_stat = (m1 - m2) / se
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df_val = _builtin_max(num / (denom + 1e-14), 1.0)
    p_val = 2.0 * (1.0 - scistats.t.cdf(abs(t_stat), df_val))
    return (float(t_stat), float(p_val))


# --- Standard Library: Reduktionen (min, max, argmin, argmax) ---



import cmath as _cmath
import math as _math

# Quantum constants added during master-quantum merge
PAULI_I = [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]
PAULI_X = [[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]]
PAULI_Y = [[0 + 0j, -1j], [1j, 0 + 0j]]
PAULI_Z = [[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]]
PAULI_H = [[1 / _math.sqrt(2) + 0j, 1 / _math.sqrt(2) + 0j], [1 / _math.sqrt(2) + 0j, -1 / _math.sqrt(2) + 0j]]
_T_MAT = [[1 + 0j, 0 + 0j], [0 + 0j, _cmath.exp(1j * _math.pi / 4)]]
_S_MAT = [[1 + 0j, 0 + 0j], [0 + 0j, 1j]]


def _graph_laplacian_sparse(adj_sp, normalized):
    """Sparse-Implementierung. Liefert COO-Tensor."""
    adj_sp = adj_sp.coalesce()
    N = adj_sp.shape[0]
    idx = adj_sp.indices()
    vals = adj_sp.values().float()
    deg = torch.zeros(N, dtype=vals.dtype, device=vals.device)
    deg.scatter_add_(0, idx[0], vals)
    if normalized:
        d_inv_sqrt = torch.where(deg > 0, deg.clamp(min=1e-12).pow(-0.5),
                                  torch.zeros_like(deg))
        scaled_off = -vals * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
        diag_idx = torch.arange(N, device=vals.device)
        diag_v = (deg > 0).to(vals.dtype)
        all_i = torch.cat([idx[0], diag_idx])
        all_j = torch.cat([idx[1], diag_idx])
        all_v = torch.cat([scaled_off, diag_v])
    else:
        diag_idx = torch.arange(N, device=vals.device)
        all_i = torch.cat([diag_idx, idx[0]])
        all_j = torch.cat([diag_idx, idx[1]])
        all_v = torch.cat([deg, -vals])
    return torch.sparse_coo_tensor(torch.stack([all_i, all_j]), all_v, (N, N)).coalesce()

def graph_laplacian(adj, normalized=False):
    """Berechnet den Graph-Laplacian fuer eine Adjazenzmatrix.

    Kombinatorisch (Default):  L = D - A
        L[i,i] =  deg(i)
        L[i,j] = -A[i,j]   (i != j)

    Normalisiert (symmetrisch): L_sym = I - D^{-1/2} A D^{-1/2}
        Eigenwerte in [0, 2]; nuetzlich fuer spektrale Verfahren auf Graphen
        unterschiedlicher Dichte.

    Eingabe `adj`: dichte (N,N)-Matrix, sparse torch.Tensor, oder geschachtelte
    Liste. Quadratisch, nicht-negative Eintraege; bei ungerichteten Graphen
    symmetrisch. Ausgabe ist sparse, wenn `adj` sparse ist, sonst dicht ÔÇö direkt
    einsetzbar in `cg`, `gmres`, `bicgstab` (Heat-Diffusion, Eigenvektoren, etc.).
    """
    if hasattr(adj, "is_sparse") and adj.is_sparse:
        return _graph_laplacian_sparse(adj, normalized)
    adj_t = _to_tensor(adj)
    if not isinstance(adj_t, torch.Tensor):
        raise TypeError(f"graph_laplacian: Eingabe muss tensor-artig sein, erhielt {type(adj).__name__}.")
    adj_t = adj_t.float()
    if adj_t.dim() != 2 or adj_t.shape[0] != adj_t.shape[1]:
        raise ValueError(
            f"graph_laplacian: Adjazenz muss quadratisch (N,N) sein, "
            f"erhalten {tuple(adj_t.shape)}."
        )
    deg = adj_t.sum(dim=1)
    if normalized:
        d_inv_sqrt = torch.where(deg > 0, deg.clamp(min=1e-12).pow(-0.5),
                                  torch.zeros_like(deg))
        scaled = d_inv_sqrt.unsqueeze(1) * adj_t * d_inv_sqrt.unsqueeze(0)
        eye = torch.eye(adj_t.shape[0], dtype=adj_t.dtype, device=adj_t.device)
        return eye - scaled
    return torch.diag(deg) - adj_t

def _milp_scalar(x):
    """Extrahiert eine reine Zahl aus Quantity/Tensor/Skalar fuer Bounds/Constraints."""
    if isinstance(x, Quantity):
        return float(x.value)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)

def _milp_units_compat(u1, u2, where):
    """Wirft ValueError, wenn u1 und u2 nicht dimensional kompatibel sind.
    Liefert die fuehrende Einheit (Linke-Seite-Konvention)."""
    if not u1 or not u2:
        return u1 or u2
    if u1 == u2:
        return u1
    if _normalize_unit_for_compare(u1) == _normalize_unit_for_compare(u2):
        return u1
    d1 = _get_dimension(u1)
    d2 = _get_dimension(u2)
    if d1 is not None and d1 == d2:
        return u1
    if _units_dimensionally_equal(u1, u2):
        return u1
    raise ValueError(f"MILP-Einheiten passen nicht in {where}: [{u1}] vs [{u2}].")

class _MILPVariable:
    """Entscheidungsvariable. Hashbar via Identitaet (kein __eq__-Override)."""
    __slots__ = ("name", "lower", "upper", "integer", "unit")

    def __init__(self, name, lower=None, upper=None, integer=False):
        self.name = str(name)
        self.lower = lower
        self.upper = upper
        self.integer = bool(integer)
        u = ""
        for v in (lower, upper):
            if isinstance(v, Quantity) and v.unit:
                u = v.unit
                break
        self.unit = u

    def __repr__(self):
        suf = f" [{self.unit}]" if self.unit else ""
        return f"Variable({self.name!r}{suf})"

    def _as_expr(self):
        return _MILPExpression({self: 1.0}, unit=self.unit)

    def __add__(self, other): return self._as_expr() + other
    def __radd__(self, other): return self._as_expr() + other
    def __sub__(self, other): return self._as_expr() - other
    def __rsub__(self, other): return _MILPExpression(offset=0.0) - self._as_expr() + other if False else self._as_expr().__rsub__(other)
    def __mul__(self, other): return self._as_expr() * other
    def __rmul__(self, other): return self._as_expr() * other
    def __truediv__(self, other): return self._as_expr() / other
    def __neg__(self): return _MILPExpression({self: -1.0}, unit=self.unit)
    def __ge__(self, other): return _MILPConstraint(self._as_expr(), ">=", other)
    def __le__(self, other): return _MILPConstraint(self._as_expr(), "<=", other)

class _MILPExpression:
    """Lineare Kombination sum(coef * Variable) + offset, mit Einheit."""
    __slots__ = ("coeffs", "offset", "unit")

    def __init__(self, coeffs=None, offset=0.0, unit=""):
        self.coeffs = dict(coeffs) if coeffs else {}
        self.offset = float(offset)
        self.unit = str(unit)

    def _coerce(self, other):
        if isinstance(other, _MILPVariable):
            return _MILPExpression({other: 1.0}, unit=other.unit)
        if isinstance(other, _MILPExpression):
            return other
        if isinstance(other, Quantity):
            return _MILPExpression(offset=other.value, unit=other.unit or "")
        if isinstance(other, (int, float)):
            return _MILPExpression(offset=float(other))
        return None

    def __add__(self, other):
        oth = self._coerce(other)
        if oth is None: return NotImplemented
        unit = _milp_units_compat(self.unit, oth.unit, "Addition")
        coeffs = dict(self.coeffs)
        for v, c in oth.coeffs.items():
            coeffs[v] = coeffs.get(v, 0.0) + c
        return _MILPExpression(coeffs, self.offset + oth.offset, unit)
    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        oth = self._coerce(other)
        if oth is None: return NotImplemented
        unit = _milp_units_compat(self.unit, oth.unit, "Subtraktion")
        coeffs = dict(self.coeffs)
        for v, c in oth.coeffs.items():
            coeffs[v] = coeffs.get(v, 0.0) - c
        return _MILPExpression(coeffs, self.offset - oth.offset, unit)
    def __rsub__(self, other):
        oth = self._coerce(other)
        if oth is None: return NotImplemented
        return oth.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (_MILPVariable, _MILPExpression)):
            other_expr = other if isinstance(other, _MILPExpression) else other._as_expr()
            if self.coeffs and other_expr.coeffs:
                raise ValueError(
                    f"MILP: Produkt zweier Variabler ist nichtlinear "
                    f"({list(self.coeffs.keys())[0].name} * {list(other_expr.coeffs.keys())[0].name})."
                )
            if not self.coeffs:
                scalar, scalar_unit = self.offset, self.unit
                expr = other_expr
            else:
                scalar, scalar_unit = other_expr.offset, other_expr.unit
                expr = self
            new_unit = _unit_mul(expr.unit, scalar_unit) if scalar_unit else expr.unit
            return _MILPExpression(
                {v: c * scalar for v, c in expr.coeffs.items()},
                expr.offset * scalar,
                new_unit,
            )
        if isinstance(other, Quantity):
            scalar = other.value
            new_unit = _unit_mul(self.unit, other.unit) if other.unit else self.unit
            return _MILPExpression(
                {v: c * scalar for v, c in self.coeffs.items()},
                self.offset * scalar,
                new_unit,
            )
        if isinstance(other, (int, float)):
            return _MILPExpression(
                {v: c * float(other) for v, c in self.coeffs.items()},
                self.offset * float(other),
                self.unit,
            )
        return NotImplemented
    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            inv = 1.0 / float(other)
            return _MILPExpression({v: c * inv for v, c in self.coeffs.items()},
                                   self.offset * inv, self.unit)
        if isinstance(other, Quantity):
            inv = 1.0 / other.value
            new_unit = _unit_div(self.unit, other.unit) if other.unit else self.unit
            return _MILPExpression({v: c * inv for v, c in self.coeffs.items()},
                                   self.offset * inv, new_unit)
        return NotImplemented

    def __neg__(self):
        return _MILPExpression({v: -c for v, c in self.coeffs.items()}, -self.offset, self.unit)

    def __ge__(self, other): return _MILPConstraint(self, ">=", other)
    def __le__(self, other): return _MILPConstraint(self, "<=", other)

class _MILPConstraint:
    """Normalisiert: sum(coef * var) op bound, op in {<=, >=, ==}."""
    __slots__ = ("coeffs", "bound", "op", "unit")

    def __init__(self, lhs, op, rhs):
        if isinstance(lhs, _MILPVariable):
            lhs = lhs._as_expr()
        if not isinstance(lhs, _MILPExpression):
            raise TypeError(f"MILP-Constraint linke Seite: erwartet Variable/Expression, bekam {type(lhs).__name__}.")
        rhs_expr = lhs._coerce(rhs)
        if rhs_expr is None:
            raise TypeError(f"MILP-Constraint rechte Seite: erwartet Variable/Expression/Quantity/Zahl, bekam {type(rhs).__name__}.")
        _milp_units_compat(lhs.unit, rhs_expr.unit, "Constraint")
        diff = lhs - rhs_expr
        self.coeffs = dict(diff.coeffs)
        self.bound = -diff.offset
        self.op = op
        self.unit = diff.unit

    def __repr__(self):
        terms = " + ".join(f"{c}*{v.name}" for v, c in self.coeffs.items()) or "0"
        return f"Constraint({terms} {self.op} {self.bound} [{self.unit}])"

def Variable(name, lower=None, upper=None, integer=False):
    """Erzeugt eine MILP-Entscheidungsvariable.

    Beispiel:
        x = Variable("x", lower=0[km], upper=1000[km])
        y = Variable("trucks", lower=1, integer=true)
        cost = 2.5[Ôé¼/km] * x + 1000[Ôé¼] * y
        result = optimize_milp(cost, [x + 200[km]*y >= 500[km]])
    """
    return _MILPVariable(name=name, lower=lower, upper=upper, integer=integer)

def optimize_milp(objective, constraints=None, sense="minimize"):
    """Loest ein (gemischt-)ganzzahliges lineares Programm in deklarativer DSL.

    objective:    _MILPExpression oder _MILPVariable.
    constraints:  Liste von _MILPConstraint (per `>=`/`<=`), optional.
    sense:        'minimize' (default) oder 'maximize'.

    Rueckgabe: dict {<var_name>: optimaler_wert, ..., "objective": Z, "status": msg}.
    """
    try:
        from scipy.optimize import milp as _milp, LinearConstraint, Bounds  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("optimize_milp benoetigt scipy>=1.9.")
    import numpy as _np  # type: ignore[reportMissingImports]

    if isinstance(objective, _MILPVariable):
        obj_expr = objective._as_expr()
    elif isinstance(objective, _MILPExpression):
        obj_expr = objective
    else:
        raise TypeError(
            f"optimize_milp: objective muss Variable oder MILP-Expression sein, "
            f"bekam {type(objective).__name__}."
        )

    # constraints kann eine Python-Liste sein ODER ein leerer torch.Tensor
    # (Dedekind transpiliert `[]` zu `torch.tensor([])`). Normalisieren:
    if constraints is None:
        cons_list = []
    elif hasattr(constraints, "numel"):
        cons_list = [] if constraints.numel() == 0 else list(constraints)
    else:
        cons_list = list(constraints)
    for i, con in enumerate(cons_list):
        if not isinstance(con, _MILPConstraint):
            raise TypeError(
                f"optimize_milp: constraints[{i}] ist {type(con).__name__}, "
                f"erwartet Constraint (Variable >=/<= Wert)."
            )

    var_order = []
    var_idx = {}
    def _reg(v):
        if v not in var_idx:
            var_idx[v] = len(var_order)
            var_order.append(v)
    for v in obj_expr.coeffs:
        _reg(v)
    for con in cons_list:
        for v in con.coeffs:
            _reg(v)
    N = len(var_order)
    if N == 0:
        raise ValueError("optimize_milp: keine Variablen in objective/constraints gefunden.")

    c = _np.zeros(N, dtype=float)
    for v, coef in obj_expr.coeffs.items():
        c[var_idx[v]] = coef
    if sense == "maximize":
        c = -c
    elif sense != "minimize":
        raise ValueError(f"optimize_milp: sense muss 'minimize'/'maximize' sein, nicht {sense!r}.")

    lb = _np.full(N, -_np.inf)
    ub = _np.full(N, _np.inf)
    integrality = _np.zeros(N, dtype=int)
    for i, v in enumerate(var_order):
        if v.lower is not None:
            lb[i] = _milp_scalar(v.lower)
        if v.upper is not None:
            ub[i] = _milp_scalar(v.upper)
        if v.integer:
            integrality[i] = 1
    bounds = Bounds(lb=lb, ub=ub)

    ub_rows, ub_b = [], []
    eq_rows, eq_b = [], []
    for con in cons_list:
        row = _np.zeros(N, dtype=float)
        for v, coef in con.coeffs.items():
            row[var_idx[v]] = coef
        bnd = _milp_scalar(con.bound)
        if con.op == "<=":
            ub_rows.append(row); ub_b.append(bnd)
        elif con.op == ">=":
            ub_rows.append(-row); ub_b.append(-bnd)
        elif con.op == "==":
            eq_rows.append(row); eq_b.append(bnd)
        else:
            raise ValueError(f"optimize_milp: Operator {con.op!r} nicht unterstuetzt.")

    sci_constraints = []
    if ub_rows:
        sci_constraints.append(LinearConstraint(_np.array(ub_rows), -_np.inf, _np.array(ub_b)))
    if eq_rows:
        sci_constraints.append(LinearConstraint(_np.array(eq_rows), _np.array(eq_b), _np.array(eq_b)))

    res = _milp(c, constraints=sci_constraints if sci_constraints else None,
                bounds=bounds, integrality=integrality)
    if not res.success:
        raise RuntimeError(f"optimize_milp: keine Loesung. {res.message}")

    out = {v.name: float(res.x[i]) for i, v in enumerate(var_order)}
    obj_val = float(res.fun)
    if sense == "maximize":
        obj_val = -obj_val
    obj_val += obj_expr.offset
    out["objective"] = obj_val
    out["status"] = res.message
    return out

def _md_convert_to_unit(q, target_unit, dimension, arg_name):
    """Validiert Quantity-Dimension und liefert numerischen Wert in target_unit."""
    if not isinstance(q, Quantity):
        raise TypeError(
            f"md_simulate_lj: {arg_name} muss Quantity sein (z. B. 0.34[nm]), "
            f"bekam {type(q).__name__}."
        )
    dim = _get_dimension(q.unit)
    expected_dim = _get_dimension(target_unit)
    if dim is None or dim != expected_dim:
        raise ValueError(
            f"md_simulate_lj: {arg_name}={q} hat falsche Dimension; "
            f"erwartet kompatibel zu [{target_unit}] ({expected_dim})."
        )
    base = _convert_to_base(q.value, q.unit, dim)
    return _convert_from_base(base, target_unit, dim)

def md_simulate_lj(n_particles, sigma, epsilon, mass, temperature, dt, n_steps,
                    box_size=None, friction=1.0, seed=None):
    """Lennard-Jones Molekulardynamik-Simulation via OpenMM, mit Unit-Validierung.

    Anders als ein direkter OpenMM-Aufruf prueft diese Funktion die Dimensionen
    aller Eingabeparameter VOR dem C++-Kernel ÔÇö kein stilles Mischen von kcal/mol
    mit eV oder ├à mit nm.

    Parameter:
      n_particles:  int, Anzahl Teilchen.
      sigma:        Quantity in Laengeneinheit [nm, Angstrom, pm, m, ...].
      epsilon:      Quantity in molarer Energie [kJ/mol, kcal/mol, ...].
      mass:         Quantity in Masse [amu, g, kg, ...].
      temperature:  Quantity in [K] ÔÇö Langevin-Bad.
      dt:           Quantity in Zeit [fs, ps, ns, ...] ÔÇö Integrations-Zeitschritt.
      n_steps:      int ÔÇö Anzahl Integrationsschritte.
      box_size:     Optional Quantity in Laenge ÔÇö kubische periodische Box.
      friction:     float in 1/ps ÔÇö Langevin-Reibung (Default 1.0).
      seed:         Optional int ÔÇö RNG-Seed fuer reproduzierbare Trajektorien.

    Rueckgabe: dict mit
      positions:        torch.Tensor (n_particles, 3) ÔÇö Endpositionen in nm.
      potential_energy: Quantity in [kJ/mol].
      kinetic_energy:   Quantity in [kJ/mol].
      temperature:      Quantity in [K] ÔÇö gemessene End-Temperatur.
      n_steps:          ausgefuehrte Schritte.
    """
    try:
        import openmm as _mm  # type: ignore[import-untyped]
        import openmm.unit as _ommu  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "md_simulate_lj benoetigt openmm. Installation: pip install openmm"
        )
    import numpy as _np  # type: ignore[reportMissingImports]

    # Unit-Validierung + Konvertierung in OpenMM-Standards (nm, kJ/mol, amu, ps, K)
    sigma_nm = _md_convert_to_unit(sigma, "nm", "length", "sigma")
    eps_kjm = _md_convert_to_unit(epsilon, "kJ/mol", "molar_energy", "epsilon")
    mass_g_per_mol = _md_convert_to_unit(mass, "g", "mass", "mass")
    # OpenMM: amu numerisch gleich g/mol fuer Avogadro-Skalierung
    mass_amu = mass_g_per_mol / 1.66053906660e-24
    if not isinstance(temperature, Quantity) or temperature.unit != "K":
        raise ValueError(f"md_simulate_lj: temperature muss in [K] sein, bekam {temperature}.")
    temp_K = float(temperature.value)
    dt_ps = _md_convert_to_unit(dt, "ps", "time", "dt")
    box_nm = _md_convert_to_unit(box_size, "nm", "length", "box_size") if box_size is not None else None

    # System aufbauen
    system = _mm.System()
    for _ in range(int(n_particles)):
        system.addParticle(mass_amu * _ommu.amu)

    nb = _mm.NonbondedForce()
    cutoff = 2.5 * sigma_nm
    for _ in range(int(n_particles)):
        nb.addParticle(0.0, sigma_nm * _ommu.nanometer, eps_kjm * _ommu.kilojoule_per_mole)
    if box_nm is not None:
        box_vec = box_nm * _np.eye(3)
        system.setDefaultPeriodicBoxVectors(*(v * _ommu.nanometer for v in box_vec))
        nb.setNonbondedMethod(_mm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(_builtin_min(cutoff, 0.49 * box_nm) * _ommu.nanometer)
    else:
        nb.setNonbondedMethod(_mm.NonbondedForce.CutoffNonPeriodic)
        nb.setCutoffDistance(cutoff * _ommu.nanometer)
    system.addForce(nb)

    # Anfangspositionen: regelmaessiges Gitter ÔÇö vermeidet Ueberlappungen
    # und resultierende NaN-Energien (LJ-r^-12 explodiert bei r -> 0).
    if seed is not None:
        _np.random.seed(int(seed))
    extent = box_nm if box_nm is not None else _builtin_max(5.0 * sigma_nm, 1.0)
    # Gitter-Abstand >= 1.05 * sigma (LJ-Gleichgewicht ~ 2^(1/6)*sigma Ôëê 1.122*sigma)
    side = int(_np.ceil(int(n_particles) ** (1.0 / 3.0)))
    spacing = extent / _builtin_max(side, 1)
    grid_min = 1.05 * sigma_nm
    if spacing < grid_min:
        spacing = grid_min
        extent = spacing * side  # Box ggf. vergroessern
    positions = _np.zeros((int(n_particles), 3), dtype=float)
    k = 0
    for i in range(side):
        for j in range(side):
            for kk in range(side):
                if k >= int(n_particles):
                    break
                positions[k] = [(i + 0.5) * spacing, (j + 0.5) * spacing, (kk + 0.5) * spacing]
                k += 1
            if k >= int(n_particles):
                break
        if k >= int(n_particles):
            break
    # Leichte Stoerung gegen perfekte Gitter-Symmetrie
    positions += 0.02 * spacing * (_np.random.rand(int(n_particles), 3) - 0.5)

    integrator = _mm.LangevinIntegrator(
        temp_K * _ommu.kelvin,
        float(friction) / _ommu.picosecond,
        dt_ps * _ommu.picosecond,
    )
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))

    ctx = _mm.Context(system, integrator)
    ctx.setPositions(positions * _ommu.nanometer)
    ctx.setVelocitiesToTemperature(temp_K * _ommu.kelvin)

    integrator.step(int(n_steps))

    state = ctx.getState(getPositions=True, getEnergy=True, getVelocities=True)
    pos_nm = _np.asarray(state.getPositions(asNumpy=True).value_in_unit(_ommu.nanometer))
    pe = float(state.getPotentialEnergy().value_in_unit(_ommu.kilojoule_per_mole))
    ke = float(state.getKineticEnergy().value_in_unit(_ommu.kilojoule_per_mole))
    # Kinetic Energy -> Temperatur: KE = (3/2)*N*k_B*T => T = 2*KE / (3*N*k_B)
    # k_B = 8.314e-3 kJ/(mol*K) in molaren Einheiten
    k_B_mol = 0.008314462618
    T_measured = 2.0 * ke / (3.0 * n_particles * k_B_mol)

    return {
        "positions": torch.as_tensor(pos_nm, dtype=torch.float32),
        "potential_energy": Quantity(pe, "kJ/mol"),
        "kinetic_energy": Quantity(ke, "kJ/mol"),
        "temperature": Quantity(T_measured, "K"),
        "n_steps": int(n_steps),
    }

def _unwrap_q(x):
    """Unwraps Quantity to float (dimensionslos oder rad)."""
    if isinstance(x, Quantity):
        if x.unit not in ("", "rad", None):
            raise ValueError(
                f"Rotationswinkel muss dimensionslos oder [rad] sein, bekam [{x.unit}]."
            )
        return float(x.value) if hasattr(x.value, 'item') else float(x.value)
    if hasattr(x, 'item'):
        return float(x.item())
    return float(x)

def _kron2(a, b):
    """Kronecker-Produkt zweier komplexer 2D-Listen."""
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])
    result = [[0+0j] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    result[i * rb + k][j * cb + l] = a[i][j] * b[k][l]
    return result

def _mat_vec(mat, vec):
    """Matrix-Vektor-Multiplikation (komplexe Listen)."""
    n = len(mat)
    return [sum(mat[i][j] * vec[j] for j in range(n)) for i in range(n)]

def _single_qubit_gate_unitary(gate_mat, qubit, n):
    """Erstellt 2^n x 2^n Unitary fuer ein 1-Qubit-Gatter auf qubit `qubit`.

    Konvention: qubit 0 = LSB (rightmost in kron product).
    kron wird von MSB nach LSB aufgebaut (von links nach rechts).
    Qubit q befindet sich an Kron-Position n-1-q von links.
    """
    result = [[1+0j]]
    for q in range(n - 1, -1, -1):  # n-1 (MSB) down to 0 (LSB)
        if q == qubit:
            m = gate_mat
        else:
            m = PAULI_I
        result = _kron2(result, m)
    return result

def _cx_unitary(control, target, n):
    """CNOT-Unitary fuer gegebene Qubit-Indizes."""
    dim = 1 << n
    U = [[0+0j] * dim for _ in range(dim)]
    for state in range(dim):
        ctrl_bit = (state >> control) & 1
        if ctrl_bit == 1:
            new_state = state ^ (1 << target)
        else:
            new_state = state
        U[new_state][state] = 1+0j
    return U

def _cz_unitary(control, target, n):
    """CZ-Unitary."""
    dim = 1 << n
    U = [[complex(i == j) for j in range(dim)] for i in range(dim)]
    for state in range(dim):
        c_bit = (state >> control) & 1
        t_bit = (state >> target) & 1
        if c_bit == 1 and t_bit == 1:
            U[state][state] = -1+0j
    return U

def _swap_unitary(q0, q1, n):
    """SWAP-Unitary."""
    dim = 1 << n
    U = [[0+0j] * dim for _ in range(dim)]
    for state in range(dim):
        b0 = (state >> q0) & 1
        b1 = (state >> q1) & 1
        if b0 != b1:
            new_state = state ^ (1 << q0) ^ (1 << q1)
        else:
            new_state = state
        U[new_state][state] = 1+0j
    return U

def _rx_mat(theta):
    c = _cmath.cos(theta / 2)
    s = _cmath.sin(theta / 2)
    return [[c, -1j * s], [-1j * s, c]]

def _ry_mat(theta):
    c = _cmath.cos(theta / 2)
    s = _cmath.sin(theta / 2)
    return [[c, -s], [s, c]]

def _rz_mat(theta):
    return [[_cmath.exp(-1j * theta / 2), 0], [0, _cmath.exp(1j * theta / 2)]]

def statevec_sim(circuit, shots: int = 0):
    """Reiner Statevector-Simulator fuer QuantumCircuit.

    Args:
        circuit: QuantumCircuit-Objekt
        shots: 0 = gibt Statevector (Liste komplexer Amplituden) zurueck.
               > 0 = gibt Mess-Ergebnis-Dict mit Bitstringen und Haeufigkeiten zurueck.

    Returns:
        Statevector (list[complex]) oder dict[str, int] (bei shots > 0).
    """
    shots = int(shots.item() if hasattr(shots, 'item') else shots)
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"statevec_sim: Erwartet QuantumCircuit, bekam {type(circuit).__name__}.")
    n = circuit.n_qubits
    dim = 1 << n
    # Startzustand |0...0>
    sv = [0+0j] * dim
    sv[0] = 1+0j

    for name, qubits, params in circuit._gates:
        if name == "H":
            U = _single_qubit_gate_unitary(PAULI_H, qubits[0], n)
        elif name == "X":
            U = _single_qubit_gate_unitary(PAULI_X, qubits[0], n)
        elif name == "Y":
            U = _single_qubit_gate_unitary(PAULI_Y, qubits[0], n)
        elif name == "Z":
            U = _single_qubit_gate_unitary(PAULI_Z, qubits[0], n)
        elif name == "T":
            U = _single_qubit_gate_unitary(_T_MAT, qubits[0], n)
        elif name == "S":
            U = _single_qubit_gate_unitary(_S_MAT, qubits[0], n)
        elif name == "RX":
            U = _single_qubit_gate_unitary(_rx_mat(params[0]), qubits[0], n)
        elif name == "RY":
            U = _single_qubit_gate_unitary(_ry_mat(params[0]), qubits[0], n)
        elif name == "RZ":
            U = _single_qubit_gate_unitary(_rz_mat(params[0]), qubits[0], n)
        elif name == "CX":
            U = _cx_unitary(qubits[0], qubits[1], n)
        elif name == "CZ":
            U = _cz_unitary(qubits[0], qubits[1], n)
        elif name == "SWAP":
            U = _swap_unitary(qubits[0], qubits[1], n)
        else:
            raise ValueError(f"Unbekanntes Gatter '{name}' im Simulator.")
        sv = _mat_vec(U, sv)

    if shots <= 0:
        return sv

    # Messe alle Qubits (oder nur die gemessenen)
    probs = [abs(a) ** 2 for a in sv]
    import random
    counts = {}
    measure_qubits = [q for q, _ in circuit._measurements] if circuit._measurements else list(range(n))
    for _ in range(shots):
        r = random.random()
        cumsum = 0.0
        idx = dim - 1
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                idx = i
                break
        # Extrahiere gemessene Bits (MSB = Qubit 0)
        bits = format(idx, f'0{n}b')
        key = ''.join(bits[q] for q in sorted(measure_qubits))
        counts[key] = counts.get(key, 0) + 1
    return counts

def statevec_probs(circuit):
    """Gibt Wahrscheinlichkeiten |¤ê_i|┬▓ als Liste zurueck."""
    sv = statevec_sim(circuit)
    return [abs(a) ** 2 for a in sv]

def statevec_expectation(circuit, observable):
    """Berechnet Ôƒ¿¤ê|O|¤êÔƒ® fuer einen Pauli-String-Observable.

    Args:
        circuit: QuantumCircuit
        observable: String wie "ZI", "XX", "ZZ" (Pauli-Produkt)

    Returns:
        Erwartungswert (float)
    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("statevec_expectation: Erwartet QuantumCircuit.")
    n = circuit.n_qubits
    if not isinstance(observable, str) or len(observable) != n:
        raise ValueError(
            f"Observable muss ein {n}-Zeichen-Pauli-String sein (z.B. 'ZI'), bekam {observable!r}."
        )
    sv = statevec_sim(circuit)
    # Baue Observ.-Matrix als Tensor-Produkt
    obs_mat = [[1+0j]]
    for ch in observable:
        obs_mat = _kron2(obs_mat, _pauli(ch))
    obs_sv = _mat_vec(obs_mat, sv)
    return sum((a.conjugate() * b).real for a, b in zip(sv, obs_sv))

def vqe_circuit(n_qubits, n_layers, params):
    """Erstellt einen parametrisierten Ansatz-Schaltkreis fuer VQE.

    Hardware-efficient Ansatz: abwechselnd Ry-Schicht und CNOT-Kette.
    params: Liste/Tensor der Laenge n_qubits * n_layers (Ry-Winkel).

    Returns:
        QuantumCircuit (kann mit statevec_expectation ausgewertet werden)
    """
    if isinstance(n_qubits, Quantity):
        n_qubits = int(float(n_qubits.value))
    if isinstance(n_layers, Quantity):
        n_layers = int(float(n_layers.value))
    n_qubits = int(n_qubits)
    n_layers = int(n_layers)

    # Flatten params
    if hasattr(params, 'tolist'):
        flat = params.reshape(-1).tolist()
    elif isinstance(params, list):
        flat = params
    else:
        flat = list(params)

    expected = n_qubits * n_layers
    if len(flat) != expected:
        raise ValueError(
            f"vqe_circuit: Erwartet {expected} Parameter (n_qubits={n_qubits} * n_layers={n_layers}), "
            f"bekam {len(flat)}."
        )

    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            theta = flat[idx] if not hasattr(flat[idx], 'item') else flat[idx].item()
            qc.ry(float(theta), q)
            idx += 1
        # Entangle: lineare Kette
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc

def vqe_energy(params, n_qubits, n_layers, hamiltonian_terms):
    """Berechnet Energie Ôƒ¿¤ê(╬©)|H|¤ê(╬©)Ôƒ® fuer VQE-Schleife.

    Args:
        params: Tensor der Winkel (grad() faehig via torch)
        n_qubits: Anzahl Qubits
        n_layers: Anzahl VQE-Schichten
        hamiltonian_terms: Liste von (koeffizient, pauli_string) ÔÇö z.B. [(0.5, "ZI"), (-0.5, "IZ")]

    Returns:
        Energie als float
    """
    import torch as _torch
    # params als Python-Liste (detachieren falls Tensor)
    if hasattr(params, 'detach'):
        flat = params.detach().tolist()
        if isinstance(flat, float):
            flat = [flat]
    else:
        flat = list(params) if hasattr(params, '__iter__') else [float(params)]

    qc = vqe_circuit(n_qubits, n_layers, flat)
    energy = 0.0
    for coeff, pauli_str in hamiltonian_terms:
        c = float(coeff.value if isinstance(coeff, Quantity) else coeff)
        energy += c * statevec_expectation(qc, pauli_str)
    return energy
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
        result = np.trapezoid(y_np, x_np) if hasattr(np, 'trapezoid') else np.trapz(y_np, x_np)
    else:
        result = np.trapezoid(y_np) if hasattr(np, 'trapezoid') else np.trapz(y_np)
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

class OptimizeResult(tuple):
    def __new__(cls, x, fun):
        return super().__new__(cls, (x, fun))
    @property
    def x(self):
        return self[0]
    @property
    def fun(self):
        return self[1]

def minimize(f, x0, method="gd", lr=0.01, steps=500):
    """
    Mehrdimensionale Minimierung von f(x). x0: Startvektor (1D-Tensor oder Liste).
    method: "gd" (Gradient Descent), "adam" oder "lbfgs". Rückgabe: OptimizeResult (x_opt, f_opt) behaving like a tuple.
    """
    x = _to_tensor(x0).float().clone().detach().requires_grad_(True)
    if x.dim() != 1:
        x = x.flatten()
    n_params = x.numel()
    
    method_lower = str(method).lower()
    if method_lower == "lbfgs":
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
    elif method_lower == "adam":
        optimizer = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            out = f(x)
            out = _to_tensor(out).float()
            loss = out.sum() if out.numel() > 1 else out
            loss.backward()
            optimizer.step()
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
    return OptimizeResult(x.detach(), f_val)

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

def riemann_sum(f, a, b, n=100, method="midpoint"):
    """
    Riemann-Summe: Approximation von int_a^b f(x) dx.
    f: Callable mit einem Argument (Tensor); a, b: Integrationsgrenzen; n: Anzahl Teilintervalle.
    method: "left" (links), "right" (rechts), "midpoint" (Mittelpunkt, default).
    Rückgabe: Skalar-Tensor; differenzierbar bzgl. in f verwendeter Parameter.
    """
    a_val = float(_to_tensor(a).float().squeeze().item())
    b_val = float(_to_tensor(b).float().squeeze().item())
    n_int = _builtin_max(1, int(n))
    dx = (b_val - a_val) / n_int
    if method == "left":
        x = torch.linspace(a_val, b_val - dx, n_int)
    elif method == "right":
        x = torch.linspace(a_val + dx, b_val, n_int)
    else:  # midpoint
        x = torch.linspace(a_val + dx / 2.0, b_val - dx / 2.0, n_int)
    y = f(x)
    y = _to_tensor(y).float().flatten()
    if y.numel() != n_int:
        raise ValueError("riemann_sum: f(x) muss gleiche Länge wie Stützstellen haben.")
    return (dx * y.sum()).squeeze()

def zeta(s):
    """
    Riemann-Zeta-Funktion ζ(s) = Σ_{n=1}^∞ 1/n^s.
    s: reell oder komplex; Konvergenz für Re(s) > 1; analytische Fortsetzung für s ≠ 1.
    Nutzt scipy.special.zeta. Für s=1: harmonische Reihe divergiert → inf.
    """
    try:
        import scipy.special as sc  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("zeta(s) erfordert scipy. Bitte installieren: pip install scipy")
    s_t = _to_tensor(s).float()
    _abs = builtins.abs  # Python built-in für komplexe Zahlen
    if s_t.dim() == 0:
        s_val = complex(s_t.item())
        if _abs(s_val - 1.0) < 1e-12:
            return torch.tensor(float("inf"), dtype=torch.float32)
        out = sc.zeta(s_val, 1)  # Riemann zeta (q=1)
        return torch.tensor(float(out.real), dtype=torch.float32)
    out = torch.zeros_like(s_t)
    flat = s_t.flatten()
    for i in range(flat.numel()):
        v = complex(flat[i].item())
        if _abs(v - 1.0) < 1e-12:
            out.flatten()[i] = float("inf")
        else:
            out.flatten()[i] = float(sc.zeta(v, 1).real)
    return out

def volume_revolution_x(f, a, b, n=100):
    """
    Rotationskörper: Volumen bei Rotation von y=f(x) um die x-Achse.
    V = pi * int_a^b f(x)^2 dx (Kreisscheiben-Methode).
    f: Callable mit Tensor-Argument; a, b: Integrationsgrenzen; n: Stützstellen.
    Rückgabe: Skalar-Tensor; differenzierbar.
    """
    def g(x):
        y = f(x)
        y = _to_tensor(y).float().flatten()
        return y * y
    return (pi * integrate(g, a, b, n)).squeeze()

def volume_revolution_y(f, a, b, n=100):
    """
    Rotationskörper: Volumen bei Rotation von y=f(x) um die y-Achse (Mantel-Methode).
    V = 2*pi * int_a^b x * f(x) dx. Gültig für f(x)>=0, a,b>0 (oder angepasste Grenzen).
    f: Callable mit Tensor-Argument; a, b: Integrationsgrenzen; n: Stützstellen.
    Rückgabe: Skalar-Tensor; differenzierbar.
    """
    def g(x):
        y = f(x)
        y = _to_tensor(y).float().flatten()
        return x * y
    return (2.0 * pi * integrate(g, a, b, n)).squeeze()

def volume_revolution_vertical(f, a, b, x0, n=100):
    """
    Rotationskörper: Rotation von y=f(x) um vertikale Achse x=x0 (parallel zur y-Achse).
    V = 2*pi * int_a^b |x - x0| * f(x) dx (Mantel-Methode).
    f: Callable mit Tensor-Argument; a, b: Integrationsgrenzen; x0: Achsenposition; n: Stützstellen.
    """
    x0_val = float(_to_tensor(x0).float().squeeze().item())

    def g(x):
        y = f(x)
        y = _to_tensor(y).float().flatten()
        r = torch.abs(x.float().flatten() - x0_val)
        return r * y
    return (2.0 * pi * integrate(g, a, b, n)).squeeze()

def volume_revolution_horizontal(f, a, b, y0, n=100):
    """
    Rotationskörper: Rotation von y=f(x) um horizontale Achse y=y0 (parallel zur x-Achse).
    V = pi * int_a^b (f(x) - y0)^2 dx (Kreisscheiben-Methode).
    f sollte vollständig auf einer Seite von y0 liegen (nicht schneiden).
    """
    y0_val = float(_to_tensor(y0).float().squeeze().item())

    def g(x):
        y = f(x)
        y = _to_tensor(y).float().flatten()
        d = y - y0_val
        return d * d
    return (pi * integrate(g, a, b, n)).squeeze()

def pappus_volume_vertical(f, a, b, x0, n=100):
    """
    Satz von Pappus (Volumen): Rotation einer Fläche um vertikale Achse x=x0.
    V = 2*pi * R * A, wobei A = int f dx, R = |x̄ - x0|, x̄ = (1/A)*int x*f dx (Schwerpunkt).
    Äquivalent zu volume_revolution_vertical, aber über Schwerpunkt formuliert.
    """
    x0_val = float(_to_tensor(x0).float().squeeze().item())
    a_val = float(_to_tensor(a).float().squeeze().item())
    b_val = float(_to_tensor(b).float().squeeze().item())
    n_int = _builtin_max(2, int(n))
    x = torch.linspace(a_val, b_val, n_int)
    y = _to_tensor(f(x)).float().flatten()
    if y.numel() != n_int:
        raise ValueError("pappus_volume_vertical: f(x) muss gleiche Länge wie Stützstellen haben.")
    dx = (b_val - a_val) / (n_int - 1.0)
    A = (dx / 2.0) * (y[0] + 2.0 * y[1:-1].sum() + y[-1])
    My = (dx / 2.0) * ((x[0] * y[0]) + 2.0 * (x[1:-1] * y[1:-1]).sum() + (x[-1] * y[-1]))
    x_bar = My / (A + 1e-12)
    R = torch.abs(x_bar - x0_val)
    return (2.0 * pi * R * A).squeeze()

def pappus_volume_horizontal(f, a, b, y0, n=100):
    """
    Satz von Pappus (Volumen): Rotation einer Fläche um horizontale Achse y=y0.
    V = 2*pi * R * A, wobei A = int f dx, R = |ȳ - y0|, ȳ = (1/(2A))*int f^2 dx (y-Koordinate des Schwerpunkts).
    Äquivalent zu volume_revolution_horizontal, aber über Schwerpunkt formuliert.
    """
    y0_val = float(_to_tensor(y0).float().squeeze().item())
    a_val = float(_to_tensor(a).float().squeeze().item())
    b_val = float(_to_tensor(b).float().squeeze().item())
    n_int = _builtin_max(2, int(n))
    x = torch.linspace(a_val, b_val, n_int)
    y = _to_tensor(f(x)).float().flatten()
    if y.numel() != n_int:
        raise ValueError("pappus_volume_horizontal: f(x) muss gleiche Länge wie Stützstellen haben.")
    dx = (b_val - a_val) / (n_int - 1.0)
    A = (dx / 2.0) * (y[0] + 2.0 * y[1:-1].sum() + y[-1])
    y_sq = y * y
    Mx = (dx / 2.0) * (y_sq[0] + 2.0 * y_sq[1:-1].sum() + y_sq[-1]) * 0.5
    y_bar = Mx / (A + 1e-12)
    R = torch.abs(y_bar - y0_val)
    return (2.0 * pi * R * A).squeeze()

# --- Standard Library: Uncertainty Propagation (Gaussian) ---
# Fehlerfortpflanzung für Wissenschaftler: value ± std; Gauß'sche Näherung für +, -, *, /, ^.



def partial(u, x, order=1):
    """Berechnet partielle Ableitung(en) von `u` nach `x` via Autograd.

    Anders als `grad(fn, x)` (das eine Funktion an einer Stelle differenziert)
    arbeitet `partial` mit BEREITS BERECHNETEN Tensoren: `u = net(x)` wurde
    ausgewertet, `x` traegt `requires_grad=True`. Liefert Ôêéu/Ôêéx.

    - `u`: torch.Tensor (Skalar oder beliebige Form); wird ggf. summiert fuer
       skalare Backward-Aggregation, sodass der Gradient elementweise Ôêéu_i/Ôêéx bleibt.
    - `x`: Leaf-Tensor mit requires_grad=True (typisch via `.with_grad()`).
    - `order`: Ableitungsordnung (1, 2, ...). Rekursive Anwendung.

    Beispiel ÔÇö Heat-Equation Residuum:
        x = linspace(0, 1, 100).reshape(-1, 1).with_grad()
        u = net(x)
        u_x  = partial(u, x)
        u_xx = partial(u, x, order=2)
        residual = u_t - alpha * u_xx
    """
    if not isinstance(u, torch.Tensor):
        raise TypeError(f"partial: u muss torch.Tensor sein, bekam {type(u).__name__}.")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"partial: x muss torch.Tensor sein, bekam {type(x).__name__}.")
    if not x.requires_grad:
        raise ValueError(
            "partial: x.requires_grad ist False. Markiere den Eingabe-Tensor mit "
            "`.with_grad()` (z. B. `x = linspace(0,1,N).reshape(-1,1).with_grad()`)."
        )
    cur = u
    for _ in range(int(order)):
        target = cur if cur.dim() == 0 else cur.sum()
        grads = torch.autograd.grad(
            target, x, create_graph=True, retain_graph=True, allow_unused=False
        )[0]
        if grads is None:
            raise ValueError(
                "partial: Ableitung ist None. `u` haengt vermutlich nicht von `x` ab "
                "(z. B. Konstante uebergeben, oder Vorwaerts-Pass hat den Graph verloren)."
            )
        cur = grads
    return cur

class MultiVector:
    """3D Geometric Algebra G(3,0) Multivector mit 8 reellen Komponenten.

    Komponenten-Indizes (Bit-Pattern, kanonisch):
        0 = Skalar (1)
        1 = e1     2 = e2     4 = e3        (Vektoren)
        3 = e12    5 = e13    6 = e23       (Bivektoren ÔÇö orientierte Flaechen)
        7 = e123                              (Pseudoskalar ÔÇö orientiertes Volumen)

    Operationen:
        a + b, a - b, -a                  : komponentenweise
        a * b                              : geometrisches Produkt (zentrale GA-Operation)
        scalar * a, a * scalar             : Skalar-Multiplikation
        a.wedge(b)                         : outer/wedge product (Grad-Erhoehung)
        a.dot(b)                           : inner/dot product (Grad-Reduktion)
        a.reverse()                        : Reverse ÔÇö Sign pro Grad k: (-1)^(k(k-1)/2)
        a.grade(n)                         : Extrahiert nur Grad-n-Teil
        a.norm()                           : sqrt(<a * ~a>_0)
    """
    BASIS_NAMES = ["", "e1", "e2", "e12", "e3", "e13", "e23", "e123"]
    GRADES = [0, 1, 1, 2, 1, 2, 2, 3]  # Anzahl Bits pro Index = Grad

    __slots__ = ("c",)

    def __init__(self, components=None):
        if components is None:
            self.c = [0.0] * 8
        elif isinstance(components, MultiVector):
            self.c = list(components.c)
        else:
            comps = list(components)
            if len(comps) != 8:
                raise ValueError(
                    f"MultiVector erwartet 8 Komponenten, bekam {len(comps)}."
                )
            self.c = [float(x) for x in comps]

    @staticmethod
    def _gp_basis(a_bits, b_bits):
        """Geometrisches Produkt zweier Basis-Blades in G(3,0):
        Liefert (sign, result_bits). In G(3,0) annihilieren sich identische
        Faktoren (e_i * e_i = +1), daher result = a XOR b. Vorzeichen aus
        der Anzahl noetiger Swaps."""
        sign = 1
        for i in range(2, -1, -1):
            if a_bits & (1 << i):
                for j in range(i):
                    if b_bits & (1 << j):
                        sign = -sign
        return sign, a_bits ^ b_bits

    def __add__(self, other):
        if isinstance(other, (int, float)):
            r = MultiVector(self.c)
            r.c[0] += float(other)
            return r
        if isinstance(other, MultiVector):
            return MultiVector([a + b for a, b in zip(self.c, other.c)])
        return NotImplemented
    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            r = MultiVector(self.c)
            r.c[0] -= float(other)
            return r
        if isinstance(other, MultiVector):
            return MultiVector([a - b for a, b in zip(self.c, other.c)])
        return NotImplemented
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            r = MultiVector()
            r.c[0] = float(other)
            return r - self
        return NotImplemented

    def __neg__(self):
        return MultiVector([-x for x in self.c])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MultiVector([x * float(other) for x in self.c])
        if isinstance(other, MultiVector):
            r = [0.0] * 8
            for i in range(8):
                ai = self.c[i]
                if ai == 0.0:
                    continue
                for j in range(8):
                    bj = other.c[j]
                    if bj == 0.0:
                        continue
                    sign, k = MultiVector._gp_basis(i, j)
                    r[k] += sign * ai * bj
            return MultiVector(r)
        return NotImplemented
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def grade(self, n):
        """Extrahiert den Grad-n-Teil (0..3)."""
        n = int(n)
        return MultiVector([c if g == n else 0.0
                            for c, g in zip(self.c, MultiVector.GRADES)])

    def reverse(self):
        """Reverse: kehrt die Reihenfolge aller Basisvektoren um.
        Sign pro Grad k: (-1)^(k(k-1)/2). Fuer 3D: Grad 0,1: +, Grad 2,3: -."""
        signs = [(-1) ** (g * (g - 1) // 2) for g in MultiVector.GRADES]
        return MultiVector([s * c for s, c in zip(signs, self.c)])

    def wedge(self, other):
        """Outer (wedge) product: nur Grad-erhoehende Anteile des geom. Produkts.
        a ^ b hat Grad |grade(a) + grade(b)| (sofern nicht reduzierend)."""
        if isinstance(other, (int, float)):
            return MultiVector([c * float(other) for c in self.c])
        if not isinstance(other, MultiVector):
            return NotImplemented
        r = [0.0] * 8
        for i in range(8):
            ai = self.c[i]
            if ai == 0.0:
                continue
            for j in range(8):
                bj = other.c[j]
                if bj == 0.0:
                    continue
                sign, k = MultiVector._gp_basis(i, j)
                if MultiVector.GRADES[k] == MultiVector.GRADES[i] + MultiVector.GRADES[j]:
                    r[k] += sign * ai * bj
        return MultiVector(r)

    def dot(self, other):
        """Inner (dot) product: Grad-reduzierende Anteile des geom. Produkts.
        a . b hat Grad ||grade(a) - grade(b)||."""
        if isinstance(other, (int, float)):
            r = MultiVector(self.c)
            return r * float(other) if False else MultiVector([float(other) * self.c[0] if i == 0 else 0.0 for i in range(8)])
        if not isinstance(other, MultiVector):
            return NotImplemented
        r = [0.0] * 8
        for i in range(8):
            ai = self.c[i]
            if ai == 0.0:
                continue
            for j in range(8):
                bj = other.c[j]
                if bj == 0.0:
                    continue
                gi, gj = MultiVector.GRADES[i], MultiVector.GRADES[j]
                if gi == 0 or gj == 0:
                    continue  # Skalar . X ist nur Multiplikation, nicht inner
                sign, k = MultiVector._gp_basis(i, j)
                if MultiVector.GRADES[k] == abs(gi - gj):
                    r[k] += sign * ai * bj
        return MultiVector(r)

    def norm(self):
        """Magnitude = sqrt(<a * ~a>_0)."""
        s = (self * self.reverse()).c[0]
        if s < 0:
            return 0.0
        return s ** 0.5

    def scalar_part(self):
        """Reine Skalar-Komponente als float."""
        return float(self.c[0])

    def __repr__(self):
        parts = []
        for i, c in enumerate(self.c):
            if abs(c) < 1e-12:
                continue
            name = MultiVector.BASIS_NAMES[i]
            if not name:
                parts.append(f"{c:g}")
            elif abs(c - 1.0) < 1e-12:
                parts.append(name)
            elif abs(c + 1.0) < 1e-12:
                parts.append(f"-{name}")
            else:
                parts.append(f"{c:g}*{name}")
        if not parts:
            return "MultiVector(0)"
        return " + ".join(parts).replace("+ -", "- ")

def scalar(s):
    """Konstruktor: skalarer Multivector."""
    mv = MultiVector()
    mv.c[0] = float(s)
    return mv

def vector(x, y, z):
    """Konstruktor: 3D-Vektor als Multivector (x*e1 + y*e2 + z*e3)."""
    mv = MultiVector()
    mv.c[1] = float(x)  # e1
    mv.c[2] = float(y)  # e2
    mv.c[4] = float(z)  # e3
    return mv

def bivector(b12, b13, b23):
    """Konstruktor: Bivektor (orientierte Flaeche) b12*e12 + b13*e13 + b23*e23."""
    mv = MultiVector()
    mv.c[3] = float(b12)
    mv.c[5] = float(b13)
    mv.c[6] = float(b23)
    return mv

def pseudoscalar(s):
    """Konstruktor: Pseudoskalar (orientiertes Volumen) s*e123."""
    mv = MultiVector()
    mv.c[7] = float(s)
    return mv

def multivector(s, e1, e2, e3, e12, e13, e23, e123):
    """Konstruktor: vollstaendiger Multivector mit allen 8 Komponenten."""
    return MultiVector([s, e1, e2, e12, e3, e13, e23, e123])

def rotor(angle, b12, b13, b23):
    """Konstruiert einen Rotor R = exp(-angle/2 * B) fuer Rotation in der durch
    B = b12*e12 + b13*e13 + b23*e23 gegebenen Ebene.

    Der Bivektor sollte normalisiert sein (||B|| = 1) ÔÇö typisch:
      e12-Ebene (xy):  rotor(angle, 1, 0, 0)
      e13-Ebene (xz):  rotor(angle, 0, 1, 0)
      e23-Ebene (yz):  rotor(angle, 0, 0, 1)

    Anwendung via `rotate(v, R)`: v' = R v ~R.
    """
    import math as _math
    half = float(angle) / 2.0
    cos_h = _math.cos(half)
    sin_h = _math.sin(half)
    r = MultiVector()
    r.c[0] = cos_h
    r.c[3] = -sin_h * float(b12)
    r.c[5] = -sin_h * float(b13)
    r.c[6] = -sin_h * float(b23)
    return r

def rotate(v, R):
    """Rotation eines Multivectors via Sandwich-Produkt: v' = R v ~R.
    Fuer einen Unit-Rotor (||R|| = 1) ist ~R = R^{-1}."""
    if not isinstance(v, MultiVector):
        raise TypeError(f"rotate: v muss MultiVector sein, bekam {type(v).__name__}.")
    if not isinstance(R, MultiVector):
        raise TypeError(f"rotate: R muss MultiVector sein, bekam {type(R).__name__}.")
    return R * v * R.reverse()
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


# --- Standard Library: Medizin, Pharmakologie, Epidemiologie (Quick Wins) ---

def hill_equation(dose, Emax, EC50, n=1.0):
    """
    Hill-Gleichung: E = Emax * dose^n / (EC50^n + dose^n).
    dose, Emax, EC50, n: Tensor oder Skalar; differenzierbar.
    n: Steilheit (Hill-Koeffizient); n=1 entspricht Michaelis-Menten.
    """
    dose = _to_tensor(dose).float()
    Emax = _to_tensor(Emax).float()
    EC50 = _to_tensor(EC50).float()
    n_val = _to_tensor(n).float()
    return Emax * (dose ** n_val) / ((EC50 ** n_val) + (dose ** n_val))


def one_compartment_pk(C0, ke, t):
    """
    Ein-Kompartiment-Modell (Elimination 1. Ordnung): C(t) = C0 * exp(-ke*t).
    C0: Anfangskonzentration, ke: Eliminationskonstante (1/Zeit), t: Zeitgitter (1D).
    Rückgabe: Konzentration zu jedem Zeitpunkt (differenzierbar).
    """
    C0 = _to_tensor(C0).float()
    ke = _to_tensor(ke).float()
    t = _to_tensor(t).float()
    return C0 * torch.exp(-ke * t)


def two_compartment_pk(C0, k12, k21, ke, t):
    """
    Zwei-Kompartiment-Modell (IV Bolus in das zentrale Kompartiment):
    C1(t) = A * exp(-alpha * t) + B * exp(-beta * t)
    C0: Anfangskonzentration im zentralen Kompartiment
    k12: Transferrate von zentral zu peripher (1/Zeit)
    k21: Transferrate von peripher zu zentral (1/Zeit)
    ke: Eliminationsrate aus dem zentralen Kompartiment (1/Zeit)
    t: Zeitgitter (1D)
    Rückgabe: Konzentration im zentralen Kompartiment C1(t) (differenzierbar).
    """
    C0 = _to_tensor(C0).float()
    k12 = _to_tensor(k12).float()
    k21 = _to_tensor(k21).float()
    ke = _to_tensor(ke).float()
    t = _to_tensor(t).float()

    sum_k = ke + k12 + k21
    prod_k = ke * k21

    disc = sum_k ** 2 - 4.0 * prod_k
    disc = torch.clamp(disc, min=0.0)
    sqrt_disc = torch.sqrt(disc)

    alpha = 0.5 * (sum_k + sqrt_disc)
    beta = 0.5 * (sum_k - sqrt_disc)

    denom = torch.where(sqrt_disc < 1e-15, torch.tensor(1e-15, dtype=alpha.dtype, device=alpha.device), sqrt_disc)
    A = C0 * (alpha - k21) / denom
    B = C0 * (k21 - beta) / denom

    return A * torch.exp(-alpha * t) + B * torch.exp(-beta * t)


def half_life(ke):
    """
    Halbwertszeit aus Eliminationskonstante: t1/2 = ln(2) / ke.
    ke: Eliminationskonstante (1/Zeit). Rückgabe: Halbwertszeit (differenzierbar).
    """
    ke = _to_tensor(ke).float()
    return (0.6931471805599453) / ke  # ln(2)


def sir_model(S0, I0, R0, beta, gamma, t):
    """
    SIR-Kompartimentmodell: dS/dt = -beta*S*I, dI/dt = beta*S*I - gamma*I, dR/dt = gamma*I.
    S0, I0, R0: Anfangsbestände (Susceptible, Infected, Recovered). beta: Kontaktrate, gamma: Erholungsrate.
    t: Zeitgitter (1D). Rückgabe: (len(t), 3) – Spalten S, I, R.
    """
    def rhs(_, y):
        S, I, R = y[0], y[1], y[2]
        N = S + I + R
        if N < 1e-14:
            return torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)])
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return torch.stack([dS, dI, dR])
    y0_t = torch.tensor([float(S0), float(I0), float(R0)], dtype=torch.float32)
    return ode_solve(rhs, y0_t, t)


def basic_reproduction_number(beta, gamma):
    """
    Basisreproduktionszahl R0 = beta / gamma (SIR-Modell, homogene Population).
    beta: Kontaktrate, gamma: Erholungsrate. Rückgabe: R0 (differenzierbar).
    """
    beta = _to_tensor(beta).float()
    gamma = _to_tensor(gamma).float()
    return beta / gamma


def confidence_interval(x, alpha=0.05):
    """
    Konfidenzintervall für den Mittelwert (t-Verteilung).
    x: 1D-Tensor oder Liste. alpha: Signifikanzniveau (z.B. 0.05 für 95% CI).
    Rückgabe: (untere_Grenze, obere_Grenze).
    """
    import scipy.stats as scistats  # type: ignore[import-untyped]
    t = _to_tensor(x).float().flatten()
    n = t.numel()
    if n < 2:
        raise ValueError("confidence_interval: mindestens 2 Beobachtungen nötig.")
    m = t.mean().item()
    s = t.std(unbiased=True).item()
    se = s / (n ** 0.5)
    df = n - 1
    t_crit = scistats.t.ppf(1.0 - alpha / 2.0, df)
    lo = m - t_crit * se
    hi = m + t_crit * se
    return (float(lo), float(hi))


def odds_ratio(a, b, c, d):
    """
    Odds Ratio aus 2x2-Kontingenztafel:   | a  b |
                                          | c  d |
    OR = (a*d) / (b*c). Rückgabe: Odds Ratio (float).
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    if b * c < 1e-20:
        return float('inf') if a * d > 0 else float('nan')
    return (a * d) / (b * c)


def sensitivity_specificity(TP, FN, FP, TN):
    """
    Sensitivität, Spezifität, positiver/negativer prädiktiver Wert aus 2x2-Tabelle.
    TP, FN, FP, TN: True/False Positives/Negatives.
    Rückgabe: (sensitivity, specificity, PPV, NPV).
    """
    TP, FN, FP, TN = float(TP), float(FN), float(FP), float(TN)
    sens = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    spec = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
    ppv = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    npv = TN / (TN + FN) if (TN + FN) > 0 else float('nan')
    return (sens, spec, ppv, npv)

# --- Quick Wins: Musik (cents, equal temperament) ---
def cents_to_ratio(cents):
    """Cent zu Frequenzverhältnis: ratio = 2^(cents/1200). 100 cent = Halbton."""
    c = _to_tensor(cents).float()
    return torch.pow(2.0, c / 1200.0)


def ratio_to_cents(ratio):
    """Frequenzverhältnis zu Cent: cents = 1200 * log2(ratio)."""
    r = _to_tensor(ratio).float()
    return 1200.0 * torch.log2(torch.clamp(r, min=1e-30))


def equal_temperament(n, a4_hz=440.0):
    """
    Frequenz des n-ten Halbtons in gleichstufiger Stimmung (A4 = 440 Hz Referenz).
    n=0 → A4, n=12 → A5, n=-12 → A3. Rückgabe in Hz.
    """
    n_val = int(n) if isinstance(n, (int, float)) else int(_to_tensor(n).item())
    return a4_hz * (2.0 ** (n_val / 12.0))


# --- Quick Wins: Ökonomie (discount_factor, cobb_douglas, solow_rhs) ---
def discount_factor(r, t, discrete=False):
    """
    Barwertfaktor: continuous exp(-r*t) oder discrete 1/(1+r)^t.
    r: Zinssatz, t: Zeit; differenzierbar.
    """
    r_t = _to_tensor(r).float()
    t_t = _to_tensor(t).float()
    if discrete:
        return 1.0 / torch.pow(1.0 + r_t, t_t)
    return torch.exp(-r_t * t_t)


def cobb_douglas(K, L, alpha, A=1.0):
    """
    Cobb-Douglas Produktionsfunktion: Y = A * K^alpha * L^(1-alpha).
    K: Kapital, L: Arbeit, alpha: Kapitalanteil; differenzierbar.
    """
    K_t = _to_tensor(K).float()
    L_t = _to_tensor(L).float()
    a = _to_tensor(alpha).float()
    A_t = _to_tensor(A).float()
    return A_t * torch.pow(K_t, a) * torch.pow(L_t, 1.0 - a)


def solow_rhs(K, s, delta, n, g, alpha):
    """
    RHS für Solow-Wachstumsmodell: dK/dt = s*Y - (delta+n+g)*K mit Y = K^alpha.
    K: Kapital, s: Sparquote, delta: Abschreibung, n: Bevölkerungswachstum,
    g: technologischer Fortschritt, alpha: Kapitalanteil.
    Nutzung: fn rhs(t, y) { return [solow_rhs(y[0], s, delta, n, g, alpha)] }; ode_solve(rhs, [K0], t).
    """
    K_t = _to_tensor(K).float()
    Y = torch.pow(K_t, float(alpha))
    return float(s) * Y - (float(delta) + float(n) + float(g)) * K_t


# --- Quick Wins: Geologie (darcy_velocity) ---
def darcy_velocity(K, grad_P, mu):
    """
    Darcy-Gesetz: v = -(K/mu) * grad_P. Durchlässigkeit K [m² oder D], Viskosität mu [Pa·s], Druckgradient grad_P [Pa/m].
    Rückgabe: Geschwindigkeit [m/s]. K, grad_P, mu: Skalar oder Tensor; differenzierbar.
    """
    K_t = _to_tensor(K).float()
    g_t = _to_tensor(grad_P).float()
    mu_t = _to_tensor(mu).float()
    return -(K_t / mu_t) * g_t


# --- Quick Wins: Werkstoffe (Johnson-Mehl-Avrami) ---
def johnson_mehl_avrami(t, k, n):
    """
    Johnson-Mehl-Avrami-Kolmogorov: Umwandlungsanteil f(t) = 1 - exp(-(k*t)^n).
    t: Zeit, k: Geschwindigkeitskonstante, n: Avrami-Exponent; differenzierbar.
    """
    t_t = _to_tensor(t).float()
    k_t = _to_tensor(k).float()
    n_t = _to_tensor(n).float()
    return 1.0 - torch.exp(-torch.pow(k_t * t_t, n_t))


def avrami_rate(t, k, n):
    """
    Zeitableitung der JMAK-Umwandlung: df/dt = n*k*(k*t)^(n-1) * exp(-(k*t)^n).
    RHS für ode_solve: fn rhs(t, y) { return [avrami_rate(t, k, n)] }; ode_solve(rhs, [0], t).
    """
    t_t = _to_tensor(t).float()
    k_t = _to_tensor(k).float()
    n_t = _to_tensor(n).float()
    kt = k_t * t_t
    return n_t * k_t * torch.pow(kt, n_t - 1.0) * torch.exp(-torch.pow(kt, n_t))


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

# --- Standard Library: Life Sciences Extension (Phase 1) ---

def _get_bond_order(bond_char, aromatic1, aromatic2):
    if bond_char == '-': return 1.0
    if bond_char == '=': return 2.0
    if bond_char == '#': return 3.0
    if bond_char == ':': return 1.5
    if aromatic1 and aromatic2: return 1.5
    return 1.0

def _handle_ring(ring_num, last_idx, pending_bond, atoms, bonds, ring_closures):
    if last_idx is None:
        return
    if ring_num in ring_closures:
        partner_idx, r_bond_char = ring_closures.pop(ring_num)
        bond_char = pending_bond or r_bond_char
        order = _get_bond_order(bond_char, atoms[last_idx]["aromatic"], atoms[partner_idx]["aromatic"])
        bonds.append((last_idx, partner_idx, order))
    else:
        ring_closures[ring_num] = (last_idx, pending_bond)

def _parse_smiles(smiles):
    import re
    smiles = str(smiles).strip()
    atoms = []
    bonds = []
    branch_stack = []
    ring_closures = {}
    
    i = 0
    n = len(smiles)
    last_idx = None
    pending_bond = None
    
    while i < n:
        c = smiles[i]
        
        if c == '(':
            branch_stack.append(last_idx)
            i += 1
            continue
        elif c == ')':
            if branch_stack:
                last_idx = branch_stack.pop()
            i += 1
            continue
        elif c in ['-', '=', '#', ':']:
            pending_bond = c
            i += 1
            continue
        elif c == '.':
            last_idx = None
            pending_bond = None
            i += 1
            continue
        elif c == '%':
            if i + 2 < n and smiles[i+1:i+3].isdigit():
                ring_num = int(smiles[i+1:i+3])
                i += 3
            else:
                i += 1
                continue
            _handle_ring(ring_num, last_idx, pending_bond, atoms, bonds, ring_closures)
            pending_bond = None
            continue
        elif c.isdigit():
            ring_num = int(c)
            i += 1
            _handle_ring(ring_num, last_idx, pending_bond, atoms, bonds, ring_closures)
            pending_bond = None
            continue
        elif c == '[':
            j = smiles.find(']', i)
            if j == -1:
                raise ValueError(f"SMILES-Parser: Fehlende schließende Klammer bei Position {i}")
            content = smiles[i+1:j]
            i = j + 1
            
            m = re.match(r"^(\d*)([A-Z][a-z]?|[a-z])(@+)?(H\d*)?([\+\-]\d*|\d*[\+\-]|\++|\-+)?", content)
            if not m:
                elem = "C"
                aromatic = False
                explicit_h = 0
                charge = 0
            else:
                elem_raw = m.group(2)
                aromatic = elem_raw.islower()
                elem = elem_raw.capitalize()
                
                h_str = m.group(4)
                explicit_h = 0
                if h_str:
                    explicit_h = 1 if len(h_str) == 1 else int(h_str[1:])
                    
                charge_str = m.group(5)
                charge = 0
                if charge_str:
                    if charge_str == '+': charge = 1
                    elif charge_str == '-': charge = -1
                    elif '+' in charge_str:
                        parts = charge_str.replace('+', '')
                        charge = int(parts) if parts else charge_str.count('+')
                    elif '-' in charge_str:
                        parts = charge_str.replace('-', '')
                        charge = -int(parts) if parts else -charge_str.count('-')
                        
            new_atom = {
                "elem": elem,
                "aromatic": aromatic,
                "charge": charge,
                "explicit_h": explicit_h,
                "is_bracket": True
            }
            atoms.append(new_atom)
            new_idx = len(atoms) - 1
            
            if last_idx is not None:
                order = _get_bond_order(pending_bond, atoms[last_idx]["aromatic"], aromatic)
                bonds.append((last_idx, new_idx, order))
                
            last_idx = new_idx
            pending_bond = None
            continue
        else:
            elem_raw = None
            if smiles[i:i+2] in ['Cl', 'Br', 'cl', 'br']:
                elem_raw = smiles[i:i+2]
                i += 2
            elif c in ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'I', 'b', 'c', 'n', 'o', 'p', 's']:
                elem_raw = c
                i += 1
            else:
                i += 1
                continue
                
            aromatic = elem_raw.islower()
            elem = elem_raw.capitalize()
            
            new_atom = {
                "elem": elem,
                "aromatic": aromatic,
                "charge": 0,
                "explicit_h": 0,
                "is_bracket": False
            }
            atoms.append(new_atom)
            new_idx = len(atoms) - 1
            
            if last_idx is not None:
                order = _get_bond_order(pending_bond, atoms[last_idx]["aromatic"], aromatic)
                bonds.append((last_idx, new_idx, order))
                
            last_idx = new_idx
            pending_bond = None
            continue
            
    STANDARD_VALENCES = {
        "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3,
        "F": 1, "Cl": 1, "Br": 1, "I": 1
    }
    
    counts = {}
    for idx, atom in enumerate(atoms):
        elem = atom["elem"]
        if atom["is_bracket"]:
            h_count = atom["explicit_h"]
        else:
            conn_bonds = [order for (i1, i2, order) in bonds if i1 == idx or i2 == idx]
            tot_order = sum(conn_bonds)
            val = STANDARD_VALENCES.get(elem, 0)
            h_count = int(round(_builtin_max(0.0, float(val) - tot_order)))
            
        counts[elem] = counts.get(elem, 0) + 1
        if h_count > 0:
            counts["H"] = counts.get("H", 0) + h_count
            
    return counts, atoms, bonds

def smiles_molecular_weight(smiles):
    """
    Berechnet das Molekulargewicht eines SMILES-Strings (g/mol).
    smiles: String. Rückgabe: Quantity(Wert, "g/mol").
    """
    counts, _, _ = _parse_smiles(smiles)
    mw = sum(ATOMIC_MASSES.get(elem, 0.0) * count for elem, count in counts.items())
    return Quantity(mw, "g/mol")

def pubchem_get_molecular_formula(name):
    """
    Fragt die PubChem API nach der Summenformel eines Verbindungsnamens.
    Weicht bei Verbindungsfehlern transparent auf ein internes Wörterbuch aus.
    """
    n = str(name).strip().lower()
    
    PUBCHEM_CACHE = {
        "aspirin": "C9H8O4",
        "caffeine": "C8H10N4O2",
        "water": "H2O",
        "wasser": "H2O",
        "ethanol": "C2H6O",
        "ibuprofen": "C13H18O2",
        "paracetamol": "C8H9NO2",
        "glucose": "C6H12O6",
        "methane": "CH4",
        "methan": "CH4",
        "benzene": "C6H6",
        "benzol": "C6H6",
        "pyridine": "C5H5N",
        "nicotine": "C10H14N2",
        "urea": "CH4N2O",
        "harnstoff": "CH4N2O",
        "penicillin": "C16H18N2O4S",
        "taxol": "C47H51NO14"
    }
    
    try:
        import urllib.request
        import urllib.parse
        import json
        safe_name = urllib.parse.quote(n)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/MolecularFormula/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'DedekindLifeSciences/1.0'})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            formula = data["PropertyTable"]["Properties"][0]["MolecularFormula"]
            return str(formula)
    except Exception:
        if n in PUBCHEM_CACHE:
            return PUBCHEM_CACHE[n]
        raise ValueError(f"pubchem_get_molecular_formula: Verbindung fehlgeschlagen und Name '{name}' nicht im lokalen Cache.")

def chembl_get_ic50(target, compound):
    """
    Fragt die ChEMBL-Datenbank nach dem IC50-Wert einer Substanz gegen ein Target.
    Weicht bei Verbindungsfehlern transparent auf ein internes Wörterbuch aus.
    """
    tgt = str(target).strip().lower()
    cmpd = str(compound).strip().lower()
    
    CHEMBL_CACHE = {
        ("cox-1", "aspirin"): 100000.0,
        ("chembl1914", "aspirin"): 15000.0,
        ("cox-2", "ibuprofen"): 10000.0,
        ("egfr", "gefitinib"): 33.0,
        ("chembl203", "gefitinib"): 33.0,
        ("herg", "aspirin"): 100000.0,
        ("chembl240", "aspirin"): 100000.0
    }
    
    try:
        import urllib.request
        import urllib.parse
        import json
        safe_tgt = urllib.parse.quote(tgt)
        safe_cmpd = urllib.parse.quote(cmpd)
        url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={safe_tgt}&molecule_pref_name__iexact={safe_cmpd}&standard_type=IC50"
        if not tgt.startswith("chembl"):
            url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_components__target_protein__keyword={safe_tgt}&molecule_pref_name__iexact={safe_cmpd}&standard_type=IC50"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'DedekindLifeSciences/1.0'})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            activities = data.get("activities", [])
            if activities:
                for act in activities:
                    val = act.get("standard_value")
                    if val is not None:
                        return Quantity(float(val), "nM")
            raise ValueError("Keine Aktivitäten gefunden.")
    except Exception:
        key = (tgt, cmpd)
        if key in CHEMBL_CACHE:
            return Quantity(CHEMBL_CACHE[key], "nM")
        raise ValueError(f"chembl_get_ic50: Verbindung fehlgeschlagen und Paar ({target}, {compound}) nicht im lokalen Cache.")


# --- Standard Library: Life Sciences Extension (Phase 2) ---

def smith_waterman_alignment(seq1, seq2, match_score=2.0, mismatch_penalty=-1.0, gap_penalty=-1.0):
    """
    Lokales Sequenzalignment nach dem Smith-Waterman-Algorithmus in PyTorch.
    seq1, seq2: Strings oder 1D-Tensors.
    Rückgabe: Dict mit "score" (Tensor), "aligned_seq1" und "aligned_seq2" (Strings oder Listen).
    """
    is_str = isinstance(seq1, str) and isinstance(seq2, str)
    
    if isinstance(seq1, str):
        s1_arr = list(seq1)
    elif hasattr(seq1, "tolist"):
        s1_arr = seq1.tolist()
    else:
        s1_arr = list(seq1)

    if isinstance(seq2, str):
        s2_arr = list(seq2)
    elif hasattr(seq2, "tolist"):
        s2_arr = seq2.tolist()
    else:
        s2_arr = list(seq2)

    M, N = len(s1_arr), len(s2_arr)
    
    device = "cpu"
    if hasattr(seq1, "device"):
        device = seq1.device
    elif hasattr(seq2, "device"):
        device = seq2.device

    import torch
    H = torch.zeros((M + 1, N + 1), device=device, dtype=torch.float32)
    
    match = float(match_score)
    mismatch = float(mismatch_penalty)
    gap = float(gap_penalty)
    
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            val_match = match if s1_arr[i - 1] == s2_arr[j - 1] else mismatch
            diag = H[i - 1, j - 1] + val_match
            up = H[i - 1, j] + gap
            left = H[i, j - 1] + gap
            H[i, j] = torch.max(
                torch.tensor(0.0, device=device),
                torch.max(diag, torch.max(up, left))
            )
            
    max_val = torch.max(H)
    max_pos = torch.argmax(H)
    i_max, j_max = int(max_pos.item() // (N + 1)), int(max_pos.item() % (N + 1))
    
    aligned1 = []
    aligned2 = []
    
    curr_i, curr_j = i_max, j_max
    
    while curr_i > 0 and curr_j > 0:
        val = H[curr_i, curr_j].item()
        if val == 0.0:
            break
            
        val_match = match if s1_arr[curr_i - 1] == s2_arr[curr_j - 1] else mismatch
        
        if abs(H[curr_i - 1, curr_j - 1].item() + val_match - val) < 1e-4:
            aligned1.append(s1_arr[curr_i - 1])
            aligned2.append(s2_arr[curr_j - 1])
            curr_i -= 1
            curr_j -= 1
        elif abs(H[curr_i - 1, curr_j].item() + gap - val) < 1e-4:
            aligned1.append(s1_arr[curr_i - 1])
            aligned2.append("-" if is_str else None)
            curr_i -= 1
        elif abs(H[curr_i, curr_j - 1].item() + gap - val) < 1e-4:
            aligned1.append("-" if is_str else None)
            aligned2.append(s2_arr[curr_j - 1])
            curr_j -= 1
        else:
            aligned1.append(s1_arr[curr_i - 1])
            aligned2.append(s2_arr[curr_j - 1])
            curr_i -= 1
            curr_j -= 1
            
    aligned1.reverse()
    aligned2.reverse()
    
    if is_str:
        aligned1_out = "".join(aligned1)
        aligned2_out = "".join(aligned2)
    else:
        aligned1_out = aligned1
        aligned2_out = aligned2
        
    return {
        "score": max_val,
        "aligned_seq1": aligned1_out,
        "aligned_seq2": aligned2_out
    }

def protein_structure_parse(path_or_content):
    """
    Parst eine PDB- oder mmCIF-Proteinstruktur (aus Datei oder String).
    Rückgabe: DataFrame mit x, y, z Spalten in 'angstrom'.
    """
    import os
    content = ""
    is_file = False
    if isinstance(path_or_content, str):
        if len(path_or_content) < 500 and ("/" in path_or_content or "\\" in path_or_content or path_or_content.lower().endswith(('.pdb', '.cif', '.mmcif'))):
            if os.path.exists(path_or_content):
                is_file = True
                with open(path_or_content, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
    if not is_file:
        content = str(path_or_content)
        
    is_cif = "_atom_site." in content
    
    group_list = []
    atom_id_list = []
    atom_name_list = []
    res_name_list = []
    chain_id_list = []
    res_seq_list = []
    x_list = []
    y_list = []
    z_list = []
    occupancy_list = []
    b_factor_list = []
    element_list = []
    
    if is_cif:
        lines = content.splitlines()
        in_loop = False
        headers = []
        data_rows = []
        i = 0
        n_lines = len(lines)
        while i < n_lines:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line == "loop_":
                in_loop = True
                headers = []
                data_rows = []
                i += 1
                while i < n_lines:
                    nxt = lines[i].strip()
                    if nxt.startswith("_atom_site."):
                        headers.append(nxt)
                        i += 1
                    else:
                        break
                while i < n_lines:
                    nxt = lines[i].strip()
                    if not nxt or nxt.startswith("#") or nxt == "loop_":
                        break
                    import shlex
                    try:
                        parts = shlex.split(nxt)
                    except Exception:
                        parts = nxt.split()
                    if len(parts) >= len(headers):
                        data_rows.append(parts[:len(headers)])
                    i += 1
                if any(h.startswith("_atom_site.") for h in headers):
                    break
            else:
                i += 1
        
        if headers:
            h_map = {h: idx for idx, h in enumerate(headers)}
            
            def get_val(row, *keys):
                for k in keys:
                    if k in h_map:
                        return row[h_map[k]]
                return None
                
            for row in data_rows:
                group = get_val(row, "_atom_site.group_PDB") or "ATOM"
                
                atom_id_str = get_val(row, "_atom_site.id") or "0"
                atom_id = int(atom_id_str)
                
                element = get_val(row, "_atom_site.type_symbol") or ""
                
                atom_name = get_val(row, "_atom_site.label_atom_id", "_atom_site.auth_atom_id") or ""
                
                res_name = get_val(row, "_atom_site.label_comp_id", "_atom_site.auth_comp_id") or ""
                
                chain_id = get_val(row, "_atom_site.auth_asym_id", "_atom_site.label_asym_id") or ""
                
                res_seq_str = get_val(row, "_atom_site.auth_seq_id", "_atom_site.label_seq_id") or "0"
                res_seq = int(''.join(filter(str.isdigit, res_seq_str)) or "0")
                
                x_str = get_val(row, "_atom_site.Cartn_x") or "0.0"
                y_str = get_val(row, "_atom_site.Cartn_y") or "0.0"
                z_str = get_val(row, "_atom_site.Cartn_z") or "0.0"
                
                occ_str = get_val(row, "_atom_site.occupancy") or "1.0"
                b_str = get_val(row, "_atom_site.B_iso_or_equiv") or "0.0"
                
                group_list.append(group)
                atom_id_list.append(atom_id)
                atom_name_list.append(atom_name)
                res_name_list.append(res_name)
                chain_id_list.append(chain_id)
                res_seq_list.append(res_seq)
                x_list.append(float(x_str))
                y_list.append(float(y_str))
                z_list.append(float(z_str))
                occupancy_list.append(float(occ_str))
                b_factor_list.append(float(b_str))
                element_list.append(element)
    else:
        for line in content.splitlines():
            if line.startswith("ATOM  ") or line.startswith("HETATM"):
                group = line[0:6].strip()
                try:
                    atom_id = int(line[6:11].strip())
                except ValueError:
                    atom_id = 0
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    res_seq = 0
                try:
                    x = float(line[30:38].strip())
                except ValueError:
                    x = 0.0
                try:
                    y = float(line[38:46].strip())
                except ValueError:
                    y = 0.0
                try:
                    z = float(line[46:54].strip())
                except ValueError:
                    z = 0.0
                try:
                    occupancy = float(line[54:60].strip())
                except ValueError:
                    occupancy = 1.0
                try:
                    b_factor = float(line[60:66].strip())
                except ValueError:
                    b_factor = 0.0
                element = line[76:78].strip()
                if not element:
                    if atom_name:
                        element = atom_name[0]
                        if element.isdigit():
                            element = atom_name[1] if len(atom_name) > 1 else ""
                
                group_list.append(group)
                atom_id_list.append(atom_id)
                atom_name_list.append(atom_name)
                res_name_list.append(res_name)
                chain_id_list.append(chain_id)
                res_seq_list.append(res_seq)
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
                occupancy_list.append(occupancy)
                b_factor_list.append(b_factor)
                element_list.append(element)
                
    import torch
    data = {
        "group": group_list,
        "atom_id": torch.tensor(atom_id_list, dtype=torch.int32),
        "atom_name": atom_name_list,
        "res_name": res_name_list,
        "chain_id": chain_id_list,
        "res_seq": torch.tensor(res_seq_list, dtype=torch.int32),
        "x": torch.tensor(x_list, dtype=torch.float32),
        "y": torch.tensor(y_list, dtype=torch.float32),
        "z": torch.tensor(z_list, dtype=torch.float32),
        "occupancy": torch.tensor(occupancy_list, dtype=torch.float32),
        "b_factor": torch.tensor(b_factor_list, dtype=torch.float32),
        "element": element_list
    }
    
    return DataFrame(data, units={"x": "angstrom", "y": "angstrom", "z": "angstrom"})


# --- Standard Library: Astrophysik & Kosmologie ---

def solve_kepler(M, e):
    """
    Löst die Kepler-Gleichung E - e * sin(E) = M für die exzentrische Anomalie E.
    M: Mittlere Anomalie (Skalar, Tensor oder Quantity; in rad oder deg).
    e: Exzentrizität (Skalar, Tensor oder Quantity; 0 <= e < 1).
    Rückgabe: Exzentrische Anomalie E (rad, voll differenzierbar).
    """
    if isinstance(M, Quantity):
        M_val = _convert_to_base(M.value, M.unit, "angle")
    else:
        M_val = M
    if isinstance(e, Quantity):
        e_val = e.value
    else:
        e_val = e
    M_t = _to_tensor(M_val).float()
    e_t = _to_tensor(e_val).float()
    E = M_t.clone()
    for _ in range(6):
        E = E - (E - e_t * torch.sin(E) - M_t) / (1.0 - e_t * torch.cos(E))
    return E


def redshift_to_velocity(z):
    """
    Berechnet die Fluchtgeschwindigkeit v aus der Rotverschiebung z (relativistisch).
    z: Rotverschiebung (dimensionslos; Skalar, Tensor oder Quantity).
    Rückgabe: Fluchtgeschwindigkeit als Quantity [m/s] (wenn Skalar) oder Tensor.
    """
    if isinstance(z, Quantity):
        z_val = z.value
    else:
        z_val = z
    z_t = _to_tensor(z_val).float()
    c_val = float(c.value)  # 299792458.0 m/s
    v_val = c_val * (((1.0 + z_t)**2 - 1.0) / ((1.0 + z_t)**2 + 1.0))
    if z_t.requires_grad:
        return v_val
    if v_val.numel() == 1:
        return Quantity(float(v_val.item()), "m/s")
    return v_val


def schwarzschild_radius(M):
    """
    Berechnet den Schwarzschild-Radius Rs für eine Masse M.
    M: Masse (Skalar, Tensor oder Quantity [kg] / [M_sun]).
    Rückgabe: Schwarzschild-Radius als Quantity [m] (wenn Skalar) oder Tensor.
    """
    if isinstance(M, Quantity):
        M_val = _convert_to_base(M.value, M.unit, "mass")
    else:
        M_val = M
    M_t = _to_tensor(M_val).float()
    G_val = float(G.value)  # 6.6743e-11
    c_val = float(c.value)  # 299792458.0
    r_val = (2.0 * G_val * M_t) / (c_val ** 2)
    if M_t.requires_grad:
        return r_val
    if r_val.numel() == 1:
        return Quantity(float(r_val.item()), "m")
    return r_val


def stellar_luminosity(M_solar):
    """
    Berechnet die Leuchtkraft L (in solaren Leuchtkräften L_sun) aus der Masse M (in solaren Massen M_sun oder kg).
    M_solar: Masse des Hauptreihensterns (Skalar, Tensor oder Quantity).
    Rückgabe: Leuchtkraft L als Quantity [L_sun] (wenn Skalar) oder Tensor.
    """
    if isinstance(M_solar, Quantity):
        # Konvertiere erst in kg (Basis) und dann in M_sun
        M_kg = _convert_to_base(M_solar.value, M_solar.unit, "mass")
        M_val = M_kg / 1.98847e30
    else:
        M_val = M_solar
    M_t = _to_tensor(M_val).float()
    cond1 = M_t < 0.43
    cond2 = (M_t >= 0.43) & (M_t < 2.0)
    cond3 = (M_t >= 2.0) & (M_t < 20.0)
    
    L_val = torch.where(cond1, 0.23 * torch.pow(M_t, 2.3),
            torch.where(cond2, torch.pow(M_t, 4.0),
            torch.where(cond3, 1.4 * torch.pow(M_t, 3.5),
            32000.0 * M_t)))
    if M_t.requires_grad:
        return L_val
    if L_val.numel() == 1:
        return Quantity(float(L_val.item()), "L_sun")
    return L_val


# --- Standard Library: Meteorologie, Klimatologie & Geowissenschaften ---

def coriolis_parameter(latitude):
    """
    Berechnet den Coriolis-Parameter f = 2 * Omega * sin(lat) für eine Breite.
    latitude: Geographische Breite (Skalar, Tensor oder Quantity [deg]/[rad]).
    Rückgabe: f als Quantity [s^-1] oder Tensor.
    """
    if isinstance(latitude, Quantity):
        lat_val = _convert_to_base(latitude.value, latitude.unit, "angle")
    else:
        lat_val = latitude
    lat_t = _to_tensor(lat_val).float()
    omega = 7.2921159e-5  # rad/s, Winkelgeschwindigkeit der Erdrotation
    f_val = 2.0 * omega * torch.sin(lat_t)
    if lat_t.requires_grad:
        return f_val
    if f_val.numel() == 1:
        return Quantity(float(f_val.item()), "s^-1")
    return f_val


def saturated_vapor_pressure(T):
    """
    Berechnet den Sättigungsdampfdruck von Wasserdampf über flüssigem Wasser nach der Magnus-Formel.
    T: Temperatur (Skalar, Tensor oder Quantity; K).
    Rückgabe: Sättigungsdampfdruck als Quantity [Pa] oder Tensor.
    """
    if isinstance(T, Quantity):
        T_val = _convert_to_base(T.value, T.unit, "temperature")
    else:
        T_val = T
    T_t = _to_tensor(T_val).float()
    T_c = T_t - 273.15  # Umrechnung in Grad Celsius für Magnus-Formel
    es_val = 611.2 * torch.exp(17.67 * T_c / (T_c + 243.5))
    if T_t.requires_grad:
        return es_val
    if es_val.numel() == 1:
        return Quantity(float(es_val.item()), "Pa")
    return es_val


def dew_point(T, RH):
    """
    Berechnet den Taupunkt T_d aus der Temperatur T und der relativen Luftfeuchtigkeit RH.
    T: Temperatur (Skalar, Tensor oder Quantity [K]).
    RH: Relative Luftfeuchtigkeit (Skalar, Tensor; 0 bis 100).
    Rückgabe: Taupunkt als Quantity [K] oder Tensor.
    """
    if isinstance(T, Quantity):
        T_val = _convert_to_base(T.value, T.unit, "temperature")
    else:
        T_val = T
    if isinstance(RH, Quantity):
        RH_val = RH.value
    else:
        RH_val = RH
    T_t = _to_tensor(T_val).float()
    RH_t = _to_tensor(RH_val).float()
    
    T_c = T_t - 273.15
    RH_clamped = torch.clamp(RH_t, min=1e-5)
    y = torch.log(RH_clamped / 100.0) + (17.67 * T_c) / (T_c + 243.5)
    Td_c = (243.5 * y) / (17.67 - y)
    Td_val = Td_c + 273.15
    if T_t.requires_grad or RH_t.requires_grad:
        return Td_val
    if Td_val.numel() == 1:
        return Quantity(float(Td_val.item()), "K")
    return Td_val


def seismic_wave_velocities(K, G, rho):
    """
    Berechnet die Ausbreitungsgeschwindigkeiten von P-Wellen (Kompressionswellen) und S-Wellen (Scherwellen).
    K: Kompressionsmodul (Bulk modulus, Pa oder Quantity).
    G: Schermodul (Shear modulus, Pa oder Quantity).
    rho: Dichte (Density, kg/m^3 oder Quantity; unterstützt auch g/cm^3).
    Rückgabe: Liste [v_p, v_s] als Quantities [m/s] oder Tensoren.
    """
    if isinstance(K, Quantity):
        K_val = _convert_to_base(K.value, K.unit, "pressure")
    else:
        K_val = K
    if isinstance(G, Quantity):
        G_val = _convert_to_base(G.value, G.unit, "pressure")
    else:
        G_val = G
    if isinstance(rho, Quantity):
        u = str(rho.unit).strip()
        if u in ("g/cm^3", "g/cm^3"):
            rho_val = rho.value * 1000.0
        else:
            rho_val = rho.value
    else:
        rho_val = rho
        
    K_t = _to_tensor(K_val).float()
    G_t = _to_tensor(G_val).float()
    rho_t = _to_tensor(rho_val).float()
    
    vp_val = torch.sqrt((K_t + 4.0/3.0 * G_t) / rho_t)
    vs_val = torch.sqrt(G_t / rho_t)
    
    if K_t.requires_grad or G_t.requires_grad or rho_t.requires_grad:
        return [vp_val, vs_val]
    if vp_val.numel() == 1:
        return [Quantity(float(vp_val.item()), "m/s"), Quantity(float(vs_val.item()), "m/s")]
    return [vp_val, vs_val]






def fidelity(sv1, sv2):
    """Fidelitaet |Ôƒ¿¤ê1|¤ê2Ôƒ®|┬▓ zwischen zwei Statevektoren."""
    if len(sv1) != len(sv2):
        raise ValueError(f"fidelity: Vektoren muessen gleiche Laenge haben, bekam {len(sv1)} und {len(sv2)}.")
    inner = sum(a.conjugate() * b for a, b in zip(sv1, sv2))
    return abs(inner) ** 2

def entropy_von_neumann(probs):
    """Von-Neumann-Entropie S = -╬ú p_i log2(p_i) (aus Wahrscheinlichkeiten)."""
    if hasattr(probs, 'tolist'):
        probs = probs.tolist()
    s = 0.0
    for p in probs:
        p = abs(float(p))
        if p > 1e-15:
            s -= p * _math.log2(p)
    return s

def schmidt_rank(sv, n_a):
    """Schmidt-Rang des Statevektors fuer bipartite Zerlegung A|B.

    Args:
        sv: Statevector (Liste komplexer Zahlen, Laenge 2^n)
        n_a: Anzahl Qubits in Teilsystem A

    Returns:
        Schmidt-Rang (int)
    """
    import math
    n = int(math.log2(len(sv)))
    n_b = n - n_a
    dim_a = 1 << n_a
    dim_b = 1 << n_b
    # Reshape als Matrix dim_a x dim_b
    mat = [[0+0j] * dim_b for _ in range(dim_a)]
    for idx, amp in enumerate(sv):
        i = idx >> n_b
        j = idx & (dim_b - 1)
        mat[i][j] = amp
    # SVD via Python (kein scipy/numpy noetig fuer kleine Systeme)
    try:
        import torch as _t
        T = _t.tensor([[mat[i][j] for j in range(dim_b)] for i in range(dim_a)],
                      dtype=_t.complex64)
        sv_vals = _t.linalg.svdvals(T)
        return int((sv_vals > 1e-8).sum().item())
    except Exception:
        # Fallback: Rang der Matrix via Gauss
        return _matrix_rank_approx(mat, dim_a, dim_b)

def _matrix_rank_approx(mat, rows, cols, tol=1e-8):
    """Naiver Gauss-Rank (fuer kleine Matrizen wenn torch fehlt)."""
    m = [row[:] for row in mat]
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if abs(m[row][col]) > tol:
                pivot = row
                break
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        for row in range(rows):
            if row != rank and abs(m[row][col]) > tol:
                factor = m[row][col] / m[rank][col]
                m[row] = [m[row][j] - factor * m[rank][j] for j in range(cols)]
        rank += 1
    return rank

def qubit_frequency_check(freq):
    """Prueft, dass freq eine Frequenz-Einheit ([GHz], [MHz], [THz]) hat.

    Gibt den Wert in GHz zurueck.
    """
    if not isinstance(freq, Quantity):
        raise TypeError(
            f"qubit_frequency_check: Erwartet Quantity mit Frequenz-Einheit ([GHz]), "
            f"bekam {type(freq).__name__}."
        )
    freq_units = {"Hz", "kHz", "MHz", "GHz", "THz"}
    if freq.unit not in freq_units:
        raise TypeError(
            f"qubit_frequency_check: Einheit [{freq.unit}] ist keine Frequenzeinheit. "
            f"Erlaubt: {freq_units}."
        )
    # Konvertiere zu GHz
    dim_info = DIMENSION_TO_BASE["frequency"]
    base_val = float(freq.value) * dim_info[1].get(freq.unit, 1.0)  # -> Hz
    return Quantity(base_val / 1e9, "GHz")

def coherence_time_check(t):
    """Prueft Kohaerenzzeit (muss [us], [ns], [ms] oder [s] sein)."""
    if not isinstance(t, Quantity):
        raise TypeError(
            f"coherence_time_check: Erwartet Quantity mit Zeit-Einheit ([us]), "
            f"bekam {type(t).__name__}."
        )
    time_units = {"s", "ms", "us", "ns", "ps", "fs"}
    if t.unit not in time_units:
        raise TypeError(
            f"coherence_time_check: Einheit [{t.unit}] ist keine Zeiteinheit."
        )
    return t

def energy_gap_check(E):
    """Prueft Energieluecke (muss [eV], [meV], [J] sein)."""
    if not isinstance(E, Quantity):
        raise TypeError(
            f"energy_gap_check: Erwartet Quantity mit Energie-Einheit ([eV]), "
            f"bekam {type(E).__name__}."
        )
    energy_units = {"J", "kJ", "MJ", "eV", "meV", "MeV"}
    if E.unit not in energy_units:
        raise TypeError(
            f"energy_gap_check: Einheit [{E.unit}] ist keine Energieeinheit."
        )
    return E
def read_file(path):
    """
    Reads a file as text (UTF-8).
    path: string (file path).
    Returns: string (content).
    """
    p = str(path)
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(path, content):
    """
    Writes text to a file (UTF-8); overwrites if it exists.
    path: string (file path). content: string (content).
    """
    p = str(path)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(str(content))


def file_exists(path):
    """
    Checks whether a file exists.
    path: string (file path). Returns: bool.
    """
    import os
    return os.path.isfile(str(path))


def http_get(url):
    """
    HTTP GET request; returns response text (UTF-8).
    url: string (e.g. \"https://example.com\").
    """
    import urllib.request
    req = urllib.request.Request(str(url), method='GET')
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8')


def http_post(url, data):
    """
    HTTP POST request. data: string (body) or dict/list (sent as JSON).
    url: string. Returns: response text (UTF-8).
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
    Parses a JSON string into an object (dict/list); access e.g. obj[\"key\"].
    s: string (valid JSON).
    """
    import json
    return json.loads(str(s))


def json_stringify(obj):
    """
    Converts an object (dict, list, number, string) into a JSON string.
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
    Checks a condition; raises AssertionError when condition is false.
    assert(condition) or assert(condition, "error message").
    """
    val = condition
    if hasattr(val, "item") and (hasattr(val, "numel") or hasattr(val, "size")):
        size = val.numel() if hasattr(val, "numel") else val.size
        val = val.item() if size == 1 else bool(val.all().item())
    elif hasattr(val, "__len__") and len(val) == 1:
        val = val[0]
    if not bool(val):
        raise AssertionError(message if message is not None else "Assertion failed")

# --- Standard library: symbolic differentiation (embedded, no import) ---
# Simple AST for expressions
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
    """Tokenizer without re: character by character."""
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
                raise ValueError("Invalid number: " + s[start:i])
            tokens.append(("NUMBER", value))
            continue
        if c.isalpha() or c == "_":
            start = i
            while i < n and (s[i].isalnum() or s[i] == "_"):
                i += 1
            tokens.append(("ID", s[start:i]))
            continue
        raise ValueError("Unexpected character: " + repr(c))
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
                raise ValueError("Missing ')'")
            return e
        raise ValueError(f"Unexpected token: {p}")

def _sym_parse(expr_str):
    s = expr_str.strip().replace(" ", "")
    if not s:
        raise ValueError("Empty expression")
    tokens = _sym_tokenize(s)
    if not tokens:
        raise ValueError("No valid tokens")
    p = _SymParser(tokens)
    e = p._parse_expr()
    if p.pos < len(tokens):
        raise ValueError("Trailing content after expression")
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
    Symbolic differentiation: differentiates the expression expr with respect to var.
    expr: string, e.g. 'x^2 + sin(x)', 'exp(x)*log(x)'.
    var: name of the variable, e.g. 'x'.
    Returns: derivative as string.
    """
    expr = str(expr).strip()
    var = str(var).strip()
    if not var:
        raise ValueError("Variable must not be empty")
    ast = _sym_parse(expr)
    d = _sym_diff(ast, var)
    d = _sym_simplify(d)
    return _sym_to_string(d)


def integrate_sym(expr, var):
    """
    Indefinite (symbolic) integral: integral of expr d(var).
    expr: string, e.g. 'x^2', 'sin(x)', 'exp(x)*x'.
    var: name of the integration variable, e.g. 'x'.
    Returns: antiderivative as string (without constant of integration C).
    Uses SymPy for robust symbolic integration.
    """
    try:
        import sympy  # type: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("integrate_sym requires sympy. Please install: pip install sympy")
    expr_str = str(expr).strip().replace(" ", "")
    var_str = str(var).strip()
    if not var_str:
        raise ValueError("Integration variable must not be empty")
    # Convert Dedekind syntax (^) to SymPy (**)
    expr_sympy = expr_str.replace("^", "**")
    try:
        x = sympy.Symbol(var_str)
        sym_expr = sympy.sympify(expr_sympy)
        result = sympy.integrate(sym_expr, x)
        out = str(result)
        # Optional: convert ** back to ^ for Dedekind consistency
        return out.replace("**", "^")
    except Exception as e:
        raise ValueError(f"integrate_sym: could not integrate '{expr}' with respect to {var_str}: {e}") from e

# --- Standard Library: Jacobian / Hessian (Autograd) ---

def jacobian(f, x):
    """
    Jacobian matrix of f at point x. f: R^n -> R^m (callable, tensor -> tensor).
    x: 1D tensor of length n. Returns: tensor (m, n); row i = gradient of f_i.
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
    Hessian matrix of f at point x. f: R^n -> R (scalar).
    x: 1D tensor of length n. Returns: tensor (n, n).
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
# Plots are collected in _dedekind_plots and returned by the server to the IDE.

_dedekind_plots = []

def _plot_ndarray(x, y=None, title=None, xlabel=None, ylabel=None, kind="line", xscale="linear", yscale="linear"):
    """Internal: creates a plot and appends it as base64 PNG to _dedekind_plots. kind: line, scatter."""
    try:
        import base64
        import io
        import matplotlib  # type: ignore[import-untyped]
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    except ImportError:
        print("plot(): matplotlib not installed. pip install matplotlib")
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
    Plots data (line) and displays it in Dedekind Studio.
    plot(y) -- y over index; plot(x, y) -- y over x.
    xscale, yscale: "linear" (default) or "log".
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
    Scatter plot: points (x, y). scatter(y) -- y over index; scatter(x, y) -- points.
    Displayed in Dedekind Studio.
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
    Contour plot. X, Y: 1D or 2D coordinates (e.g. linspace); Z: 2D matrix (values).
    levels: number of contour lines (default 10). Displayed in Dedekind Studio.
    """
    import numpy as np  # type: ignore[reportMissingImports]
    X_t = _to_tensor(X).float().detach().cpu().numpy()
    Y_t = _to_tensor(Y).float().detach().cpu().numpy()
    Z_t = _to_tensor(Z).float().detach().cpu().numpy()
    if Z_t.ndim != 2:
        raise ValueError("contour: Z must be a 2D matrix.")
    if X_t.ndim == 1 and Y_t.ndim == 1:
        X_t, Y_t = np.meshgrid(X_t, Y_t)
    _plot_contour_inner(X_t, Y_t, Z_t, title=title, xlabel=xlabel, ylabel=ylabel, levels=levels)


# ============================================================================
# Reproducibility: seed manager + data hash
# ============================================================================

def seed(n):
    """Sets a deterministic seed in Python (`random`), NumPy, and PyTorch.
    Usage: `seed(42)` at the very start of a script; random operations become reproducible."""
    try:
        import random as _random
        _random.seed(int(n))
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore[reportMissingImports]
        _np.random.seed(int(n) & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        torch.manual_seed(int(n))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(n))
    except Exception:
        pass
    return int(n)


def data_hash(x):
    """Returns the SHA256 hex digest of an input (tensor, list, dict, number, string, DataFrame).
    Useful for reproducibility logs: `print("Input hash:", data_hash(data))`."""
    import hashlib
    import json
    if isinstance(x, DataFrame):
        payload = json.dumps({"cols": x.columns, "units": x._units, "data": [list(c) for c in x._cols]},
                             sort_keys=True, default=_json_default).encode("utf-8")
    elif hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.detach().cpu().numpy()
        payload = arr.tobytes() + str(arr.shape).encode("utf-8") + str(arr.dtype).encode("utf-8")
    elif isinstance(x, (list, tuple, dict)):
        payload = json.dumps(x, sort_keys=True, default=_json_default).encode("utf-8")
    elif isinstance(x, (bytes, bytearray)):
        payload = bytes(x)
    else:
        payload = str(x).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(o):
    if isinstance(o, Quantity):
        return {"__quantity__": True, "value": o.value, "unit": o.unit}
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# ============================================================================
# DataFrame + tabular I/O (CSV nativ; Parquet/HDF5/NetCDF optional)
# ============================================================================

class DataFrame:
    """Lightweight column-oriented table with optional per-column units.

    Construction:
      df = DataFrame({"t": [0.0, 1.0, 2.0], "T": [20.0, 22.0, 25.0]}, units={"t": "s", "T": "K"})
      df["T"]              -> list of values
      df.units["T"]        -> "K"
      df.rows              -> iterator over rows (dict)
      df.column_with_unit("T") -> list of Quantity values
    """
    def __init__(self, data=None, units=None):
        if data is None:
            data = {}
        def _to_plain_list(seq):
            # Torch tensor: .tolist() returns plain Python numbers.
            if hasattr(seq, "detach") and hasattr(seq, "cpu") and hasattr(seq, "tolist"):
                return list(seq.detach().cpu().tolist())
            out = []
            for v in seq:
                if hasattr(v, "item") and not isinstance(v, (str, bytes)):
                    try:
                        out.append(v.item())
                        continue
                    except Exception:
                        pass
                out.append(v)
            return out
        if isinstance(data, dict):
            cols_names = list(data.keys())
            cols = [_to_plain_list(data[c]) for c in cols_names]
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
            cols_names = list(data[0].keys())
            cols = [_to_plain_list([row.get(c) for row in data]) for c in cols_names]
        else:
            raise TypeError("DataFrame: data must be a dict of lists or a list of dicts.")
        n_rows = _builtin_max([len(c) for c in cols]) if cols else 0
        for c in cols:
            if len(c) != n_rows:
                raise ValueError("DataFrame: all columns must have the same length.")
        self.columns = cols_names
        self._cols = cols
        self._units = {}
        if units:
            for k, v in dict(units).items():
                if v:
                    self._units[k] = str(v)
        self.n_rows = n_rows

    @property
    def units(self):
        return dict(self._units)

    @property
    def shape(self):
        return (self.n_rows, len(self.columns))

    def __len__(self):
        return self.n_rows

    def __getitem__(self, key):
        if key not in self.columns:
            raise KeyError(f"DataFrame: column '{key}' not found. Available: {self.columns}.")
        return list(self._cols[self.columns.index(key)])

    def __contains__(self, key):
        return key in self.columns

    def column_with_unit(self, key):
        """Column as a list of Quantity values (uses self.units[key] if present)."""
        vals = self[key]
        u = self._units.get(key, "")
        if not u:
            return [Quantity(float(v), "") if isinstance(v, (int, float)) else v for v in vals]
        return [Quantity(float(v), u) if isinstance(v, (int, float)) else v for v in vals]

    @property
    def rows(self):
        for i in range(self.n_rows):
            yield {c: self._cols[j][i] for j, c in enumerate(self.columns)}

    def head(self, n=5):
        n = int(n)
        head_cols = {c: self._cols[j][:n] for j, c in enumerate(self.columns)}
        return DataFrame(head_cols, units=self._units)

    def __repr__(self):
        # Compact representation with units in column headers.
        if self.n_rows == 0:
            return f"DataFrame(0 rows, columns={self.columns})"
        widths = []
        headers = []
        for j, c in enumerate(self.columns):
            label = c if c not in self._units else f"{c} [{self._units[c]}]"
            headers.append(label)
            col_strs = [str(v) for v in self._cols[j][:_builtin_min(self.n_rows, 10)]]
            widths.append(_builtin_max(len(label), *(len(s) for s in col_strs)) if col_strs else len(label))
        sep = "  "
        lines = [sep.join(h.ljust(w) for h, w in zip(headers, widths))]
        lines.append(sep.join("-" * w for w in widths))
        for i in range(_builtin_min(self.n_rows, 10)):
            row = [str(self._cols[j][i]).ljust(widths[j]) for j in range(len(self.columns))]
            lines.append(sep.join(row))
        if self.n_rows > 10:
            lines.append(f"... ({self.n_rows} rows total)")
        return "\n".join(lines)


def _csv_parse_unit_in_header(header):
    """Header 'T [K]' -> ('T', 'K'); 'T' -> ('T', None)."""
    s = str(header).strip()
    if s.endswith("]") and "[" in s:
        idx = s.rfind("[")
        name = s[:idx].strip()
        unit = s[idx + 1:-1].strip()
        if name:
            return name, (unit or None)
    return s, None


def read_csv(path, units=None, has_header=True):
    """Reads a CSV file into a DataFrame.
    - The header row may contain units in the format `name [unit]`; these are stored in `df.units`.
    - `units` (optional dict) overrides detected units.
    - Numeric values are converted to float (fallback: leave as string).
    """
    import csv
    p = str(path)
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return DataFrame({}, units={})
    detected_units = {}
    if has_header:
        raw_header = rows[0]
        col_names = []
        for h in raw_header:
            name, unit = _csv_parse_unit_in_header(h)
            col_names.append(name)
            if unit:
                detected_units[name] = unit
        body = rows[1:]
    else:
        col_names = [f"col{i}" for i in range(len(rows[0]))]
        body = rows
    cols = [[] for _ in col_names]
    for r in body:
        for j in range(len(col_names)):
            v = r[j] if j < len(r) else ""
            try:
                cols[j].append(float(v))
            except (TypeError, ValueError):
                cols[j].append(v)
    data = {c: cols[j] for j, c in enumerate(col_names)}
    eff_units = dict(detected_units)
    if units:
        for k, v in dict(units).items():
            if v:
                eff_units[k] = str(v)
    return DataFrame(data, units=eff_units)


def write_csv(path, df, include_units_in_header=True):
    """Schreibt eine DataFrame als CSV. Bei `include_units_in_header=True` wird `name [unit]` ausgegeben."""
    import csv
    if not isinstance(df, DataFrame):
        raise TypeError("write_csv: df muss DataFrame sein.")
    p = str(path)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = []
        for c in df.columns:
            if include_units_in_header and c in df._units:
                header.append(f"{c} [{df._units[c]}]")
            else:
                header.append(c)
        w.writerow(header)
        for row in df.rows:
            w.writerow([row[c] for c in df.columns])


def read_parquet(path):
    """Reads a Parquet file (requires pyarrow). Units are not persisted; only columns."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("read_parquet: pyarrow not installed (pip install pyarrow).")
    table = pq.read_table(str(path))
    data = {name: table.column(name).to_pylist() for name in table.column_names}
    return DataFrame(data)


def write_parquet(path, df):
    """Writes a DataFrame as Parquet (requires pyarrow)."""
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("write_parquet: pyarrow not installed (pip install pyarrow).")
    if not isinstance(df, DataFrame):
        raise TypeError("write_parquet: df must be a DataFrame.")
    table = pa.table({c: df._cols[j] for j, c in enumerate(df.columns)})
    pq.write_table(table, str(path))


def read_hdf5(path, dataset=None):
    """Reads an HDF5 dataset (requires h5py). If `dataset=None`: the first dataset is selected."""
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("read_hdf5: h5py not installed (pip install h5py).")
    import numpy as _np  # type: ignore[reportMissingImports]
    with h5py.File(str(path), "r") as f:
        if dataset is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError("read_hdf5: file contains no datasets.")
            dataset = keys[0]
        arr = _np.array(f[dataset])
    return arr


def write_hdf5(path, data, dataset="data"):
    """Writes a tensor/array to an HDF5 file (requires h5py)."""
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("write_hdf5: h5py not installed (pip install h5py).")
    import numpy as _np  # type: ignore[reportMissingImports]
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = _np.asarray(data)
    with h5py.File(str(path), "w") as f:
        f.create_dataset(str(dataset), data=arr)


def read_netcdf(path, variable=None):
    """Reads a NetCDF variable (requires netCDF4). If `variable=None`: the first variable is selected."""
    try:
        import netCDF4  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("read_netcdf: netCDF4 not installed (pip install netCDF4).")
    import numpy as _np  # type: ignore[reportMissingImports]
    ds = netCDF4.Dataset(str(path), "r")
    try:
        if variable is None:
            names = list(ds.variables.keys())
            if not names:
                raise ValueError("read_netcdf: file contains no variables.")
            variable = names[0]
        arr = _np.array(ds.variables[variable][:])
    finally:
        ds.close()
    return arr


# ============================================================================
# Unit annotations for function signatures (`fn f(x: [m]) -> [J]`)
# ============================================================================

# Derived SI units -> base dimensions (kg, m, s, A, K, mol, cd)
# We only track the 7 SI base dimensions.
_DERIVED_UNIT_TO_BASE = {
    "":     {},
    "1":    {},
    "kg":   {"kg": 1},
    "g":    {"kg": 1},
    "t":    {"kg": 1},
    "mg":   {"kg": 1},
    "m":    {"m": 1},
    "cm":   {"m": 1},
    "km":   {"m": 1},
    "mm":   {"m": 1},
    "dm":   {"m": 1},
    "s":    {"s": 1},
    "min":  {"s": 1},
    "h":    {"s": 1},
    "ms":   {"s": 1},
    "A":    {"A": 1},
    "mA":   {"A": 1},
    "kA":   {"A": 1},
    "uA":   {"A": 1},
    "muA":  {"A": 1},
    "K":    {"K": 1},
    "mK":   {"K": 1},
    "mol":  {"mol": 1},
    "mmol": {"mol": 1},
    "kmol": {"mol": 1},
    "cd":   {"cd": 1},
    "mcd":  {"cd": 1},
    "rad":  {},
    "deg":  {},
    # Abgeleitet:
    "N":   {"kg": 1, "m": 1, "s": -2},
    "kN":  {"kg": 1, "m": 1, "s": -2},
    "MN":  {"kg": 1, "m": 1, "s": -2},
    "Pa":  {"kg": 1, "m": -1, "s": -2},
    "kPa": {"kg": 1, "m": -1, "s": -2},
    "MPa": {"kg": 1, "m": -1, "s": -2},
    "GPa": {"kg": 1, "m": -1, "s": -2},
    "hPa": {"kg": 1, "m": -1, "s": -2},
    "bar": {"kg": 1, "m": -1, "s": -2},
    "atm": {"kg": 1, "m": -1, "s": -2},
    "J":   {"kg": 1, "m": 2, "s": -2},
    "kJ":  {"kg": 1, "m": 2, "s": -2},
    "MJ":  {"kg": 1, "m": 2, "s": -2},
    "Wh":  {"kg": 1, "m": 2, "s": -2},
    "kWh": {"kg": 1, "m": 2, "s": -2},
    "W":   {"kg": 1, "m": 2, "s": -3},
    "kW":  {"kg": 1, "m": 2, "s": -3},
    "MW":  {"kg": 1, "m": 2, "s": -3},
    "L_sun": {"kg": 1, "m": 2, "s": -3},
    "V":   {"kg": 1, "m": 2, "s": -3, "A": -1},
    "mV":  {"kg": 1, "m": 2, "s": -3, "A": -1},
    "kV":  {"kg": 1, "m": 2, "s": -3, "A": -1},
    "Hz":  {"s": -1},
    "kHz": {"s": -1},
    "MHz": {"s": -1},
    "GHz": {"s": -1},
    "Bq":  {"s": -1},
    "C":   {"A": 1, "s": 1},
    "mC":  {"A": 1, "s": 1},
    "uC":  {"A": 1, "s": 1},
    "ohm": {"kg": 1, "m": 2, "s": -3, "A": -2},
    "kohm": {"kg": 1, "m": 2, "s": -3, "A": -2},
    "Mohm": {"kg": 1, "m": 2, "s": -3, "A": -2},
    "F":   {"kg": -1, "m": -2, "s": 4, "A": 2},
    "L":   {"m": 3},
    "mL":  {"m": 3},
    "M":   {"mol": 1, "m": -3},  # mol/L = 1000 mol/m³, gleiche Dimension
    "mM":  {"mol": 1, "m": -3},
    "uM":  {"mol": 1, "m": -3},
    "nM":  {"mol": 1, "m": -3},
    "g/L": {"kg": 1, "m": -3},
    "mg/mL": {"kg": 1, "m": -3},
    "percent_wv": {"kg": 1, "m": -3},
    "Gy":  {"m": 2, "s": -2},
    "mGy": {"m": 2, "s": -2},
    "Sv":  {"m": 2, "s": -2},
    "mSv": {"m": 2, "s": -2},
    "um":  {"m": 1},
    "nm":  {"m": 1},
    "pm":  {"m": 1},
    "angstrom": {"m": 1},
    "AU":  {"m": 1},
    "ly":  {"m": 1},
    "pc":  {"m": 1},
    "D":   {"m": 2},
    "mD":  {"m": 2},
    "ug":  {"kg": 1},
    "Da":  {"kg": 1},
    "amu": {"kg": 1},
    "M_sun": {"kg": 1},
    "us":  {"s": 1},
    "ns":  {"s": 1},
    "d":   {"s": 1},
    "yr":  {"s": 1},
    "a":   {"s": 1},
    "umol": {"mol": 1},
    "nmol": {"mol": 1},
    "cal":  {"kg": 1, "m": 2, "s": -2},
    "kcal": {"kg": 1, "m": 2, "s": -2},
    "eV":   {"kg": 1, "m": 2, "s": -2},
    "meV":  {"kg": 1, "m": 2, "s": -2},
    "keV":  {"kg": 1, "m": 2, "s": -2},
    "MeV":  {"kg": 1, "m": 2, "s": -2},
    "GeV":  {"kg": 1, "m": 2, "s": -2},
    "T":    {"kg": 1, "s": -2, "A": -1},
    "G":    {"kg": 1, "s": -2, "A": -1},
    "Wb":   {"kg": 1, "m": 2, "s": -2, "A": -1},
    "H":    {"kg": 1, "m": 2, "s": -2, "A": -2},
    "mH":   {"kg": 1, "m": 2, "s": -2, "A": -2},
    "uH":   {"kg": 1, "m": 2, "s": -2, "A": -2},
    "lm":   {"cd": 1},
    "lx":   {"cd": 1, "m": -2},
    "kat":  {"mol": 1, "s": -1},
    "U":    {"mol": 1, "s": -1},
    "Angstrom": {"m": 1},
    "fm":   {"m": 1},
    "ps":   {"s": 1},
    "fs":   {"s": 1},
    "THz":  {"s": -1},
    "kJ/mol": {"kg": 1, "m": 2, "s": -2, "mol": -1},
    "kcal/mol": {"kg": 1, "m": 2, "s": -2, "mol": -1},
    "J/mol": {"kg": 1, "m": 2, "s": -2, "mol": -1},
    "eV/atom": {"kg": 1, "m": 2, "s": -2, "mol": -1},
    "Hartree/mol": {"kg": 1, "m": 2, "s": -2, "mol": -1},
}


def _parse_unit_to_base_dims(unit_str):
    """Converts a unit string ('kg*m^2/s^2', '(kg)*(m/s)*(m/s)', 'J', ...) into
    a dict {base_unit: exponent} over the 7 SI base dimensions.

    Tokenizer-like: splits on *, /, ^ and parentheses; each atomic unit token
    is resolved to base dimensions via _DERIVED_UNIT_TO_BASE. Unknown tokens
    produce None (comparison then fails).
    """
    s = (unit_str or "").strip()
    if not s:
        return {}
    # Tokenize: parentheses, *, /, ^, atomic units (letter sequences), numbers (for exponents).
    import re as _re
    tokens = _re.findall(r"\(|\)|\*|/|\^|-?\d+|[A-Za-z][A-Za-z_]*", s)
    if not tokens:
        return {}
    pos = [0]

    def parse_factor():
        if pos[0] >= len(tokens):
            return None
        t = tokens[pos[0]]
        if t == "(":
            pos[0] += 1
            val = parse_term()
            if pos[0] < len(tokens) and tokens[pos[0]] == ")":
                pos[0] += 1
            return val
        # Atomic unit or number
        if t and t[0].isalpha():
            pos[0] += 1
            base = _DERIVED_UNIT_TO_BASE.get(t)
            if base is None:
                return None
            # Optional: ^exp
            if pos[0] < len(tokens) and tokens[pos[0]] == "^":
                pos[0] += 1
                if pos[0] < len(tokens):
                    try:
                        exp = int(tokens[pos[0]])
                    except ValueError:
                        try:
                            exp = float(tokens[pos[0]])
                        except ValueError:
                            return None
                    pos[0] += 1
                    return {k: v * exp for k, v in base.items()}
            return dict(base)
        # Plain number as a factor (e.g. constant 1) -> dimensionless
        try:
            int(t)
            pos[0] += 1
            return {}
        except ValueError:
            return None

    def parse_term():
        left = parse_factor()
        if left is None:
            return None
        while pos[0] < len(tokens) and tokens[pos[0]] in ("*", "/"):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_factor()
            if right is None:
                return None
            sign = 1 if op == "*" else -1
            for k, v in right.items():
                left[k] = left.get(k, 0) + sign * v
        return left

    result = parse_term()
    if result is None:
        return None
    # Remove zero exponents
    return {k: v for k, v in result.items() if v != 0}


def _units_dimensionally_equal(u1, u2):
    """True if u1 and u2 reduce to the same SI base dimensions (e.g. J == kg*m^2/s^2)."""
    d1 = _parse_unit_to_base_dims(u1)
    d2 = _parse_unit_to_base_dims(u2)
    if d1 is None or d2 is None:
        return False
    return d1 == d2


def _coerce_to_expected_unit(value, expected_unit, context_label):
    """Coerces `value` to `expected_unit` if dimensionally compatible.

    Paths (in order):
    1) String equality (same unit) -> unchanged
    2) Same unique dimension (length, mass, time, ...) -> numeric conversion
    3) Same SI base dimensions (kg*m^2/s^2 == J) -> value adopted as Quantity in expected_unit
    4) Plain number + expected_unit -> wrap with unit
    Otherwise raises ValueError.
    """
    expected = (expected_unit or "").strip()
    if isinstance(value, Quantity):
        if not expected:
            return value
        if value.unit == expected:
            return value
        dim_have = _get_dimension(value.unit)
        dim_want = _get_dimension(expected)
        if dim_have is not None and dim_have == dim_want:
            v_base = _convert_to_base(value.value, value.unit, dim_have)
            v_target = _convert_from_base(v_base, expected, dim_want)
            return Quantity(v_target, expected)
        n_have = _normalize_unit_for_compare(value.unit)
        n_want = _normalize_unit_for_compare(expected)
        if n_have == n_want:
            return Quantity(value.value, expected)
        if _units_dimensionally_equal(value.unit, expected):
            return Quantity(value.value, expected)
        raise ValueError(
            f"Unit error in {context_label}: expected [{expected}], got [{value.unit}]."
        )
    if isinstance(value, (int, float)):
        if expected:
            return Quantity(float(value), expected)
        return value
    return value


def _check_signature_unit(value, expected_unit, fn_name, arg_name):
    return _coerce_to_expected_unit(value, expected_unit, f"{fn_name}({arg_name})")


def _check_return_unit(value, expected_unit, fn_name):
    return _coerce_to_expected_unit(value, expected_unit, f"return of {fn_name}")


# ============================================================================
# Unit-aware plotting: lists of Quantity automatically receive axis labels "[unit]"
# ============================================================================

def _strip_unit_from_seq(seq):
    """If `seq` is a list/tuple of Quantity, returns (values_list, unit_str); otherwise (seq, None)."""
    if isinstance(seq, (list, tuple)) and seq and all(isinstance(v, Quantity) for v in seq):
        unit = seq[0].unit
        # If units vary across the list, leave it neutral.
        if all(v.unit == unit for v in seq):
            return [v.value for v in seq], unit
    if isinstance(seq, Quantity):
        return seq.value, seq.unit
    return seq, None


def _append_unit_label(label, unit):
    if not unit:
        return label
    if label and "[" not in str(label):
        return f"{label} [{unit}]"
    if not label:
        return f"[{unit}]"
    return label


# Wrap plot/scatter/contour so they detect Quantity inputs.
_dedekind_plot_inner = plot
_dedekind_scatter_inner = scatter
_dedekind_contour_inner = contour


def plot(x=None, y=None, title=None, xlabel=None, ylabel=None, xscale="linear", yscale="linear"):
    """Like `plot`, but when x/y are lists of Quantity, values are extracted and units used as axis labels."""
    new_x, ux = _strip_unit_from_seq(x) if x is not None else (None, None)
    new_y, uy = _strip_unit_from_seq(y) if y is not None else (None, None)
    if y is None and x is not None and ux:
        # plot(y_quantity_list) — Single-Argument-Variante
        new_y, uy = new_x, ux
        new_x, ux = None, None
    xlabel = _append_unit_label(xlabel, ux)
    ylabel = _append_unit_label(ylabel, uy)
    return _dedekind_plot_inner(new_x, new_y, title=title, xlabel=xlabel, ylabel=ylabel,
                                xscale=xscale, yscale=yscale)


def scatter(x=None, y=None, title=None, xlabel=None, ylabel=None):
    """Wie `scatter`, mit automatischer Einheiten-Beschriftung."""
    new_x, ux = _strip_unit_from_seq(x) if x is not None else (None, None)
    new_y, uy = _strip_unit_from_seq(y) if y is not None else (None, None)
    if y is None and x is not None and ux:
        new_y, uy = new_x, ux
        new_x, ux = None, None
    xlabel = _append_unit_label(xlabel, ux)
    ylabel = _append_unit_label(ylabel, uy)
    return _dedekind_scatter_inner(new_x, new_y, title=title, xlabel=xlabel, ylabel=ylabel)


def contour(X, Y, Z, title=None, xlabel=None, ylabel=None, levels=10):
    """Wie `contour`, mit automatischer Einheiten-Beschriftung der Achsen."""
    new_X, ux = _strip_unit_from_seq(X)
    new_Y, uy = _strip_unit_from_seq(Y)
    xlabel = _append_unit_label(xlabel, ux)
    ylabel = _append_unit_label(ylabel, uy)
    return _dedekind_contour_inner(new_X, new_Y, Z, title=title, xlabel=xlabel, ylabel=ylabel, levels=levels)


# ============================================================================
# Benchmarking & Profiling: Wandzeit + Speicherprofil als Built-ins
# ============================================================================

class BenchmarkResult:
    """Ergebnis von `benchmark(fn, n=...)` mit mean/std/min/max Sekunden und n Wiederholungen."""
    def __init__(self, label, mean_s, std_s, min_s, max_s, n, last_result=None):
        self.label = label
        self.mean_s = float(mean_s)
        self.std_s = float(std_s)
        self.min_s = float(min_s)
        self.max_s = float(max_s)
        self.n = int(n)
        self.last_result = last_result

    def __repr__(self):
        return (f"benchmark({self.label or 'fn'}): mean={self.mean_s*1e3:.3f}ms "
                f"± {self.std_s*1e3:.3f}ms  min={self.min_s*1e3:.3f}ms  "
                f"max={self.max_s*1e3:.3f}ms  (n={self.n})")


def benchmark(fn, n=10, warmup=2, label=None):
    """Misst Wandzeit einer Null-Argument-Funktion über n Wiederholungen (default 10, plus warmup-Runs).
    Verwendung: `r = benchmark(fn() => meine_arbeit(), n=50)`. Achtung: `fn` muss aufrufbar sein
    (in Dedekind: anonyme Lambda via einer normalen Funktion ohne Argumente)."""
    import time
    if not callable(fn):
        raise TypeError("benchmark: fn muss eine aufrufbare Funktion ohne Argumente sein.")
    for _ in range(int(warmup)):
        try: fn()
        except Exception: pass
    times = []
    last = None
    for _ in range(int(n)):
        t0 = time.perf_counter()
        last = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean_s = sum(times) / len(times)
    var = sum((x - mean_s) ** 2 for x in times) / _builtin_max(1, len(times) - 1)
    std_s = var ** 0.5
    return BenchmarkResult(label, mean_s, std_s, _builtin_min(times), _builtin_max(times), len(times), last)


class ProfileResult:
    """Ergebnis von `profile(fn)`: Wandzeit, optional Peak-Speicher (Bytes) und Top-Funktionen aus cProfile."""
    def __init__(self, label, wall_s, peak_bytes, top_funcs, returned):
        self.label = label
        self.wall_s = float(wall_s)
        self.peak_bytes = int(peak_bytes) if peak_bytes is not None else None
        self.top_funcs = list(top_funcs)
        self.returned = returned

    def __repr__(self):
        lines = [f"profile({self.label or 'fn'}): {self.wall_s*1e3:.3f}ms"]
        if self.peak_bytes is not None:
            lines.append(f"  peak memory: {self.peak_bytes/1024:.1f} KiB")
        if self.top_funcs:
            lines.append("  top calls (cumulative):")
            for name, n_calls, cum_s in self.top_funcs[:5]:
                lines.append(f"    {cum_s*1e3:8.3f}ms  {n_calls:6d}x  {name}")
        return "\n".join(lines)


def profile(fn, label=None, top=5):
    """Profiliert eine Null-Argument-Funktion (Wandzeit + Peak-Speicher per `tracemalloc` + cProfile-Top-Calls)."""
    import time
    import tracemalloc
    import cProfile
    import pstats
    if not callable(fn):
        raise TypeError("profile: fn muss eine aufrufbare Funktion ohne Argumente sein.")
    tracemalloc.start()
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    try:
        returned = fn()
    finally:
        pr.disable()
        t1 = time.perf_counter()
        _curr, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    stats = pstats.Stats(pr).sort_stats("cumulative")
    top_funcs = []
    for func, (cc, _nc, _tt, ct, _callers) in list(stats.stats.items())[:int(top)]:
        fname = f"{func[0].rsplit('/', 1)[-1]}:{func[1]}({func[2]})"
        top_funcs.append((fname, cc, ct))
    return ProfileResult(label, t1 - t0, peak, top_funcs, returned)


def time_block(label, fn):
    """Misst und gibt Wandzeit einer Null-Argument-Funktion einmal aus; gibt das Ergebnis zurück.
    Praktisch für Ad-hoc-Messungen: `result = time_block("solve", fn() => ode_solve(...))`."""
    import time
    if not callable(fn):
        raise TypeError("time_block: fn muss eine aufrufbare Funktion ohne Argumente sein.")
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    print(f"[time_block] {label}: {(t1 - t0)*1e3:.3f} ms")
    return out


# ============================================================================
# JIT-Backend: torch.compile-Wrapper als realistischer Schritt Richtung AOT
# ============================================================================

class RobustJITWrapper:
    def __init__(self, original_fn, compiled_fn):
        self.original_fn = original_fn
        self.compiled_fn = compiled_fn
        self.failed = False

    def __call__(self, *args, **kwargs):
        if not self.failed:
            try:
                return self.compiled_fn(*args, **kwargs)
            except Exception as e:
                # Robust fallback for runtime compilation errors (e.g. cl.exe not found)
                self.failed = True
        return self.original_fn(*args, **kwargs)

def jit(fn):
    """Versucht, `fn` mit `torch.compile` zu beschleunigen (falls verfügbar); fällt sonst auf die
    Original-Funktion zurück. Realistischer Zwischenschritt Richtung AOT: nutzt TorchInductor als
    Backend, sodass Dedekind-Code, der zu PyTorch-Operationen transpiliert wird, durch denselben
    Compiler-Stack läuft wie reines PyTorch-Modell-Code."""
    if not callable(fn):
        raise TypeError("jit: fn muss eine aufrufbare Funktion sein.")
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        # PyTorch < 2.0 oder Stub: einfach Original zurückgeben.
        return fn
    try:
        compiled = compiler(fn)
        return RobustJITWrapper(fn, compiled)
    except Exception:
        return fn



# ============================================================================
# Stochastische DGLn (SDEs): Euler-Maruyama und Milstein
# ============================================================================

def sde_solve(drift, diffusion, y0, t, method="euler_maruyama", seed_value=None):
    """Löst dY = drift(t, Y) dt + diffusion(t, Y) dW (Itô-Interpretation).

    - `drift(t, y)`, `diffusion(t, y)`: callables, Rückgabe wie y (Skalar/Tensor).
    - `y0`: Anfangsbedingung; `t`: 1D-Zeitgitter (z. B. `linspace(0, 1, 1001)`).
    - `method`: `"euler_maruyama"` (default) oder `"milstein"` (1D mit numerischer Ableitung
      von diffusion bzgl. y; höhere Konvergenzordnung 1.0 statt 0.5).
    - `seed_value`: optionaler int für deterministische Pfade (sonst aktueller globaler Seed).
    """
    y0 = _to_tensor(y0).float()
    t_grid = _to_tensor(t).float().flatten()
    if t_grid.numel() < 2:
        raise ValueError("sde_solve: t braucht mindestens 2 Stützstellen.")
    if seed_value is not None:
        torch.manual_seed(int(seed_value))
    out = [y0]
    y = y0.clone()
    is_scalar = (y0.dim() == 0)
    for i in range(t_grid.numel() - 1):
        t_cur = t_grid[i]
        dt = (t_grid[i + 1] - t_grid[i]).item()
        sqrt_dt = abs(dt) ** 0.5
        if is_scalar:
            dW = torch.randn((), dtype=y.dtype, device=y.device) * sqrt_dt
        else:
            dW = torch.randn(y.shape, dtype=y.dtype, device=y.device) * sqrt_dt
        a = drift(t_cur, y)
        b = diffusion(t_cur, y)
        a_t = _to_tensor(a).float()
        b_t = _to_tensor(b).float()
        if method == "milstein":
            # 1D-Milstein: Y += a dt + b dW + 0.5 b b' (dW^2 - dt)
            eps = 1e-4 * (1.0 + float(y.detach().abs().mean().item() if y.numel() > 0 else 1.0))
            b_plus = _to_tensor(diffusion(t_cur, y + eps)).float()
            b_minus = _to_tensor(diffusion(t_cur, y - eps)).float()
            db_dy = (b_plus - b_minus) / (2.0 * eps)
            y = y + a_t * dt + b_t * dW + 0.5 * b_t * db_dy * (dW * dW - dt)
        else:
            y = y + a_t * dt + b_t * dW
        out.append(y)
    return torch.stack(out, dim=0)


# ============================================================================
# Erweiterte Optimierung: least_squares, nichtlineare Constraints, MILP
# ============================================================================

def least_squares(residuals, x0, jacobian=None, bounds=None, method="trf"):
    """Nichtlineare Kleinste-Quadrate: minimiert ||residuals(x)||² über x.

    - `residuals(x)`: Callable, liefert Residuen-Vektor (Liste/Tensor).
    - `x0`: Startwerte. `jacobian` optional (Callable, sonst numerisch).
    - `bounds`: (low, high) oder None (unbeschränkt).
    - `method`: `"trf"` (Trust-Region Reflective, default), `"lm"` (Levenberg-Marquardt; unbeschränkt),
      `"dogbox"`.
    Rückgabe: Dict mit Keys `x`, `cost`, `nfev`, `success`, `message`.
    """
    try:
        from scipy.optimize import least_squares as _ls  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("least_squares benötigt scipy.")
    import numpy as _np  # type: ignore[reportMissingImports]
    x0_np = _np.asarray(_to_tensor(x0).detach().cpu().numpy(), dtype=float).reshape(-1)

    def _res_wrap(x):
        r = residuals(x)
        if hasattr(r, "detach"):
            r = r.detach().cpu().numpy()
        return _np.asarray(r, dtype=float).reshape(-1)

    kwargs = {"method": method}
    if jacobian is not None:
        def _jac_wrap(x):
            j = jacobian(x)
            if hasattr(j, "detach"):
                j = j.detach().cpu().numpy()
            return _np.asarray(j, dtype=float)
        kwargs["jac"] = _jac_wrap
    else:
        # PyTorch-Default-dtype ist float32; scipy's Default-Step von ~1.5e-8 für die numerische
        # Jacobi-Approximation liegt unter float32-Epsilon und liefert dann Null-Gradient.
        # Wir verwenden einen Schritt von 1e-4, der sowohl float32 als auch float64 stabil ist.
        kwargs["diff_step"] = 1e-4
    if bounds is not None:
        kwargs["bounds"] = bounds
    res = _ls(_res_wrap, x0_np, **kwargs)
    return {
        "x": res.x.tolist(),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "success": bool(res.success),
        "message": str(res.message),
    }


def minimize_constrained(f, x0, constraints=None, bounds=None, method="SLSQP", tol=1e-8):
    """Nichtlineare Optimierung mit Constraints (SLSQP/trust-constr/COBYLA).

    - `f(x)`: skalare Zielfunktion.
    - `constraints`: Liste von Dicts `{"type": "eq"|"ineq", "fun": g}` (ineq = g(x) >= 0).
    - `bounds`: Liste von `(low, high)` pro Variable.
    - `method`: "SLSQP" (default), "trust-constr", "COBYLA".
    Rückgabe: Dict mit `x`, `fun`, `success`, `message`, `nit`.
    """
    try:
        from scipy.optimize import minimize as _min  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("minimize_constrained benötigt scipy.")
    import numpy as _np  # type: ignore[reportMissingImports]
    x0_np = _np.asarray(_to_tensor(x0).detach().cpu().numpy(), dtype=float).reshape(-1)

    def _f_wrap(x):
        v = f(x)
        if hasattr(v, "detach"):
            v = v.detach().cpu().item()
        return float(v)

    _cons = []
    if constraints:
        for c in constraints:
            ctype = c["type"] if isinstance(c, dict) else c[0]
            cfun = c["fun"] if isinstance(c, dict) else c[1]

            def _wrap(fun):
                def inner(x):
                    v = fun(x)
                    if hasattr(v, "detach"):
                        v = v.detach().cpu().numpy()
                    return _np.asarray(v, dtype=float).reshape(-1)
                return inner
            _cons.append({"type": ctype, "fun": _wrap(cfun)})

    kwargs = {"method": method, "tol": tol}
    if _cons:
        kwargs["constraints"] = _cons
    if bounds is not None:
        # Convert any tensor/array bounds to float tuples for scipy optimize
        bounds_cleaned = []
        for b in bounds:
            if hasattr(b, "tolist"):
                b_list = b.tolist()
            else:
                try:
                    b_list = [float(x) for x in b]
                except TypeError:
                    b_list = list(b)
            bounds_cleaned.append((float(b_list[0]), float(b_list[1])))
        kwargs["bounds"] = bounds_cleaned
    res = _min(_f_wrap, x0_np, **kwargs)
    return {
        "x": res.x.tolist(),
        "fun": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", 0)),
    }


def milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, integrality=None):
    """(Gemischt-)Ganzzahlige lineare Optimierung: min c·x mit A_ub x ≤ b_ub, A_eq x = b_eq.

    - `integrality`: Liste/Vektor mit 0 (kontinuierlich), 1 (Integer), 2 (semicontinuous),
      3 (semi-integer) pro Variable; None = alles kontinuierlich (entspricht regulärem LP).
    - `bounds`: Liste von `(low, high)` (None = unbeschränkt).
    Rückgabe: Dict mit `x`, `fun`, `success`, `message`.
    """
    try:
        from scipy.optimize import milp as _milp, LinearConstraint, Bounds  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("milp benötigt scipy>=1.9 (scipy.optimize.milp).")
    import numpy as _np  # type: ignore[reportMissingImports]
    c_np = _np.asarray(_to_tensor(c).detach().cpu().numpy(), dtype=float).reshape(-1)

    constraints = []
    if A_ub is not None and b_ub is not None:
        A = _np.asarray(_to_tensor(A_ub).detach().cpu().numpy(), dtype=float)
        b = _np.asarray(_to_tensor(b_ub).detach().cpu().numpy(), dtype=float).reshape(-1)
        constraints.append(LinearConstraint(A, -_np.inf, b))
    if A_eq is not None and b_eq is not None:
        A = _np.asarray(_to_tensor(A_eq).detach().cpu().numpy(), dtype=float)
        b = _np.asarray(_to_tensor(b_eq).detach().cpu().numpy(), dtype=float).reshape(-1)
        constraints.append(LinearConstraint(A, b, b))

    n = c_np.size
    if bounds is not None:
        lows = _np.array([b[0] if b[0] is not None else -_np.inf for b in bounds], dtype=float)
        highs = _np.array([b[1] if b[1] is not None else _np.inf for b in bounds], dtype=float)
        bnds = Bounds(lb=lows, ub=highs)
    else:
        bnds = Bounds(lb=_np.full(n, -_np.inf), ub=_np.full(n, _np.inf))

    if integrality is not None:
        integ = _np.asarray(_to_tensor(integrality).detach().cpu().numpy(), dtype=int).reshape(-1)
    else:
        integ = _np.zeros(n, dtype=int)

    res = _milp(c_np, constraints=constraints if constraints else None,
                bounds=bnds, integrality=integ)
    return {
        "x": (res.x.tolist() if res.x is not None else None),
        "fun": (float(res.fun) if res.fun is not None else None),
        "success": bool(res.success),
        "message": str(res.message),
    }


# ============================================================================
# FEM-Primitiven: Dreiecksgitter + lineare Galerkin-Assemblierung (Poisson 2D)
# ============================================================================

def mesh_unit_square(n):
    """Strukturiertes Dreiecksgitter auf [0,1]² mit (n+1)² Knoten und 2·n² Dreiecken.

    Rückgabe Dict mit:
      - `nodes`: (Nn, 2)-Tensor der Knotenkoordinaten
      - `elements`: (Ne, 3)-Tensor mit Knotenindizes pro Dreieck
      - `boundary`: 1D-Tensor mit Indizes der Randknoten
      - `n`: Original-n
    """
    import numpy as _np  # type: ignore[reportMissingImports]
    n_i = int(n)
    if n_i < 1:
        raise ValueError("mesh_unit_square: n muss >= 1 sein.")
    xs = _np.linspace(0.0, 1.0, n_i + 1)
    ys = _np.linspace(0.0, 1.0, n_i + 1)
    nodes = _np.array([[x, y] for y in ys for x in xs], dtype=float)
    elements = []
    for j in range(n_i):
        for i in range(n_i):
            v0 = j * (n_i + 1) + i
            v1 = v0 + 1
            v2 = v0 + (n_i + 1)
            v3 = v2 + 1
            elements.append([v0, v1, v3])
            elements.append([v0, v3, v2])
    elements = _np.array(elements, dtype=int)
    boundary = []
    for k in range(nodes.shape[0]):
        x, y = nodes[k, 0], nodes[k, 1]
        if x == 0.0 or x == 1.0 or y == 0.0 or y == 1.0:
            boundary.append(k)
    boundary = _np.array(boundary, dtype=int)
    return {
        "nodes": torch.from_numpy(nodes).float(),
        "elements": torch.from_numpy(elements).long(),
        "boundary": torch.from_numpy(boundary).long(),
        "n": n_i,
    }


def fem_assemble_stiffness(mesh):
    """Assembliert die Steifigkeitsmatrix K für ∫∇φ_i·∇φ_j dx auf linearen Dreiecks-Elementen.
    Rückgabe: dichte (Nn, Nn)-Matrix (für kleine Probleme; für große Meshes sparse_laplacian_2d nutzen)."""
    import numpy as _np  # type: ignore[reportMissingImports]
    nodes = mesh["nodes"].detach().cpu().numpy()
    elements = mesh["elements"].detach().cpu().numpy()
    Nn = nodes.shape[0]
    K = _np.zeros((Nn, Nn), dtype=float)
    for tri in elements:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        xk, yk = nodes[k]
        # Fläche per Determinante.
        det = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
        area = 0.5 * abs(det)
        if area < 1e-15:
            continue
        # Gradienten der Hütchen-Funktionen (lineare Dreiecke).
        b = _np.array([yj - yk, yk - yi, yi - yj]) / det
        c = _np.array([xk - xj, xi - xk, xj - xi]) / det
        Ke = area * (_np.outer(b, b) + _np.outer(c, c))
        idx = (i, j, k)
        for a in range(3):
            for bIdx in range(3):
                K[idx[a], idx[bIdx]] += Ke[a, bIdx]
    return torch.from_numpy(K).float()


def fem_assemble_load(mesh, f_source):
    """Assembliert den Lastvektor F_i = ∫ f φ_i dx, mit `f_source(x, y)` Skalar pro Punkt.
    Verwendet Mittelpunkts-Quadratur pro Dreieck (1 Punkt, Ordnung 2 für linear)."""
    import numpy as _np  # type: ignore[reportMissingImports]
    nodes = mesh["nodes"].detach().cpu().numpy()
    elements = mesh["elements"].detach().cpu().numpy()
    Nn = nodes.shape[0]
    F = _np.zeros(Nn, dtype=float)
    for tri in elements:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        xk, yk = nodes[k]
        det = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
        area = 0.5 * abs(det)
        cx = (xi + xj + xk) / 3.0
        cy = (yi + yj + yk) / 3.0
        f_val = float(f_source(cx, cy))
        contrib = f_val * area / 3.0
        F[i] += contrib
        F[j] += contrib
        F[k] += contrib
    return torch.from_numpy(F).float()


def fem_poisson_2d(mesh, f_source, dirichlet_value=0.0):
    """Löst -Δu = f auf einem Dreiecksgitter mit homogenen (oder konstanten) Dirichlet-Randwerten.
    Rückgabe: 1D-Tensor `u` mit Nn Knotenwerten.

    Beispiel:
      mesh = mesh_unit_square(20)
      u = fem_poisson_2d(mesh, fn(x, y) => 1.0, dirichlet_value=0.0)
    """
    import numpy as _np  # type: ignore[reportMissingImports]
    K = fem_assemble_stiffness(mesh).detach().cpu().numpy()
    F = fem_assemble_load(mesh, f_source).detach().cpu().numpy()
    boundary = mesh["boundary"].detach().cpu().numpy()
    Nn = K.shape[0]
    g = float(dirichlet_value)
    # Dirichlet: setze Knotenwert = g durch Zeilen/Spalten-Substitution.
    free = _np.setdiff1d(_np.arange(Nn), boundary)
    u = _np.full(Nn, g, dtype=float)
    K_ff = K[_np.ix_(free, free)]
    F_f = F[free] - K[_np.ix_(free, boundary)] @ u[boundary]
    u_free = _np.linalg.solve(K_ff, F_f)
    u[free] = u_free
    return torch.from_numpy(u).float()


# ============================================================================
# Tiefere Symbolik: solve, simplify_sym, series (Taylor-Entwicklung)
# ============================================================================

def _require_sympy():
    try:
        import sympy  # type: ignore[reportMissingImports]
        return sympy
    except ImportError:
        raise ImportError("Symbolische Operation erfordert sympy. Bitte installieren: pip install sympy")


def _sympify_expr(sympy_mod, expr_str):
    """Parst einen Dedekind-typischen Mathe-String mit `^` als Potenz zu einem SymPy-Ausdruck."""
    s = str(expr_str).replace("^", "**").strip()
    return sympy_mod.sympify(s)


def solve_sym(equation, var):
    """Löst eine Gleichung symbolisch nach `var` (für numerische LGS: siehe `solve(A, b)`).

    - `equation`: String, entweder `"lhs = rhs"` oder Ausdruck `"f(x)"` (interpretiert als `f(x) = 0`).
      Mehrere Gleichungen können als Liste übergeben werden, dann zusammen mit Liste von Variablen.
    - `var`: Variablenname als String, oder Liste von Strings (für Systeme).
    Rückgabe: Liste von Lösungs-Strings (für eine Variable) bzw. Liste von Dicts (für Systeme).

    Beispiele:
      solve_sym("x^2 - 4", "x")            -> ["-2", "2"]
      solve_sym("x^2 + 1", "x")            -> ["-I", "I"]   (komplexe Wurzeln)
      solve_sym(["x + y - 3", "x - y - 1"], ["x", "y"])  -> [{x: 2, y: 1}]
    """
    sp = _require_sympy()

    def _to_eq(e):
        s = str(e)
        if "=" in s and "==" not in s:
            lhs, _, rhs = s.partition("=")
            return sp.Eq(_sympify_expr(sp, lhs), _sympify_expr(sp, rhs))
        return _sympify_expr(sp, s)

    if isinstance(equation, (list, tuple)):
        eqs = [_to_eq(e) for e in equation]
        if isinstance(var, (list, tuple)):
            vars_sp = [sp.Symbol(str(v)) for v in var]
        else:
            vars_sp = sp.Symbol(str(var))
        sols = sp.solve(eqs, vars_sp, dict=True)
        return [{str(k): str(v) for k, v in d.items()} for d in sols]

    eq = _to_eq(equation)
    var_sp = sp.Symbol(str(var))
    sols = sp.solve(eq, var_sp)
    return [str(s) for s in sols]


def simplify_sym(expr):
    """Symbolische Vereinfachung via SymPy (Ausmultiplizieren, Kürzen, Trigonometrie etc.).

    Beispiele:
      simplify_sym("sin(x)^2 + cos(x)^2")   -> "1"
      simplify_sym("(x^2 - 1) / (x - 1)")   -> "x + 1"
      simplify_sym("log(exp(x))")            -> "x"  (nur für reelle x; SymPy ist konservativ)
    """
    sp = _require_sympy()
    s = _sympify_expr(sp, expr)
    return str(sp.simplify(s))


def series(expr, var, x0=0, n=6):
    """Taylor-Reihenentwicklung von `expr` in `var` um `x0` bis Ordnung `n` (exklusive O-Term).

    Beispiele:
      series("sin(x)", "x", 0, 8)
        -> "x - x**3/6 + x**5/120 - x**7/5040"
      series("exp(x)", "x", 0, 5)
        -> "1 + x + x**2/2 + x**3/6 + x**4/24"
    """
    sp = _require_sympy()
    e = _sympify_expr(sp, expr)
    v = sp.Symbol(str(var))
    s = sp.series(e, v, float(x0), int(n)).removeO()
    return str(s)


# ============================================================================
# Sparse iterative Solver: cg, gmres, bicgstab + Jacobi-/ILU-Preconditioner
# ============================================================================

def _to_scipy_sparse(A):
    """Wandelt Tensor/numpy-Array/CSR-Matrix in scipy.sparse CSR um (float64)."""
    import numpy as _np  # type: ignore[reportMissingImports]
    try:
        import scipy.sparse as _sps  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("Sparse-Solver benötigen scipy.")
    if _sps.issparse(A):
        return A.tocsr().astype(float)
    if hasattr(A, "is_sparse") and A.is_sparse:
        A_dense = A.to_dense().detach().cpu().numpy()
    elif hasattr(A, "detach"):
        A_dense = A.detach().cpu().numpy()
    else:
        A_dense = _np.asarray(A)
    return _sps.csr_matrix(_np.asarray(A_dense, dtype=float))


def _to_numpy_vector(b):
    import numpy as _np  # type: ignore[reportMissingImports]
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    return _np.asarray(b, dtype=float).reshape(-1)


def _call_scipy_iterative(method, A_sp, b_np, x0_np, tol, max_iter, M, callback):
    """Ruft scipy-Krylov-Solver auf; behandelt API-Wechsel `tol` -> `rtol` (scipy >= 1.12)."""
    try:
        return method(A_sp, b_np, x0=x0_np, rtol=float(tol),
                      maxiter=int(max_iter), M=M, callback=callback)
    except TypeError:
        # Älteres scipy: nur `tol` verfügbar.
        return method(A_sp, b_np, x0=x0_np, tol=float(tol),
                      maxiter=int(max_iter), M=M, callback=callback)


def _iterative_solve_dispatch(method_name, A, b, x0, tol, max_iter, preconditioner):
    """Gemeinsame Aufruf-Mechanik für die drei Krylov-Solver."""
    try:
        import scipy.sparse.linalg as _ssl  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("Sparse-Solver benötigen scipy.")
    A_sp = _to_scipy_sparse(A)
    b_np = _to_numpy_vector(b)
    x0_np = _to_numpy_vector(x0) if x0 is not None else None
    M = preconditioner if preconditioner is not None else None
    method = getattr(_ssl, method_name)
    counter = [0]

    def _cb(*_a, **_kw):
        counter[0] += 1

    x, info = _call_scipy_iterative(method, A_sp, b_np, x0_np, tol, max_iter, M, _cb)
    residual_norm = float(((A_sp @ x) - b_np).__abs__().max())
    converged = int(info) == 0
    return {
        "x": x.tolist(),
        "converged": converged,
        "iterations": int(counter[0]),
        "info": int(info),
        "residual_inf": residual_norm,
    }


def cg(A, b, x0=None, tol=1e-8, max_iter=1000, preconditioner=None):
    """Conjugate Gradient für symmetrisch-positiv-definite Systeme A x = b.
    Rückgabe: Dict mit `x`, `converged`, `iterations`, `info`, `residual_inf`."""
    return _iterative_solve_dispatch("cg", A, b, x0, tol, max_iter, preconditioner)


def gmres(A, b, x0=None, tol=1e-8, max_iter=1000, preconditioner=None):
    """GMRES (Generalized Minimum Residual) für allgemeine (nicht-symmetrische) Systeme A x = b."""
    return _iterative_solve_dispatch("gmres", A, b, x0, tol, max_iter, preconditioner)


def bicgstab(A, b, x0=None, tol=1e-8, max_iter=1000, preconditioner=None):
    """BiCGSTAB für allgemeine Systeme A x = b; oft günstiger als GMRES bei großen Matrizen."""
    return _iterative_solve_dispatch("bicgstab", A, b, x0, tol, max_iter, preconditioner)


def jacobi_preconditioner(A):
    """Diagonaler (Jacobi-)Vorkonditionierer: M^{-1} = diag(1 / a_ii). Beschleunigt cg/gmres/bicgstab
    typischerweise um den Faktor 2–10×, wenn A diagonal-dominant ist.
    Rückgabe: scipy.sparse.linalg.LinearOperator, direkt als `preconditioner=` einsetzbar."""
    try:
        import scipy.sparse.linalg as _ssl  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("jacobi_preconditioner benötigt scipy.")
    import numpy as _np  # type: ignore[reportMissingImports]
    A_sp = _to_scipy_sparse(A)
    diag = _np.asarray(A_sp.diagonal(), dtype=float)
    diag = _np.where(diag != 0, 1.0 / diag, 1.0)
    n = A_sp.shape[0]
    return _ssl.LinearOperator(
        (n, n),
        matvec=lambda x: _np.asarray(diag * x, dtype=float),
        dtype=float,
    )


def ilu_preconditioner(A, drop_tol=1e-4, fill_factor=10):
    """Incomplete-LU-Vorkonditionierer (spilu). Stärker als Jacobi, besonders für nicht-symmetrische
    Probleme; `drop_tol` und `fill_factor` steuern Sparsity vs Genauigkeit."""
    try:
        import scipy.sparse.linalg as _ssl  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("ilu_preconditioner benötigt scipy.")
    A_sp = _to_scipy_sparse(A).tocsc()
    ilu = _ssl.spilu(A_sp, drop_tol=float(drop_tol), fill_factor=float(fill_factor))
    n = A_sp.shape[0]
    return _ssl.LinearOperator((n, n), matvec=ilu.solve, dtype=float)


# ============================================================================
# Reproducible-Notebook-Export: .ddk -> standalone HTML/Markdown
# ============================================================================




# Bioinformatics constants added during master-quantum merge
_SEQ_ALPHABETS = {'DNA': set('ACGTN'), 'RNA': set('ACGUN'), 'PROTEIN': set('ACDEFGHIKLMNPQRSTVWYBZX*')}
_PROTEIN_CODON_TABLE = {'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L', 'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M', 'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*', 'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W', 'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}
_DNA_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}


def _rebuild_additive_unit_sets():
    """Aktualisiert ADDITIVE_DIMENSION_UNIT_SETS in-place nach ├änderungen an DIMENSION_TO_BASE."""
    ADDITIVE_DIMENSION_UNIT_SETS.clear()
    for _b, tab in DIMENSION_TO_BASE.values():
        ADDITIVE_DIMENSION_UNIT_SETS.append(frozenset(tab.keys()))

def unwrap(x):
    """Entfernt Einheit/Wrapper und liefert den nackten numerischen Wert (float/int/Tensor).
    - Quantity(value, unit)  -> value (float)
    - UncertainQuantity      -> value (float; std wird verworfen)
    - 0-d torch.Tensor       -> .item()
    - Liste/Tuple            -> elementweise unwrap
    Sonst: passthrough.
    Wirft NICHTS; sicher in jeder Hot-Loop-Position aufrufbar.
    Nutzung: ersten Schritt einer per-jit/grad/fit-gerufenen Funktion, um Quantity-Overhead
    zu vermeiden ÔÇö die Compile-Zeit-Einheitenpruefung hat die Dimensionen bereits validiert."""
    if isinstance(x, Quantity):
        return x.value
    if 'UncertainQuantity' in globals() and isinstance(x, globals()['UncertainQuantity']):
        return x.value
    if hasattr(x, "shape") and hasattr(x, "numel"):
        try:
            if x.numel() == 1:
                return x.item()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return type(x)(unwrap(item) for item in x)
    return x

def quantum_circuit(n_qubits):
    """Erstellt einen neuen QuantumCircuit mit n_qubits Qubits.

    Beispiel:
        qc = quantum_circuit(2)
        qc.h(0).cx(0, 1)
        sv = statevec_sim(qc)
    """
    if isinstance(n_qubits, Quantity):
        if n_qubits.unit not in ("", None):
            raise ValueError(
                f"quantum_circuit: n_qubits muss dimensionslos sein, bekam [{n_qubits.unit}]."
            )
        n_qubits = int(float(n_qubits.value))
    else:
        n_qubits = int(n_qubits)
    return QuantumCircuit(n_qubits)

def bell_state(which: int = 0):
    """Erstellt einen der vier Bell-Zustaende als QuantumCircuit.

    which=0: |╬ª+Ôƒ® = (|00Ôƒ®+|11Ôƒ®)/ÔêÜ2
    which=1: |╬ª-Ôƒ® = (|00Ôƒ®-|11Ôƒ®)/ÔêÜ2
    which=2: |╬¿+Ôƒ® = (|01Ôƒ®+|10Ôƒ®)/ÔêÜ2
    which=3: |╬¿-Ôƒ® = (|01Ôƒ®-|10Ôƒ®)/ÔêÜ2
    """
    which = int(which.item() if hasattr(which, 'item') else which)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    if which == 1:
        qc.z(0)
    elif which == 2:
        qc.x(1)
    elif which == 3:
        qc.z(0)
        qc.x(1)
    return qc

def ghz_state(n_qubits: int):
    """GHZ-Zustand: (|0...0Ôƒ® + |1...1Ôƒ®)/ÔêÜ2 fuer n Qubits."""
    n_qubits = int(n_qubits.item() if hasattr(n_qubits, 'item') else n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc

def grover_circuit(n_qubits: int, target_state: int, n_iterations: int = None):
    """Erstellt Grover-Suchalgorithmus-Schaltkreis.

    Args:
        n_qubits: Anzahl Qubits
        target_state: Gesuchter Zustand als Integer (0 bis 2^n - 1)
        n_iterations: Grover-Iterationen (default: pi/4 * sqrt(2^n))

    Returns:
        QuantumCircuit
    """
    n_qubits = int(n_qubits.item() if hasattr(n_qubits, 'item') else n_qubits)
    target_state = int(target_state.item() if hasattr(target_state, 'item') else target_state)
    dim = 1 << n_qubits
    if target_state < 0 or target_state >= dim:
        raise ValueError(f"grover_circuit: target_state={target_state} ausserhalb [0, {dim-1}].")

    if n_iterations is None:
        n_iterations = int(_builtin_max(1, round(_math.pi / 4 * _math.sqrt(dim))))
    n_iterations = int(n_iterations.item() if hasattr(n_iterations, 'item') else n_iterations)

    qc = QuantumCircuit(n_qubits)

    # Initialisierung: Hadamard auf alle Qubits
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(n_iterations):
        # Orakel: Phase-Flip fuer |targetÔƒ®
        # Implementiert via Z auf korrekten Qubits (vereinfacht fuer native Simulation)
        bits = format(target_state, f'0{n_qubits}b')
        # X-Gatter auf Qubits die 0 sind (Bitumkehr fuer Multi-CZ)
        for q in range(n_qubits):
            if bits[q] == '0':
                qc.x(q)
        # Mehrstufige CZ-Kette (approximiert Multi-Qubit-Controlled-Z)
        for q in range(n_qubits - 1):
            qc.cz(q, n_qubits - 1)
        for q in range(n_qubits):
            if bits[q] == '0':
                qc.x(q)

        # Diffusion-Operator
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits):
            qc.x(q)
        for q in range(n_qubits - 1):
            qc.cz(q, n_qubits - 1)
        for q in range(n_qubits):
            qc.x(q)
        for q in range(n_qubits):
            qc.h(q)

    return qc

def gc_content(dna):
    """Liefert den Anteil G+C in einer DNA-Sequenz (0..1).
    Akzeptiert auch RNA ÔÇö N wird ignoriert."""
    if not isinstance(dna, str):
        raise TypeError(f"gc_content: erwarte String, erhalten {type(dna).__name__}.")
    s = dna.upper()
    if not s:
        return 0.0
    valid = [c for c in s if c in "ACGTU"]
    if not valid:
        return 0.0
    gc = sum(1 for c in valid if c in "GC")
    return gc / len(valid)

def reverse_complement(dna):
    """Liefert das Reverse-Complement einer DNA-Sequenz."""
    if not isinstance(dna, str):
        raise TypeError(f"reverse_complement: erwarte String, erhalten {type(dna).__name__}.")
    s = dna.upper()
    bad = [c for c in s if c not in _DNA_COMPLEMENT]
    if bad:
        raise ValueError(
            f"reverse_complement: ungueltiges DNA-Zeichen {bad[0]!r} "
            f"(erlaubt: A, C, G, T, N)."
        )
    return "".join(_DNA_COMPLEMENT[c] for c in reversed(s))

def transcribe(dna):
    """DNA -> RNA: ersetzt T durch U."""
    if not isinstance(dna, str):
        raise TypeError(f"transcribe: erwarte String, erhalten {type(dna).__name__}.")
    return dna.upper().replace("T", "U")

def translate(rna, stop_at_stop=True):
    """RNA -> Protein (1-Letter-Code). Liest in 3er-Codons; unbekannte Codons -> 'X'.
    stop_at_stop=True (Default): bricht beim ersten Stop-Codon (*) ab.
    """
    if not isinstance(rna, str):
        raise TypeError(f"translate: erwarte String, erhalten {type(rna).__name__}.")
    s = rna.upper().replace("T", "U")
    out = []
    for i in range(0, len(s) - 2, 3):
        codon = s[i:i+3]
        aa = _PROTEIN_CODON_TABLE.get(codon, "X")
        if aa == "*" and stop_at_stop:
            break
        out.append(aa)
    return "".join(out)

def k_mer_count(seq, k):
    """Liefert ein Dict {k_mer: count} fuer alle ueberlappenden k-Mere."""
    if not isinstance(seq, str):
        raise TypeError(f"k_mer_count: erwarte String, erhalten {type(seq).__name__}.")
    k = int(k)
    if k < 1:
        raise ValueError(f"k_mer_count: k muss >= 1 sein, bekam {k}.")
    s = seq.upper()
    counts = {}
    for i in range(0, len(s) - k + 1):
        kmer = s[i:i+k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts

def smiles_descriptors(smiles):
    """Molekulare Deskriptoren aus einer SMILES-Notation via rdkit.
    Liefert Dict mit: mw [g/mol], logp, num_atoms, num_heavy_atoms, num_rings,
    num_aromatic_rings, hbd, hba, tpsa [Angstrom^2] (oder [Angstrom]), num_rotatable_bonds.
    Falls rdkit nicht installiert ist, erfolgt ein Fallback auf den integrierten Parser
    für die Basis-Deskriptoren (mw, logp, hbd, hba). Ringe und TPSA bleiben dabei
    auf 0/None gesetzt; wer diese Felder zwingend braucht, muss rdkit installieren
    (`pip install rdkit`)."""
    if not isinstance(smiles, str):
        raise TypeError(f"smiles_descriptors: erwarte String, erhalten {type(smiles).__name__}.")
    try:
        from rdkit import Chem  # type: ignore[import-untyped]
        from rdkit.Chem import Descriptors, Lipinski  # type: ignore[import-untyped]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"smiles_descriptors: ungueltige SMILES {smiles!r}.")
        return {
            "mw": Quantity(float(Descriptors.MolWt(mol)), "g/mol"),
            "logp": float(Descriptors.MolLogP(mol)),
            "num_atoms": int(mol.GetNumAtoms()),
            "num_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "num_rings": int(Descriptors.RingCount(mol)),
            "num_aromatic_rings": int(Descriptors.NumAromaticRings(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "tpsa": Quantity(float(Descriptors.TPSA(mol)), "Angstrom"),
            "num_rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        }
    except ImportError:
        counts, atoms, bonds = _parse_smiles(smiles)
        mw = sum(ATOMIC_MASSES.get(elem, 0.0) * count for elem, count in counts.items())
        hbd = 0
        hba = 0
        STANDARD_VALENCES = {
            "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3,
            "F": 1, "Cl": 1, "Br": 1, "I": 1
        }
        logp = 0.0
        for idx, atom in enumerate(atoms):
            elem = atom["elem"]
            aromatic = atom["aromatic"]
            if atom["is_bracket"]:
                h_count = atom["explicit_h"]
            else:
                conn_bonds = [order for (i1, i2, order) in bonds if i1 == idx or i2 == idx]
                tot_order = sum(conn_bonds)
                val = STANDARD_VALENCES.get(elem, 0)
                h_count = int(round(_builtin_max(0.0, float(val) - tot_order)))
            if elem in ["N", "O"]:
                hba += 1
                if h_count > 0:
                    hbd += 1
            elif elem == "F":
                hba += 1
            if elem == "C":
                if aromatic:
                    logp += 0.35
                else:
                    logp += 0.40
            elif elem == "O":
                if h_count > 0:
                    logp -= 1.50
                else:
                    logp -= 0.40
            elif elem == "N":
                if h_count > 0:
                    logp -= 1.00
                elif aromatic:
                    logp -= 0.50
                else:
                    logp -= 0.70
            elif elem == "F":
                logp += 0.14
            elif elem == "Cl":
                logp += 0.56
            elif elem == "Br":
                logp += 0.88
            elif elem == "I":
                logp += 1.25
            elif elem == "S":
                logp += 0.20
            elif elem == "P":
                logp -= 0.40
            logp += h_count * 0.11
        return {
            "mw": Quantity(mw, "g/mol"),
            "logp": float(logp),
            "num_atoms": len(atoms),
            "num_heavy_atoms": sum(1 for a in atoms if a["elem"] != "H"),
            "num_rings": 0,
            "num_aromatic_rings": 0,
            "hbd": int(hbd),
            "hba": int(hba),
            "tpsa": Quantity(0.0, "Angstrom"),
            "num_rotatable_bonds": 0,
        }

def lipinski_rule_of_five(smiles):
    """Lipinskis 'Rule of Five' fuer orale Bioverfuegbarkeit.
    Liefert Dict mit Werten, Boolean-Checks und Anzahl Verletzungen."""
    desc = smiles_descriptors(smiles)
    mw = desc["mw"].value
    logp = desc["logp"]
    hbd = desc["hbd"]
    hba = desc["hba"]
    checks = {
        "mw_le_500":  mw <= 500.0,
        "logp_le_5":  logp <= 5.0,
        "hbd_le_5":   hbd <= 5,
        "hba_le_10":  hba <= 10,
    }
    violations = sum(1 for ok in checks.values() if not ok)
    return {
        "mw": desc["mw"],
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "checks": checks,
        "violations": violations,
        "passes": violations <= 1,
    }

def _notebook_in_progress_set():
    """Process-wide re-entry tracking set. Stored on sys so it persists across exec contexts
    (each exec(compile_source(...)) would otherwise get a fresh ml_runtime copy)."""
    import sys as _sys
    key = "_dedekind_notebook_export_in_progress"
    s = getattr(_sys, key, None)
    if s is None:
        s = set()
        setattr(_sys, key, s)
    return s


def export_notebook(source_path, output_path=None, format="html", title=None,
                    include_hash=True, capture_plots=True):
    """Runs a .ddk file and writes a standalone file (HTML or Markdown) bundling
    source code, stdout output, plots (base64 PNG), and a SHA-256 hash of the source.

    Parameters:
      source_path: path to the .ddk file.
      output_path: target file (default: `<source>.html` or `.md`).
      format: "html" or "md".
      title: optional title; default = filename without extension.
      include_hash: includes a SHA-256 hash of the source in the output (reproducibility).
      capture_plots: collects `_dedekind_plots` and embeds them as base64 PNG.

    Returns: path to the generated file.
    """
    import os
    import sys
    import io
    import hashlib
    src_path = os.path.abspath(str(source_path))
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"export_notebook: file not found: {src_path}")
    # Re-entry guard: if the source file exports itself, it would call itself infinitely while
    # executing. We mark the path and return a stub on re-entry.
    in_progress = _notebook_in_progress_set()
    if src_path in in_progress:
        return output_path or (os.path.splitext(src_path)[0] +
                               (".html" if str(format).lower() == "html" else ".md"))
    in_progress.add(src_path)
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    fmt = str(format).lower()
    if fmt not in ("html", "md", "markdown"):
        raise ValueError("export_notebook: format must be 'html' or 'md'.")
    if output_path is None:
        ext = "html" if fmt == "html" else "md"
        output_path = os.path.splitext(src_path)[0] + f".{ext}"
    title_str = str(title) if title else os.path.basename(os.path.splitext(src_path)[0])
    src_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    # Compile the source and execute it in an isolated globals dict; capture stdout.
    # We import locally to avoid circular imports when inlining.
    try:
        from dedekind import compile_source  # type: ignore[import-not-found]
        py_code = compile_source(source, filepath=src_path)
        old_stdout = sys.stdout
        captured = io.StringIO()
        exec_globals = {}
        try:
            sys.stdout = captured
            exec(py_code, exec_globals)
        finally:
            sys.stdout = old_stdout
        stdout_text = captured.getvalue()
        plots_b64 = []
        if capture_plots:
            plots_b64 = list(exec_globals.get("_dedekind_plots", []))

        if fmt == "html":
            content = _render_notebook_html(title_str, source, stdout_text, plots_b64,
                                            src_hash if include_hash else None,
                                            os.path.basename(src_path))
        else:
            content = _render_notebook_markdown(title_str, source, stdout_text, plots_b64,
                                                src_hash if include_hash else None,
                                                os.path.basename(src_path))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path
    finally:
        in_progress.discard(src_path)


def _render_notebook_html(title, source, stdout_text, plots_b64, src_hash, src_name):
    import html as _html
    import datetime
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:900px;margin:2em auto;padding:0 1em;color:#222;}",
        "h1,h2{border-bottom:1px solid #ddd;padding-bottom:.2em;}",
        "pre{background:#f6f8fa;padding:1em;border-radius:6px;overflow-x:auto;"
        "font-family:'SF Mono',Consolas,monospace;font-size:.9em;}",
        ".meta{color:#666;font-size:.85em;margin:.5em 0;}",
        ".plot{margin:1em 0;text-align:center;}",
        ".plot img{max-width:100%;border:1px solid #ddd;border-radius:4px;}",
        ".hash{font-family:monospace;font-size:.8em;color:#888;"
        "word-break:break-all;background:#f6f8fa;padding:.3em .5em;border-radius:4px;}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{_html.escape(title)}</h1>",
        f'<div class="meta">Source: <code>{_html.escape(src_name)}</code> · '
        f'Generated: {datetime.datetime.now().isoformat(timespec="seconds")}</div>',
    ]
    if src_hash:
        parts.append(f'<div class="meta">SHA-256: <span class="hash">{src_hash}</span></div>')
    parts.append("<h2>Source</h2>")
    parts.append(f"<pre><code>{_html.escape(source)}</code></pre>")
    parts.append("<h2>Output</h2>")
    parts.append(f"<pre>{_html.escape(stdout_text) if stdout_text else '(no output)'}</pre>")
    if plots_b64:
        parts.append(f"<h2>Plots ({len(plots_b64)})</h2>")
        for i, b64 in enumerate(plots_b64, 1):
            parts.append(f'<div class="plot"><img alt="Plot {i}" '
                         f'src="data:image/png;base64,{b64}"></div>')
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def _render_notebook_markdown(title, source, stdout_text, plots_b64, src_hash, src_name):
    import datetime
    lines = [
        f"# {title}",
        "",
        f"*Source:* `{src_name}`  ·  *Generated:* {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    if src_hash:
        lines.append(f"*SHA-256:* `{src_hash}`")
        lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("```")
    lines.append(source.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Output")
    lines.append("")
    lines.append("```")
    lines.append((stdout_text or "(no output)").rstrip())
    lines.append("```")
    lines.append("")
    if plots_b64:
        lines.append(f"## Plots ({len(plots_b64)})")
        lines.append("")
        for i, b64 in enumerate(plots_b64, 1):
            lines.append(f"![Plot {i}](data:image/png;base64,{b64})")
            lines.append("")
    return "\n".join(lines) + "\n"


# ============================================================================
# Paper-mode output: print_table with LaTeX booktabs / Markdown / CSV + +/- for UncertainQuantity
# ============================================================================

def _format_cell_value(v, precision=4):
    """Formats a cell: UncertainQuantity -> 'val +/- std [unit]', Quantity -> 'val [unit]'."""
    if isinstance(v, UncertainQuantity):
        unit = f" [{v.unit}]" if getattr(v, "unit", "") else ""
        return f"{v.value:.{precision}g} ± {v.std:.{precision}g}{unit}"
    if isinstance(v, Quantity):
        unit = f" [{v.unit}]" if v.unit else ""
        return f"{v.value:.{precision}g}{unit}"
    if isinstance(v, float):
        return f"{v:.{precision}g}"
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            iv = v.item()
            return _format_cell_value(iv, precision)
        except Exception:
            pass
    return str(v)


def _format_cell_latex(v, precision=4):
    """Like _format_cell_value, but LaTeX-ready (`\\pm`, `\\,[\\mathrm{...}]`)."""
    if isinstance(v, UncertainQuantity):
        unit = f"\\,[\\mathrm{{{v.unit}}}]" if getattr(v, "unit", "") else ""
        return f"${v.value:.{precision}g} \\pm {v.std:.{precision}g}{unit}$"
    if isinstance(v, Quantity):
        unit = f"\\,[\\mathrm{{{v.unit}}}]" if v.unit else ""
        return f"${v.value:.{precision}g}{unit}$"
    if isinstance(v, float):
        return f"${v:.{precision}g}$"
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return _format_cell_latex(v.item(), precision)
        except Exception:
            pass
    return str(v)


def print_table(rows, headers=None, format="markdown", precision=4, caption=None, label=None):
    """Erzeugt eine Tabelle in einem von vier Formaten und gibt sie via `print()` aus.

    - `rows`: Liste von Listen/Tupeln *oder* eine `DataFrame`.
    - `headers`: optional Liste von Spaltennamen (wenn rows kein DataFrame ist).
    - `format`: `"markdown"` (default), `"latex"` (booktabs), `"csv"`, `"plain"` (ASCII-Tabelle).
    - `precision`: signifikante Stellen für float/Quantity (default 4).
    - `caption`, `label`: nur für LaTeX (Tabellen-Caption und `\\label{...}`).

    UncertainQuantity wird automatisch als `val ± std [unit]` formatiert,
    Quantity als `val [unit]`. Für LaTeX werden ± und Einheiten mathmodisch gesetzt.
    """
    if isinstance(rows, DataFrame):
        df = rows
        headers = list(df.columns)
        # Einheiten in Header für Märkdown/Plain/CSV — LaTeX bleibt einheitslos im Header.
        units_map = dict(df._units)
        rows = [[df._cols[j][i] for j in range(len(headers))] for i in range(df.n_rows)]
    else:
        units_map = {}
        if headers is None:
            headers = [f"col{i+1}" for i in range(len(rows[0]) if rows else 0)]
        else:
            headers = list(headers)
    fmt = str(format).lower()
    if fmt == "latex":
        out = _table_latex(rows, headers, precision, caption, label)
    elif fmt == "csv":
        out = _table_csv(rows, headers, precision, units_map)
    elif fmt == "plain":
        out = _table_plain(rows, headers, precision, units_map)
    else:
        out = _table_markdown(rows, headers, precision, units_map)
    print(out)
    return out


def _decorate_header_with_unit(name, units_map):
    u = units_map.get(name) if units_map else None
    return f"{name} [{u}]" if u else name


def _table_markdown(rows, headers, precision, units_map):
    hdr = [_decorate_header_with_unit(h, units_map) for h in headers]
    body = [[_format_cell_value(v, precision) for v in row] for row in rows]
    widths = [_builtin_max(len(hdr[j]),
                           *(len(b[j]) for b in body) if body else (len(hdr[j]),))
              for j in range(len(headers))]
    lines = []
    lines.append("| " + " | ".join(hdr[j].ljust(widths[j]) for j in range(len(headers))) + " |")
    lines.append("|" + "|".join("-" * (widths[j] + 2) for j in range(len(headers))) + "|")
    for b in body:
        lines.append("| " + " | ".join(b[j].ljust(widths[j]) for j in range(len(headers))) + " |")
    return "\n".join(lines)


def _table_latex(rows, headers, precision, caption, label):
    cols = "l" * len(headers)
    lines = [r"\begin{table}[h]", r"\centering"]
    if caption:
        lines.append(r"\caption{" + str(caption) + "}")
    if label:
        lines.append(r"\label{" + str(label) + "}")
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(str(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = [_format_cell_latex(v, precision) for v in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _table_csv(rows, headers, precision, units_map):
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([_decorate_header_with_unit(h, units_map) for h in headers])
    for row in rows:
        w.writerow([_format_cell_value(v, precision) for v in row])
    return buf.getvalue().rstrip("\n")


def _table_plain(rows, headers, precision, units_map):
    hdr = [_decorate_header_with_unit(h, units_map) for h in headers]
    body = [[_format_cell_value(v, precision) for v in row] for row in rows]
    widths = [_builtin_max(len(hdr[j]),
                           *(len(b[j]) for b in body) if body else (len(hdr[j]),))
              for j in range(len(headers))]
    sep = "  "
    lines = [sep.join(hdr[j].ljust(widths[j]) for j in range(len(headers)))]
    lines.append(sep.join("-" * widths[j] for j in range(len(headers))))
    for b in body:
        lines.append(sep.join(b[j].ljust(widths[j]) for j in range(len(headers))))
    return "\n".join(lines)
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
# --- Computational Fluid Dynamics (CFD) via Lattice Boltzmann Method ---
#
# D2Q9 BGK lattice-Boltzmann Solver, vollständig differenzierbar.
# Tier-1-Refactor (2026-05): physikalisch korrekte Drag/Lift via
# Momentum-Exchange-Method, tau-Stabilitätscheck, cachefreie innere
# Schleife, Reynolds-Helper, Karman-Vortex-Benchmark.

import math
import torch

# --- D2Q9 Konstanten (Modul-Ebene, einmal allokiert) -------------------

_LBM_CX_LIST = [0, 1, 0, -1, 0, 1, -1, -1, 1]
_LBM_CY_LIST = [0, 0, 1, 0, -1, 1, 1, -1, -1]
_LBM_W_LIST = [4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9,
               1.0/36, 1.0/36, 1.0/36, 1.0/36]
# Bounce-Back: für jede Richtung b die entgegengesetzte Richtung -b
_LBM_OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# MRT-Transformationsmatrix (Lallemand & Luo 2000, d'Humières et al. 2002)
# Reihen sind die 9 Momente: rho, e, eps, jx, qx, jy, qy, pxx, pxy.
# Spalten entsprechen den 9 D2Q9-Richtungen.
_LBM_MRT_M = [
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1],   # rho     (m0, konserviert)
    [-4, -1, -1, -1, -1,  2,  2,  2,  2],   # e       (m1, Energie)
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1],   # eps     (m2, Energie²)
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1],   # jx      (m3, Impuls x — konserviert)
    [ 0, -2,  0,  2,  0,  1, -1, -1,  1],   # qx      (m4, Wärmestrom x)
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1],   # jy      (m5, Impuls y — konserviert)
    [ 0,  0, -2,  0,  2,  1,  1, -1, -1],   # qy      (m6, Wärmestrom y)
    [ 0,  1, -1,  1, -1,  0,  0,  0,  0],   # pxx     (m7, Spannungs-Diag — Scherviskosität)
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1],   # pxy     (m8, Spannungs-Off — Scherviskosität)
]


def _to_double_tensor(v):
    t = _to_tensor(v)
    return t.double()


def _lbm_constants(device, dtype):
    """Liefert (cx, cy, w) als 9x1x1-Tensoren auf dem gewünschten Device/Dtype.
    Wird einmal pro LbmSimulation in __init__ aufgerufen und gecacht."""
    cx = torch.tensor(_LBM_CX_LIST, dtype=dtype, device=device).view(9, 1, 1)
    cy = torch.tensor(_LBM_CY_LIST, dtype=dtype, device=device).view(9, 1, 1)
    w = torch.tensor(_LBM_W_LIST, dtype=dtype, device=device).view(9, 1, 1)
    return cx, cy, w


def lbm_equilibrium(rho, u, device=None):
    """Berechnet die D2Q9 Gleichgewichtsverteilung f_eq.

    rho: shape (nx, ny) oder (1, ny); u: shape (2, nx, ny) oder (2, 1, ny).
    Rückgabe: f_eq mit shape (9, *rho.shape).
    """
    if device is None:
        device = rho.device
    ux = u[0]
    uy = u[1]
    u2 = ux * ux + uy * uy
    cx, cy, w = _lbm_constants(device, torch.float64)
    c_dot_u = cx * ux + cy * uy
    feq = w * rho * (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u * c_dot_u - 1.5 * u2)
    return feq


class LbmSimulation:
    """Vollständig differenzierbarer 2D-D2Q9 Lattice-Boltzmann-Solver.

    Parameter
    ---------
    nx, ny : int
        Gittergröße in Lattice-Einheiten.
    tau : float
        Scher-Relaxationszeit. Muss > 0.5 sein (Stabilität). nu_lattice = (tau - 0.5)/3.
        Bei MRT ist tau die Scherviskositäts-Relaxation (s7=s8=1/tau).
    obstacle_mask : Tensor[nx, ny] | None
        Mask in [0, 1]: 1 = solides Hindernis, 0 = freie Strömung. Soft-Bounce-Back
        per Default; "hard" als bounce_back-Parameter schärft die Mask auf {0, 1}
        für physikalisch korrekte Halfway-Bounce-Back-Behandlung.
    inlet_u : float
        Eingangsgeschwindigkeit (Lattice-Einheit), wird sowohl für Initialisierung
        als auch (per Default) für die Inlet-BC genutzt. Default 0.0.
    collision : str
        "bgk" (default): Bhatnagar-Gross-Krook Single-Relaxation. Schnell,
        differenzierbar, stabil bis Re~300. "mrt" (Multiple-Relaxation-Time
        nach Lallemand & Luo 2000): höhere Stabilität (Re~2000), ~30% langsamer
        wegen Matrix-Transformation in Momentenraum.
    bounce_back : str
        "soft" (default): glatte Maske-Mischung, differenzierbar in M. Geeignet
        für Shape-Optimierung. "hard": M wird auf {0,1} geschärft (Halfway-
        Bounce-Back). Physikalisch korrektere C_D-Werte; Gradient bzgl. M ist
        durchgereicht, aber durch die Schärfung quasi-binär.
    """

    # Empfohlene Maximalgeschwindigkeit (Mach-Limit ~ 0.1 für niedrige Kompressibilität)
    MAX_LATTICE_U = 0.1

    def __init__(self, nx, ny, tau, obstacle_mask=None, inlet_u=0.0,
                 collision="bgk", bounce_back="soft", smagorinsky_constant=0.0):
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        self._inlet_u = float(inlet_u)
        self.collision_model = str(collision).lower()
        self.bounce_back_model = str(bounce_back).lower()
        self.smagorinsky_constant = smagorinsky_constant
        if self.collision_model not in ("bgk", "mrt"):
            raise ValueError(
                f"LbmSimulation: collision='{collision}' unbekannt, "
                f"erwartet 'bgk' oder 'mrt'."
            )
        if self.bounce_back_model not in ("soft", "hard"):
            raise ValueError(
                f"LbmSimulation: bounce_back='{bounce_back}' unbekannt, "
                f"erwartet 'soft' oder 'hard'."
            )

        if self.tau <= 0.5:
            raise ValueError(
                f"LbmSimulation: tau={self.tau} muss > 0.5 sein "
                f"(BGK-Stabilität; nu_lattice = (tau-0.5)/3 muss > 0)."
            )
        if abs(self._inlet_u) > self.MAX_LATTICE_U:
            # Kein Fehler, nur Warnung über Print — Mach-Zahl in LBM ist u*sqrt(3)
            print(
                f"[LbmSimulation] Warnung: |inlet_u|={abs(self._inlet_u):.3g} > "
                f"{self.MAX_LATTICE_U} → Mach-Effekte möglich, niedrigere Geschwindigkeit empfohlen."
            )

        # Obstacle-Maske: leerer Tensor (1-elementig mit 0.0) oder (nx, ny)
        if obstacle_mask is not None:
            mask_t = _to_double_tensor(obstacle_mask)
            if mask_t.numel() == 1 and float(mask_t.flatten()[0]) == 0.0:
                self.obstacle_mask = torch.zeros(
                    (self.nx, self.ny), dtype=torch.float64
                )
            else:
                self.obstacle_mask = mask_t
                if self.obstacle_mask.shape != (self.nx, self.ny):
                    raise ValueError(
                        f"Obstacle mask shape must be ({self.nx}, {self.ny}), "
                        f"got {tuple(self.obstacle_mask.shape)}."
                    )
        else:
            self.obstacle_mask = torch.zeros(
                (self.nx, self.ny), dtype=torch.float64
            )

        device = self.obstacle_mask.device
        dtype = torch.float64

        # Konstanten einmal allokieren — wiederverwendet pro Step und in get_drag_lift
        self._cx, self._cy, self._w = _lbm_constants(device, dtype)

        # MRT-Setup: Transformationsmatrix M (9,9), Inverse, Diagonal-Relaxation S
        if self.collision_model == "mrt":
            M_np = _LBM_MRT_M
            M_t = torch.tensor(M_np, dtype=dtype, device=device)  # (9, 9)
            self._mrt_M = M_t
            # Inverse via geschlossener Formel (M ist orthogonal bis auf Skalierung,
            # M_inv = M^T * diag(1/||row||²))
            row_norm_sq = (M_t * M_t).sum(dim=1)  # (9,)
            self._mrt_Minv = M_t.t() / row_norm_sq.view(1, 9)
            # Relaxationsraten (Lallemand & Luo 2000):
            # s0 = s3 = s5 = 0 (rho, jx, jy konserviert)
            # s1 = s2 = 1.4 (e, eps — Bulk-Viskosität & energy²)
            # s4 = s6 = 1.2 (qx, qy — Geist-Moden)
            # s7 = s8 = 1/tau (Scherviskosität)
            s7 = 1.0 / self.tau
            s_rates = [0.0, 1.4, 1.4, 0.0, 1.2, 0.0, 1.2, s7, s7]
            self._mrt_S = torch.tensor(s_rates, dtype=dtype, device=device).view(9, 1, 1)
        else:
            self._mrt_M = None
            self._mrt_Minv = None
            self._mrt_S = None

        # Initialisierung: rho=1, u=inlet_u im Fluid, u=0 im Solid
        rho_init = torch.ones((self.nx, self.ny), dtype=dtype, device=device)
        u_init = torch.zeros((2, self.nx, self.ny), dtype=dtype, device=device)
        # Fluid-only initialisieren (sonst springen Populationen im Solid herum)
        fluid = 1.0 - self.obstacle_mask
        u_init[0] = self._inlet_u * fluid

        self.f = lbm_equilibrium(rho_init, u_init, device)

    def _equilibrium_from_macro(self, rho, ux, uy):
        """Lokales Equilibrium aus rho, ux, uy (gleiche Shapes erlaubt) ohne erneutes
        Anlegen der cx/cy/w-Tensoren."""
        c_dot_u = self._cx * ux + self._cy * uy
        u2 = ux * ux + uy * uy
        return self._w * rho * (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u * c_dot_u - 1.5 * u2)

    def set_obstacle_mask(self, mask):
        self.obstacle_mask = _to_double_tensor(mask)
        if self.obstacle_mask.shape != (self.nx, self.ny):
            raise ValueError(
                f"Obstacle mask shape must be ({self.nx}, {self.ny})."
            )

    def _mrt_collide(self, f, rho, ux, uy):
        """MRT-Collision: Transform → Relax in Momentum Space → Transform Back.

        Lallemand & Luo (2000) Standard-Rates; konservierte Momente (rho, jx, jy)
        haben s=0, Scherviskosität ist tau (s7=s8=1/tau).
        """
        # f: (9, nx, ny) → reshape für 9x9 Matrix-Multiplikation
        f_flat = f.view(9, -1)               # (9, N)
        m = self._mrt_M @ f_flat              # (9, N) Momente

        # Equilibriums-Momente (Lallemand & Luo 2000, normiert auf rho)
        rho_flat = rho.view(-1)               # (N,)
        jx = (rho * ux).view(-1)              # (N,)
        jy = (rho * uy).view(-1)              # (N,)
        rho_safe = rho_flat + 1e-15
        jx2_p_jy2 = (jx * jx + jy * jy) / rho_safe

        m_eq = torch.zeros_like(m)
        m_eq[0] = rho_flat
        m_eq[1] = -2.0 * rho_flat + 3.0 * jx2_p_jy2
        m_eq[2] = rho_flat - 3.0 * jx2_p_jy2
        m_eq[3] = jx
        m_eq[4] = -jx
        m_eq[5] = jy
        m_eq[6] = -jy
        m_eq[7] = (jx * jx - jy * jy) / rho_safe
        m_eq[8] = jx * jy / rho_safe

        # Relaxation in Momentenraum
        S = self._mrt_S.view(9, 1)            # (9, 1)
        m_new = m - S * (m - m_eq)

        # Rücktransformation
        f_new_flat = self._mrt_Minv @ m_new   # (9, N)
        return f_new_flat.view(9, self.nx, self.ny)

    def step(self, inlet_u=None):
        """Ein LBM-Zeitschritt: Stream → Inlet/Outlet-BC → Macroscopic → Collide → Bounce."""
        if inlet_u is None:
            inlet_u_val = self._inlet_u
        else:
            inlet_u_val = float(inlet_u) if not isinstance(inlet_u, torch.Tensor) else inlet_u

        # 1. Streaming via torch.roll (out-of-place, autograd-freundlich)
        f_streamed = torch.stack([
            torch.roll(
                self.f[i],
                shifts=(int(_LBM_CX_LIST[i]), int(_LBM_CY_LIST[i])),
                dims=(0, 1),
            )
            for i in range(9)
        ])

        # 2. Inlet (x=0): Gleichgewicht mit prescribed inlet_u
        ny = self.ny
        device = self.f.device
        dtype = self.f.dtype
        rho_col = torch.ones((ny,), dtype=dtype, device=device)
        u_col = torch.zeros((2, ny), dtype=dtype, device=device)
        if isinstance(inlet_u_val, torch.Tensor):
            u_col = torch.stack([
                torch.ones((ny,), dtype=dtype, device=device) * inlet_u_val,
                torch.zeros((ny,), dtype=dtype, device=device),
            ])
        else:
            u_col[0] = inlet_u_val
        feq_inlet = lbm_equilibrium(
            rho_col.unsqueeze(0), u_col.unsqueeze(1), device
        )  # (9, 1, ny)
        f_inlet = torch.cat([feq_inlet, f_streamed[:, 1:, :]], dim=1)

        # 3. Outlet (x=nx-1): Zero-Gradient (Kopie vom vorletzten in den letzten Knoten)
        f_outlet = torch.cat([f_inlet[:, :-1, :], f_inlet[:, -2:-1, :]], dim=1)

        # 4. Makroskopische Felder
        rho = torch.sum(f_outlet, dim=0)
        rho_safe = rho + 1e-15
        ux = torch.sum(f_outlet * self._cx, dim=0) / rho_safe
        uy = torch.sum(f_outlet * self._cy, dim=0) / rho_safe

        # 5. Collision (BGK oder MRT)
        if self.collision_model == "mrt":
            f_coll = self._mrt_collide(f_outlet, rho, ux, uy)
        else:
            feq = self._equilibrium_from_macro(rho, ux, uy)
            use_smag = False
            if isinstance(self.smagorinsky_constant, torch.Tensor):
                use_smag = float(self.smagorinsky_constant.detach().item()) > 0.0
            else:
                use_smag = float(self.smagorinsky_constant) > 0.0
                
            if use_smag:
                f_neq = f_outlet - feq
                Q_xx = torch.sum(self._cx.view(9, 1, 1)**2 * f_neq, dim=0)
                Q_yy = torch.sum(self._cy.view(9, 1, 1)**2 * f_neq, dim=0)
                Q_xy = torch.sum(self._cx.view(9, 1, 1) * self._cy.view(9, 1, 1) * f_neq, dim=0)
                Q_bar = torch.sqrt(Q_xx**2 + Q_yy**2 + 2.0 * Q_xy**2 + 1e-12)
                temp = 18.0 * math.sqrt(2.0) * (self.smagorinsky_constant**2) * Q_bar / rho_safe
                tau_eff = 0.5 * (self.tau + torch.sqrt(self.tau**2 + temp))
                f_coll = f_outlet - (1.0 / tau_eff.unsqueeze(0)) * f_neq
            else:
                f_coll = f_outlet - (1.0 / self.tau) * (f_outlet - feq)

        # 6. Bounce-Back: an Hindernis-Zellen wird f durch f[opposite] ersetzt.
        #    Soft-Mask: M bleibt in [0,1], Mischung mit Verlauf.
        #    Hard:     M wird auf {0,1} geschärft (Halfway-Bounce-Back, scharf).
        f_bounce = f_outlet[_LBM_OPPOSITE]
        if self.bounce_back_model == "hard":
            # Schwellenwert 0.5: differenzierbar via float-Cast, aber Gradient = 0
            # (für Shape-Optimierung "soft" verwenden)
            M_sharp = (self.obstacle_mask >= 0.5).to(self.obstacle_mask.dtype)
            M = M_sharp.unsqueeze(0)
        else:
            M = self.obstacle_mask.unsqueeze(0)
        self.f = (1.0 - M) * f_coll + M * f_bounce

    def run(self, steps, inlet_u=None):
        for _ in range(int(steps)):
            self.step(inlet_u)

    def get_velocity(self):
        rho = torch.sum(self.f, dim=0)
        rho_safe = rho + 1e-15
        ux = torch.sum(self.f * self._cx, dim=0) / rho_safe
        uy = torch.sum(self.f * self._cy, dim=0) / rho_safe
        return torch.stack([ux, uy], dim=0)

    def get_density(self):
        return torch.sum(self.f, dim=0)

    def get_pressure(self):
        """Druck p = c_s^2 * rho mit c_s^2 = 1/3 in Lattice-Einheiten."""
        return torch.sum(self.f, dim=0) / 3.0

    def get_drag_lift(self, target_mask=None):
        """Drag und Lift via Momentum-Exchange-Method (MEM).

        Für jeden Lattice-Link (x, c_b), der vom Fluid (1-M(x)) in den Solid
        (M(x+c_b)) zeigt, ist die pro Schritt übertragene Impulskomponente
        in Richtung d gegeben durch  2 * c_b^d * f_b(x).

            F_d = 2 * Σ_{x,b}  c_b^d * f_b(x) * (1 - M(x)) * M(x + c_b)

        Das ist die physikalisch korrekte halbweg-Bounce-Back-Formel, sie ist
        differenzierbar in M und f.  Bei sigmoiden Soft-Masken wird sie um die
        Interface-Zellen herum konzentriert; Bulk-Solid trägt nicht bei.

        Rückgabe: Tensor [Fx, Fy] in Lattice-Einheiten (Mass * Velocity / Step).
        """
        if target_mask is None:
            M = self.obstacle_mask
        else:
            mask_t = _to_double_tensor(target_mask)
            # Legacy: [0.0] als Sentinel für "use stored mask"
            if mask_t.numel() == 1 and float(mask_t.flatten()[0]) == 0.0:
                M = self.obstacle_mask
            else:
                M = mask_t

        # Berechne Σ_b c_b^d * f_b(x) * w_b(x) mit w_b(x) = (1-M(x)) * M(x+c_b)
        one_minus_M = 1.0 - M  # (nx, ny)
        Fx = torch.zeros((), dtype=self.f.dtype, device=self.f.device)
        Fy = torch.zeros((), dtype=self.f.dtype, device=self.f.device)
        # b=0 (Ruhe) trägt nicht bei (c_0 = 0)
        for b in range(1, 9):
            cxb = _LBM_CX_LIST[b]
            cyb = _LBM_CY_LIST[b]
            # M(x + c_b): shift M um (-c_b) → an Position x steht M(x+c_b)
            M_shift = torch.roll(M, shifts=(-cxb, -cyb), dims=(0, 1))
            w = one_minus_M * M_shift
            contrib = (self.f[b] * w).sum()
            Fx = Fx + 2.0 * cxb * contrib
            Fy = Fy + 2.0 * cyb * contrib

        return torch.stack([Fx, Fy])


# --- Thermal LBM: Double-Distribution D2Q9 (Velocity) + D2Q5 (Temperatur) ---
#
# Ansatz: Passive-scalar Temperaturverteilung g_i mit eigener Relaxationszeit
# tau_T, gekoppelt an die Geschwindigkeit durch Boussinesq-Buoyancy
# F_buoy = ρ·g·β·(T - T_ref) entlang -y. Das ist das Standard-Modell für
# Rayleigh-Bénard-Konvektion, Wärmetauscher und natürliche Konvektion.

_LBM_T_CX_LIST = [0, 1, 0, -1, 0]              # D2Q5: rest, ost, nord, west, süd
_LBM_T_CY_LIST = [0, 0, 1, 0, -1]
_LBM_T_W_LIST = [1.0/3, 1.0/6, 1.0/6, 1.0/6, 1.0/6]
_LBM_T_OPPOSITE = [0, 3, 4, 1, 2]


def _thermal_constants(device, dtype):
    cx = torch.tensor(_LBM_T_CX_LIST, dtype=dtype, device=device).view(5, 1, 1)
    cy = torch.tensor(_LBM_T_CY_LIST, dtype=dtype, device=device).view(5, 1, 1)
    w = torch.tensor(_LBM_T_W_LIST, dtype=dtype, device=device).view(5, 1, 1)
    return cx, cy, w


def _thermal_equilibrium(T, ux, uy, device, dtype):
    """D2Q5-Gleichgewicht für passiven Skalar T."""
    cx, cy, w = _thermal_constants(device, dtype)
    c_dot_u = cx * ux + cy * uy
    return w * T * (1.0 + 3.0 * c_dot_u)


class ThermalLbmSimulation:
    """Double-Distribution Thermal LBM mit Boussinesq-Buoyancy-Kopplung.

    Geschwindigkeit: D2Q9 BGK (wie LbmSimulation). Temperatur: D2Q5 BGK als
    passiver Skalar. Buoyancy-Force F_y = -ρ·g·β·(T-T_ref) wird im
    Geschwindigkeits-Kollisionsschritt via "Shift-Velocity"-Methode addiert.

    Parameter
    ---------
    nx, ny : int
        Gittergröße in Lattice-Einheiten.
    tau_u, tau_T : float
        Relaxationszeiten für Velocity (kin. Viskosität ν=(τ_u-0.5)/3)
        und Temperatur (Wärmediffusivität α=(τ_T-0.5)/3).
    T_hot, T_cold : float
        Dirichlet-Temperaturen für unten (y=0) bzw. oben (y=ny-1).
    gravity_beta : float
        Produkt g·β (Beschleunigung × thermischer Ausdehnungskoeffizient).
        Steuert die Rayleigh-Zahl Ra = g·β·ΔT·H³/(ν·α).
    obstacle_mask : (nx, ny)-Tensor oder None
        Soft-Mask für Hindernisse (z.B. beheizte Zylinder). Wird per
        Bounce-Back behandelt.
    T_obstacle : float oder None
        Falls gesetzt, wird die Temperatur in Hindernis-Zellen darauf
        festgehalten (Dirichlet-BC am Hindernis).
    """

    def __init__(self, nx, ny, tau_u, tau_T, T_hot=1.0, T_cold=0.0,
                 gravity_beta=0.001, obstacle_mask=None, T_obstacle=None,
                 T_init=None, inlet_u=0.0, T_inlet=None, bc_x="periodic", bc_y="walls"):
        self.nx = int(nx)
        self.ny = int(ny)
        if tau_u <= 0.5:
            raise ValueError(
                f"ThermalLbmSimulation: tau_u={tau_u} ≤ 0.5 ist instabil. "
                "Wähle tau_u > 0.5."
            )
        if tau_T <= 0.5:
            raise ValueError(
                f"ThermalLbmSimulation: tau_T={tau_T} ≤ 0.5 ist instabil. "
                "Wähle tau_T > 0.5."
            )
        self.tau_u = float(tau_u)
        self.tau_T = float(tau_T)
        self.T_hot = float(T_hot)
        self.T_cold = float(T_cold)
        self.T_ref = 0.5 * (self.T_hot + self.T_cold)
        self.gravity_beta = _to_double_tensor(gravity_beta)
        
        if T_obstacle is None or float(T_obstacle) < 0.0:
            self.T_obstacle = None
        else:
            self.T_obstacle = float(T_obstacle)

        self.inlet_u = float(inlet_u)
        self.T_inlet = float(T_inlet) if T_inlet is not None else None
        self.bc_x = str(bc_x).lower()
        self.bc_y = str(bc_y).lower()

        if obstacle_mask is None:
            self.obstacle_mask = torch.zeros((self.nx, self.ny), dtype=torch.float64)
        else:
            mask = _to_double_tensor(obstacle_mask)
            if mask.numel() == 1 and float(mask.flatten()[0]) == 0.0:
                self.obstacle_mask = torch.zeros(
                    (self.nx, self.ny), dtype=torch.float64
                )
            else:
                self.obstacle_mask = mask
                if self.obstacle_mask.shape != (self.nx, self.ny):
                    raise ValueError(
                        f"ThermalLbmSimulation: obstacle_mask Shape {tuple(mask.shape)} "
                        f"passt nicht zu ({self.nx},{self.ny})."
                    )

        device = self.obstacle_mask.device
        dtype = torch.float64

        # Initiale Temperatur: linearer Verlauf T_hot (unten) → T_cold (oben)
        if T_init is None:
            j = torch.arange(self.ny, dtype=dtype, device=device)
            T_profile = self.T_hot + (self.T_cold - self.T_hot) * j / _builtin_max(self.ny - 1, 1)
            self.T = T_profile.view(1, self.ny).expand(self.nx, self.ny).contiguous()
        else:
            self.T = _to_double_tensor(T_init).clone()

        # Initiale Geschwindigkeit = 0, Dichte = 1
        rho_init = torch.ones((self.nx, self.ny), dtype=dtype, device=device)
        u_init = torch.zeros((2, self.nx, self.ny), dtype=dtype, device=device)
        fluid = 1.0 - self.obstacle_mask
        if self.bc_x == "inlet_outlet":
            u_init[0] = self.inlet_u * fluid

        # Equilibria
        self.f = lbm_equilibrium(rho_init, u_init, device)
        self.g = _thermal_equilibrium(self.T, u_init[0], u_init[1], device, dtype)

        # Cached constants
        self._cx_u, self._cy_u, self._w_u = _lbm_constants(device, dtype)
        self._cx_t, self._cy_t, self._w_t = _thermal_constants(device, dtype)
        self._device = device
        self._dtype = dtype

    def step(self, inlet_u=None):
        """Ein Thermal-LBM-Zeitschritt: Stream → BCs (x) → Macroscopic → Buoyancy → Collide → BCs (y) → Bounce."""
        nx, ny = self.nx, self.ny
        device, dtype = self._device, self._dtype

        if inlet_u is None:
            inlet_u_val = self.inlet_u
        else:
            inlet_u_val = float(inlet_u) if not isinstance(inlet_u, torch.Tensor) else inlet_u

        # 1. Streaming für f (D2Q9) und g (D2Q5)
        f_streamed = torch.stack([
            torch.roll(self.f[i],
                       shifts=(int(_LBM_CX_LIST[i]), int(_LBM_CY_LIST[i])),
                       dims=(0, 1))
            for i in range(9)
        ])
        g_streamed = torch.stack([
            torch.roll(self.g[i],
                       shifts=(int(_LBM_T_CX_LIST[i]), int(_LBM_T_CY_LIST[i])),
                       dims=(0, 1))
            for i in range(5)
        ])

        # 2. Inlet/Outlet-BC in x (falls bc_x == "inlet_outlet")
        if self.bc_x == "inlet_outlet":
            # Velocity inlet/outlet
            rho_col = torch.ones((ny,), dtype=dtype, device=device)
            u_col = torch.zeros((2, ny), dtype=dtype, device=device)
            if isinstance(inlet_u_val, torch.Tensor):
                u_col = torch.stack([
                    torch.ones((ny,), dtype=dtype, device=device) * inlet_u_val,
                    torch.zeros((ny,), dtype=dtype, device=device),
                ])
            else:
                u_col[0] = inlet_u_val
            feq_inlet = lbm_equilibrium(
                rho_col.unsqueeze(0), u_col.unsqueeze(1), device
            )  # (9, 1, ny)
            f_inlet = torch.cat([feq_inlet, f_streamed[:, 1:, :]], dim=1)
            f_bc = torch.cat([f_inlet[:, :-1, :], f_inlet[:, -2:-1, :]], dim=1)

            # Temperature inlet/outlet
            T_inlet_val = self.T_inlet if self.T_inlet is not None else self.T_hot
            T_col = torch.full((ny,), T_inlet_val, dtype=dtype, device=device)
            geq_inlet = _thermal_equilibrium(
                T_col.unsqueeze(0),
                u_col[0].unsqueeze(0),
                u_col[1].unsqueeze(0),
                device,
                dtype
            )  # (5, 1, ny)
            g_inlet = torch.cat([geq_inlet, g_streamed[:, 1:, :]], dim=1)
            g_bc = torch.cat([g_inlet[:, :-1, :], g_inlet[:, -2:-1, :]], dim=1)
        else:
            f_bc = f_streamed
            g_bc = g_streamed

        # 3. Makroskopische Felder
        rho = torch.sum(f_bc, dim=0)
        rho_safe = rho + 1e-15
        ux = torch.sum(f_bc * self._cx_u, dim=0) / rho_safe
        uy = torch.sum(f_bc * self._cy_u, dim=0) / rho_safe
        T = torch.sum(g_bc, dim=0)

        # 4. Boussinesq-Buoyancy mit Shift-Velocity-Forcing.
        F_y_per_rho = -self.gravity_beta * (T - self.T_ref)
        ux_eq = ux
        uy_eq = uy + self.tau_u * F_y_per_rho

        # 5. Collision: f (BGK mit shifted velocity), g (BGK passiv mit echter Geschwindigkeit)
        feq = lbm_equilibrium(rho, torch.stack([ux_eq, uy_eq]), device)
        f_coll = f_bc - (1.0 / self.tau_u) * (f_bc - feq)

        geq = _thermal_equilibrium(T, ux, uy, device, dtype)
        g_coll = g_bc - (1.0 / self.tau_T) * (g_bc - geq)

        # 6. Boundary Conditions in y:
        wall_bottom = torch.zeros((nx, ny), dtype=dtype, device=device)
        wall_top = torch.zeros((nx, ny), dtype=dtype, device=device)
        wall_bottom[:, 0] = 1.0
        wall_top[:, -1] = 1.0
        wall_mask = wall_bottom + wall_top   # 1 an Top und Bottom

        # Velocity Bounce-Back an Wänden (immer)
        f_bounce = f_coll[_LBM_OPPOSITE]
        f_coll = (1.0 - wall_mask.unsqueeze(0)) * f_coll + wall_mask.unsqueeze(0) * f_bounce

        if self.bc_y == "insulated":
            # Adiabatic Neumann boundary condition: swap g2 and g4 at y=0 and y=ny-1 post-collision
            g_coll_new = g_coll.clone()
            g_coll_new[2, :, 0] = g_coll[4, :, 0]
            g_coll_new[4, :, 0] = g_coll[2, :, 0]
            g_coll_new[4, :, ny - 1] = g_coll[2, :, ny - 1]
            g_coll_new[2, :, ny - 1] = g_coll[4, :, ny - 1]
            g_coll = g_coll_new
        else:
            # Temperatur-Dirichlet: an Wand-Cells g neu setzen aus Equilibrium mit T_wall
            T_wall = T.clone()
            T_wall = torch.where(wall_bottom.bool(),
                                 torch.full_like(T_wall, self.T_hot), T_wall)
            T_wall = torch.where(wall_top.bool(),
                                 torch.full_like(T_wall, self.T_cold), T_wall)
            u_wall = torch.zeros((2, nx, ny), dtype=dtype, device=device)
            g_wall_eq = _thermal_equilibrium(T_wall, u_wall[0], u_wall[1], device, dtype)
            wall_mask_t = wall_mask.unsqueeze(0)
            g_coll = (1.0 - wall_mask_t) * g_coll + wall_mask_t * g_wall_eq

        # 7. Hindernis-Bounce-Back (Soft-Mask) für f
        M = self.obstacle_mask.unsqueeze(0)
        f_bounce_obs = f_coll[_LBM_OPPOSITE]
        f_coll = (1.0 - M) * f_coll + M * f_bounce_obs

        # 8. Falls T_obstacle gesetzt: Temperatur in Hindernis-Zellen festhalten
        if self.T_obstacle is not None:
            T_obs_field = torch.full_like(T, self.T_obstacle)
            u_zero = torch.zeros((2, nx, ny), dtype=dtype, device=device)
            g_obs_eq = _thermal_equilibrium(T_obs_field, u_zero[0], u_zero[1], device, dtype)
            M_t = self.obstacle_mask.unsqueeze(0)
            g_coll = (1.0 - M_t) * g_coll + M_t * g_obs_eq

        self.f = f_coll
        self.g = g_coll
        self.T = torch.sum(self.g, dim=0)

    def run(self, steps, inlet_u=None):
        for _ in range(int(steps)):
            self.step(inlet_u)

    def get_temperature(self):
        return torch.sum(self.g, dim=0)

    def get_velocity(self):
        rho = torch.sum(self.f, dim=0)
        rho_safe = rho + 1e-15
        ux = torch.sum(self.f * self._cx_u, dim=0) / rho_safe
        uy = torch.sum(self.f * self._cy_u, dim=0) / rho_safe
        return torch.stack([ux, uy])

    def nusselt_number(self):
        """Globale Nusselt-Zahl: Nu = 1 + <u_y·T'>·H / (α·ΔT)
        — Standard-Definition für Rayleigh-Bénard. Misst Konvektions- vs.
        Konduktions-Wärmetransport. Nu=1 → reine Konduktion."""
        T = self.get_temperature()
        u = self.get_velocity()
        uy = u[1]
        delta_T = self.T_hot - self.T_cold
        if abs(delta_T) < 1e-12:
            return torch.tensor(1.0, dtype=self._dtype, device=self._device)
        alpha = (self.tau_T - 0.5) / 3.0    # thermische Diffusivität (Lattice)
        H = float(self.ny - 1)
        # Nu = 1 + H/(α·ΔT) · <u_y·(T - T_ref)>
        return 1.0 + H / (alpha * delta_T) * torch.mean(uy * (T - self.T_ref))


# --- Runtime-Brücke zum Dedekind-Compiler ------------------------------

def lbm_simulation_impl(nx, ny, tau, obstacle_mask=None, inlet_u=0.0,
                        collision="bgk", bounce_back="soft", smagorinsky_constant=0.0):
    return LbmSimulation(nx, ny, tau, obstacle_mask, inlet_u,
                         collision=collision, bounce_back=bounce_back,
                         smagorinsky_constant=smagorinsky_constant)


def simulation_step_impl(sim, inlet_u=None):
    # Wenn inlet_u explizit übergeben wird, nutze ihn; sonst Default der Simulation
    if inlet_u is None or (isinstance(inlet_u, (int, float)) and inlet_u == 0.0 and sim._inlet_u != 0.0):
        # Spezialfall: alte Signatur ohne Default-Verhalten — wir nutzen 0.0 nur wenn
        # die Simulation auch mit 0 initialisiert war
        sim.step(inlet_u)
    else:
        sim.step(inlet_u)
    return sim


def simulation_run_impl(sim, steps, inlet_u=None):
    sim.run(steps, inlet_u)
    return sim


# --- Thermal LBM Bridge ------------------------------------------------

def lbm_thermal_simulation_impl(nx, ny, tau_u, tau_T, T_hot=1.0, T_cold=0.0,
                                 gravity_beta=0.001, obstacle_mask=None,
                                 T_obstacle=None, inlet_u=0.0, T_inlet=None,
                                 bc_x="periodic", bc_y="walls"):
    """Erzeugt eine ThermalLbmSimulation für gekoppelte Strömung+Wärme.

    Beispiele:
      Rayleigh-Bénard: heiß unten / kalt oben, Ra = g·β·ΔT·H³/(ν·α)
      Wärmetauscher:   beheizter Zylinder als Hindernis im Kanal
    """
    return ThermalLbmSimulation(nx, ny, tau_u, tau_T, T_hot, T_cold,
                                 gravity_beta, obstacle_mask, T_obstacle,
                                 inlet_u=inlet_u, T_inlet=T_inlet,
                                 bc_x=bc_x, bc_y=bc_y)


def thermal_set_temperature_impl(sim, T_field):
    """Setzt die Anfangs-Temperaturverteilung manuell (re-initialisiert g aus
    Equilibrium). Wird typischerweise verwendet, um eine kleine Perturbation
    zur Symmetriebrechung in das Rayleigh-Bénard-Startfeld einzubringen."""
    T_t = _to_double_tensor(T_field).clone()
    if T_t.shape != (sim.nx, sim.ny):
        raise ValueError(
            f"thermal_set_temperature: Form {tuple(T_t.shape)} passt nicht zu "
            f"({sim.nx}, {sim.ny})."
        )
    sim.T = T_t
    u_zero = torch.zeros((2, sim.nx, sim.ny), dtype=sim._dtype, device=sim._device)
    sim.g = _thermal_equilibrium(T_t, u_zero[0], u_zero[1],
                                  sim._device, sim._dtype)
    return sim


def thermal_step_impl(sim, inlet_u=None):
    sim.step(inlet_u)
    return sim


def thermal_run_impl(sim, steps, inlet_u=None):
    sim.run(steps, inlet_u)
    return sim


def thermal_get_temperature_impl(sim):
    return sim.get_temperature()


def thermal_get_velocity_impl(sim):
    return sim.get_velocity()


def thermal_nusselt_impl(sim):
    return sim.nusselt_number()


def thermal_rayleigh_impl(tau_u, tau_T, delta_T, gravity_beta, height):
    """Berechnet Rayleigh-Zahl: Ra = g·β·ΔT·H³/(ν·α). Validierungs-Helfer.
    Bei Ra > Ra_c ≈ 1708 setzt Konvektion ein (kritische Rayleigh-Zahl)."""
    nu = (float(tau_u) - 0.5) / 3.0
    alpha = (float(tau_T) - 0.5) / 3.0
    return float(gravity_beta) * float(delta_T) * float(height) ** 3 / (nu * alpha)


def simulation_get_velocity_impl(sim):
    return sim.get_velocity()


def simulation_get_density_impl(sim):
    return sim.get_density()


def simulation_get_pressure_impl(sim):
    return sim.get_pressure()


def simulation_get_drag_lift_impl(sim, target_mask=None):
    return sim.get_drag_lift(target_mask)


def simulation_set_obstacle_impl(sim, mask):
    sim.set_obstacle_mask(mask)
    return sim


# --- Reynolds & Einheiten-Helper ---------------------------------------

def lbm_reynolds_impl(u_lattice, l_lattice, tau):
    """Reynolds-Zahl aus Lattice-Größen: Re = u*L / nu, nu = (tau-0.5)/3."""
    tau_val = float(tau) if not isinstance(tau, torch.Tensor) else float(tau.item())
    if tau_val <= 0.5:
        raise ValueError(f"lbm_reynolds: tau={tau_val} muss > 0.5 sein.")
    nu = (tau_val - 0.5) / 3.0
    u_val = float(u_lattice) if not isinstance(u_lattice, torch.Tensor) else float(u_lattice.item())
    l_val = float(l_lattice) if not isinstance(l_lattice, torch.Tensor) else float(l_lattice.item())
    return u_val * l_val / nu


def lbm_tau_from_reynolds_impl(re, u_lattice, l_lattice):
    """Inverse: gegebene Re, u, L → tau für stabile BGK-Simulation."""
    re_v = float(re) if not isinstance(re, torch.Tensor) else float(re.item())
    u_v = float(u_lattice) if not isinstance(u_lattice, torch.Tensor) else float(u_lattice.item())
    l_v = float(l_lattice) if not isinstance(l_lattice, torch.Tensor) else float(l_lattice.item())
    if re_v <= 0:
        raise ValueError("lbm_tau_from_reynolds: Re muss > 0 sein.")
    nu = u_v * l_v / re_v
    tau = 3.0 * nu + 0.5
    if tau <= 0.5:
        raise ValueError(
            f"lbm_tau_from_reynolds: berechnetes tau={tau:.4f} <= 0.5 (instabil); "
            f"erhöhe Re oder reduziere u_lattice/l_lattice."
        )
    return tau


# --- Soft-Mask-Geometrien (differenzierbare Indikator-Funktionen) ------

def lbm_soft_cylinder_mask_impl(nx, ny, cx, cy, r, alpha=1.0):
    nx_val = int(nx)
    ny_val = int(ny)
    r_t = _to_double_tensor(r)
    cx_t = _to_double_tensor(cx)
    cy_t = _to_double_tensor(cy)
    alpha_t = _to_double_tensor(alpha)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(ny_val, dtype=torch.float64),
        torch.arange(nx_val, dtype=torch.float64),
        indexing='ij',
    )
    x_coords = x_coords.t()
    y_coords = y_coords.t()

    dist_sq = (x_coords - cx_t) ** 2 + (y_coords - cy_t) ** 2
    mask = torch.sigmoid(-(dist_sq - r_t ** 2) / alpha_t)
    return mask


def lbm_soft_ellipse_mask_impl(nx, ny, cx, cy, a, b, alpha=0.1):
    """
    Differenzierbare Soft-Ellipse-Maske. Halbachsen a (x-Richtung), b (y-Richtung).
    Voll differenzierbar bezüglich cx, cy, a, b. Ideal für Shape-Optimierung:
    bei festem Volumen (a*b = const) den Drag durch Variation von a/b minimieren.
    """
    nx_val = int(nx)
    ny_val = int(ny)
    cx_t = _to_double_tensor(cx)
    cy_t = _to_double_tensor(cy)
    a_t = _to_double_tensor(a)
    b_t = _to_double_tensor(b)
    alpha_t = _to_double_tensor(alpha)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(ny_val, dtype=torch.float64),
        torch.arange(nx_val, dtype=torch.float64),
        indexing='ij',
    )
    x_coords = x_coords.t()
    y_coords = y_coords.t()

    # Normalisierte Ellipsendistanz² (=1 am Rand)
    d_norm_sq = ((x_coords - cx_t) / a_t) ** 2 + ((y_coords - cy_t) / b_t) ** 2
    # Sigmoid um Ellipsenrand: ~1 innen, ~0 außen, weicher Übergang
    mask = torch.sigmoid(-(d_norm_sq - 1.0) / alpha_t)
    return mask


def lbm_fourier_shape_mask_impl(nx, ny, cx, cy, r0, a_coeffs, b_coeffs, alpha=0.5):
    """
    Differenzierbare Soft-Maske mit Fourier-parametrisiertem Rand:

        r(θ) = r0 · (1 + Σ_{k=1..K} a_k·cos(k·θ) + b_k·sin(k·θ))

    Erlaubt komplexe 2D-Topologien (Tragflächenprofile, Tropfen, Rotorblätter)
    mit K freien harmonischen Modi pro Cos/Sin. Voll differenzierbar bezüglich
    cx, cy, r0 und jedem Element von a_coeffs, b_coeffs. K kann zwischen
    a_coeffs und b_coeffs unterschiedlich sein — fehlende Koeffizienten
    werden mit 0 aufgefüllt.

    Parameter
    ---------
    nx, ny : int
        Gittergröße in Lattice-Einheiten.
    cx, cy : float / Tensor
        Form-Zentrum (Lattice-Koordinaten).
    r0 : float / Tensor
        Basisradius (Lattice-Einheiten).
    a_coeffs, b_coeffs : 1D-Tensor oder Liste
        Cosinus- bzw. Sinus-Fourier-Koeffizienten. Index 0 entspricht dem
        ersten Harmonischen (k=1).
    alpha : float
        Glättungsbreite des Sigmoid-Übergangs am Rand.
    """
    nx_val = int(nx)
    ny_val = int(ny)
    cx_t = _to_double_tensor(cx)
    cy_t = _to_double_tensor(cy)
    r0_t = _to_double_tensor(r0)
    alpha_t = _to_double_tensor(alpha)
    a_t = _to_double_tensor(a_coeffs).flatten()
    b_t = _to_double_tensor(b_coeffs).flatten()

    y_coords, x_coords = torch.meshgrid(
        torch.arange(ny_val, dtype=torch.float64),
        torch.arange(nx_val, dtype=torch.float64),
        indexing='ij',
    )
    x_coords = x_coords.t()
    y_coords = y_coords.t()

    dx_grid = x_coords - cx_t
    dy_grid = y_coords - cy_t
    # Numerisch stabiler Radius (vermeidet sqrt(0) für Gradient bei Zentrum)
    rho = torch.sqrt(dx_grid * dx_grid + dy_grid * dy_grid + 1e-12)
    theta = torch.atan2(dy_grid, dx_grid)

    # Harmonische Summe; gilt für a_t und b_t unabhängig (gleiche oder
    # unterschiedliche Längen)
    harmonics = torch.ones_like(rho)  # k=0-Konstante = 1 (eingebaut)
    K_a = a_t.numel()
    K_b = b_t.numel()
    for k in range(1, _builtin_max(K_a, K_b) + 1):
        if k - 1 < K_a:
            harmonics = harmonics + a_t[k - 1] * torch.cos(k * theta)
        if k - 1 < K_b:
            harmonics = harmonics + b_t[k - 1] * torch.sin(k * theta)

    r_border = r0_t * harmonics
    # Soft Mask: 1 wenn rho < r_border, 0 sonst, glatter Übergang
    mask = torch.sigmoid(-(rho - r_border) / alpha_t)
    return mask


def lbm_soft_airfoil_mask_impl(nx, ny, t, c, beta, x_start, x_end, y_center, alpha=1.0):
    nx_val = int(nx)
    ny_val = int(ny)
    t_t = _to_double_tensor(t)
    c_t = _to_double_tensor(c)
    beta_t = _to_double_tensor(beta)
    xs_t = _to_double_tensor(x_start)
    xe_t = _to_double_tensor(x_end)
    yc_t = _to_double_tensor(y_center)
    alpha_t = _to_double_tensor(alpha)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(ny_val, dtype=torch.float64),
        torch.arange(nx_val, dtype=torch.float64),
        indexing='ij',
    )
    x_coords = x_coords.t()
    y_coords = y_coords.t()

    L = xe_t - xs_t
    x_norm = (x_coords - xs_t) / L
    x_norm_clamped = torch.clamp(x_norm, 0.0, 1.0)

    # NACA-artige Dickenverteilung y_t(x)
    y_t = (
        t_t
        * torch.sqrt(x_norm_clamped)
        * (1.0 - x_norm_clamped)
        * (1.0 + beta_t * x_norm_clamped)
        * L
    )
    # Wölbungslinie y_c(x)
    y_c = yc_t + 4.0 * c_t * x_norm_clamped * (1.0 - x_norm_clamped) * L

    d_v = torch.abs(y_coords - y_c) - y_t
    mask_inside = torch.sigmoid(-d_v / alpha_t)
    mask_x_start = torch.sigmoid((x_coords - xs_t) / alpha_t)
    mask_x_end = torch.sigmoid((xe_t - x_coords) / alpha_t)
    return mask_inside * mask_x_start * mask_x_end


def add_wind_tunnel_walls_impl(mask):
    m = _to_double_tensor(mask).clone()
    # Top/Bottom-Wände als feste Hindernisse (in Soft-Mask-Form: M=1)
    m[:, 0] = 1.0
    m[:, -1] = 1.0
    return m


# --- Karman-Vortex-Benchmark -------------------------------------------

def lbm_karman_benchmark_impl(re=100.0, nx=240, ny=80, n_steps=4000, warmup_frac=0.4):
    """Karman-Vortex-Benchmark: Zylinder in Kanal, Re=u*D/nu.

    Misst Strouhal-Zahl St = f*D/u und mittleren Drag-Koeffizienten C_D.
    Literaturwerte bei Re=100: St ≈ 0.16-0.17, C_D ≈ 1.3-1.4.

    Parameter
    ---------
    re : float
        Reynolds-Zahl basierend auf Zylinderdurchmesser D und Inlet u.
    nx, ny : int
        Domänen-Größe in Lattice-Einheiten.
    n_steps : int
        Gesamte Zeitschritte. Die erste warmup_frac wird für die
        Mittelwertbildung verworfen.
    warmup_frac : float
        Anteil der Schritte, die vor der Auswertung verworfen werden (0..1).

    Rückgabe
    --------
    dict mit Keys:
        "re", "nx", "ny", "n_steps",
        "D" (Lattice-Durchmesser), "u_inlet",
        "tau", "nu_lattice",
        "C_D_mean", "C_L_amp",
        "strouhal", "shedding_period_steps",
        "lift_history" (Tensor)  — zur Plot-Validierung
    """
    re_v = float(re)
    nx_v = int(nx)
    ny_v = int(ny)
    n_steps_v = int(n_steps)

    # Geometrie: Zylinder mit D ≈ ny/8 etwas links vom Eingang
    D = _builtin_max(8.0, ny_v / 8.0)
    cx_pos = nx_v * 0.20
    # 1.5 Lattice-Knoten von der Mitte verschoben → erzwingt Symmetriebruch und
    # beschleunigt das Einschwingen der Karman-Wirbelstraße um ~10x
    cy_pos = ny_v * 0.5 + 1.5
    u_inlet = 0.05  # konservativ unter Mach-Limit
    tau = lbm_tau_from_reynolds_impl(re_v, u_inlet, D)
    nu_lattice = (tau - 0.5) / 3.0

    with torch.no_grad():
        cylinder_mask = lbm_soft_cylinder_mask_impl(nx_v, ny_v, cx_pos, cy_pos, D / 2.0, 0.6)
        # Kanal-Wände oben/unten brechen die periodische y-Wraparound-Symmetrie
        full_mask = add_wind_tunnel_walls_impl(cylinder_mask)
        sim = LbmSimulation(nx_v, ny_v, tau, full_mask, inlet_u=u_inlet)

        warmup = int(warmup_frac * n_steps_v)
        lift_history = torch.zeros(n_steps_v, dtype=torch.float64)
        drag_history = torch.zeros(n_steps_v, dtype=torch.float64)
        for step in range(n_steps_v):
            sim.step()
            # Drag/Lift nur auf den Zylinder, nicht auf die Kanalwände
            fxy = sim.get_drag_lift(target_mask=cylinder_mask)
            drag_history[step] = fxy[0]
            lift_history[step] = fxy[1]

    # Auswertung nur auf Post-Warmup-Daten
    lift_post = lift_history[warmup:]
    drag_post = drag_history[warmup:]

    # Strouhal: dominante Frequenz im Lift-Signal via FFT
    sig = lift_post - lift_post.mean()
    n_sig = sig.numel()
    spectrum = torch.fft.rfft(sig)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)
    if power.numel() > 1:
        power[0] = 0.0
        k_peak = int(torch.argmax(power).item())
        if k_peak > 0:
            freq_lattice = k_peak / float(n_sig)  # Zyklen pro Schritt
            shedding_period = 1.0 / freq_lattice
        else:
            freq_lattice = 0.0
            shedding_period = float("inf")
    else:
        freq_lattice = 0.0
        shedding_period = float("inf")

    strouhal = freq_lattice * D / u_inlet

    # Drag-Koeffizient: C_D = F_x / (0.5 * rho * u^2 * D), rho ≈ 1 in Lattice
    rho_ref = 1.0
    C_D_mean = float(drag_post.mean().item()) / (0.5 * rho_ref * u_inlet ** 2 * D)
    C_L_amp = float(lift_post.abs().max().item()) / (0.5 * rho_ref * u_inlet ** 2 * D)

    return {
        "re": re_v,
        "nx": nx_v,
        "ny": ny_v,
        "n_steps": n_steps_v,
        "D": float(D),
        "u_inlet": float(u_inlet),
        "tau": float(tau),
        "nu_lattice": float(nu_lattice),
        "C_D_mean": C_D_mean,
        "C_L_amp": C_L_amp,
        "strouhal": float(strouhal),
        "shedding_period_steps": float(shedding_period),
        "lift_history": lift_history,
    }


# --- Einheiten-bewusste API ---------------------------------------------
#
# Forschende denken in physikalischen Einheiten (m, m/s, m²/s, Pa, N).
# LBM rechnet intern in Lattice-Einheiten.  PhysicalLbmSimulation übersetzt
# zwischen beiden Welten und enthält den Solver `LbmSimulation` als
# Implementierungsdetail.
#
# Standard-Konvertierungen (Krüger et al., "The Lattice Boltzmann Method"):
#   dx           = L_domain / nx           [m / lattice unit]
#   dt           = u_lat * dx / u_phys     [s / lattice step]
#   nu_lattice   = nu_phys * dt / dx^2     [dimensionslos]
#   tau          = 3 * nu_lattice + 0.5    [BGK]
# Output-Umrechnung:
#   u_phys       = u_lat   * (dx / dt)
#   p_phys       = p_lat   * rho_phys * (dx / dt)^2     [Pa]
#   F_phys (2D)  = F_lat   * rho_phys * dx^3 / dt^2     [N/m, force per depth]

def _lbm_q_value_unit(x):
    """Liefert (value, unit_string) für Quantity oder (float(x), '') für rohe Zahl."""
    if isinstance(x, Quantity):
        return float(x.value), str(x.unit).strip()
    return float(x), ""


def _lbm_to_si_length(x, name="length"):
    v, u = _lbm_q_value_unit(x)
    if u in ("", "m"):
        return v
    table = {"cm": 0.01, "mm": 0.001, "km": 1000.0, "um": 1e-6, "nm": 1e-9}
    if u in table:
        return v * table[u]
    raise ValueError(
        f"LBM Physical API: Parameter '{name}' braucht Längeneinheit "
        f"(m, cm, mm, km, um, nm), bekam [{u}]."
    )


def _lbm_to_si_velocity(x, name="velocity"):
    v, u = _lbm_q_value_unit(x)
    if u in ("", "m/s"):
        return v
    table = {"km/h": 1.0 / 3.6, "cm/s": 0.01, "mm/s": 0.001}
    if u in table:
        return v * table[u]
    raise ValueError(
        f"LBM Physical API: Parameter '{name}' braucht Geschwindigkeit "
        f"(m/s, km/h, cm/s, mm/s), bekam [{u}]."
    )


def _lbm_to_si_kviscosity(x, name="nu"):
    """Kinematische Viskosität in m²/s. Stokes (St) und Centistokes (cSt) erlaubt."""
    v, u = _lbm_q_value_unit(x)
    if u in ("", "m^2/s", "m**2/s", "m²/s"):
        return v
    table = {
        "cm^2/s": 1e-4, "cm**2/s": 1e-4, "cm²/s": 1e-4,
        "mm^2/s": 1e-6, "mm**2/s": 1e-6, "mm²/s": 1e-6,
        "St": 1e-4, "cSt": 1e-6,
    }
    if u in table:
        return v * table[u]
    raise ValueError(
        f"LBM Physical API: Parameter '{name}' braucht kinematische Viskosität "
        f"(m^2/s, mm^2/s, cSt, St), bekam [{u}]."
    )


def _lbm_to_si_density(x, name="rho"):
    v, u = _lbm_q_value_unit(x)
    if u in ("", "kg/m^3", "kg/m**3", "kg/m³"):
        return v
    table = {
        "g/cm^3": 1000.0, "g/cm**3": 1000.0, "g/cm³": 1000.0,
        "g/L": 1.0, "g/l": 1.0, "kg/L": 1000.0,
    }
    if u in table:
        return v * table[u]
    raise ValueError(
        f"LBM Physical API: Parameter '{name}' braucht Dichte "
        f"(kg/m^3, g/cm^3), bekam [{u}]."
    )


def _lbm_to_si_time(x, name="time"):
    v, u = _lbm_q_value_unit(x)
    if u in ("", "s"):
        return v
    table = {"ms": 1e-3, "us": 1e-6, "ns": 1e-9,
             "min": 60.0, "h": 3600.0}
    if u in table:
        return v * table[u]
    raise ValueError(
        f"LBM Physical API: Parameter '{name}' braucht Zeit "
        f"(s, ms, min, h), bekam [{u}]."
    )


class PhysicalLbmSimulation:
    """Einheiten-bewusster Wrapper um LbmSimulation.

    Eingabe in SI-Einheiten (m, m/s, m²/s, kg/m³, s), Ausgabe als Quantity
    in physikalischen Einheiten. Interne Lattice-Skalierung wird automatisch
    so gewählt, dass die Mach-Bedingung (u_lattice ≈ 0.05, Default) erfüllt
    ist und tau im stabilen BGK-Bereich liegt.

    Domänen-Geometrie: Rechteckiger Kanal, x = Strömungsrichtung, y = quer.
    Auflösung wird über nx in x vorgegeben; ny folgt aus dem Seitenverhältnis,
    oder kann explizit gesetzt werden.

    Force-Output ist 2D, also pro Tiefeneinheit (Einheit: N/m).
    """

    # Konservative Ziel-Lattice-Geschwindigkeit (u_max ≤ 0.1 für niedrige Mach)
    DEFAULT_U_LATTICE = 0.05

    def __init__(self, domain_x, domain_y, nx, inlet_u, nu,
                 rho=1.225, ny=None, u_lattice_target=None,
                 collision="bgk", bounce_back="soft", smagorinsky_constant=0.0):
        self.collision_model = str(collision).lower()
        self.bounce_back_model = str(bounce_back).lower()
        self.smagorinsky_constant = smagorinsky_constant
        self.L_x = _lbm_to_si_length(domain_x, "domain_x")
        self.L_y = _lbm_to_si_length(domain_y, "domain_y")
        self.U_in = _lbm_to_si_velocity(inlet_u, "inlet_u")
        self.nu_phys = _lbm_to_si_kviscosity(nu, "nu")
        self.rho_phys = _lbm_to_si_density(rho, "rho")

        if self.L_x <= 0 or self.L_y <= 0:
            raise ValueError("PhysicalLbmSimulation: domain_x/domain_y müssen > 0 sein.")
        if self.U_in <= 0:
            raise ValueError("PhysicalLbmSimulation: inlet_u muss > 0 sein.")
        if self.nu_phys <= 0:
            raise ValueError("PhysicalLbmSimulation: nu muss > 0 sein.")

        self.nx = int(nx)
        if self.nx < 8:
            raise ValueError(f"PhysicalLbmSimulation: nx={self.nx} zu klein (min 8).")

        # dx aus Domänenlänge und Auflösung
        self.dx = self.L_x / self.nx  # [m / lattice unit]
        # ny aus Seitenverhältnis (oder explizit)
        if ny is None:
            self.ny = _builtin_max(8, int(round(self.L_y / self.dx)))
        else:
            self.ny = int(ny)

        # dt aus Mach-Bedingung
        u_lat = float(u_lattice_target) if u_lattice_target is not None else self.DEFAULT_U_LATTICE
        if u_lat <= 0 or u_lat > 0.1:
            raise ValueError(
                f"PhysicalLbmSimulation: u_lattice_target={u_lat} muss in (0, 0.1] liegen "
                f"(Mach-Stabilität)."
            )
        self.u_lattice = u_lat
        self.dt = u_lat * self.dx / self.U_in  # [s / lattice step]

        # Lattice-Viskosität und tau
        self.nu_lattice = self.nu_phys * self.dt / (self.dx * self.dx)
        self.tau = 3.0 * self.nu_lattice + 0.5

        if self.tau <= 0.5:
            # nx zur Erreichung von tau ≥ 0.55 vorschlagen
            # tau = 0.5 + 3*nu*dt/dx^2; dt = u_lat*dx/U; → tau = 0.5 + 3*nu*u_lat/(U*dx)
            # für tau ≥ 0.55: dx ≤ 3*nu*u_lat / (0.05*U)  → nx ≥ L_x / dx
            dx_safe = 3.0 * self.nu_phys * self.u_lattice / (0.05 * self.U_in)
            nx_suggest = _builtin_max(self.nx + 1, int(self.L_x / dx_safe) + 1) if dx_safe > 0 else self.nx * 4
            raise ValueError(
                f"PhysicalLbmSimulation: tau={self.tau:.4f} ≤ 0.5 (BGK instabil). "
                f"Vorschlag: nx ≥ {nx_suggest} (statt {self.nx}); alternativ "
                f"u_lattice_target reduzieren oder nu größer wählen."
            )
        if self.tau < 0.55:
            print(
                f"[PhysicalLbmSimulation] Warnung: tau={self.tau:.4f} im grenzwertigen "
                f"Bereich (<0.55). Numerische Artefakte möglich; höhere Auflösung empfohlen."
            )
        if self.tau > 2.0:
            print(
                f"[PhysicalLbmSimulation] Hinweis: tau={self.tau:.3f} > 2.0 deutet auf "
                f"unnötig grobe Auflösung hin; nx oder u_lattice_target reduzieren."
            )

        # Conversion factors für Output
        self._velocity_factor = self.dx / self.dt           # u_lat → m/s
        self._pressure_factor = self.rho_phys * self._velocity_factor ** 2
        # 2D-Kraft pro Tiefe: rho * dx^3 / dt^2 = rho * dx * (dx/dt)^2
        self._force_factor_2d = self.rho_phys * self.dx * self._velocity_factor ** 2

        # Underlying Solver (BGK/MRT + soft/hard bounce-back werden durchgereicht)
        self.sim = LbmSimulation(
            self.nx, self.ny, self.tau,
            obstacle_mask=None,
            inlet_u=self.u_lattice,
            collision=self.collision_model,
            bounce_back=self.bounce_back_model,
            smagorinsky_constant=self.smagorinsky_constant,
        )

    def reynolds(self, length_char=None):
        """Re = U * L / nu. Default-Länge ist domain_y (Kanalbreite)."""
        L = self.L_y if length_char is None else _lbm_to_si_length(length_char, "length_char")
        return self.U_in * L / self.nu_phys

    def set_cylinder(self, cx, cy, radius, alpha=None):
        """Setzt einen runden Hindernis. cx, cy, radius in physikalischen Längeneinheiten."""
        cx_lat = _lbm_to_si_length(cx, "cx") / self.dx
        cy_lat = _lbm_to_si_length(cy, "cy") / self.dx
        r_lat = _lbm_to_si_length(radius, "radius") / self.dx
        if r_lat < 3:
            raise ValueError(
                f"PhysicalLbmSimulation.set_cylinder: Radius {r_lat:.2f} Lattice-Einheiten "
                f"zu klein für saubere Aufloesung (min ~3). Höhere nx oder größerer Radius."
            )
        alpha_lat = 0.8 if alpha is None else float(alpha)
        mask = lbm_soft_cylinder_mask_impl(self.nx, self.ny, cx_lat, cy_lat, r_lat, alpha_lat)
        self.sim.set_obstacle_mask(mask)
        return self

    def add_walls(self):
        """Schaltet die periodischen y-Wände aus, indem Top/Bottom als Solid markiert werden."""
        new_mask = add_wind_tunnel_walls_impl(self.sim.obstacle_mask)
        self.sim.set_obstacle_mask(new_mask)
        return self

    def step(self):
        self.sim.step()
        return self

    def run_steps(self, n_steps):
        self.sim.run(int(n_steps))
        return self

    def run_time(self, t):
        """Läuft für eine physikalische Zeit t (Quantity in [s] oder rohe Zahl)."""
        t_phys = _lbm_to_si_time(t, "t")
        n_steps = _builtin_max(1, int(round(t_phys / self.dt)))
        self.sim.run(n_steps)
        return self

    # --- Output mit physikalischen Einheiten ---

    def get_velocity(self):
        """Geschwindigkeitsfeld (2, nx, ny) als Quantity in [m/s]."""
        u_lat = self.sim.get_velocity()
        return Quantity(u_lat * self._velocity_factor, "m/s")

    def get_pressure(self):
        """Druckfeld (nx, ny) als Quantity in [Pa]."""
        p_lat = self.sim.get_pressure()
        return Quantity(p_lat * self._pressure_factor, "Pa")

    def get_density(self):
        """Massendichte als Quantity in [kg/m^3]."""
        rho_lat = self.sim.get_density()
        return Quantity(rho_lat * self.rho_phys, "kg/m^3")

    def get_drag_lift(self, target_mask=None):
        """Drag (Fx) und Lift (Fy) auf das Hindernis als Quantity-Tensor in [N/m].

        2D-Force, also pro Tiefeneinheit (Spannweitenlänge). Mit gegebener
        Tiefe einfach multiplizieren: total_force = result * depth[m].
        """
        F_lat = self.sim.get_drag_lift(target_mask)
        return Quantity(F_lat * self._force_factor_2d, "N/m")

    def summary(self):
        """Liefert dict mit Setup-Übersicht für Debug/Logs."""
        return {
            "domain_x_m": self.L_x,
            "domain_y_m": self.L_y,
            "nx": self.nx,
            "ny": self.ny,
            "dx_m": self.dx,
            "dt_s": self.dt,
            "inlet_u_m_per_s": self.U_in,
            "nu_m2_per_s": self.nu_phys,
            "rho_kg_per_m3": self.rho_phys,
            "tau": self.tau,
            "u_lattice": self.u_lattice,
            "nu_lattice": self.nu_lattice,
            "Reynolds_Ly": self.reynolds(),
        }


# --- Runtime-Brücke (von .ddk-Code aus aufrufbar) ---

def lbm_physical_impl(domain_x, domain_y, nx, inlet_u, nu, rho=1.225,
                      ny=None, u_lattice=None, collision="bgk", bounce_back="soft",
                      smagorinsky_constant=0.0):
    # Aus .ddk-Wrappern kommen 0/0.0 als Sentinel für "keine Angabe"
    if isinstance(ny, (int, float)) and ny == 0:
        ny = None
    if isinstance(u_lattice, (int, float)) and u_lattice == 0:
        u_lattice = None
    return PhysicalLbmSimulation(domain_x, domain_y, nx, inlet_u, nu,
                                 rho=rho, ny=ny, u_lattice_target=u_lattice,
                                 collision=collision, bounce_back=bounce_back,
                                 smagorinsky_constant=smagorinsky_constant)


def lbm_physical_set_cylinder_impl(sim, cx, cy, radius):
    sim.set_cylinder(cx, cy, radius)
    return sim


def lbm_physical_add_walls_impl(sim):
    sim.add_walls()
    return sim


def lbm_physical_run_steps_impl(sim, n_steps):
    sim.run_steps(n_steps)
    return sim


def lbm_physical_run_time_impl(sim, t):
    sim.run_time(t)
    return sim


def lbm_physical_velocity_impl(sim):
    return sim.get_velocity()


def lbm_physical_pressure_impl(sim):
    return sim.get_pressure()


def lbm_physical_reynolds_impl(sim, length_char=None):
    # Sentinel: [0.0] / 0 → kein expliziter L_char, Default = domain_y
    if length_char is None:
        return sim.reynolds(None)
    if isinstance(length_char, (int, float)) and length_char == 0:
        return sim.reynolds(None)
    if hasattr(length_char, "numel") and length_char.numel() == 1:
        if float(length_char.flatten()[0]) == 0.0:
            return sim.reynolds(None)
    if isinstance(length_char, list) and len(length_char) == 1 and length_char[0] == 0.0:
        return sim.reynolds(None)
    return sim.reynolds(length_char)


def lbm_physical_drag_lift_impl(sim, target_mask=None):
    return sim.get_drag_lift(target_mask)


def lbm_physical_summary_impl(sim):
    return sim.summary()


# --- D3Q19 Konstanten ----------------------------------------------------
_D3Q19_CX_LIST = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
_D3Q19_CY_LIST = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1]
_D3Q19_CZ_LIST = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1]
_D3Q19_W_LIST = [
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
]
_D3Q19_OPPOSITE = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]


def _lbm3d_constants(device, dtype):
    """Liefert (cx, cy, cz, w) als 19x1x1x1-Tensoren auf dem gewünschten Device/Dtype."""
    cx = torch.tensor(_D3Q19_CX_LIST, dtype=dtype, device=device).view(19, 1, 1, 1)
    cy = torch.tensor(_D3Q19_CY_LIST, dtype=dtype, device=device).view(19, 1, 1, 1)
    cz = torch.tensor(_D3Q19_CZ_LIST, dtype=dtype, device=device).view(19, 1, 1, 1)
    w = torch.tensor(_D3Q19_W_LIST, dtype=dtype, device=device).view(19, 1, 1, 1)
    return cx, cy, cz, w


def lbm3d_equilibrium(rho, u, device=None):
    """Berechnet die D3Q19 Gleichgewichtsverteilung f_eq in 3D.
    rho: shape (nx, ny, nz) oder (1, ny, nz); u: shape (3, nx, ny, nz) oder (3, 1, ny, nz).
    Rückgabe: f_eq mit shape (19, *rho.shape).
    """
    if device is None:
        device = rho.device
    ux = u[0]
    uy = u[1]
    uz = u[2]
    u2 = ux * ux + uy * uy + uz * uz
    cx, cy, cz, w = _lbm3d_constants(device, torch.float64)
    c_dot_u = cx * ux + cy * uy + cz * uz
    feq = w * rho * (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u * c_dot_u - 1.5 * u2)
    return feq


class Lbm3dSimulation:
    """Vollständig differenzierbarer 3D-D3Q19 Lattice-Boltzmann-Solver.
    
    Grid-Größe: (nx, ny, nz).
    tau: Relaxationszeit (> 0.5).
    """
    MAX_LATTICE_U = 0.1

    def __init__(self, nx, ny, nz, tau, obstacle_mask=None, inlet_u=0.0,
                 collision="bgk", bounce_back="soft", smagorinsky_constant=0.0):
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.tau = float(tau)
        self._inlet_u = float(inlet_u)
        self.collision_model = str(collision).lower()
        self.bounce_back_model = str(bounce_back).lower()
        self.smagorinsky_constant = smagorinsky_constant
        
        if self.tau <= 0.5:
            raise ValueError(f"Lbm3dSimulation: tau={self.tau} muss > 0.5 sein.")
            
        if obstacle_mask is not None:
            mask_t = _to_double_tensor(obstacle_mask)
            if mask_t.numel() == 1 and float(mask_t.flatten()[0]) == 0.0:
                self.obstacle_mask = torch.zeros((self.nx, self.ny, self.nz), dtype=torch.float64)
            else:
                self.obstacle_mask = mask_t
                if self.obstacle_mask.shape != (self.nx, self.ny, self.nz):
                    raise ValueError(f"Obstacle mask shape must be ({self.nx}, {self.ny}, {self.nz}), got {tuple(self.obstacle_mask.shape)}.")
        else:
            self.obstacle_mask = torch.zeros((self.nx, self.ny, self.nz), dtype=torch.float64)
            
        device = self.obstacle_mask.device
        dtype = torch.float64
        self._cx, self._cy, self._cz, self._w = _lbm3d_constants(device, dtype)
        
        # Initialisierung
        rho_init = torch.ones((self.nx, self.ny, self.nz), dtype=dtype, device=device)
        u_init = torch.zeros((3, self.nx, self.ny, self.nz), dtype=dtype, device=device)
        u_init[0] = self._inlet_u
        
        self.f = lbm3d_equilibrium(rho_init, u_init, device)

    def set_obstacle_mask(self, mask):
        mask_t = _to_double_tensor(mask)
        if mask_t.shape != (self.nx, self.ny, self.nz):
            raise ValueError(f"Obstacle mask shape must be ({self.nx}, {self.ny}, {self.nz}), got {tuple(mask_t.shape)}")
        self.obstacle_mask = mask_t

    def step(self, inlet_u=None):
        if inlet_u is None:
            inlet_u_val = self._inlet_u
        else:
            inlet_u_val = float(inlet_u) if not isinstance(inlet_u, torch.Tensor) else inlet_u

        # 1. Streaming in 3D
        f_streamed = torch.stack([
            torch.roll(
                self.f[i],
                shifts=(int(_D3Q19_CX_LIST[i]), int(_D3Q19_CY_LIST[i]), int(_D3Q19_CZ_LIST[i])),
                dims=(0, 1, 2),
            )
            for i in range(19)
        ])

        # 2. Inlet (x=0)
        ny = self.ny
        nz = self.nz
        device = self.f.device
        dtype = self.f.dtype
        
        rho_col = torch.ones((ny, nz), dtype=dtype, device=device)
        u_col = torch.zeros((3, ny, nz), dtype=dtype, device=device)
        if isinstance(inlet_u_val, torch.Tensor):
            u_col = torch.stack([
                torch.ones((ny, nz), dtype=dtype, device=device) * inlet_u_val,
                torch.zeros((ny, nz), dtype=dtype, device=device),
                torch.zeros((ny, nz), dtype=dtype, device=device),
            ])
        else:
            u_col[0] = inlet_u_val
            
        feq_inlet = lbm3d_equilibrium(
            rho_col.unsqueeze(0), u_col.unsqueeze(1), device
        )  # (19, 1, ny, nz)
        f_inlet = torch.cat([feq_inlet, f_streamed[:, 1:, :, :]], dim=1)

        # 3. Outlet (x=nx-1)
        f_outlet = torch.cat([f_inlet[:, :-1, :, :], f_inlet[:, -2:-1, :, :]], dim=1)

        # 4. Macroscopic
        rho = torch.sum(f_outlet, dim=0)
        rho_safe = rho + 1e-15
        ux = torch.sum(f_outlet * self._cx, dim=0) / rho_safe
        uy = torch.sum(f_outlet * self._cy, dim=0) / rho_safe
        uz = torch.sum(f_outlet * self._cz, dim=0) / rho_safe

        # 5. Collision (BGK)
        u_stack = torch.stack([ux, uy, uz])
        feq = lbm3d_equilibrium(rho, u_stack, device)
        use_smag = False
        if isinstance(self.smagorinsky_constant, torch.Tensor):
            use_smag = float(self.smagorinsky_constant.detach().item()) > 0.0
        else:
            use_smag = float(self.smagorinsky_constant) > 0.0
            
        if use_smag:
            f_neq = f_outlet - feq
            Q_xx = torch.sum(self._cx**2 * f_neq, dim=0)
            Q_yy = torch.sum(self._cy**2 * f_neq, dim=0)
            Q_zz = torch.sum(self._cz**2 * f_neq, dim=0)
            Q_xy = torch.sum(self._cx * self._cy * f_neq, dim=0)
            Q_xz = torch.sum(self._cx * self._cz * f_neq, dim=0)
            Q_yz = torch.sum(self._cy * self._cz * f_neq, dim=0)
            
            Q_bar = torch.sqrt(Q_xx**2 + Q_yy**2 + Q_zz**2 + 2.0 * (Q_xy**2 + Q_xz**2 + Q_yz**2) + 1e-12)
            temp = 18.0 * math.sqrt(2.0) * (self.smagorinsky_constant**2) * Q_bar / rho_safe
            tau_eff = 0.5 * (self.tau + torch.sqrt(self.tau**2 + temp))
            f_coll = f_outlet - (1.0 / tau_eff.unsqueeze(0)) * f_neq
        else:
            f_coll = f_outlet - (1.0 / self.tau) * (f_outlet - feq)

        # 6. Bounce-Back at obstacles
        f_bounce = f_outlet[_D3Q19_OPPOSITE]
        if self.bounce_back_model == "hard":
            M_sharp = (self.obstacle_mask >= 0.5).to(self.obstacle_mask.dtype)
            M = M_sharp.unsqueeze(0)
        else:
            M = self.obstacle_mask.unsqueeze(0)
            
        self.f = (1.0 - M) * f_coll + M * f_bounce

    def run(self, steps, inlet_u=None):
        for _ in range(int(steps)):
            self.step(inlet_u)

    def get_velocity(self):
        rho = torch.sum(self.f, dim=0)
        rho_safe = rho + 1e-15
        ux = torch.sum(self.f * self._cx, dim=0) / rho_safe
        uy = torch.sum(self.f * self._cy, dim=0) / rho_safe
        uz = torch.sum(self.f * self._cz, dim=0) / rho_safe
        return torch.stack([ux, uy, uz])

    def get_density(self):
        return torch.sum(self.f, dim=0)

    def get_drag_lift(self, target_mask=None):
        """Impulsaustausch-Methode (MEM) in 3D.
        
        Rückgabe: Tensor [Fx, Fy, Fz]
        """
        if target_mask is None:
            M = self.obstacle_mask
        else:
            mask_t = _to_double_tensor(target_mask)
            if mask_t.numel() == 1 and float(mask_t.flatten()[0]) == 0.0:
                M = self.obstacle_mask
            else:
                M = mask_t

        one_minus_M = 1.0 - M
        Fx = torch.zeros((), dtype=self.f.dtype, device=self.f.device)
        Fy = torch.zeros((), dtype=self.f.dtype, device=self.f.device)
        Fz = torch.zeros((), dtype=self.f.dtype, device=self.f.device)
        
        for b in range(1, 19):
            cxb = _D3Q19_CX_LIST[b]
            cyb = _D3Q19_CY_LIST[b]
            czb = _D3Q19_CZ_LIST[b]
            M_shift = torch.roll(M, shifts=(-cxb, -cyb, -czb), dims=(0, 1, 2))
            w = one_minus_M * M_shift
            contrib = (self.f[b] * w).sum()
            Fx = Fx + 2.0 * cxb * contrib
            Fy = Fy + 2.0 * cyb * contrib
            Fz = Fz + 2.0 * czb * contrib

        return torch.stack([Fx, Fy, Fz])


def lbm3d_simulation_impl(nx, ny, nz, tau, obstacle_mask=None, inlet_u=0.0, smagorinsky_constant=0.0):
    return Lbm3dSimulation(nx, ny, nz, tau, obstacle_mask=obstacle_mask, inlet_u=inlet_u,
                           smagorinsky_constant=smagorinsky_constant)


def lbm3d_step_impl(sim, inlet_u=None):
    sim.step(inlet_u)
    return sim


def lbm3d_run_impl(sim, steps, inlet_u=None):
    sim.run(steps, inlet_u)
    return sim


def lbm3d_velocity_impl(sim):
    return sim.get_velocity()


def lbm3d_density_impl(sim):
    return sim.get_density()


def lbm3d_drag_lift_force_impl(sim, target_mask=None):
    if target_mask is None:
        return sim.get_drag_lift(None)
    if hasattr(target_mask, "numel") and target_mask.numel() == 1 and float(target_mask.flatten()[0]) == 0.0:
        return sim.get_drag_lift(None)
    if isinstance(target_mask, list) and len(target_mask) == 1 and target_mask[0] == 0.0:
        return sim.get_drag_lift(None)
    return sim.get_drag_lift(target_mask)


def lbm3d_set_obstacle_impl(sim, mask):
    sim.set_obstacle_mask(mask)
    return sim


def load_stl_triangles(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    is_ascii = False
    if len(data) > 5 and data[:5] == b'solid':
        preview = data[:1000].lower()
        if b'facet' in preview and b'outer' in preview:
            is_ascii = True
            
    triangles = []
    
    if is_ascii:
        lines = data.decode('utf-8', errors='ignore').split('\n')
        curr_tri = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'vertex':
                curr_tri.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(curr_tri) == 3:
                    triangles.append(curr_tri)
                    curr_tri = []
    else:
        if len(data) < 84:
            raise ValueError("Invalid binary STL file: too short.")
        import struct
        num_triangles = struct.unpack('<I', data[80:84])[0]
        expected_size = 84 + num_triangles * 50
        if len(data) < expected_size:
            raise ValueError(f"Invalid binary STL file: size {len(data)} is less than expected {expected_size}.")
        
        offset = 84
        for _ in range(num_triangles):
            tri_data = struct.unpack('<9f', data[offset+12:offset+48])
            v1 = list(tri_data[0:3])
            v2 = list(tri_data[3:6])
            v3 = list(tri_data[6:9])
            triangles.append([v1, v2, v3])
            offset += 50
            
    return torch.tensor(triangles, dtype=torch.float64)


def voxelize_stl_impl(stl_path, nx, ny, nz, scale_to_fit=True, padding=2.0):
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    triangles = load_stl_triangles(stl_path)
    
    if triangles.shape[0] == 0:
        return torch.zeros((nx, ny, nz), dtype=torch.float64)
        
    if scale_to_fit:
        min_bounds = triangles.view(-1, 3).min(dim=0)[0]
        max_bounds = triangles.view(-1, 3).max(dim=0)[0]
        size = max_bounds - min_bounds
        max_size = size.max()
        if max_size < 1e-8:
            max_size = 1.0
            
        target_size_x = nx - 2.0 * padding
        target_size_y = ny - 2.0 * padding
        target_size_z = nz - 2.0 * padding
        
        scale = _builtin_min(target_size_x / size[0].item() if size[0] > 1e-8 else float('inf'),
                    target_size_y / size[1].item() if size[1] > 1e-8 else float('inf'),
                    target_size_z / size[2].item() if size[2] > 1e-8 else float('inf'))
        
        center = 0.5 * (min_bounds + max_bounds)
        grid_center = torch.tensor([nx / 2.0, ny / 2.0, nz / 2.0], dtype=torch.float64)
        
        triangles = (triangles - center.view(1, 1, 3)) * scale + grid_center.view(1, 1, 3)
        
    mask = torch.zeros((nx, ny, nz), dtype=torch.float64)
    M = triangles.shape[0]
    
    for m in range(M):
        tri = triangles[m]
        A, B, C = tri[0], tri[1], tri[2]
        
        AB = B - A
        AC = C - A
        Nx = AB[1]*AC[2] - AB[2]*AC[1]
        Ny = AB[2]*AC[0] - AB[0]*AC[2]
        Nz = AB[0]*AC[1] - AB[1]*AC[0]
        
        if abs(Nz.item()) < 1e-8:
            continue
            
        min_x = int(math.floor(_builtin_min(A[0].item(), B[0].item(), C[0].item())))
        max_x = int(math.ceil(_builtin_max(A[0].item(), B[0].item(), C[0].item())))
        min_y = int(math.floor(_builtin_min(A[1].item(), B[1].item(), C[1].item())))
        max_y = int(math.ceil(_builtin_max(A[1].item(), B[1].item(), C[1].item())))
        
        min_x = _builtin_max(0, min_x)
        max_x = _builtin_min(nx - 1, max_x)
        min_y = _builtin_max(0, min_y)
        max_y = _builtin_min(ny - 1, max_y)
        
        ax, ay, az = A[0].item(), A[1].item(), A[2].item()
        bx, by, bz = B[0].item(), B[1].item(), B[2].item()
        cx, cy, cz = C[0].item(), C[1].item(), C[2].item()
        
        nx_val, ny_val, nz_val = Nx.item(), Ny.item(), Nz.item()
        
        for ix in range(min_x, max_x + 1):
            px = ix + 0.123456789
            for iy in range(min_y, max_y + 1):
                py = iy + 0.456789123
                
                d1 = (px - bx)*(ay - by) - (ax - bx)*(py - by)
                d2 = (px - cx)*(by - cy) - (bx - cx)*(py - cy)
                d3 = (px - ax)*(cy - ay) - (cx - ax)*(py - ay)
                
                has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                
                if not (has_neg and has_pos):
                    z_inter = az - (nx_val * (px - ax) + ny_val * (py - ay)) / nz_val
                    limit_iz = int(math.floor(z_inter - 0.5))
                    limit_iz = _builtin_min(nz - 1, _builtin_max(0, limit_iz))
                    if z_inter > 0.5:
                        mask[ix, iy, 0:limit_iz+1] = 1.0 - mask[ix, iy, 0:limit_iz+1]
                        
    return mask


def lbm3d_soft_sphere_mask_impl(nx, ny, nz, cx, cy, cz, r, alpha=1.0):
    nx_v = int(nx)
    ny_v = int(ny)
    nz_v = int(nz)
    r_val = float(r) if not hasattr(r, "item") else r
    cx_val = float(cx) if not hasattr(cx, "item") else cx
    cy_val = float(cy) if not hasattr(cy, "item") else cy
    cz_val = float(cz) if not hasattr(cz, "item") else cz
    alpha_val = float(alpha)
    
    x = torch.arange(nx_v, dtype=torch.float64).view(nx_v, 1, 1)
    y = torch.arange(ny_v, dtype=torch.float64).view(1, ny_v, 1)
    z = torch.arange(nz_v, dtype=torch.float64).view(1, 1, nz_v)
    
    dist = torch.sqrt((x - cx_val)**2 + (y - cy_val)**2 + (z - cz_val)**2 + 1e-12)
    mask = torch.sigmoid((r_val - dist) / alpha_val)
    return mask


def lbm3d_soft_cylinder_mask_impl(nx, ny, nz, cx, cy, r, axis=0, alpha=1.0):
    nx_v = int(nx)
    ny_v = int(ny)
    nz_v = int(nz)
    r_val = float(r) if not hasattr(r, "item") else r
    cx_val = float(cx) if not hasattr(cx, "item") else cx
    cy_val = float(cy) if not hasattr(cy, "item") else cy
    alpha_val = float(alpha)
    axis_val = int(axis)
    
    x = torch.arange(nx_v, dtype=torch.float64).view(nx_v, 1, 1)
    y = torch.arange(ny_v, dtype=torch.float64).view(1, ny_v, 1)
    z = torch.arange(nz_v, dtype=torch.float64).view(1, 1, nz_v)
    
    if axis_val == 0:
        dist = torch.sqrt((y - cx_val)**2 + (z - cy_val)**2 + 1e-12)
    elif axis_val == 1:
        dist = torch.sqrt((x - cx_val)**2 + (z - cy_val)**2 + 1e-12)
    else:
        dist = torch.sqrt((x - cx_val)**2 + (y - cy_val)**2 + 1e-12)
        
    mask = torch.sigmoid((r_val - dist) / alpha_val)
    return mask

# Dedekind Standard Library: Structural Mechanics & Topology Optimization
import torch
import math
import builtins

class StructuralMesh2D:
    def __init__(self, nx, ny):
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_elem = self.nx * self.ny
        self.n_nodes = (self.nx + 1) * (self.ny + 1)
        self.n_dof = 2 * self.n_nodes

        # Generate element degree-of-freedom indices
        import numpy as _np
        edof = _np.zeros((self.n_elem, 8), dtype=_np.int64)
        for ex in range(self.nx):
            for ey in range(self.ny):
                el = ex * self.ny + ey
                # Nodes mapping
                n1 = ex * (self.ny + 1) + ey
                n2 = (ex + 1) * (self.ny + 1) + ey
                n3 = (ex + 1) * (self.ny + 1) + (ey + 1)
                n4 = ex * (self.ny + 1) + (ey + 1)
                edof[el] = [
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n3, 2 * n3 + 1,
                    2 * n4, 2 * n4 + 1
                ]
        self.edof = torch.from_numpy(edof)

        # Precompute flat indices for stiffness matrix assembly
        row_indices = self.edof.unsqueeze(2).repeat(1, 1, 8)
        col_indices = self.edof.unsqueeze(1).repeat(1, 8, 1)
        self.flat_indices = (row_indices * self.n_dof + col_indices).flatten()


def _get_k0(nu=0.3):
    # Element stiffness matrix for bilinear Q4 plane stress square element
    k = [
        0.5 - nu / 6.0,
        0.125 + nu / 8.0,
        -0.25 - nu / 12.0,
        -0.125 + 0.375 * nu,
        -0.25 + nu / 12.0,
        -0.125 - nu / 8.0,
        nu / 6.0,
        0.125 - 0.375 * nu
    ]
    K1 = torch.tensor([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ], dtype=torch.float64)

    # Alternate sign matrix for K2
    S = torch.tensor([
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0]
    ], dtype=torch.float64)

    K2 = K1 * S
    K0 = (K1 + K2 * nu) / (1.0 - nu**2)
    return K0


def structural_mesh_2d_impl(nx, ny):
    return StructuralMesh2D(nx, ny)


def structural_solve_2d_impl(mesh, densities, loads, fixed_dofs, E0=1.0, Emin=1e-9, nu=0.3, penal=3.0):
    loads = _to_tensor(loads).double()
    device = loads.device
    n_dof = mesh.n_dof
    
    fixed_dofs = _to_tensor(fixed_dofs).long().to(device)
    
    # Material interpolation (SIMP)
    x = _to_tensor(densities).double().to(device)
    E = Emin + (x ** penal) * (E0 - Emin)
    
    # Local stiffness matrix
    K0 = _get_k0(nu).to(device)
    
    # Global assembly
    Ke_all = E[:, None, None] * K0[None, :, :]
    
    K_flat = torch.zeros(n_dof * n_dof, dtype=torch.float64, device=device)
    K_flat.index_add_(0, mesh.flat_indices.to(device), Ke_all.flatten())
    K = K_flat.view(n_dof, n_dof)
    
    # Boundary conditions via submatrix partitioning
    fixed_set = set(int(d) for d in fixed_dofs)
    free_dofs = torch.tensor([d for d in range(n_dof) if d not in fixed_set], dtype=torch.long, device=device)
    
    K_free = K[free_dofs][:, free_dofs]
    F_free = loads[free_dofs]
    
    # Solve system differentiably
    U_free = torch.linalg.solve(K_free, F_free)
    
    # Rebuild full displacement vector
    U = torch.zeros(n_dof, dtype=torch.float64, device=device)
    U = U.scatter(0, free_dofs, U_free)
    return U


def structural_compliance_2d_impl(mesh, densities, loads, fixed_dofs, E0=1.0, Emin=1e-9, nu=0.3, penal=3.0):
    loads_t = _to_tensor(loads).double()
    U = structural_solve_2d_impl(mesh, densities, loads_t, fixed_dofs, E0, Emin, nu, penal)
    return torch.dot(U, loads_t)


def topo_opt_oc_2d_impl(mesh, loads, fixed_dofs, volfrac, steps=50, penal=3.0, rmin=1.5):
    import torch.nn.functional as F
    loads = _to_tensor(loads).double()
    device = loads.device
    nx, ny = mesh.nx, mesh.ny
    
    steps = int(steps)
    volfrac = float(volfrac)
    penal = float(penal)
    rmin = float(rmin)
    
    # Initial design densities
    x = torch.full((ny, nx), volfrac, dtype=torch.float64, device=device)
    vol_target = volfrac * nx * ny
    
    # Prepare filtering kernel
    r_int = int(math.floor(rmin))
    size = 2 * r_int + 1
    kernel = torch.zeros((size, size), dtype=torch.float64, device=device)
    for i in range(size):
        for j in range(size):
            dx = i - r_int
            dy = j - r_int
            dist = math.sqrt(dx*dx + dy*dy)
            kernel[i, j] = builtins.max(0.0, rmin - dist)
            
    w = kernel.view(1, 1, size, size)
    ones = torch.ones((1, 1, ny, nx), dtype=torch.float64, device=device)
    norm = F.conv2d(ones, w, padding=r_int)
    
    # Local stiffness matrix
    K0 = _get_k0(nu=0.3).to(device)
    
    for _ in range(steps):
        # Design density filter
        x_4d = x.view(1, 1, ny, nx)
        x_phys = (F.conv2d(x_4d, w, padding=r_int) / norm).view(-1)
        
        # Finite element analysis
        U = structural_solve_2d_impl(mesh, x_phys, loads, fixed_dofs, penal=penal)
        
        # Strain energy of elements
        U_e = U[mesh.edof.to(device)]
        strain_energy = torch.sum((U_e @ K0) * U_e, dim=1)
        
        # Compliance sensitivities
        dc = -penal * (x_phys ** (penal - 1.0)) * strain_energy
        
        # Adjoint filtering of sensitivities
        dc_phys_4d = dc.view(1, 1, ny, nx)
        dc_design = F.conv2d(dc_phys_4d / norm, w, padding=r_int).view(ny, nx)
        
        # Bisection search for Lagrange Multiplier lambda
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        
        x_new = x.clone()
        # Cap bisection iterations at 100 to avoid infinite loops
        for _ in range(100):
            if (l2 - l1) / (l1 + 1e-12) <= 1e-4:
                break
            lmid = 0.5 * (l1 + l2)
            B = -dc_design / lmid
            B = torch.clamp(B, min=1e-12)
            B_eta = torch.sqrt(B)
            
            x_proposal = torch.clamp(
                x * B_eta,
                min=torch.max(torch.tensor(1e-9, device=device), x - move),
                max=torch.min(torch.tensor(1.0, device=device), x + move)
            )
            
            if x_proposal.sum().item() > vol_target:
                l1 = lmid
            else:
                l2 = lmid
            x_new = x_proposal
            
        x = x_new
        
    # Return physical densities
    x_4d = x.view(1, 1, ny, nx)
    x_phys = (F.conv2d(x_4d, w, padding=r_int) / norm).view(-1)
    return x_phys


def print_structural_topology_2d_impl(densities, nx, ny):
    import numpy as _np
    import builtins
    x = _to_tensor(densities).detach().cpu().numpy()
    nx = int(nx)
    ny = int(ny)
    
    grid = _np.zeros((ny, nx))
    for ex in range(nx):
        for ey in range(ny):
            el = ex * ny + ey
            if el < len(x):
                grid[ey, ex] = x[el]
                
    chars = [" ", "░", "▒", "▓", "█"]
    builtins.print("+" + "-" * nx + "+")
    for ey in range(ny):
        row_str = "|"
        for ex in range(nx):
            val = grid[ey, ex]
            val = builtins.min(builtins.max(val, 0.0), 1.0)
            idx = int(builtins.round(val * 4))
            row_str += chars[idx]
        row_str += "|"
        builtins.print(row_str)
    builtins.print("+" + "-" * nx + "+")


def structural_solve_truss_2d_impl(nodes, elements, E, A, loads, fixed_dofs):
    nodes = _to_tensor(nodes).double()
    elements = _to_tensor(elements).long()
    E = _to_tensor(E).double()
    A = _to_tensor(A).double()
    loads = _to_tensor(loads).double()
    fixed_dofs = _to_tensor(fixed_dofs).long()
    
    device = nodes.device
    num_nodes = len(nodes)
    num_elements = len(elements)
    n_dof = 2 * num_nodes
    
    if E.dim() == 0 or (E.dim() == 1 and E.numel() == 1):
        E = E.expand(num_elements)
    if A.dim() == 0 or (A.dim() == 1 and A.numel() == 1):
        A = A.expand(num_elements)
        
    n1 = elements[:, 0]
    n2 = elements[:, 1]
    
    dx = nodes[n2, 0] - nodes[n1, 0]
    dy = nodes[n2, 1] - nodes[n1, 1]
    L = torch.sqrt(dx*dx + dy*dy)
    c = dx / L
    s = dy / L
    
    k_coeff = (E * A) / L
    
    row1 = torch.stack([c*c, c*s, -c*c, -c*s], dim=-1)
    row2 = torch.stack([c*s, s*s, -c*s, -s*s], dim=-1)
    row3 = torch.stack([-c*c, -c*s, c*c, c*s], dim=-1)
    row4 = torch.stack([-c*s, -s*s, c*s, s*s], dim=-1)
    ke = torch.stack([row1, row2, row3, row4], dim=-2) * k_coeff[:, None, None]
    
    edof = torch.stack([2*n1, 2*n1+1, 2*n2, 2*n2+1], dim=-1)
    
    row_indices = edof.unsqueeze(2).repeat(1, 1, 4)
    col_indices = edof.unsqueeze(1).repeat(1, 4, 1)
    flat_indices = (row_indices * n_dof + col_indices).flatten()
    
    K_flat = torch.zeros(n_dof * n_dof, dtype=torch.float64, device=device)
    K_flat.index_add_(0, flat_indices.to(device), ke.flatten())
    K = K_flat.view(n_dof, n_dof)
    
    fixed_set = set(int(d) for d in fixed_dofs)
    free_dofs = torch.tensor([d for d in range(n_dof) if d not in fixed_set], dtype=torch.long, device=device)
    
    K_free = K[free_dofs][:, free_dofs]
    F_free = loads[free_dofs]
    
    U_free = torch.linalg.solve(K_free, F_free)
    
    U = torch.zeros(n_dof, dtype=torch.float64, device=device)
    U = U.scatter(0, free_dofs, U_free)
    return U


def structural_truss_stress_2d_impl(nodes, elements, E, U):
    nodes = _to_tensor(nodes).double()
    elements = _to_tensor(elements).long()
    E = _to_tensor(E).double()
    U = _to_tensor(U).double()
    
    num_elements = len(elements)
    if E.dim() == 0 or (E.dim() == 1 and E.numel() == 1):
        E = E.expand(num_elements)
        
    n1 = elements[:, 0]
    n2 = elements[:, 1]
    
    dx = nodes[n2, 0] - nodes[n1, 0]
    dy = nodes[n2, 1] - nodes[n1, 1]
    L = torch.sqrt(dx*dx + dy*dy)
    c = dx / L
    s = dy / L
    
    ux1 = U[2*n1]
    uy1 = U[2*n1+1]
    ux2 = U[2*n2]
    uy2 = U[2*n2+1]
    
    stresses = (E / L) * (-c * ux1 - s * uy1 + c * ux2 + s * uy2)
    return stresses


def structural_modal_2d_impl(mesh, densities, fixed_dofs, rho=1.0, num_modes=5):
    densities = _to_tensor(densities).double()
    fixed_dofs = _to_tensor(fixed_dofs).long()
    device = densities.device
    n_dof = mesh.n_dof
    num_modes = int(num_modes)
    rho = float(rho)
    
    E0, Emin, nu, penal = 1.0, 1e-9, 0.3, 3.0
    E = Emin + (densities ** penal) * (E0 - Emin)
    K0 = _get_k0(nu).to(device)
    Ke_all = E[:, None, None] * K0[None, :, :]
    
    K_flat = torch.zeros(n_dof * n_dof, dtype=torch.float64, device=device)
    K_flat.index_add_(0, mesh.flat_indices.to(device), Ke_all.flatten())
    K = K_flat.view(n_dof, n_dof)
    
    M_diag = torch.zeros(n_dof, dtype=torch.float64, device=device)
    element_mass = densities * rho / 4.0
    mass_contrib = element_mass.unsqueeze(1).repeat(1, 8)
    M_diag.index_add_(0, mesh.edof.to(device).flatten(), mass_contrib.flatten())
    
    fixed_set = set(int(d) for d in fixed_dofs)
    free_dofs = torch.tensor([d for d in range(n_dof) if d not in fixed_set], dtype=torch.long, device=device)
    
    K_free = K[free_dofs][:, free_dofs]
    M_free_diag = M_diag[free_dofs]
    
    M_free_diag_clamped = torch.clamp(M_free_diag, min=1e-9)
    D = 1.0 / torch.sqrt(M_free_diag_clamped)
    K_free_tilde = D.unsqueeze(1) * K_free * D.unsqueeze(0)
    
    lambdas, V_tilde = torch.linalg.eigh(K_free_tilde)
    
    num_free = len(free_dofs)
    n_sel = builtins.min(num_modes, num_free)
    
    lambdas_sel = lambdas[:n_sel]
    V_tilde_sel = V_tilde[:, :n_sel]
    
    V_free = D.unsqueeze(1) * V_tilde_sel
    V_full = torch.zeros(n_dof, n_sel, dtype=torch.float64, device=device)
    V_full[free_dofs, :] = V_free
    
    lambdas_clamped = torch.clamp(lambdas_sel, min=0.0)
    frequencies = torch.sqrt(lambdas_clamped) / (2.0 * math.pi)
    
    return frequencies, V_full


def concrete_beam_capacity_impl(b, h, d, As, fprime_c, fy, Es=200e9):
    b = _to_tensor(b).double()
    h = _to_tensor(h).double()
    d = _to_tensor(d).double()
    As = _to_tensor(As).double()
    fprime_c = _to_tensor(fprime_c).double()
    fy = _to_tensor(fy).double()
    Es = _to_tensor(Es).double()
    
    device = b.device
    f_c_MPa = fprime_c / 1e6
    
    beta_1 = torch.where(
        f_c_MPa <= 28.0,
        torch.tensor(0.85, dtype=torch.float64, device=device),
        torch.clamp(0.85 - 0.05 * (f_c_MPa - 28.0) / 7.0, min=0.65, max=0.85)
    )
    
    a = (As * fy) / (0.85 * fprime_c * b + 1e-12)
    c_trial = a / beta_1
    eps_cu = 0.003
    eps_s_trial = eps_cu * (d - c_trial) / (c_trial + 1e-12)
    eps_y = fy / Es
    
    A_q = 0.85 * fprime_c * b * beta_1
    B_q = As * Es * eps_cu
    C_q = -As * Es * eps_cu * d
    c_quadratic = (-B_q + torch.sqrt(B_q*B_q - 4.0 * A_q * C_q + 1e-12)) / (2.0 * A_q + 1e-12)
    
    c_actual = torch.where(eps_s_trial >= eps_y, c_trial, c_quadratic)
    a_actual = c_actual * beta_1
    eps_s_actual = eps_cu * (d - c_actual) / (c_actual + 1e-12)
    fs_actual = torch.where(eps_s_trial >= eps_y, fy, Es * eps_s_actual)
    
    Mn = fs_actual * As * (d - 0.5 * a_actual)
    
    phi = torch.where(
        eps_s_actual >= 0.005,
        torch.tensor(0.90, dtype=torch.float64, device=device),
        torch.where(
            eps_s_actual <= 0.002,
            torch.tensor(0.65, dtype=torch.float64, device=device),
            0.65 + 0.25 * (eps_s_actual - 0.002) / 0.003
        )
    )
    
    Md = phi * Mn
    return Mn, Md, eps_s_actual, c_actual, phi


def steel_buckling_check_impl(A, r, L, K, E, fy):
    A = _to_tensor(A).double()
    r = _to_tensor(r).double()
    L = _to_tensor(L).double()
    K = _to_tensor(K).double()
    E = _to_tensor(E).double()
    fy = _to_tensor(fy).double()
    
    device = A.device
    
    lambda_val = K * L / r
    Fe = (math.pi * math.pi * E) / (lambda_val * lambda_val + 1e-12)
    lambda_c = 4.71 * torch.sqrt(E / fy)
    
    Fcr = torch.where(
        lambda_val <= lambda_c,
        (0.658 ** (fy / Fe)) * fy,
        0.877 * Fe
    )
    
    Pn = Fcr * A
    Pd = 0.90 * Pn
    Pa = Pn / 1.67
    
    return Pn, Pd, Pa, Fe, Fcr, lambda_val


# Dedekind Standard Library: Differentiable Heat Transfer & Thermodynamics
import torch
import math
import builtins

class ThermalMesh2D:
    def __init__(self, nx, ny):
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_elem = self.nx * self.ny
        self.n_nodes = (self.nx + 1) * (self.ny + 1)

        import numpy as _np
        edof = _np.zeros((self.n_elem, 4), dtype=_np.int64)
        for ex in range(self.nx):
            for ey in range(self.ny):
                el = ex * self.ny + ey
                # Nodes mapping
                n1 = ex * (self.ny + 1) + ey
                n2 = (ex + 1) * (self.ny + 1) + ey
                n3 = (ex + 1) * (self.ny + 1) + (ey + 1)
                n4 = ex * (self.ny + 1) + (ey + 1)
                edof[el] = [n1, n2, n3, n4]
        self.edof = torch.from_numpy(edof)

        # Precompute flat indices for global stiffness assembly
        row_indices = self.edof.unsqueeze(2).repeat(1, 1, 4)
        col_indices = self.edof.unsqueeze(1).repeat(1, 4, 1)
        self.flat_indices = (row_indices * self.n_nodes + col_indices).flatten()


def thermal_mesh_2d_impl(nx, ny):
    return ThermalMesh2D(nx, ny)


def thermal_solve_2d_impl(mesh, conductivities, heat_sources, fixed_nodes, fixed_temps, k_min=1e-6):
    conductivities = _to_tensor(conductivities).double()
    device = conductivities.device
    
    n_nodes = mesh.n_nodes
    fixed_nodes = _to_tensor(fixed_nodes).long().to(device)
    fixed_temps = _to_tensor(fixed_temps).double().to(device)
    heat_sources = _to_tensor(heat_sources).double().to(device)
    
    # Enforce minimum conductivity
    k = torch.clamp(conductivities, min=k_min)
    
    # Local stiffness matrix K0 for unit square bilinear Q4 element
    K0 = torch.tensor([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0]
    ], dtype=torch.float64, device=device) / 6.0
    
    # Scale local matrices
    Ke = k[:, None, None] * K0[None, :, :]
    
    # Global assembly
    K_flat = torch.zeros(n_nodes * n_nodes, dtype=torch.float64, device=device)
    K_flat.index_add_(0, mesh.flat_indices.to(device), Ke.flatten())
    K = K_flat.view(n_nodes, n_nodes)
    
    # Boundary conditions
    fixed_set = set(int(n) for n in fixed_nodes)
    free_nodes = torch.tensor([n for n in range(n_nodes) if n not in fixed_set], dtype=torch.long, device=device)
    
    K_free = K[free_nodes][:, free_nodes]
    if len(fixed_nodes) > 0:
        F_free = heat_sources[free_nodes] - K[free_nodes][:, fixed_nodes] @ fixed_temps
    else:
        F_free = heat_sources[free_nodes]
        
    T_free = torch.linalg.solve(K_free, F_free)
    
    # Reconstruct temperature vector
    T = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    if len(fixed_nodes) > 0:
        T = T.scatter(0, fixed_nodes, fixed_temps)
    T = T.scatter(0, free_nodes, T_free)
    
    return T


def thermal_solve_transient_2d_impl(mesh, conductivities, capacities, initial_temps, heat_sources, fixed_nodes, fixed_temps, dt, steps, k_min=1e-6):
    conductivities = _to_tensor(conductivities).double()
    device = conductivities.device
    
    n_nodes = mesh.n_nodes
    fixed_nodes = _to_tensor(fixed_nodes).long().to(device)
    fixed_temps = _to_tensor(fixed_temps).double().to(device)
    
    k = torch.clamp(conductivities, min=k_min)
    c_e = _to_tensor(capacities).double().to(device)
    
    # Assemble K
    K0 = torch.tensor([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0]
    ], dtype=torch.float64, device=device) / 6.0
    Ke = k[:, None, None] * K0[None, :, :]
    
    K_flat = torch.zeros(n_nodes * n_nodes, dtype=torch.float64, device=device)
    K_flat.index_add_(0, mesh.flat_indices.to(device), Ke.flatten())
    K = K_flat.view(n_nodes, n_nodes)
    
    # Lumped mass matrix (diagonal)
    M = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    for i in range(4):
        M.index_add_(0, mesh.edof[:, i].to(device), c_e / 4.0)
        
    M_dt = M / float(dt)
    
    fixed_set = set(int(n) for n in fixed_nodes)
    free_nodes = torch.tensor([n for n in range(n_nodes) if n not in fixed_set], dtype=torch.long, device=device)
    
    T = _to_tensor(initial_temps).double().clone().to(device)
    if len(fixed_nodes) > 0:
        T[fixed_nodes] = fixed_temps
        
    Q = _to_tensor(heat_sources).double().to(device)
    steps = int(steps)
    if Q.dim() == 1:
        Q_all = Q.unsqueeze(0).repeat(steps, 1)
    else:
        Q_all = Q
        
    # K_star = K + diag(M_dt)
    K_star = K.clone()
    diag_indices = torch.arange(n_nodes, device=device)
    K_star[diag_indices, diag_indices] += M_dt
    
    K_star_free = K_star[free_nodes][:, free_nodes]
    
    for step in range(steps):
        Q_step = Q_all[step]
        F_star = Q_step + M_dt * T
        
        if len(fixed_nodes) > 0:
            F_star_free = F_star[free_nodes] - K_star[free_nodes][:, fixed_nodes] @ fixed_temps
        else:
            F_star_free = F_star[free_nodes]
            
        T_free = torch.linalg.solve(K_star_free, F_star_free)
        
        T = torch.zeros(n_nodes, dtype=torch.float64, device=device)
        if len(fixed_nodes) > 0:
            T = T.scatter(0, fixed_nodes, fixed_temps)
        T = T.scatter(0, free_nodes, T_free)
        
    return T


def topo_opt_thermal_oc_2d_impl(mesh, heat_sources, fixed_nodes, fixed_temps, volfrac, steps=50, penal=3.0, rmin=1.5, k0=1.0, kmin=1e-6):
    import torch.nn.functional as F
    heat_sources = _to_tensor(heat_sources).double()
    device = heat_sources.device
    nx, ny = mesh.nx, mesh.ny
    
    steps = int(steps)
    volfrac = float(volfrac)
    penal = float(penal)
    rmin = float(rmin)
    k0 = float(k0)
    kmin = float(kmin)
    
    # Initial design densities
    x = torch.full((ny, nx), volfrac, dtype=torch.float64, device=device)
    vol_target = volfrac * nx * ny
    
    # Prepare filtering kernel
    r_int = int(math.floor(rmin))
    size = 2 * r_int + 1
    kernel = torch.zeros((size, size), dtype=torch.float64, device=device)
    for i in range(size):
        for j in range(size):
            dx = i - r_int
            dy = j - r_int
            dist = math.sqrt(dx*dx + dy*dy)
            kernel[i, j] = builtins.max(0.0, rmin - dist)
            
    w = kernel.view(1, 1, size, size)
    ones = torch.ones((1, 1, ny, nx), dtype=torch.float64, device=device)
    norm = F.conv2d(ones, w, padding=r_int)
    
    # Local stiffness matrix K0 for unit square element
    K0 = torch.tensor([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0]
    ], dtype=torch.float64, device=device) / 6.0
    
    for _ in range(steps):
        # Design density filter
        x_4d = x.view(1, 1, ny, nx)
        x_phys = (F.conv2d(x_4d, w, padding=r_int) / norm).view(-1)
        
        # Finite element steady-state thermal solve
        # Thermal conductivity interpolated with SIMP-like scheme: k = kmin + x_phys^penal * (k0 - kmin)
        k_interp = kmin + (x_phys ** penal) * (k0 - kmin)
        T = thermal_solve_2d_impl(mesh, k_interp, heat_sources, fixed_nodes, fixed_temps, k_min=kmin)
        
        # Thermal energy of elements: T_e^T @ K0 @ T_e
        T_e = T[mesh.edof.to(device)]
        element_energy = torch.sum((T_e @ K0) * T_e, dim=1)
        
        # Compliance sensitivities
        dc = -penal * (x_phys ** (penal - 1.0)) * (k0 - kmin) * element_energy
        
        # Adjoint filtering of sensitivities
        dc_phys_4d = dc.view(1, 1, ny, nx)
        dc_design = F.conv2d(dc_phys_4d / norm, w, padding=r_int).view(ny, nx)
        
        # Bisection search for Lagrange Multiplier lambda
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        
        x_new = x.clone()
        for _ in range(100):
            if (l2 - l1) / (l1 + 1e-12) <= 1e-4:
                break
            lmid = 0.5 * (l1 + l2)
            B = -dc_design / lmid
            B = torch.clamp(B, min=1e-12)
            B_eta = torch.sqrt(B)
            
            x_proposal = torch.clamp(
                x * B_eta,
                min=torch.max(torch.tensor(1e-9, device=device), x - move),
                max=torch.min(torch.tensor(1.0, device=device), x + move)
            )
            
            if x_proposal.sum().item() > vol_target:
                l1 = lmid
            else:
                l2 = lmid
            x_new = x_proposal
            
        x = x_new
        
    # Return physical densities
    x_4d = x.view(1, 1, ny, nx)
    x_phys = (F.conv2d(x_4d, w, padding=r_int) / norm).view(-1)
    return x_phys


def print_thermal_topology_2d_impl(densities, nx, ny):
    import numpy as _np
    import builtins
    x = _to_tensor(densities).detach().cpu().numpy()
    nx = int(nx)
    ny = int(ny)
    
    grid = _np.zeros((ny, nx))
    for ex in range(nx):
        for ey in range(ny):
            el = ex * ny + ey
            if el < len(x):
                grid[ey, ex] = x[el]
                
    chars = [" ", "░", "▒", "▓", "█"]
    builtins.print("+" + "-" * nx + "+")
    for ey in range(ny):
        row_str = "|"
        for ex in range(nx):
            val = grid[ey, ex]
            val = builtins.min(builtins.max(val, 0.0), 1.0)
            idx = int(builtins.round(val * 4))
            row_str += chars[idx]
        row_str += "|"
        builtins.print(row_str)
    builtins.print("+" + "-" * nx + "+")
# Dedekind Standard Library: Differentiable Space Physics & Orbital Mechanics
import torch
import math
import builtins

def compute_gravitational_acceleration(pos, masses, G, softening=1e-9):
    # pos: [N, dim]
    # masses: [N]
    # G: float
    # Return: [N, dim]
    N = pos.shape[0]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, dim]
    dist_sq = torch.sum(diff ** 2, dim=-1) + softening  # [N, N]
    dist_cubed = dist_sq ** 1.5  # [N, N]
    
    term = masses.unsqueeze(0).unsqueeze(-1) * diff / dist_cubed.unsqueeze(-1)  # [N, N, dim]
    acc = G * torch.sum(term, dim=1)  # [N, dim]
    return acc

def _get_space_val(v, dim=None):
    if isinstance(v, Quantity):
        if dim is not None:
            return _convert_to_base(v.value, v.unit, dim)
        return v.value
    return v

def n_body_simulate_impl(positions_init, velocities_init, masses, dt, n_steps, G=6.6743e-11, softening=1e-9):
    positions_init = _to_tensor(positions_init).double()
    velocities_init = _to_tensor(velocities_init).double()
    masses = _to_tensor(masses).double()
    
    G_val = float(_get_space_val(G))
    dt_val = float(_get_space_val(dt, "time"))
    n_steps = int(n_steps)
    softening_val = float(_get_space_val(softening))
    
    pos_list = [positions_init]
    vel_list = [velocities_init]
    
    curr_pos = positions_init
    curr_vel = velocities_init
    
    for _ in range(n_steps - 1):
        # RK4 step
        # k1
        k1_pos = curr_vel
        k1_vel = compute_gravitational_acceleration(curr_pos, masses, G_val, softening_val)
        
        # k2
        k2_pos_arg = curr_pos + 0.5 * dt_val * k1_pos
        k2_pos = curr_vel + 0.5 * dt_val * k1_vel
        k2_vel = compute_gravitational_acceleration(k2_pos_arg, masses, G_val, softening_val)
        
        # k3
        k3_pos_arg = curr_pos + 0.5 * dt_val * k2_pos
        k3_pos = curr_vel + 0.5 * dt_val * k2_vel
        k3_vel = compute_gravitational_acceleration(k3_pos_arg, masses, G_val, softening_val)
        
        # k4
        k4_pos_arg = curr_pos + dt_val * k3_pos
        k4_pos = curr_vel + dt_val * k3_vel
        k4_vel = compute_gravitational_acceleration(k4_pos_arg, masses, G_val, softening_val)
        
        # Update
        curr_pos = curr_pos + (dt_val / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
        curr_vel = curr_vel + (dt_val / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
        
        pos_list.append(curr_pos)
        vel_list.append(curr_vel)
        
    return {
        "positions": torch.stack(pos_list),
        "velocities": torch.stack(vel_list)
    }

def kepler_solve_impl(M, ecc, max_iter=10, tol=1e-12):
    M_val = _to_tensor(_get_space_val(M, "angle")).double()
    ecc_val = _to_tensor(_get_space_val(ecc)).double()
    
    E = M_val + ecc_val * torch.sin(M_val)
    for _ in range(int(max_iter)):
        f = E - ecc_val * torch.sin(E) - M_val
        df = 1.0 - ecc_val * torch.cos(E)
        df = torch.where(df == 0.0, torch.ones_like(df) * 1e-15, df)
        E = E - f / df
    return E

def kepler_to_cartesian_impl(a, ecc, inc, Omega, omega, nu, mu):
    a_val = _to_tensor(_get_space_val(a, "length")).double()
    ecc_val = _to_tensor(_get_space_val(ecc)).double()
    inc_val = _to_tensor(_get_space_val(inc, "angle")).double()
    Omega_val = _to_tensor(_get_space_val(Omega, "angle")).double()
    omega_val = _to_tensor(_get_space_val(omega, "angle")).double()
    nu_val = _to_tensor(_get_space_val(nu, "angle")).double()
    mu_val = _to_tensor(_get_space_val(mu)).double()
    
    numerator = a_val * (1.0 - ecc_val ** 2)
    denominator = 1.0 + ecc_val * torch.cos(nu_val)
    denominator = torch.where(denominator == 0.0, torch.ones_like(denominator) * 1e-15, denominator)
    r = numerator / denominator
    
    x_p = r * torch.cos(nu_val)
    y_p = r * torch.sin(nu_val)
    z_p = torch.zeros_like(r)
    
    term1 = a_val * (1.0 - ecc_val ** 2)
    term1 = torch.where(term1 <= 0.0, torch.ones_like(term1) * 1e-15, term1)
    val_v = torch.sqrt(mu_val / term1)
    
    vx_p = -val_v * torch.sin(nu_val)
    vy_p = val_v * (ecc_val + torch.cos(nu_val))
    vz_p = torch.zeros_like(r)
    
    r_p = torch.stack([x_p, y_p, z_p], dim=-1)
    v_p = torch.stack([vx_p, vy_p, vz_p], dim=-1)
    
    cos_O = torch.cos(Omega_val)
    sin_O = torch.sin(Omega_val)
    cos_w = torch.cos(omega_val)
    sin_w = torch.sin(omega_val)
    cos_i = torch.cos(inc_val)
    sin_i = torch.sin(inc_val)
    
    rx = r_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + r_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    ry = r_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + r_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    rz = r_p[..., 0] * (sin_w * sin_i) + r_p[..., 1] * (cos_w * sin_i)
    
    vx = v_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + v_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    vy = v_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + v_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    vz = v_p[..., 0] * (sin_w * sin_i) + v_p[..., 1] * (cos_w * sin_i)
    
    return {
        "position": torch.stack([rx, ry, rz], dim=-1),
        "velocity": torch.stack([vx, vy, vz], dim=-1)
    }

def kepler_to_cartesian_from_E_impl(a, ecc, inc, Omega, omega, E, mu):
    a_val = _to_tensor(_get_space_val(a, "length")).double()
    ecc_val = _to_tensor(_get_space_val(ecc)).double()
    inc_val = _to_tensor(_get_space_val(inc, "angle")).double()
    Omega_val = _to_tensor(_get_space_val(Omega, "angle")).double()
    omega_val = _to_tensor(_get_space_val(omega, "angle")).double()
    E_val = _to_tensor(_get_space_val(E, "angle")).double()
    mu_val = _to_tensor(_get_space_val(mu)).double()
    
    x_p = a_val * (torch.cos(E_val) - ecc_val)
    ecc_term = 1.0 - ecc_val ** 2
    ecc_term = torch.where(ecc_term < 0.0, torch.zeros_like(ecc_term), ecc_term)
    y_p = a_val * torch.sqrt(ecc_term) * torch.sin(E_val)
    z_p = torch.zeros_like(x_p)
    
    r = a_val * (1.0 - ecc_val * torch.cos(E_val))
    r = torch.where(r <= 0.0, torch.ones_like(r) * 1e-15, r)
    
    term_mu_a = mu_val * a_val
    term_mu_a = torch.where(term_mu_a < 0.0, torch.zeros_like(term_mu_a), term_mu_a)
    sqrt_mu_a = torch.sqrt(term_mu_a)
    
    vx_p = -sqrt_mu_a * torch.sin(E_val) / r
    vy_p = sqrt_mu_a * torch.sqrt(ecc_term) * torch.cos(E_val) / r
    vz_p = torch.zeros_like(x_p)
    
    r_p = torch.stack([x_p, y_p, z_p], dim=-1)
    v_p = torch.stack([vx_p, vy_p, vz_p], dim=-1)
    
    cos_O = torch.cos(Omega_val)
    sin_O = torch.sin(Omega_val)
    cos_w = torch.cos(omega_val)
    sin_w = torch.sin(omega_val)
    cos_i = torch.cos(inc_val)
    sin_i = torch.sin(inc_val)
    
    rx = r_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + r_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    ry = r_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + r_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    rz = r_p[..., 0] * (sin_w * sin_i) + r_p[..., 1] * (cos_w * sin_i)
    
    vx = v_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + v_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    vy = v_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + v_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    vz = v_p[..., 0] * (sin_w * sin_i) + v_p[..., 1] * (cos_w * sin_i)
    
    return {
        "position": torch.stack([rx, ry, rz], dim=-1),
        "velocity": torch.stack([vx, vy, vz], dim=-1)
    }

def cartesian_to_kepler_impl(r, v, mu):
    r_val = _to_tensor(r).double()
    v_val = _to_tensor(v).double()
    mu_val = _to_tensor(_get_space_val(mu)).double()
    
    is_batched = r_val.dim() > 1
    if not is_batched:
        r_val = r_val.unsqueeze(0)
        v_val = v_val.unsqueeze(0)
        
    r_mag = torch.norm(r_val, dim=-1, keepdim=True)
    v_mag = torch.norm(v_val, dim=-1, keepdim=True)
    
    h = torch.cross(r_val, v_val, dim=-1)
    h_mag = torch.norm(h, dim=-1, keepdim=True)
    h_mag = torch.where(h_mag == 0.0, torch.ones_like(h_mag) * 1e-15, h_mag)
    
    n = torch.stack([-h[..., 1], h[..., 0], torch.zeros_like(h[..., 0])], dim=-1)
    n_mag = torch.norm(n, dim=-1, keepdim=True)
    n_mag = torch.where(n_mag == 0.0, torch.ones_like(n_mag) * 1e-15, n_mag)
    
    r_dot_v = torch.sum(r_val * v_val, dim=-1, keepdim=True)
    r_mag_safe = torch.where(r_mag == 0.0, torch.ones_like(r_mag) * 1e-15, r_mag)
    e_vec = (1.0 / mu_val) * ((v_mag ** 2 - mu_val / r_mag_safe) * r_val - r_dot_v * v_val)
    ecc = torch.norm(e_vec, dim=-1, keepdim=True)
    ecc_safe = torch.where(ecc == 0.0, torch.ones_like(ecc) * 1e-15, ecc)
    
    energy = 0.5 * (v_mag ** 2) - mu_val / r_mag_safe
    a = -mu_val / (2.0 * energy)
    
    inc = torch.acos(torch.clamp(h[..., 2:3] / h_mag, -1.0, 1.0))
    
    Omega = torch.atan2(n[..., 1:2], n[..., 0:1])
    Omega = torch.where(n_mag < 1e-10, torch.zeros_like(Omega), Omega)
    Omega = torch.where(Omega < 0.0, Omega + 2.0 * math.pi, Omega)
    
    n_dot_e = torch.sum(n * e_vec, dim=-1, keepdim=True)
    omega = torch.acos(torch.clamp(n_dot_e / (n_mag * ecc_safe), -1.0, 1.0))
    omega = torch.where(e_vec[..., 2:3] < 0.0, 2.0 * math.pi - omega, omega)
    
    omega_equatorial = torch.atan2(e_vec[..., 1:2], e_vec[..., 0:1])
    omega_equatorial = torch.where(omega_equatorial < 0.0, omega_equatorial + 2.0 * math.pi, omega_equatorial)
    omega = torch.where(n_mag < 1e-10, omega_equatorial, omega)
    
    e_dot_r = torch.sum(e_vec * r_val, dim=-1, keepdim=True)
    nu = torch.acos(torch.clamp(e_dot_r / (ecc_safe * r_mag_safe), -1.0, 1.0))
    nu = torch.where(r_dot_v < 0.0, 2.0 * math.pi - nu, nu)
    
    if not is_batched:
        return {
            "a": a.squeeze(0).squeeze(0),
            "ecc": ecc.squeeze(0).squeeze(0),
            "inc": inc.squeeze(0).squeeze(0),
            "Omega": Omega.squeeze(0).squeeze(0),
            "omega": omega.squeeze(0).squeeze(0),
            "nu": nu.squeeze(0).squeeze(0)
        }
    else:
        return {
            "a": a.squeeze(-1),
            "ecc": ecc.squeeze(-1),
            "inc": inc.squeeze(-1),
            "Omega": Omega.squeeze(-1),
            "omega": omega.squeeze(-1),
            "nu": nu.squeeze(-1)
        }
# Dedekind Standard Library: Atomic & Molecular Physics/Chemistry
import torch
import torch.fft
import math
import builtins

def _get_val(v):
    if hasattr(v, "value"):
        return v.value
    return v

# --- Molecular Dynamics & Chemistry ---

def _compute_lj_forces_and_pe(pos, box_size, epsilon, sigma, cutoff):
    N = pos.shape[0]
    # Pairwise differences: diff[i, j] = pos[i] - pos[j]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 3]
    
    # Minimum image convention for periodic boundary conditions
    diff = diff - box_size * torch.round(diff / box_size)
    
    dist_sq = torch.sum(diff ** 2, dim=-1)  # [N, N]
    mask = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    
    # Avoid division by zero on diagonal
    dist_sq_safe = torch.where(mask, dist_sq, torch.ones_like(dist_sq))
    dist_safe = torch.sqrt(dist_sq_safe)
    
    s_over_d = sigma / dist_safe
    s_over_d_6 = s_over_d ** 6
    s_over_d_12 = s_over_d_6 ** 2
    
    # LJ force coefficient: F_ij = coeff * (x_i - x_j)
    coeff = (24.0 * epsilon / dist_sq_safe) * (2.0 * s_over_d_12 - s_over_d_6)
    coeff = torch.where((dist_safe < cutoff) & mask, coeff, torch.zeros_like(coeff))
    
    forces = torch.sum(coeff.unsqueeze(-1) * diff, dim=1)  # [N, 3]
    
    # Potential energy
    v_lj = 4.0 * epsilon * (s_over_d_12 - s_over_d_6)
    v_lj = torch.where((dist_safe < cutoff) & mask, v_lj, torch.zeros_like(v_lj))
    pe = 0.5 * torch.sum(v_lj)
    
    return forces, pe

def molecular_lj_simulate_impl(positions, velocities, masses, box_size, dt, steps, epsilon=1.0, sigma=1.0, cutoff=2.5, thermostat_tau=-1.0, target_temp=1.0):
    pos = _to_tensor(positions).double()
    vel = _to_tensor(velocities).double()
    m = _to_tensor(masses).double()
    
    box_size_val = _to_tensor(_get_val(box_size)).double()
    dt_val = _to_tensor(_get_val(dt)).double()
    steps = int(steps)
    eps_val = _to_tensor(_get_val(epsilon)).double()
    sig_val = _to_tensor(_get_val(sigma)).double()
    cut_val = _to_tensor(_get_val(cutoff)).double()
    tau_val = _to_tensor(_get_val(thermostat_tau)).double()
    t_target_val = _to_tensor(_get_val(target_temp)).double()
    
    if m.dim() == 0 or m.numel() == 1:
        m = m.expand(pos.shape[0])
    m_col = m.unsqueeze(-1)
    
    F, pe = _compute_lj_forces_and_pe(pos, box_size_val, eps_val, sig_val, cut_val)
    
    history_pos = [pos]
    history_vel = [vel]
    history_pe = [pe]
    
    ke = 0.5 * torch.sum(m_col * vel**2)
    temp = (2.0 * ke) / (3.0 * pos.shape[0])
    history_ke = [ke]
    history_temp = [temp]
    
    curr_pos = pos
    curr_vel = vel
    curr_F = F
    
    for _ in range(steps):
        # 1. Half-step velocity update
        curr_vel = curr_vel + 0.5 * (curr_F / m_col) * dt_val
        
        # 2. Full-step position update
        curr_pos = curr_pos + curr_vel * dt_val
        
        # 3. Apply periodic boundary conditions
        curr_pos = curr_pos - box_size_val * torch.floor(curr_pos / box_size_val)
        
        # 4. Compute new forces and potential energy
        curr_F, pe = _compute_lj_forces_and_pe(curr_pos, box_size_val, eps_val, sig_val, cut_val)
        
        # 5. Second half-step velocity update
        curr_vel = curr_vel + 0.5 * (curr_F / m_col) * dt_val
        
        # 6. Calculate instantaneous kinetic energy and temperature
        ke = 0.5 * torch.sum(m_col * curr_vel**2)
        temp = (2.0 * ke) / (3.0 * curr_pos.shape[0])
        
        # 7. Apply Berendsen thermostat if active
        if tau_val.item() > 0.0:
            factor = torch.sqrt(torch.clamp(1.0 + (dt_val / tau_val) * (t_target_val / (temp + 1e-12) - 1.0), min=0.1, max=10.0))
            curr_vel = curr_vel * factor
            ke = ke * (factor ** 2)
            temp = temp * (factor ** 2)
            
        history_pos.append(curr_pos)
        history_vel.append(curr_vel)
        history_pe.append(pe)
        history_ke.append(ke)
        history_temp.append(temp)
        
    return {
        "positions": torch.stack(history_pos),
        "velocities": torch.stack(history_vel),
        "potential_energies": torch.stack(history_pe),
        "kinetic_energies": torch.stack(history_ke),
        "temperatures": torch.stack(history_temp)
    }

def morse_potential_impl(r, De, a, re):
    r = _to_tensor(r).double()
    De = _to_tensor(De).double()
    a = _to_tensor(a).double()
    re = _to_tensor(re).double()
    return De * (1.0 - torch.exp(-a * (r - re))) ** 2

def molecular_distance_impl(pos1, pos2):
    p1 = _to_tensor(pos1).double()
    p2 = _to_tensor(pos2).double()
    return torch.sqrt(torch.sum((p1 - p2) ** 2) + 1e-12)

def molecular_angle_impl(pos1, pos2, pos3):
    p1 = _to_tensor(pos1).double()
    p2 = _to_tensor(pos2).double()
    p3 = _to_tensor(pos3).double()
    
    u = p1 - p2
    v = p3 - p2
    
    u_norm = torch.sqrt(torch.sum(u ** 2) + 1e-12)
    v_norm = torch.sqrt(torch.sum(v ** 2) + 1e-12)
    
    cos_theta = torch.dot(u, v) / (u_norm * v_norm)
    cos_theta_clamped = torch.clamp(cos_theta, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    return torch.acos(cos_theta_clamped)

def molecular_dihedral_impl(pos1, pos2, pos3, pos4):
    p1 = _to_tensor(pos1).double()
    p2 = _to_tensor(pos2).double()
    p3 = _to_tensor(pos3).double()
    p4 = _to_tensor(pos4).double()
    
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    b2_norm = b2 / torch.sqrt(torch.sum(b2 ** 2) + 1e-12)
    
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    
    n1_norm = n1 / torch.sqrt(torch.sum(n1 ** 2) + 1e-12)
    n2_norm = n2 / torch.sqrt(torch.sum(n2 ** 2) + 1e-12)
    
    m1 = torch.cross(n1_norm, b2_norm, dim=-1)
    
    x = torch.dot(n1_norm, n2_norm)
    y = torch.dot(m1, n2_norm)
    
    return torch.atan2(y, x)

# --- Crystallography & Structure Analysis ---

def cryst_symmetry_apply_impl(coords, R, t):
    c = _to_tensor(coords).double()
    R_t = _to_tensor(R).double()
    t_t = _to_tensor(t).double()
    
    is_1d = (c.dim() == 1)
    if is_1d:
        c = c.unsqueeze(0)
        
    # Apply Seitz matrix (R, t) to fractional coordinates
    res = torch.matmul(c, R_t.t()) + t_t
    res = torch.remainder(res, 1.0)
    
    if is_1d:
        res = res.squeeze(0)
    return res

def cryst_generate_equivalent_atoms_impl(coords, R_ops, t_ops):
    c = _to_tensor(coords).double()
    R_ops_t = _to_tensor(R_ops).double()
    t_ops_t = _to_tensor(t_ops).double()
    
    is_1d = (c.dim() == 1)
    if is_1d:
        c = c.unsqueeze(0)
        
    # Vectorized application of S operations to N atoms
    rotated = torch.einsum("nj,skj->snk", c, R_ops_t)
    translated = rotated + t_ops_t.unsqueeze(1)
    res = torch.remainder(translated, 1.0)
    
    if is_1d:
        res = res.squeeze(1)
    return res

def cryst_find_symmetries_impl(coords, elements, R_ops, t_ops, tol=1e-3):
    c = _to_tensor(coords).double()
    R_ops_t = _to_tensor(R_ops).double()
    t_ops_t = _to_tensor(t_ops).double()
    tol_val = float(_get_val(tol))
    
    if c.dim() == 1:
        c = c.unsqueeze(0)
        
    # Map elements to integer IDs
    if isinstance(elements, torch.Tensor):
        elem_ids = elements
    else:
        if hasattr(elements, "tolist"):
            el_list = elements.tolist()
        elif hasattr(elements, "value"):
            el_list = elements.value
        else:
            el_list = list(elements)
            
        unique_el = sorted(list(set(el_list)))
        mapping = {el: i for i, el in enumerate(unique_el)}
        elem_ids = torch.tensor([mapping[el] for el in el_list], device=c.device, dtype=torch.long)
        
    # Generate all symmetry-equivalent coordinates
    eq_coords = cryst_generate_equivalent_atoms_impl(c, R_ops_t, t_ops_t)
    
    # Pairwise periodic differences
    diff = eq_coords.unsqueeze(2) - c.unsqueeze(0).unsqueeze(1)
    diff = diff - torch.round(diff)
    
    # Distance squared
    dist_sq = torch.sum(diff ** 2, dim=-1)
    
    # Element matching mask
    eq_mask = (elem_ids.unsqueeze(1) == elem_ids.unsqueeze(0))
    eq_mask = eq_mask.unsqueeze(0)
    
    # Matches must satisfy periodic distance tolerance and element type equivalence
    valid_matches = (dist_sq < tol_val ** 2) & eq_mask
    
    # An operation is a valid symmetry if every mapped atom has a match
    atom_has_match = torch.any(valid_matches, dim=2)
    op_is_symmetry = torch.all(atom_has_match, dim=1)
    
    return op_is_symmetry

def cryst_structure_factor_atoms_impl(coords, scattering_factors, hkl_indices):
    c = _to_tensor(coords).double()
    sf = _to_tensor(scattering_factors).double()
    hkl = _to_tensor(hkl_indices).double()
    
    if c.dim() == 1:
        c = c.unsqueeze(0)
    if hkl.dim() == 1:
        hkl = hkl.unsqueeze(0)
        
    # Dot product of Miller indices and fractional coordinates
    angles = 2.0 * math.pi * torch.matmul(hkl, c.t())
    
    # Complex exponentials
    phases = torch.complex(torch.cos(angles), torch.sin(angles))
    
    # Broadcast scattering factors
    if sf.dim() == 1:
        sf_bc = sf.unsqueeze(0)
    elif sf.dim() == 2 and sf.shape[1] == hkl.shape[0]:
        sf_bc = sf.t()
    else:
        sf_bc = sf.view(1, -1)
        
    # Sum over atoms to get structure factor for each reflection
    F = torch.sum(sf_bc * phases, dim=1)
    return F

def cryst_structure_factor_density_impl(density_map):
    rho = _to_tensor(density_map)
    F = torch.fft.fftn(rho)
    return F
