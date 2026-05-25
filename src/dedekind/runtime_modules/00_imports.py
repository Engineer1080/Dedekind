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
    "pressure": ("Pa", {"Pa": 1.0, "hPa": 100.0, "kPa": 1000.0, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "atm": 101325.0, "uPa": 1e-6, "muPa": 1e-6}),
    "force": ("N", {"N": 1.0, "kN": 1000.0, "MN": 1e6}),
    "permeability": ("m^2", {"m^2": 1.0, "D": 9.869233e-13, "mD": 9.869233e-16}),  # Darcy: 1 D ≈ 9.87e-13 m²
    "volume": ("L", {"L": 1.0, "mL": 0.001, "dm^3": 1.0, "m^3": 1000.0}),  # dm³ = 1 L, m³ = 1000 L
    "energy": ("J", {"J": 1.0, "kJ": 1000.0, "MJ": 1e6, "Wh": 3600.0, "kWh": 3.6e6, "eV": 1.602176634e-19, "meV": 1.602176634e-22, "keV": 1.602176634e-16, "MeV": 1.602176634e-13, "GeV": 1.602176634e-10, "cal": 4.184, "kcal": 4184.0}),
    "electric_potential": ("V", {"V": 1.0, "mV": 0.001, "uV": 1e-6, "muV": 1e-6, "kV": 1000.0}),
    "frequency": ("Hz", {"Hz": 1.0, "kHz": 1000.0, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}),
    "charge": ("C", {"C": 1.0, "mC": 0.001, "uC": 1e-6}),
    "resistance": ("ohm", {"ohm": 1.0, "kohm": 1000.0, "Mohm": 1e6}),
    "power": ("W", {"W": 1.0, "mW": 0.001, "kW": 1000.0, "MW": 1e6, "L_sun": 3.828e26}),
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


_LOG_UNITS = {
    "dB": ("", 1.0, True),
    "dB_power": ("", 1.0, True),
    "dB_amp": ("", 1.0, False),
    "dBW": ("W", 1.0, True),
    "dBm": ("W", 1e-3, True),
    "dBV": ("V", 1.0, False),
    "dBuV": ("V", 1e-6, False),
    "dBSPL": ("Pa", 20e-6, False),
}

def _get_log_dimension(unit):
    u = str(unit).strip() if unit else ""
    if u in _LOG_UNITS:
        ref_unit = _LOG_UNITS[u][0]
        if not ref_unit:
            return ""
        return _get_dimension(ref_unit)
    return _get_dimension(u)

def _log10(x):
    if isinstance(x, torch.Tensor):
        return torch.log10(x)
    import math as _math
    try:
        return _math.log10(x)
    except ValueError:
        return float("nan")

def _log_to_linear(value, unit):
    ref_unit, ref_scale, is_power = _LOG_UNITS[unit]
    factor = 10.0 if is_power else 20.0
    ratio = 10.0 ** (value / factor)
    return ratio * ref_scale, ref_unit

def _linear_to_log(value, linear_unit, log_unit):
    ref_unit, ref_scale, is_power = _LOG_UNITS[log_unit]
    dim = _get_dimension(ref_unit)
    v_base = _convert_to_base(value, linear_unit, dim)
    v_ref = _convert_from_base(v_base, ref_unit, dim)
    ratio = v_ref / ref_scale
    factor = 10.0 if is_power else 20.0
    return factor * _log10(ratio)

