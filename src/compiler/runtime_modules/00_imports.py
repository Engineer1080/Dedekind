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
    "length": ("m", {"m": 1.0, "cm": 0.01, "km": 1000.0, "mm": 0.001, "um": 1e-6, "nm": 1e-9, "angstrom": 1e-10, "pm": 1e-12, "dm": 0.1, "AU": 149597870700.0, "ly": 9460730472580800.0, "pc": 3.085677581e16}),
    "mass": ("kg", {"kg": 1.0, "g": 0.001, "t": 1000.0, "mg": 1e-6, "ug": 1e-9, "Da": 1.66053906660e-27, "amu": 1.66053906660e-27, "M_sun": 1.98847e30}),
    "time": ("s", {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "yr": 31557600.0, "a": 31557600.0, "ms": 0.001, "us": 1e-6, "ns": 1e-9}),
    "current": ("A", {"A": 1.0, "mA": 0.001, "kA": 1000.0, "uA": 1e-6, "muA": 1e-6}),
    "temperature": ("K", {"K": 1.0, "mK": 0.001}),
    "amount_of_substance": ("mol", {"mol": 1.0, "mmol": 0.001, "umol": 1e-6, "nmol": 1e-9, "kmol": 1000.0}),
    "luminous_intensity": ("cd", {"cd": 1.0, "mcd": 0.001}),
    # Abgeleitet / häufig
    "pressure": ("Pa", {"Pa": 1.0, "hPa": 100.0, "kPa": 1000.0, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "atm": 101325.0}),
    "force": ("N", {"N": 1.0, "kN": 1000.0, "MN": 1e6}),
    "permeability": ("m^2", {"m^2": 1.0, "D": 9.869233e-13, "mD": 9.869233e-16}),  # Darcy: 1 D ≈ 9.87e-13 m²
    "volume": ("L", {"L": 1.0, "mL": 0.001, "dm^3": 1.0, "m^3": 1000.0}),  # dm³ = 1 L, m³ = 1000 L
    "energy": ("J", {"J": 1.0, "kJ": 1000.0, "MJ": 1e6, "Wh": 3600.0, "kWh": 3.6e6, "eV": 1.602176634e-19, "meV": 1.602176634e-22, "keV": 1.602176634e-16, "MeV": 1.602176634e-13, "GeV": 1.602176634e-10, "cal": 4.184, "kcal": 4184.0}),
    "electric_potential": ("V", {"V": 1.0, "mV": 0.001, "kV": 1000.0}),
    "frequency": ("Hz", {"Hz": 1.0, "kHz": 1000.0, "MHz": 1e6, "GHz": 1e9}),
    "charge": ("C", {"C": 1.0, "mC": 0.001, "uC": 1e-6}),
    "resistance": ("ohm", {"ohm": 1.0, "kohm": 1000.0, "Mohm": 1e6}),
    "power": ("W", {"W": 1.0, "kW": 1000.0, "MW": 1e6, "L_sun": 3.828e26}),
    "magnetic_flux_density": ("T", {"T": 1.0, "G": 1e-4}),
    "magnetic_flux": ("Wb", {"Wb": 1.0}),
    "inductance": ("H", {"H": 1.0, "mH": 0.001, "uH": 1e-6}),
    "amount_concentration": ("M", {"M": 1.0, "mM": 0.001, "uM": 1e-6, "nM": 1e-9}),
    "absorbed_dose": ("Gy", {"Gy": 1.0, "mGy": 0.001}),
    "equivalent_dose": ("Sv", {"Sv": 1.0, "mSv": 0.001}),
    # Chemie/Biologie: Massenkonzentration (% w/v = g/100mL)
    "mass_concentration": ("g/L", {"g/L": 1.0, "mg/mL": 1.0, "percent_wv": 10.0}),  # 1% w/v = 10 g/L
    # Winkel (SI-Ergänzung: rad; deg = pi/180 rad)
    "angle": ("rad", {"rad": 1.0, "deg": 0.017453292519943295}),  # 1 deg = pi/180 rad
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
