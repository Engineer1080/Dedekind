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
    "pressure": ("Pa", {"Pa": 1.0, "kPa": 1000.0, "MPa": 1e6, "bar": 1e5, "atm": 101325.0}),
    "force": ("N", {"N": 1.0, "kN": 1000.0, "MN": 1e6}),
    "permeability": ("m^2", {"m^2": 1.0, "D": 9.869233e-13, "mD": 9.869233e-16}),  # Darcy: 1 D ≈ 9.87e-13 m²
    "volume": ("L", {"L": 1.0, "mL": 0.001, "dm^3": 1.0, "m^3": 1000.0}),  # dm³ = 1 L, m³ = 1000 L
    "energy": ("J", {"J": 1.0, "kJ": 1000.0, "MJ": 1e6, "Wh": 3600.0, "kWh": 3.6e6}),
    "electric_potential": ("V", {"V": 1.0, "mV": 0.001, "kV": 1000.0}),
    "frequency": ("Hz", {"Hz": 1.0, "kHz": 1000.0, "MHz": 1e6, "GHz": 1e9}),
    "charge": ("C", {"C": 1.0, "mC": 0.001, "uC": 1e-6}),
    "resistance": ("ohm", {"ohm": 1.0, "kohm": 1000.0, "Mohm": 1e6}),
    "power": ("W", {"W": 1.0, "kW": 1000.0, "MW": 1e6}),
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
