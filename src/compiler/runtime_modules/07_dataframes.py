class DataFrame:
    """Leichte spaltenorientierte Tabelle mit optionalen Einheiten pro Spalte.

    Konstruktion:
      df = DataFrame({"t": [0.0, 1.0, 2.0], "T": [20.0, 22.0, 25.0]}, units={"t": "s", "T": "K"})
      df["T"]              -> Liste der Werte
      df.units["T"]        -> "K"
      df.rows              -> Iterator über Zeilen (dict)
      df.column_with_unit("T") -> Liste von Quantity-Werten
    """
    def __init__(self, data=None, units=None):
        if data is None:
            data = {}
        def _to_plain_list(seq):
            # Torch-Tensor: .tolist() liefert Plain-Python-Zahlen.
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
            raise TypeError("DataFrame: data muss dict of lists oder list of dicts sein.")
        n_rows = _builtin_max([len(c) for c in cols]) if cols else 0
        for c in cols:
            if len(c) != n_rows:
                raise ValueError("DataFrame: alle Spalten müssen gleiche Länge haben.")
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
            raise KeyError(f"DataFrame: Spalte '{key}' nicht gefunden. Verfügbar: {self.columns}.")
        return list(self._cols[self.columns.index(key)])

    def __contains__(self, key):
        return key in self.columns

    def column_with_unit(self, key):
        """Spalte als Liste von Quantity-Werten (verwendet self.units[key] falls vorhanden)."""
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
        # Kompakte Darstellung mit Einheiten in Spaltenüberschriften.
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
    """Liest eine CSV-Datei in eine DataFrame ein.
    - Header-Zeile darf Einheiten im Format `name [unit]` enthalten; diese werden in `df.units` übernommen.
    - `units` (optional Dict) überschreibt erkannte Einheiten.
    - Numerische Werte werden zu float konvertiert (Fallback: String belassen).
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
    """Liest eine Parquet-Datei (benötigt pyarrow). Einheiten werden nicht persistiert; nur Spalten."""
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("read_parquet: pyarrow nicht installiert (pip install pyarrow).")
    table = pq.read_table(str(path))
    data = {name: table.column(name).to_pylist() for name in table.column_names}
    return DataFrame(data)


def write_parquet(path, df):
    """Schreibt eine DataFrame als Parquet (benötigt pyarrow)."""
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("write_parquet: pyarrow nicht installiert (pip install pyarrow).")
    if not isinstance(df, DataFrame):
        raise TypeError("write_parquet: df muss DataFrame sein.")
    table = pa.table({c: df._cols[j] for j, c in enumerate(df.columns)})
    pq.write_table(table, str(path))


def read_hdf5(path, dataset=None):
    """Liest ein HDF5-Dataset (benötigt h5py). Wenn `dataset=None`: erstes Dataset wird gewählt."""
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("read_hdf5: h5py nicht installiert (pip install h5py).")
    import numpy as _np  # type: ignore[reportMissingImports]
    with h5py.File(str(path), "r") as f:
        if dataset is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError("read_hdf5: Datei enthält keine Datasets.")
            dataset = keys[0]
        arr = _np.array(f[dataset])
    return arr


def write_hdf5(path, data, dataset="data"):
    """Schreibt ein Tensor/Array in eine HDF5-Datei (benötigt h5py)."""
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("write_hdf5: h5py nicht installiert (pip install h5py).")
    import numpy as _np  # type: ignore[reportMissingImports]
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    arr = _np.asarray(data)
    with h5py.File(str(path), "w") as f:
        f.create_dataset(str(dataset), data=arr)


def read_netcdf(path, variable=None):
    """Liest eine NetCDF-Variable (benötigt netCDF4). Wenn `variable=None`: erste Variable wird gewählt."""
    try:
        import netCDF4  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("read_netcdf: netCDF4 nicht installiert (pip install netCDF4).")
    import numpy as _np  # type: ignore[reportMissingImports]
    ds = netCDF4.Dataset(str(path), "r")
    try:
        if variable is None:
            names = list(ds.variables.keys())
            if not names:
                raise ValueError("read_netcdf: Datei enthält keine Variablen.")
            variable = names[0]
        arr = _np.array(ds.variables[variable][:])
    finally:
        ds.close()
    return arr


# ============================================================================
# Einheiten-Annotationen für Funktionssignaturen (`fn f(x: [m]) -> [J]`)
# ============================================================================

# Abgeleitete SI-Einheiten -> Basisdimensionen (kg, m, s, A, K, mol, cd)
# Wir tracken nur die 7 SI-Basisdimensionen.
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
    "F":   {"kg": -1, "m": -2, "s": 4, "A": 2},
    "L":   {"m": 3},
    "mL":  {"m": 3},
    "M":   {"mol": 1, "m": -3},  # mol/L = 1000 mol/m³, gleiche Dimension
    "Gy":  {"m": 2, "s": -2},
    "Sv":  {"m": 2, "s": -2},
}


def _parse_unit_to_base_dims(unit_str):
    """Wandelt einen Einheiten-String ('kg*m^2/s^2', '(kg)*(m/s)*(m/s)', 'J', ...) in
    ein Dict {base_unit: exponent} der 7 SI-Basisdimensionen um.

    Tokenizer-ähnlich: zerlegt nach *, /, ^ und Klammern; jeder atomare Einheiten-Token
    wird über _DERIVED_UNIT_TO_BASE in Basisdimensionen aufgelöst. Unbekannte Tokens
    erzeugen None (Vergleich schlägt dann fehl).
    """
    s = (unit_str or "").strip()
    if not s:
        return {}
    # Tokenize: Klammern, *, /, ^, atomare Einheiten (Buchstabenfolge), Zahlen (für Exponenten).
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
        # Atomare Einheit oder Zahl
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
        # Pure Zahl als Faktor (z.B. konstante 1) -> dimensionslos
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
    # Null-Exponenten entfernen
    return {k: v for k, v in result.items() if v != 0}


def _units_dimensionally_equal(u1, u2):
    """True, wenn u1 und u2 sich auf dieselbe SI-Basisdimension reduzieren (z.B. J == kg*m²/s²)."""
    d1 = _parse_unit_to_base_dims(u1)
    d2 = _parse_unit_to_base_dims(u2)
    if d1 is None or d2 is None:
        return False
    return d1 == d2


def _coerce_to_expected_unit(value, expected_unit, context_label):
    """Bringt `value` auf `expected_unit`, wenn dimensional kompatibel.

    Pfade (in Reihenfolge):
    1) String-Gleichheit (gleiche Einheit) → unverändert
    2) Gleiche eindeutige Dimension (Länge, Masse, Zeit, …) → numerisch umrechnen
    3) Gleiche SI-Basisdimensionen (kg·m²/s² == J) → Wert als Quantity in expected_unit übernehmen
    4) Reine Zahl + expected_unit → mit Einheit wrappen
    Wirft sonst ValueError.
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
            f"Einheitenfehler in {context_label}: erwartet [{expected}], erhalten [{value.unit}]."
        )
    if isinstance(value, (int, float)):
        if expected:
            return Quantity(float(value), expected)
        return value
    return value


def _check_signature_unit(value, expected_unit, fn_name, arg_name):
    return _coerce_to_expected_unit(value, expected_unit, f"{fn_name}({arg_name})")


def _check_return_unit(value, expected_unit, fn_name):
    return _coerce_to_expected_unit(value, expected_unit, f"return von {fn_name}")


# ============================================================================
# Unit-aware Plotting: Quantity-Listen erhalten automatisch Achsenbeschriftung "[unit]"
# ============================================================================

def _strip_unit_from_seq(seq):
    """Wenn `seq` eine Liste/Tuple von Quantity ist, gib (values_list, unit_str) zurück; sonst (seq, None)."""
    if isinstance(seq, (list, tuple)) and seq and all(isinstance(v, Quantity) for v in seq):
        unit = seq[0].unit
        # Wenn Einheiten in der Liste variieren, lieber neutral lassen.
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
    """Wie `plot`, aber wenn x/y Listen von Quantity sind, werden Werte extrahiert und Einheiten als Achsenbeschriftung übernommen."""
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

