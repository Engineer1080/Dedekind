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

def logspace(start, stop, steps, base=10.0):
    """Erzeugt einen 1D-Tensor mit `steps` logarithmisch verteilten Werten von base^start bis base^stop."""
    s = _to_tensor(start).float().squeeze()
    e = _to_tensor(stop).float().squeeze()
    b = float(_to_tensor(base).float().squeeze().item())
    n = int(steps)
    return torch.logspace(float(s.item()) if s.numel() == 1 else float(s.item()),
                          float(e.item()) if e.numel() == 1 else float(e.item()),
                          n, base=b)


# --- Mathematische Folgen (arithmetisch, geometrisch, allgemein) ---

def arange(start_or_stop, stop=None, step=None):
    """
    Integer-Folge wie numpy.arange.
    arange(n) → [0, 1, 2, ..., n-1]
    arange(start, stop) → [start, start+1, ..., stop-1]
    arange(start, stop, step) → [start, start+step, ...] (stop exklusive)

    Rückgabe-Dtype: int64 für reine Integer-Aufrufe (geeignet für Indexierung),
    float32 sobald ein expliziter `step` (auch ganzzahliger) angegeben ist (Konsistenz
    mit `linspace`). Tensor-Arithmetik promotet bei Mischung mit Floats automatisch.
    """
    if stop is None and step is None:
        return torch.arange(int(start_or_stop), dtype=torch.int64)
    if step is None:
        return torch.arange(int(start_or_stop), int(stop), dtype=torch.int64)
    return torch.arange(float(start_or_stop), float(stop), float(step))


def arithmetic(a0, d, n):
    """
    Arithmetische Folge: a_n = a0 + n*d für n = 0, 1, ..., n-1.
    a0: Startwert, d: Differenz, n: Anzahl Glieder.
    Rückgabe: 1D-Tensor [a0, a0+d, a0+2d, ..., a0+(n-1)d]; differenzierbar in a0, d.
    """
    a0_t = _to_tensor(a0).float().squeeze()
    d_t = _to_tensor(d).float().squeeze()
    n_val = int(n)
    k = torch.arange(n_val, dtype=torch.float32, device=a0_t.device if a0_t.dim() > 0 else None)
    return a0_t + d_t * k


def geometric(a0, r, n):
    """
    Geometrische Folge: a_n = a0 * r^n für n = 0, 1, ..., n-1.
    a0: Startwert, r: Quotient, n: Anzahl Glieder.
    Rückgabe: 1D-Tensor [a0, a0*r, a0*r^2, ..., a0*r^(n-1)]; differenzierbar in a0, r.
    """
    a0_t = _to_tensor(a0).float().squeeze()
    r_t = _to_tensor(r).float().squeeze()
    n_val = int(n)
    k = torch.arange(n_val, dtype=torch.float32, device=a0_t.device if a0_t.dim() > 0 else None)
    return a0_t * torch.pow(r_t, k)


def sequence(f, n):
    """
    Allgemeine Folge: [f(0), f(1), ..., f(n-1)].
    f: Funktion mit einem Argument (Index n); n: Anzahl Glieder.
    Nutzung: fn term(k) { return k * k }; seq = sequence(term, 10)
    Rückgabe: 1D-Tensor; f muss Skalar oder 0D-Tensor zurückgeben.
    """
    n_val = int(n)
    out = []
    for k in range(n_val):
        val = f(k)
        out.append(_to_tensor(val).float().squeeze())
    return torch.stack(out)




def labeled_tensor(data, dims, coords=None, name=None, attrs=None):
    """Erzeugt ein xarray.DataArray fuer Klima-/Geo-/Earth-Science-Workflows.

    Im Gegensatz zu nackten Tensoren tragen Labeled Tensors die *Namen* ihrer
    Achsen mit ('lat', 'lon', 'time', ...), und xarray-Operationen koennen
    namens-basiert (statt indexbasiert) arbeiten. Das verhindert die klassische
    Verwechslung 'mean ueber lat vs. mean ueber time'.

    Parameter:
      data:   Tensor/numpy-Array/Liste ÔÇö die Werte.
      dims:   Liste/Tuple von Strings ÔÇö Namen der Achsen, gleiche Reihenfolge wie data.shape.
      coords: Optional Dict {dim_name: 1D-Werte} ÔÇö Koordinaten-Achsen.
      name:   Optional Name fuer das DataArray.
      attrs:  Optional Dict ÔÇö Metadaten (z. B. {"units": "K", "crs": "EPSG:4326"}).

    Rueckgabe: xarray.DataArray.
    """
    try:
        import xarray as _xr  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("labeled_tensor benoetigt xarray. Installation: pip install xarray")
    import numpy as _np  # type: ignore[reportMissingImports]
    if hasattr(data, "detach"):
        arr = data.detach().cpu().numpy()
    else:
        arr = _np.asarray(data)
    dims_t = tuple(str(d) for d in dims)
    if len(dims_t) != arr.ndim:
        raise ValueError(
            f"labeled_tensor: data hat {arr.ndim} Achsen, dims hat {len(dims_t)} "
            f"({dims_t!r}). Anzahl muss uebereinstimmen."
        )
    coords_dict = dict(coords) if coords else {}
    return _xr.DataArray(arr, dims=dims_t, coords=coords_dict if coords_dict else None,
                          name=name, attrs=attrs or {})
