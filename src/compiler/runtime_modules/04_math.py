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
