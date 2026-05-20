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
        kwargs["bounds"] = list(bounds)
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

