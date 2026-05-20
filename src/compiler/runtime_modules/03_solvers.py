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

def pde_navier_stokes_2d(u0, v0, x, y, t, nu, bc="periodic"):
    """
    Differenzierbarer 2D-Navier-Stokes-Solver (inkompressibel): u_t + (u·∇)u = -∇p + ν∇²u, ∇·u = 0.
    Chorin-Projektionsmethode: 1) Prädiktor (Konvektion + Diffusion), 2) Druck-Poisson (FFT), 3) Projektion.
    u0, v0: Anfangsgeschwindigkeiten 2D (nx, ny); x, y: Ortsgitter; t: Zeitgitter; nu: kinematische Viskosität.
    bc: 'periodic' (default; FFT für Druck-Poisson). 'dirichlet' experimentell (Jacobi-Iteration).
    Rückgabe: (u_sol, v_sol) je (len(t), nx, ny). CFL: dt*max(|u|)/dx < 1 empfohlen.
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

    # FFT-Wellenvektoren für periodische Druck-Poisson-Lösung
    kx = torch.fft.fftfreq(nx, d=dx).to(u0.device)
    ky = torch.fft.fftfreq(ny, d=dy).to(u0.device)
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

    nt = t.numel()
    u_sol = torch.zeros(nt, nx, ny, device=u0.device, dtype=u0.dtype)
    v_sol = torch.zeros(nt, nx, ny, device=u0.device, dtype=u0.dtype)
    u_sol[0] = u0
    v_sol[0] = v0
    u_cur = u0.clone()
    v_cur = v0.clone()

    for n in range(nt - 1):
        dt = float((t[n + 1] - t[n]).item())
        if dt <= 0:
            continue
        # 1) Prädiktor: u* = u - dt*(u·∇)u + dt*ν∇²u
        conv_u = _convective_upwind(u_cur, v_cur, u_cur)
        conv_v = _convective_upwind(u_cur, v_cur, v_cur)
        lap_u = _laplacian(u_cur)
        lap_v = _laplacian(v_cur)
        u_star = u_cur + dt * (-conv_u + nu_val * lap_u)
        v_star = v_cur + dt * (-conv_v + nu_val * lap_v)
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
        u_sol[n + 1] = u_cur
        v_sol[n + 1] = v_cur

    return u_sol, v_sol

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
