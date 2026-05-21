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
    """Vollständig differenzierbarer 2D-D2Q9-BGK Lattice-Boltzmann-Solver.

    Parameter
    ---------
    nx, ny : int
        Gittergröße in Lattice-Einheiten.
    tau : float
        BGK-Relaxationszeit. Muss > 0.5 sein (Stabilität). nu_lattice = (tau - 0.5)/3.
    obstacle_mask : Tensor[nx, ny] | None
        Soft mask in [0, 1]: 1 = solides Hindernis, 0 = freie Strömung.
    inlet_u : float
        Eingangsgeschwindigkeit (Lattice-Einheit), wird sowohl für Initialisierung
        als auch (per Default) für die Inlet-BC genutzt. Default 0.0.
    """

    # Empfohlene Maximalgeschwindigkeit (Mach-Limit ~ 0.1 für niedrige Kompressibilität)
    MAX_LATTICE_U = 0.1

    def __init__(self, nx, ny, tau, obstacle_mask=None, inlet_u=0.0):
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        self._inlet_u = float(inlet_u)

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

    def step(self, inlet_u=None):
        """Ein LBM-Zeitschritt: Stream → Inlet/Outlet-BC → Macroscopic → BGK → Soft-Bounce."""
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

        # 5. BGK Collision
        feq = self._equilibrium_from_macro(rho, ux, uy)
        f_coll = f_outlet - (1.0 / self.tau) * (f_outlet - feq)

        # 6. Soft-Mask-Bounce-Back: an Hindernis-Zellen wird f durch f[opposite] ersetzt
        f_bounce = f_outlet[_LBM_OPPOSITE]
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


# --- Runtime-Brücke zum Dedekind-Compiler ------------------------------

def lbm_simulation_impl(nx, ny, tau, obstacle_mask=None, inlet_u=0.0):
    return LbmSimulation(nx, ny, tau, obstacle_mask, inlet_u)


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
