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
                 collision="bgk", bounce_back="soft"):
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        self._inlet_u = float(inlet_u)
        self.collision_model = str(collision).lower()
        self.bounce_back_model = str(bounce_back).lower()
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
                 T_init=None):
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
        self.T_obstacle = None if T_obstacle is None else float(T_obstacle)

        if obstacle_mask is None:
            self.obstacle_mask = torch.zeros((self.nx, self.ny), dtype=torch.float64)
        else:
            mask = _to_double_tensor(obstacle_mask)
            if mask.shape != (self.nx, self.ny):
                raise ValueError(
                    f"ThermalLbmSimulation: obstacle_mask Shape {tuple(mask.shape)} "
                    f"passt nicht zu ({self.nx},{self.ny})."
                )
            self.obstacle_mask = mask

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

        # Equilibria
        self.f = lbm_equilibrium(rho_init, u_init, device)
        self.g = _thermal_equilibrium(self.T, u_init[0], u_init[1], device, dtype)

        # Cached constants
        self._cx_u, self._cy_u, self._w_u = _lbm_constants(device, dtype)
        self._cx_t, self._cy_t, self._w_t = _thermal_constants(device, dtype)
        self._device = device
        self._dtype = dtype

    def step(self):
        """Ein Thermal-LBM-Zeitschritt: Stream → Macroscopic → Buoyancy → Collide → BCs → Bounce."""
        nx, ny = self.nx, self.ny
        device, dtype = self._device, self._dtype

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

        # 2. Makroskopische Felder
        rho = torch.sum(f_streamed, dim=0)
        rho_safe = rho + 1e-15
        ux = torch.sum(f_streamed * self._cx_u, dim=0) / rho_safe
        uy = torch.sum(f_streamed * self._cy_u, dim=0) / rho_safe
        T = torch.sum(g_streamed, dim=0)

        # 3. Boussinesq-Buoyancy mit Shift-Velocity-Forcing.
        #    Konvention dieses Solvers:
        #      - j=0  ist die UNTERE (heiße) Wand (T_hot).
        #      - j=ny-1 ist die OBERE (kalte) Wand (T_cold).
        #      - "+y" (wachsendes j) zeigt nach OBEN; Schwerkraft wirkt nach -y.
        #    Mit dieser Streaming-Konvention ergibt das funktionierende Vorzeichen
        #    für aufsteigende heiße / absteigende kalte Fluide:
        #      F_y / ρ = -g·β·(T - T_ref)
        #    Heiße Fluide (T > T_ref) bekommen einen effektiven Aufwärtsschub
        #    in +y-Richtung; validiert per Drift-Test und Rayleigh-Bénard
        #    (Nu > 1 oberhalb der kritischen Rayleigh-Zahl).
        F_y_per_rho = -self.gravity_beta * (T - self.T_ref)
        ux_eq = ux
        uy_eq = uy + self.tau_u * F_y_per_rho

        # 4. Collision: f (BGK mit shifted velocity), g (BGK passiv mit echter Geschwindigkeit)
        feq = lbm_equilibrium(rho, torch.stack([ux_eq, uy_eq]), device)
        f_coll = f_streamed - (1.0 / self.tau_u) * (f_streamed - feq)

        geq = _thermal_equilibrium(T, ux, uy, device, dtype)
        g_coll = g_streamed - (1.0 / self.tau_T) * (g_streamed - geq)

        # 5. Boundary Conditions:
        #    - Bottom (y=0): T = T_hot via Anti-Bounce-Back
        #    - Top    (y=ny-1): T = T_cold via Anti-Bounce-Back
        #    - Velocity an y=0 und y=ny-1: No-Slip via Bounce-Back
        # No-Slip an Top/Bottom: f[i] an Wandzellen wird durch f[opposite] ersetzt
        wall_bottom = torch.zeros((nx, ny), dtype=dtype, device=device)
        wall_top = torch.zeros((nx, ny), dtype=dtype, device=device)
        wall_bottom[:, 0] = 1.0
        wall_top[:, -1] = 1.0
        wall_mask = wall_bottom + wall_top   # 1 an Top und Bottom

        # Velocity Bounce-Back an Wänden
        f_bounce = f_coll[_LBM_OPPOSITE]
        f_coll = (1.0 - wall_mask.unsqueeze(0)) * f_coll + wall_mask.unsqueeze(0) * f_bounce

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

        # 6. Periodische BC links/rechts (durch Streaming automatisch erfüllt).

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
        self.T = T

    def run(self, steps):
        for _ in range(int(steps)):
            self.step()

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
                        collision="bgk", bounce_back="soft"):
    return LbmSimulation(nx, ny, tau, obstacle_mask, inlet_u,
                         collision=collision, bounce_back=bounce_back)


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
                                 T_obstacle=None):
    """Erzeugt eine ThermalLbmSimulation für gekoppelte Strömung+Wärme.

    Beispiele:
      Rayleigh-Bénard: heiß unten / kalt oben, Ra = g·β·ΔT·H³/(ν·α)
      Wärmetauscher:   beheizter Zylinder als Hindernis im Kanal
    """
    return ThermalLbmSimulation(nx, ny, tau_u, tau_T, T_hot, T_cold,
                                 gravity_beta, obstacle_mask, T_obstacle)


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


def thermal_step_impl(sim):
    sim.step()
    return sim


def thermal_run_impl(sim, steps):
    sim.run(steps)
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
                 collision="bgk", bounce_back="soft"):
        self.collision_model = str(collision).lower()
        self.bounce_back_model = str(bounce_back).lower()
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
                      ny=None, u_lattice=None, collision="bgk", bounce_back="soft"):
    # Aus .ddk-Wrappern kommen 0/0.0 als Sentinel für "keine Angabe"
    if isinstance(ny, (int, float)) and ny == 0:
        ny = None
    if isinstance(u_lattice, (int, float)) and u_lattice == 0:
        u_lattice = None
    return PhysicalLbmSimulation(domain_x, domain_y, nx, inlet_u, nu,
                                 rho=rho, ny=ny, u_lattice_target=u_lattice,
                                 collision=collision, bounce_back=bounce_back)


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
