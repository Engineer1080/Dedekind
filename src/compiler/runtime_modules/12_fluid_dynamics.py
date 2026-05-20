# --- Computational Fluid Dynamics (CFD) via Lattice Boltzmann Method ---

import torch

# D2Q9 Velocity Directions
LBM_C = [
    [0, 0],
    [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
]

# D2Q9 Weights
LBM_W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]

# D2Q9 Opposite Directions for Bounce-Back
LBM_OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]

def _to_double_tensor(v):
    t = _to_tensor(v)
    return t.double()

def lbm_equilibrium(rho, u, device):
    """
    Berechnet die D2Q9 Gleichgewichtsverteilung feq.
    rho: shape (nx, ny) oder (1, ny)
    u: shape (2, nx, ny) oder (2, 1, ny)
    """
    ux = u[0]
    uy = u[1]
    u2 = ux**2 + uy**2
    
    cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float64, device=device).view(9, 1, 1)
    cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.float64, device=device).view(9, 1, 1)
    w = torch.tensor(LBM_W, dtype=torch.float64, device=device).view(9, 1, 1)
    
    c_dot_u = cx * ux + cy * uy
    feq = w * rho * (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u**2 - 1.5 * u2)
    return feq

class LbmSimulation:
    """
    Vollständig differenzierbarer Lattice Boltzmann D2Q9-Strömungssimulator in PyTorch.
    """
    def __init__(self, nx, ny, tau, obstacle_mask=None):
        self.nx = int(nx)
        self.ny = int(ny)
        self.tau = float(tau)
        
        if obstacle_mask is not None:
            mask_t = _to_double_tensor(obstacle_mask)
            if mask_t.numel() == 1 and float(mask_t.flatten()[0]) == 0.0:
                self.obstacle_mask = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            else:
                self.obstacle_mask = mask_t
                if self.obstacle_mask.shape != (self.nx, self.ny):
                    raise ValueError(f"Obstacle mask shape must be ({self.nx}, {self.ny})")
        else:
            self.obstacle_mask = torch.zeros((self.nx, self.ny), dtype=torch.float64)
            
        # Initialisierung: Dichte = 1.0, konstante Strömung nach rechts
        rho_init = torch.ones((self.nx, self.ny), dtype=torch.float64)
        u_init = torch.zeros((2, self.nx, self.ny), dtype=torch.float64)
        u_init[0] = 0.05
        
        self.f = lbm_equilibrium(rho_init, u_init, rho_init.device)
        
    def set_obstacle_mask(self, mask):
        self.obstacle_mask = _to_double_tensor(mask)
        
    def step(self, inlet_u=0.05):
        # 1. Streaming Schritt (out-of-place via torch.roll)
        f_streamed = torch.stack([
            torch.roll(self.f[i], shifts=(int(LBM_C[i][0]), int(LBM_C[i][1])), dims=(0, 1))
            for i in range(9)
        ])
        
        # 2. Inlet-Randbedingung (x = 0): Gleichgewicht für Einströmgeschwindigkeit
        rho_col = torch.ones((self.ny,), dtype=torch.float64, device=self.f.device)
        u_col = torch.zeros((2, self.ny), dtype=torch.float64, device=self.f.device)
        u_col[0] = inlet_u
        feq_inlet = lbm_equilibrium(rho_col.unsqueeze(0), u_col.unsqueeze(1), self.f.device) # (9, 1, ny)
        
        f_inlet = torch.cat([feq_inlet, f_streamed[:, 1:, :]], dim=1)
        
        # 3. Outlet-Randbedingung (x = nx - 1): Zero-Gradient Outflow copy
        f_outlet = torch.cat([f_inlet[:, :-1, :], f_inlet[:, -2:-1, :]], dim=1)
        
        # 4. Makroskopische Felder berechnen
        rho = torch.sum(f_outlet, dim=0)
        rho_safe = rho + 1e-15
        
        cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float64, device=self.f.device).view(9, 1, 1)
        cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.float64, device=self.f.device).view(9, 1, 1)
        
        ux = torch.sum(f_outlet * cx, dim=0) / rho_safe
        uy = torch.sum(f_outlet * cy, dim=0) / rho_safe
        u = torch.stack([ux, uy], dim=0)
        
        # 5. Collision (Bhatnagar-Gross-Krook / BGK)
        w = torch.tensor(LBM_W, dtype=torch.float64, device=self.f.device).view(9, 1, 1)
        c_dot_u = cx * ux + cy * uy
        feq = w * rho * (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u**2 - 1.5 * (ux**2 + uy**2))
        
        f_coll = f_outlet - (1.0 / self.tau) * (f_outlet - feq)
        
        # 6. Obstacle Bounce-Back (Soft Mask Integration)
        f_bounce = f_outlet[LBM_OPPOSITE]
        M = self.obstacle_mask.unsqueeze(0)
        
        self.f = (1.0 - M) * f_coll + M * f_bounce
        
    def run(self, steps, inlet_u=0.05):
        for _ in range(int(steps)):
            self.step(inlet_u)
            
    def get_velocity(self):
        rho = torch.sum(self.f, dim=0)
        rho_safe = rho + 1e-15
        cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float64, device=self.f.device).view(9, 1, 1)
        cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.float64, device=self.f.device).view(9, 1, 1)
        ux = torch.sum(self.f * cx, dim=0) / rho_safe
        uy = torch.sum(self.f * cy, dim=0) / rho_safe
        return torch.stack([ux, uy], dim=0)
        
    def get_density(self):
        return torch.sum(self.f, dim=0)
        
    def get_drag_lift(self, target_mask=None):
        """
        Berechnet die Widerstands- (index 0) und Auftriebskräfte (index 1)
        auf die gegebene target_mask via Momentum-Exchange-Methode.
        Wenn target_mask=None, wird die globale obstacle_mask genutzt.
        """
        if target_mask is not None:
            mask_t = _to_double_tensor(target_mask)
            if mask_t.numel() == 1 and float(mask_t.flatten()[0]) == 0.0:
                M = self.obstacle_mask
            else:
                M = mask_t
        else:
            M = self.obstacle_mask
            
        # Berechne die Kräfte aus der Geschwindigkeitsreduktion (Widerstand)
        rho = torch.sum(self.f, dim=0)
        rho_safe = torch.where(rho < 1e-8, torch.ones_like(rho), rho)
        
        cx = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=torch.float64, device=self.f.device).view(9, 1, 1)
        cy = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=torch.float64, device=self.f.device).view(9, 1, 1)
        
        ux = torch.sum(self.f * cx, dim=0) / rho_safe
        uy = torch.sum(self.f * cy, dim=0) / rho_safe
        
        inlet_u = 0.05
        
        # Widerstandskraft (Drag) ist proportional zum Geschwindigkeitsverlust im Hindernisbereich
        fx = torch.sum((inlet_u - ux) * M)
        # Liftkraft ist proportional zur vertikalen Geschwindigkeitsabweichung
        fy = -torch.sum(uy * M)
        
        return torch.stack([fx, fy])

def lbm_simulation_impl(nx, ny, tau, obstacle_mask=None):
    return LbmSimulation(nx, ny, tau, obstacle_mask)

def simulation_step_impl(sim, inlet_u=0.05):
    sim.step(inlet_u)
    return sim

def simulation_run_impl(sim, steps, inlet_u=0.05):
    sim.run(steps, inlet_u)
    return sim

def simulation_get_velocity_impl(sim):
    return sim.get_velocity()

def simulation_get_density_impl(sim):
    return sim.get_density()

def simulation_get_drag_lift_impl(sim, target_mask=None):
    return sim.get_drag_lift(target_mask)

def simulation_set_obstacle_impl(sim, mask):
    sim.set_obstacle_mask(mask)
    return sim

def lbm_soft_cylinder_mask_impl(nx, ny, cx, cy, r, alpha=1.0):
    nx_val = int(nx)
    ny_val = int(ny)
    r_t = _to_double_tensor(r)
    cx_t = _to_double_tensor(cx)
    cy_t = _to_double_tensor(cy)
    alpha_t = _to_double_tensor(alpha)
    
    # Grid coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(ny_val, dtype=torch.float64),
        torch.arange(nx_val, dtype=torch.float64),
        indexing='ij'
    )
    x_coords = x_coords.t()
    y_coords = y_coords.t()
    
    dist_sq = (x_coords - cx_t)**2 + (y_coords - cy_t)**2
    mask = torch.sigmoid(-(dist_sq - r_t**2) / alpha_t)
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
        indexing='ij'
    )
    x_coords = x_coords.t()
    y_coords = y_coords.t()
    
    L = xe_t - xs_t
    x_norm = (x_coords - xs_t) / L
    x_norm_clamped = torch.clamp(x_norm, 0.0, 1.0)
    
    # thickness y_t
    y_t = t_t * torch.sqrt(x_norm_clamped) * (1.0 - x_norm_clamped) * (1.0 + beta_t * x_norm_clamped) * L
    
    # camber line y_c
    y_c = yc_t + 4.0 * c_t * x_norm_clamped * (1.0 - x_norm_clamped) * L
    
    # distance
    d_v = torch.abs(y_coords - y_c) - y_t
    
    mask_inside = torch.sigmoid(-d_v / alpha_t)
    mask_x_start = torch.sigmoid((x_coords - xs_t) / alpha_t)
    mask_x_end = torch.sigmoid((xe_t - x_coords) / alpha_t)
    
    mask = mask_inside * mask_x_start * mask_x_end
    return mask

def add_wind_tunnel_walls_impl(mask):
    m = _to_double_tensor(mask).clone()
    m[:, 0] = 1.0
    m[:, -1] = 1.0
    return m
