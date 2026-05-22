# Dedekind Standard Library: Differentiable Heat Transfer & Thermodynamics
import torch
import math
import builtins

class ThermalMesh2D:
    def __init__(self, nx, ny):
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_elem = self.nx * self.ny
        self.n_nodes = (self.nx + 1) * (self.ny + 1)

        import numpy as _np
        edof = _np.zeros((self.n_elem, 4), dtype=_np.int64)
        for ex in range(self.nx):
            for ey in range(self.ny):
                el = ex * self.ny + ey
                # Nodes mapping
                n1 = ex * (self.ny + 1) + ey
                n2 = (ex + 1) * (self.ny + 1) + ey
                n3 = (ex + 1) * (self.ny + 1) + (ey + 1)
                n4 = ex * (self.ny + 1) + (ey + 1)
                edof[el] = [n1, n2, n3, n4]
        self.edof = torch.from_numpy(edof)

        # Precompute flat indices for global stiffness assembly
        row_indices = self.edof.unsqueeze(2).repeat(1, 1, 4)
        col_indices = self.edof.unsqueeze(1).repeat(1, 4, 1)
        self.flat_indices = (row_indices * self.n_nodes + col_indices).flatten()


def thermal_mesh_2d_impl(nx, ny):
    return ThermalMesh2D(nx, ny)


def thermal_solve_2d_impl(mesh, conductivities, heat_sources, fixed_nodes, fixed_temps, k_min=1e-6):
    conductivities = _to_tensor(conductivities).double()
    device = conductivities.device
    
    n_nodes = mesh.n_nodes
    fixed_nodes = _to_tensor(fixed_nodes).long().to(device)
    fixed_temps = _to_tensor(fixed_temps).double().to(device)
    heat_sources = _to_tensor(heat_sources).double().to(device)
    
    # Enforce minimum conductivity
    k = torch.clamp(conductivities, min=k_min)
    
    # Local stiffness matrix K0 for unit square bilinear Q4 element
    K0 = torch.tensor([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0]
    ], dtype=torch.float64, device=device) / 6.0
    
    # Scale local matrices
    Ke = k[:, None, None] * K0[None, :, :]
    
    # Global assembly
    K_flat = torch.zeros(n_nodes * n_nodes, dtype=torch.float64, device=device)
    K_flat.index_add_(0, mesh.flat_indices.to(device), Ke.flatten())
    K = K_flat.view(n_nodes, n_nodes)
    
    # Boundary conditions
    fixed_set = set(int(n) for n in fixed_nodes)
    free_nodes = torch.tensor([n for n in range(n_nodes) if n not in fixed_set], dtype=torch.long, device=device)
    
    K_free = K[free_nodes][:, free_nodes]
    if len(fixed_nodes) > 0:
        F_free = heat_sources[free_nodes] - K[free_nodes][:, fixed_nodes] @ fixed_temps
    else:
        F_free = heat_sources[free_nodes]
        
    T_free = torch.linalg.solve(K_free, F_free)
    
    # Reconstruct temperature vector
    T = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    if len(fixed_nodes) > 0:
        T = T.scatter(0, fixed_nodes, fixed_temps)
    T = T.scatter(0, free_nodes, T_free)
    
    return T


def thermal_solve_transient_2d_impl(mesh, conductivities, capacities, initial_temps, heat_sources, fixed_nodes, fixed_temps, dt, steps, k_min=1e-6):
    conductivities = _to_tensor(conductivities).double()
    device = conductivities.device
    
    n_nodes = mesh.n_nodes
    fixed_nodes = _to_tensor(fixed_nodes).long().to(device)
    fixed_temps = _to_tensor(fixed_temps).double().to(device)
    
    k = torch.clamp(conductivities, min=k_min)
    c_e = _to_tensor(capacities).double().to(device)
    
    # Assemble K
    K0 = torch.tensor([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0]
    ], dtype=torch.float64, device=device) / 6.0
    Ke = k[:, None, None] * K0[None, :, :]
    
    K_flat = torch.zeros(n_nodes * n_nodes, dtype=torch.float64, device=device)
    K_flat.index_add_(0, mesh.flat_indices.to(device), Ke.flatten())
    K = K_flat.view(n_nodes, n_nodes)
    
    # Lumped mass matrix (diagonal)
    M = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    for i in range(4):
        M.index_add_(0, mesh.edof[:, i].to(device), c_e / 4.0)
        
    M_dt = M / float(dt)
    
    fixed_set = set(int(n) for n in fixed_nodes)
    free_nodes = torch.tensor([n for n in range(n_nodes) if n not in fixed_set], dtype=torch.long, device=device)
    
    T = _to_tensor(initial_temps).double().clone().to(device)
    if len(fixed_nodes) > 0:
        T[fixed_nodes] = fixed_temps
        
    Q = _to_tensor(heat_sources).double().to(device)
    steps = int(steps)
    if Q.dim() == 1:
        Q_all = Q.unsqueeze(0).repeat(steps, 1)
    else:
        Q_all = Q
        
    # K_star = K + diag(M_dt)
    K_star = K.clone()
    diag_indices = torch.arange(n_nodes, device=device)
    K_star[diag_indices, diag_indices] += M_dt
    
    K_star_free = K_star[free_nodes][:, free_nodes]
    
    for step in range(steps):
        Q_step = Q_all[step]
        F_star = Q_step + M_dt * T
        
        if len(fixed_nodes) > 0:
            F_star_free = F_star[free_nodes] - K_star[free_nodes][:, fixed_nodes] @ fixed_temps
        else:
            F_star_free = F_star[free_nodes]
            
        T_free = torch.linalg.solve(K_star_free, F_star_free)
        
        T = torch.zeros(n_nodes, dtype=torch.float64, device=device)
        if len(fixed_nodes) > 0:
            T = T.scatter(0, fixed_nodes, fixed_temps)
        T = T.scatter(0, free_nodes, T_free)
        
    return T


def topo_opt_thermal_oc_2d_impl(mesh, heat_sources, fixed_nodes, fixed_temps, volfrac, steps=50, penal=3.0, rmin=1.5, k0=1.0, kmin=1e-6):
    import torch.nn.functional as F
    heat_sources = _to_tensor(heat_sources).double()
    device = heat_sources.device
    nx, ny = mesh.nx, mesh.ny
    
    steps = int(steps)
    volfrac = float(volfrac)
    penal = float(penal)
    rmin = float(rmin)
    k0 = float(k0)
    kmin = float(kmin)
    
    # Initial design densities
    x = torch.full((ny, nx), volfrac, dtype=torch.float64, device=device)
    vol_target = volfrac * nx * ny
    
    # Prepare filtering kernel
    r_int = int(math.floor(rmin))
    size = 2 * r_int + 1
    kernel = torch.zeros((size, size), dtype=torch.float64, device=device)
    for i in range(size):
        for j in range(size):
            dx = i - r_int
            dy = j - r_int
            dist = math.sqrt(dx*dx + dy*dy)
            kernel[i, j] = builtins.max(0.0, rmin - dist)
            
    w = kernel.view(1, 1, size, size)
    ones = torch.ones((1, 1, ny, nx), dtype=torch.float64, device=device)
    norm = F.conv2d(ones, w, padding=r_int)
    
    # Local stiffness matrix K0 for unit square element
    K0 = torch.tensor([
        [ 4.0, -1.0, -2.0, -1.0],
        [-1.0,  4.0, -1.0, -2.0],
        [-2.0, -1.0,  4.0, -1.0],
        [-1.0, -2.0, -1.0,  4.0]
    ], dtype=torch.float64, device=device) / 6.0
    
    for _ in range(steps):
        # Design density filter
        x_4d = x.view(1, 1, ny, nx)
        x_phys = (F.conv2d(x_4d, w, padding=r_int) / norm).view(-1)
        
        # Finite element steady-state thermal solve
        # Thermal conductivity interpolated with SIMP-like scheme: k = kmin + x_phys^penal * (k0 - kmin)
        k_interp = kmin + (x_phys ** penal) * (k0 - kmin)
        T = thermal_solve_2d_impl(mesh, k_interp, heat_sources, fixed_nodes, fixed_temps, k_min=kmin)
        
        # Thermal energy of elements: T_e^T @ K0 @ T_e
        T_e = T[mesh.edof.to(device)]
        element_energy = torch.sum((T_e @ K0) * T_e, dim=1)
        
        # Compliance sensitivities
        dc = -penal * (x_phys ** (penal - 1.0)) * (k0 - kmin) * element_energy
        
        # Adjoint filtering of sensitivities
        dc_phys_4d = dc.view(1, 1, ny, nx)
        dc_design = F.conv2d(dc_phys_4d / norm, w, padding=r_int).view(ny, nx)
        
        # Bisection search for Lagrange Multiplier lambda
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        
        x_new = x.clone()
        for _ in range(100):
            if (l2 - l1) / (l1 + 1e-12) <= 1e-4:
                break
            lmid = 0.5 * (l1 + l2)
            B = -dc_design / lmid
            B = torch.clamp(B, min=1e-12)
            B_eta = torch.sqrt(B)
            
            x_proposal = torch.clamp(
                x * B_eta,
                min=torch.max(torch.tensor(1e-9, device=device), x - move),
                max=torch.min(torch.tensor(1.0, device=device), x + move)
            )
            
            if x_proposal.sum().item() > vol_target:
                l1 = lmid
            else:
                l2 = lmid
            x_new = x_proposal
            
        x = x_new
        
    # Return physical densities
    x_4d = x.view(1, 1, ny, nx)
    x_phys = (F.conv2d(x_4d, w, padding=r_int) / norm).view(-1)
    return x_phys


def print_thermal_topology_2d_impl(densities, nx, ny):
    import numpy as _np
    import builtins
    x = _to_tensor(densities).detach().cpu().numpy()
    nx = int(nx)
    ny = int(ny)
    
    grid = _np.zeros((ny, nx))
    for ex in range(nx):
        for ey in range(ny):
            el = ex * ny + ey
            if el < len(x):
                grid[ey, ex] = x[el]
                
    chars = [" ", "░", "▒", "▓", "█"]
    builtins.print("+" + "-" * nx + "+")
    for ey in range(ny):
        row_str = "|"
        for ex in range(nx):
            val = grid[ey, ex]
            val = builtins.min(builtins.max(val, 0.0), 1.0)
            idx = int(builtins.round(val * 4))
            row_str += chars[idx]
        row_str += "|"
        builtins.print(row_str)
    builtins.print("+" + "-" * nx + "+")
