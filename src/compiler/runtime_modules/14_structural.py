# Dedekind Standard Library: Structural Mechanics & Topology Optimization
import torch
import math
import builtins

class StructuralMesh2D:
    def __init__(self, nx, ny):
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_elem = self.nx * self.ny
        self.n_nodes = (self.nx + 1) * (self.ny + 1)
        self.n_dof = 2 * self.n_nodes

        # Generate element degree-of-freedom indices
        import numpy as _np
        edof = _np.zeros((self.n_elem, 8), dtype=_np.int64)
        for ex in range(self.nx):
            for ey in range(self.ny):
                el = ex * self.ny + ey
                # Nodes mapping
                n1 = ex * (self.ny + 1) + ey
                n2 = (ex + 1) * (self.ny + 1) + ey
                n3 = (ex + 1) * (self.ny + 1) + (ey + 1)
                n4 = ex * (self.ny + 1) + (ey + 1)
                edof[el] = [
                    2 * n1, 2 * n1 + 1,
                    2 * n2, 2 * n2 + 1,
                    2 * n3, 2 * n3 + 1,
                    2 * n4, 2 * n4 + 1
                ]
        self.edof = torch.from_numpy(edof)

        # Precompute flat indices for stiffness matrix assembly
        row_indices = self.edof.unsqueeze(2).repeat(1, 1, 8)
        col_indices = self.edof.unsqueeze(1).repeat(1, 8, 1)
        self.flat_indices = (row_indices * self.n_dof + col_indices).flatten()


def _get_k0(nu=0.3):
    # Element stiffness matrix for bilinear Q4 plane stress square element
    k = [
        0.5 - nu / 6.0,
        0.125 + nu / 8.0,
        -0.25 - nu / 12.0,
        -0.125 + 0.375 * nu,
        -0.25 + nu / 12.0,
        -0.125 - nu / 8.0,
        nu / 6.0,
        0.125 - 0.375 * nu
    ]
    K1 = torch.tensor([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ], dtype=torch.float64)

    # Alternate sign matrix for K2
    S = torch.tensor([
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0]
    ], dtype=torch.float64)

    K2 = K1 * S
    K0 = (K1 + K2 * nu) / (1.0 - nu**2)
    return K0


def structural_mesh_2d_impl(nx, ny):
    return StructuralMesh2D(nx, ny)


def structural_solve_2d_impl(mesh, densities, loads, fixed_dofs, E0=1.0, Emin=1e-9, nu=0.3, penal=3.0):
    loads = _to_tensor(loads).double()
    device = loads.device
    n_dof = mesh.n_dof
    
    fixed_dofs = _to_tensor(fixed_dofs).long().to(device)
    
    # Material interpolation (SIMP)
    x = _to_tensor(densities).double().to(device)
    E = Emin + (x ** penal) * (E0 - Emin)
    
    # Local stiffness matrix
    K0 = _get_k0(nu).to(device)
    
    # Global assembly
    Ke_all = E[:, None, None] * K0[None, :, :]
    
    K_flat = torch.zeros(n_dof * n_dof, dtype=torch.float64, device=device)
    K_flat.index_add_(0, mesh.flat_indices.to(device), Ke_all.flatten())
    K = K_flat.view(n_dof, n_dof)
    
    # Boundary conditions via submatrix partitioning
    fixed_set = set(int(d) for d in fixed_dofs)
    free_dofs = torch.tensor([d for d in range(n_dof) if d not in fixed_set], dtype=torch.long, device=device)
    
    K_free = K[free_dofs][:, free_dofs]
    F_free = loads[free_dofs]
    
    # Solve system differentiably
    U_free = torch.linalg.solve(K_free, F_free)
    
    # Rebuild full displacement vector
    U = torch.zeros(n_dof, dtype=torch.float64, device=device)
    U = U.scatter(0, free_dofs, U_free)
    return U


def structural_compliance_2d_impl(mesh, densities, loads, fixed_dofs, E0=1.0, Emin=1e-9, nu=0.3, penal=3.0):
    loads_t = _to_tensor(loads).double()
    U = structural_solve_2d_impl(mesh, densities, loads_t, fixed_dofs, E0, Emin, nu, penal)
    return torch.dot(U, loads_t)


def topo_opt_oc_2d_impl(mesh, loads, fixed_dofs, volfrac, steps=50, penal=3.0, rmin=1.5):
    import torch.nn.functional as F
    loads = _to_tensor(loads).double()
    device = loads.device
    nx, ny = mesh.nx, mesh.ny
    
    steps = int(steps)
    volfrac = float(volfrac)
    penal = float(penal)
    rmin = float(rmin)
    
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
    
    # Local stiffness matrix
    K0 = _get_k0(nu=0.3).to(device)
    
    for _ in range(steps):
        # Design density filter
        x_4d = x.view(1, 1, ny, nx)
        x_phys = (F.conv2d(x_4d, w, padding=r_int) / norm).view(-1)
        
        # Finite element analysis
        U = structural_solve_2d_impl(mesh, x_phys, loads, fixed_dofs, penal=penal)
        
        # Strain energy of elements
        U_e = U[mesh.edof.to(device)]
        strain_energy = torch.sum((U_e @ K0) * U_e, dim=1)
        
        # Compliance sensitivities
        dc = -penal * (x_phys ** (penal - 1.0)) * strain_energy
        
        # Adjoint filtering of sensitivities
        dc_phys_4d = dc.view(1, 1, ny, nx)
        dc_design = F.conv2d(dc_phys_4d / norm, w, padding=r_int).view(ny, nx)
        
        # Bisection search for Lagrange Multiplier lambda
        l1 = 0.0
        l2 = 1e9
        move = 0.2
        
        x_new = x.clone()
        # Cap bisection iterations at 100 to avoid infinite loops
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


def print_structural_topology_2d_impl(densities, nx, ny):
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

