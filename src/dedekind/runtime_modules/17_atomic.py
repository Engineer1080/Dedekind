# Dedekind Standard Library: Atomic & Molecular Physics/Chemistry
import torch
import torch.fft
import math
import builtins

def _get_val(v):
    if hasattr(v, "value"):
        return v.value
    return v

# --- Molecular Dynamics & Chemistry ---

def _compute_lj_forces_and_pe(pos, box_size, epsilon, sigma, cutoff):
    N = pos.shape[0]
    # Pairwise differences: diff[i, j] = pos[i] - pos[j]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 3]
    
    # Minimum image convention for periodic boundary conditions
    diff = diff - box_size * torch.round(diff / box_size)
    
    dist_sq = torch.sum(diff ** 2, dim=-1)  # [N, N]
    mask = ~torch.eye(N, dtype=torch.bool, device=pos.device)
    
    # Avoid division by zero on diagonal
    dist_sq_safe = torch.where(mask, dist_sq, torch.ones_like(dist_sq))
    dist_safe = torch.sqrt(dist_sq_safe)
    
    s_over_d = sigma / dist_safe
    s_over_d_6 = s_over_d ** 6
    s_over_d_12 = s_over_d_6 ** 2
    
    # LJ force coefficient: F_ij = coeff * (x_i - x_j)
    coeff = (24.0 * epsilon / dist_sq_safe) * (2.0 * s_over_d_12 - s_over_d_6)
    coeff = torch.where((dist_safe < cutoff) & mask, coeff, torch.zeros_like(coeff))
    
    forces = torch.sum(coeff.unsqueeze(-1) * diff, dim=1)  # [N, 3]
    
    # Potential energy
    v_lj = 4.0 * epsilon * (s_over_d_12 - s_over_d_6)
    v_lj = torch.where((dist_safe < cutoff) & mask, v_lj, torch.zeros_like(v_lj))
    pe = 0.5 * torch.sum(v_lj)
    
    return forces, pe

def molecular_lj_simulate_impl(positions, velocities, masses, box_size, dt, steps, epsilon=1.0, sigma=1.0, cutoff=2.5, thermostat_tau=-1.0, target_temp=1.0):
    pos = _to_tensor(positions).double()
    vel = _to_tensor(velocities).double()
    m = _to_tensor(masses).double()
    
    box_size_val = _to_tensor(_get_val(box_size)).double()
    dt_val = _to_tensor(_get_val(dt)).double()
    steps = int(steps)
    eps_val = _to_tensor(_get_val(epsilon)).double()
    sig_val = _to_tensor(_get_val(sigma)).double()
    cut_val = _to_tensor(_get_val(cutoff)).double()
    tau_val = _to_tensor(_get_val(thermostat_tau)).double()
    t_target_val = _to_tensor(_get_val(target_temp)).double()
    
    if m.dim() == 0 or m.numel() == 1:
        m = m.expand(pos.shape[0])
    m_col = m.unsqueeze(-1)
    
    F, pe = _compute_lj_forces_and_pe(pos, box_size_val, eps_val, sig_val, cut_val)
    
    history_pos = [pos]
    history_vel = [vel]
    history_pe = [pe]
    
    ke = 0.5 * torch.sum(m_col * vel**2)
    temp = (2.0 * ke) / (3.0 * pos.shape[0])
    history_ke = [ke]
    history_temp = [temp]
    
    curr_pos = pos
    curr_vel = vel
    curr_F = F
    
    for _ in range(steps):
        # 1. Half-step velocity update
        curr_vel = curr_vel + 0.5 * (curr_F / m_col) * dt_val
        
        # 2. Full-step position update
        curr_pos = curr_pos + curr_vel * dt_val
        
        # 3. Apply periodic boundary conditions
        curr_pos = curr_pos - box_size_val * torch.floor(curr_pos / box_size_val)
        
        # 4. Compute new forces and potential energy
        curr_F, pe = _compute_lj_forces_and_pe(curr_pos, box_size_val, eps_val, sig_val, cut_val)
        
        # 5. Second half-step velocity update
        curr_vel = curr_vel + 0.5 * (curr_F / m_col) * dt_val
        
        # 6. Calculate instantaneous kinetic energy and temperature
        ke = 0.5 * torch.sum(m_col * curr_vel**2)
        temp = (2.0 * ke) / (3.0 * curr_pos.shape[0])
        
        # 7. Apply Berendsen thermostat if active
        if tau_val.item() > 0.0:
            factor = torch.sqrt(torch.clamp(1.0 + (dt_val / tau_val) * (t_target_val / (temp + 1e-12) - 1.0), min=0.1, max=10.0))
            curr_vel = curr_vel * factor
            ke = ke * (factor ** 2)
            temp = temp * (factor ** 2)
            
        history_pos.append(curr_pos)
        history_vel.append(curr_vel)
        history_pe.append(pe)
        history_ke.append(ke)
        history_temp.append(temp)
        
    return {
        "positions": torch.stack(history_pos),
        "velocities": torch.stack(history_vel),
        "potential_energies": torch.stack(history_pe),
        "kinetic_energies": torch.stack(history_ke),
        "temperatures": torch.stack(history_temp)
    }

def morse_potential_impl(r, De, a, re):
    r = _to_tensor(r).double()
    De = _to_tensor(De).double()
    a = _to_tensor(a).double()
    re = _to_tensor(re).double()
    return De * (1.0 - torch.exp(-a * (r - re))) ** 2

def molecular_distance_impl(pos1, pos2):
    p1 = _to_tensor(pos1).double()
    p2 = _to_tensor(pos2).double()
    return torch.sqrt(torch.sum((p1 - p2) ** 2) + 1e-12)

def molecular_angle_impl(pos1, pos2, pos3):
    p1 = _to_tensor(pos1).double()
    p2 = _to_tensor(pos2).double()
    p3 = _to_tensor(pos3).double()
    
    u = p1 - p2
    v = p3 - p2
    
    u_norm = torch.sqrt(torch.sum(u ** 2) + 1e-12)
    v_norm = torch.sqrt(torch.sum(v ** 2) + 1e-12)
    
    cos_theta = torch.dot(u, v) / (u_norm * v_norm)
    cos_theta_clamped = torch.clamp(cos_theta, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    return torch.acos(cos_theta_clamped)

def molecular_dihedral_impl(pos1, pos2, pos3, pos4):
    p1 = _to_tensor(pos1).double()
    p2 = _to_tensor(pos2).double()
    p3 = _to_tensor(pos3).double()
    p4 = _to_tensor(pos4).double()
    
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    b2_norm = b2 / torch.sqrt(torch.sum(b2 ** 2) + 1e-12)
    
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    
    n1_norm = n1 / torch.sqrt(torch.sum(n1 ** 2) + 1e-12)
    n2_norm = n2 / torch.sqrt(torch.sum(n2 ** 2) + 1e-12)
    
    m1 = torch.cross(n1_norm, b2_norm, dim=-1)
    
    x = torch.dot(n1_norm, n2_norm)
    y = torch.dot(m1, n2_norm)
    
    return torch.atan2(y, x)

# --- Crystallography & Structure Analysis ---

def cryst_symmetry_apply_impl(coords, R, t):
    c = _to_tensor(coords).double()
    R_t = _to_tensor(R).double()
    t_t = _to_tensor(t).double()
    
    is_1d = (c.dim() == 1)
    if is_1d:
        c = c.unsqueeze(0)
        
    # Apply Seitz matrix (R, t) to fractional coordinates
    res = torch.matmul(c, R_t.t()) + t_t
    res = torch.remainder(res, 1.0)
    
    if is_1d:
        res = res.squeeze(0)
    return res

def cryst_generate_equivalent_atoms_impl(coords, R_ops, t_ops):
    c = _to_tensor(coords).double()
    R_ops_t = _to_tensor(R_ops).double()
    t_ops_t = _to_tensor(t_ops).double()
    
    is_1d = (c.dim() == 1)
    if is_1d:
        c = c.unsqueeze(0)
        
    # Vectorized application of S operations to N atoms
    rotated = torch.einsum("nj,skj->snk", c, R_ops_t)
    translated = rotated + t_ops_t.unsqueeze(1)
    res = torch.remainder(translated, 1.0)
    
    if is_1d:
        res = res.squeeze(1)
    return res

def cryst_find_symmetries_impl(coords, elements, R_ops, t_ops, tol=1e-3):
    c = _to_tensor(coords).double()
    R_ops_t = _to_tensor(R_ops).double()
    t_ops_t = _to_tensor(t_ops).double()
    tol_val = float(_get_val(tol))
    
    if c.dim() == 1:
        c = c.unsqueeze(0)
        
    # Map elements to integer IDs
    if isinstance(elements, torch.Tensor):
        elem_ids = elements
    else:
        if hasattr(elements, "tolist"):
            el_list = elements.tolist()
        elif hasattr(elements, "value"):
            el_list = elements.value
        else:
            el_list = list(elements)
            
        unique_el = sorted(list(set(el_list)))
        mapping = {el: i for i, el in enumerate(unique_el)}
        elem_ids = torch.tensor([mapping[el] for el in el_list], device=c.device, dtype=torch.long)
        
    # Generate all symmetry-equivalent coordinates
    eq_coords = cryst_generate_equivalent_atoms_impl(c, R_ops_t, t_ops_t)
    
    # Pairwise periodic differences
    diff = eq_coords.unsqueeze(2) - c.unsqueeze(0).unsqueeze(1)
    diff = diff - torch.round(diff)
    
    # Distance squared
    dist_sq = torch.sum(diff ** 2, dim=-1)
    
    # Element matching mask
    eq_mask = (elem_ids.unsqueeze(1) == elem_ids.unsqueeze(0))
    eq_mask = eq_mask.unsqueeze(0)
    
    # Matches must satisfy periodic distance tolerance and element type equivalence
    valid_matches = (dist_sq < tol_val ** 2) & eq_mask
    
    # An operation is a valid symmetry if every mapped atom has a match
    atom_has_match = torch.any(valid_matches, dim=2)
    op_is_symmetry = torch.all(atom_has_match, dim=1)
    
    return op_is_symmetry

def cryst_structure_factor_atoms_impl(coords, scattering_factors, hkl_indices):
    c = _to_tensor(coords).double()
    sf = _to_tensor(scattering_factors).double()
    hkl = _to_tensor(hkl_indices).double()
    
    if c.dim() == 1:
        c = c.unsqueeze(0)
    if hkl.dim() == 1:
        hkl = hkl.unsqueeze(0)
        
    # Dot product of Miller indices and fractional coordinates
    angles = 2.0 * math.pi * torch.matmul(hkl, c.t())
    
    # Complex exponentials
    phases = torch.complex(torch.cos(angles), torch.sin(angles))
    
    # Broadcast scattering factors
    if sf.dim() == 1:
        sf_bc = sf.unsqueeze(0)
    elif sf.dim() == 2 and sf.shape[1] == hkl.shape[0]:
        sf_bc = sf.t()
    else:
        sf_bc = sf.view(1, -1)
        
    # Sum over atoms to get structure factor for each reflection
    F = torch.sum(sf_bc * phases, dim=1)
    return F

def cryst_structure_factor_density_impl(density_map):
    rho = _to_tensor(density_map)
    F = torch.fft.fftn(rho)
    return F
