# Dedekind Standard Library: Differentiable Space Physics & Orbital Mechanics
import torch
import math
import builtins

def compute_gravitational_acceleration(pos, masses, G, softening=1e-9):
    # pos: [N, dim]
    # masses: [N]
    # G: float
    # Return: [N, dim]
    N = pos.shape[0]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [N, N, dim]
    dist_sq = torch.sum(diff ** 2, dim=-1) + softening  # [N, N]
    dist_cubed = dist_sq ** 1.5  # [N, N]
    
    term = masses.unsqueeze(0).unsqueeze(-1) * diff / dist_cubed.unsqueeze(-1)  # [N, N, dim]
    acc = G * torch.sum(term, dim=1)  # [N, dim]
    return acc

def _get_space_val(v, dim=None):
    if isinstance(v, Quantity):
        if dim is not None:
            return _convert_to_base(v.value, v.unit, dim)
        return v.value
    return v

def n_body_simulate_impl(positions_init, velocities_init, masses, dt, n_steps, G=6.6743e-11, softening=1e-9):
    positions_init = _to_tensor(positions_init).double()
    velocities_init = _to_tensor(velocities_init).double()
    masses = _to_tensor(masses).double()
    
    G_val = float(_get_space_val(G))
    dt_val = float(_get_space_val(dt, "time"))
    n_steps = int(n_steps)
    softening_val = float(_get_space_val(softening))
    
    pos_list = [positions_init]
    vel_list = [velocities_init]
    
    curr_pos = positions_init
    curr_vel = velocities_init
    
    for _ in range(n_steps - 1):
        # RK4 step
        # k1
        k1_pos = curr_vel
        k1_vel = compute_gravitational_acceleration(curr_pos, masses, G_val, softening_val)
        
        # k2
        k2_pos_arg = curr_pos + 0.5 * dt_val * k1_pos
        k2_pos = curr_vel + 0.5 * dt_val * k1_vel
        k2_vel = compute_gravitational_acceleration(k2_pos_arg, masses, G_val, softening_val)
        
        # k3
        k3_pos_arg = curr_pos + 0.5 * dt_val * k2_pos
        k3_pos = curr_vel + 0.5 * dt_val * k2_vel
        k3_vel = compute_gravitational_acceleration(k3_pos_arg, masses, G_val, softening_val)
        
        # k4
        k4_pos_arg = curr_pos + dt_val * k3_pos
        k4_pos = curr_vel + dt_val * k3_vel
        k4_vel = compute_gravitational_acceleration(k4_pos_arg, masses, G_val, softening_val)
        
        # Update
        curr_pos = curr_pos + (dt_val / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
        curr_vel = curr_vel + (dt_val / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
        
        pos_list.append(curr_pos)
        vel_list.append(curr_vel)
        
    return {
        "positions": torch.stack(pos_list),
        "velocities": torch.stack(vel_list)
    }

def kepler_solve_impl(M, ecc, max_iter=10, tol=1e-12):
    M_val = _to_tensor(_get_space_val(M, "angle")).double()
    ecc_val = _to_tensor(_get_space_val(ecc)).double()
    
    E = M_val + ecc_val * torch.sin(M_val)
    for _ in range(int(max_iter)):
        f = E - ecc_val * torch.sin(E) - M_val
        df = 1.0 - ecc_val * torch.cos(E)
        df = torch.where(df == 0.0, torch.ones_like(df) * 1e-15, df)
        E = E - f / df
    return E

def kepler_to_cartesian_impl(a, ecc, inc, Omega, omega, nu, mu):
    a_val = _to_tensor(_get_space_val(a, "length")).double()
    ecc_val = _to_tensor(_get_space_val(ecc)).double()
    inc_val = _to_tensor(_get_space_val(inc, "angle")).double()
    Omega_val = _to_tensor(_get_space_val(Omega, "angle")).double()
    omega_val = _to_tensor(_get_space_val(omega, "angle")).double()
    nu_val = _to_tensor(_get_space_val(nu, "angle")).double()
    mu_val = _to_tensor(_get_space_val(mu)).double()
    
    numerator = a_val * (1.0 - ecc_val ** 2)
    denominator = 1.0 + ecc_val * torch.cos(nu_val)
    denominator = torch.where(denominator == 0.0, torch.ones_like(denominator) * 1e-15, denominator)
    r = numerator / denominator
    
    x_p = r * torch.cos(nu_val)
    y_p = r * torch.sin(nu_val)
    z_p = torch.zeros_like(r)
    
    term1 = a_val * (1.0 - ecc_val ** 2)
    term1 = torch.where(term1 <= 0.0, torch.ones_like(term1) * 1e-15, term1)
    val_v = torch.sqrt(mu_val / term1)
    
    vx_p = -val_v * torch.sin(nu_val)
    vy_p = val_v * (ecc_val + torch.cos(nu_val))
    vz_p = torch.zeros_like(r)
    
    r_p = torch.stack([x_p, y_p, z_p], dim=-1)
    v_p = torch.stack([vx_p, vy_p, vz_p], dim=-1)
    
    cos_O = torch.cos(Omega_val)
    sin_O = torch.sin(Omega_val)
    cos_w = torch.cos(omega_val)
    sin_w = torch.sin(omega_val)
    cos_i = torch.cos(inc_val)
    sin_i = torch.sin(inc_val)
    
    rx = r_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + r_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    ry = r_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + r_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    rz = r_p[..., 0] * (sin_w * sin_i) + r_p[..., 1] * (cos_w * sin_i)
    
    vx = v_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + v_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    vy = v_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + v_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    vz = v_p[..., 0] * (sin_w * sin_i) + v_p[..., 1] * (cos_w * sin_i)
    
    return {
        "position": torch.stack([rx, ry, rz], dim=-1),
        "velocity": torch.stack([vx, vy, vz], dim=-1)
    }

def kepler_to_cartesian_from_E_impl(a, ecc, inc, Omega, omega, E, mu):
    a_val = _to_tensor(_get_space_val(a, "length")).double()
    ecc_val = _to_tensor(_get_space_val(ecc)).double()
    inc_val = _to_tensor(_get_space_val(inc, "angle")).double()
    Omega_val = _to_tensor(_get_space_val(Omega, "angle")).double()
    omega_val = _to_tensor(_get_space_val(omega, "angle")).double()
    E_val = _to_tensor(_get_space_val(E, "angle")).double()
    mu_val = _to_tensor(_get_space_val(mu)).double()
    
    x_p = a_val * (torch.cos(E_val) - ecc_val)
    ecc_term = 1.0 - ecc_val ** 2
    ecc_term = torch.where(ecc_term < 0.0, torch.zeros_like(ecc_term), ecc_term)
    y_p = a_val * torch.sqrt(ecc_term) * torch.sin(E_val)
    z_p = torch.zeros_like(x_p)
    
    r = a_val * (1.0 - ecc_val * torch.cos(E_val))
    r = torch.where(r <= 0.0, torch.ones_like(r) * 1e-15, r)
    
    term_mu_a = mu_val * a_val
    term_mu_a = torch.where(term_mu_a < 0.0, torch.zeros_like(term_mu_a), term_mu_a)
    sqrt_mu_a = torch.sqrt(term_mu_a)
    
    vx_p = -sqrt_mu_a * torch.sin(E_val) / r
    vy_p = sqrt_mu_a * torch.sqrt(ecc_term) * torch.cos(E_val) / r
    vz_p = torch.zeros_like(x_p)
    
    r_p = torch.stack([x_p, y_p, z_p], dim=-1)
    v_p = torch.stack([vx_p, vy_p, vz_p], dim=-1)
    
    cos_O = torch.cos(Omega_val)
    sin_O = torch.sin(Omega_val)
    cos_w = torch.cos(omega_val)
    sin_w = torch.sin(omega_val)
    cos_i = torch.cos(inc_val)
    sin_i = torch.sin(inc_val)
    
    rx = r_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + r_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    ry = r_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + r_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    rz = r_p[..., 0] * (sin_w * sin_i) + r_p[..., 1] * (cos_w * sin_i)
    
    vx = v_p[..., 0] * (cos_O * cos_w - sin_O * sin_w * cos_i) + v_p[..., 1] * (-cos_O * sin_w - sin_O * cos_w * cos_i)
    vy = v_p[..., 0] * (sin_O * cos_w + cos_O * sin_w * cos_i) + v_p[..., 1] * (-sin_O * sin_w + cos_O * cos_w * cos_i)
    vz = v_p[..., 0] * (sin_w * sin_i) + v_p[..., 1] * (cos_w * sin_i)
    
    return {
        "position": torch.stack([rx, ry, rz], dim=-1),
        "velocity": torch.stack([vx, vy, vz], dim=-1)
    }

def cartesian_to_kepler_impl(r, v, mu):
    r_val = _to_tensor(r).double()
    v_val = _to_tensor(v).double()
    mu_val = _to_tensor(_get_space_val(mu)).double()
    
    is_batched = r_val.dim() > 1
    if not is_batched:
        r_val = r_val.unsqueeze(0)
        v_val = v_val.unsqueeze(0)
        
    r_mag = torch.norm(r_val, dim=-1, keepdim=True)
    v_mag = torch.norm(v_val, dim=-1, keepdim=True)
    
    h = torch.cross(r_val, v_val, dim=-1)
    h_mag = torch.norm(h, dim=-1, keepdim=True)
    h_mag = torch.where(h_mag == 0.0, torch.ones_like(h_mag) * 1e-15, h_mag)
    
    n = torch.stack([-h[..., 1], h[..., 0], torch.zeros_like(h[..., 0])], dim=-1)
    n_mag = torch.norm(n, dim=-1, keepdim=True)
    n_mag = torch.where(n_mag == 0.0, torch.ones_like(n_mag) * 1e-15, n_mag)
    
    r_dot_v = torch.sum(r_val * v_val, dim=-1, keepdim=True)
    r_mag_safe = torch.where(r_mag == 0.0, torch.ones_like(r_mag) * 1e-15, r_mag)
    e_vec = (1.0 / mu_val) * ((v_mag ** 2 - mu_val / r_mag_safe) * r_val - r_dot_v * v_val)
    ecc = torch.norm(e_vec, dim=-1, keepdim=True)
    ecc_safe = torch.where(ecc == 0.0, torch.ones_like(ecc) * 1e-15, ecc)
    
    energy = 0.5 * (v_mag ** 2) - mu_val / r_mag_safe
    a = -mu_val / (2.0 * energy)
    
    inc = torch.acos(torch.clamp(h[..., 2:3] / h_mag, -1.0, 1.0))
    
    Omega = torch.atan2(n[..., 1:2], n[..., 0:1])
    Omega = torch.where(n_mag < 1e-10, torch.zeros_like(Omega), Omega)
    Omega = torch.where(Omega < 0.0, Omega + 2.0 * math.pi, Omega)
    
    n_dot_e = torch.sum(n * e_vec, dim=-1, keepdim=True)
    omega = torch.acos(torch.clamp(n_dot_e / (n_mag * ecc_safe), -1.0, 1.0))
    omega = torch.where(e_vec[..., 2:3] < 0.0, 2.0 * math.pi - omega, omega)
    
    omega_equatorial = torch.atan2(e_vec[..., 1:2], e_vec[..., 0:1])
    omega_equatorial = torch.where(omega_equatorial < 0.0, omega_equatorial + 2.0 * math.pi, omega_equatorial)
    omega = torch.where(n_mag < 1e-10, omega_equatorial, omega)
    
    e_dot_r = torch.sum(e_vec * r_val, dim=-1, keepdim=True)
    nu = torch.acos(torch.clamp(e_dot_r / (ecc_safe * r_mag_safe), -1.0, 1.0))
    nu = torch.where(r_dot_v < 0.0, 2.0 * math.pi - nu, nu)
    
    if not is_batched:
        return {
            "a": a.squeeze(0).squeeze(0),
            "ecc": ecc.squeeze(0).squeeze(0),
            "inc": inc.squeeze(0).squeeze(0),
            "Omega": Omega.squeeze(0).squeeze(0),
            "omega": omega.squeeze(0).squeeze(0),
            "nu": nu.squeeze(0).squeeze(0)
        }
    else:
        return {
            "a": a.squeeze(-1),
            "ecc": ecc.squeeze(-1),
            "inc": inc.squeeze(-1),
            "Omega": Omega.squeeze(-1),
            "omega": omega.squeeze(-1),
            "nu": nu.squeeze(-1)
        }
