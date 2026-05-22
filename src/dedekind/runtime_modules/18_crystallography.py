# Dedekind Standard Library: Crystallography & Differentiable Structure Analysis
import torch
import torch.fft
import math
import builtins

def _get_val(v):
    if hasattr(v, "value"):
        return v.value
    return v

def cryst_symmetry_apply_impl(coords, R, t):
    c = _to_tensor(coords).double()
    R_t = _to_tensor(R).double()
    t_t = _to_tensor(t).double()
    
    is_1d = (c.dim() == 1)
    if is_1d:
        c = c.unsqueeze(0)
        
    # Apply Seitz matrix (R, t) to fractional coordinates
    # c @ R^T + t is equivalent to R @ c_i + t for each row
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
    # c: [N, 3], R_ops_t: [S, 3, 3], t_ops_t: [S, 3]
    # rotated: [S, N, 3]
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
        # elements could be a list of strings, integers, etc.
        if hasattr(elements, "tolist"):
            el_list = elements.tolist()
        elif hasattr(elements, "value"):
            el_list = elements.value
        else:
            el_list = list(elements)
            
        unique_el = sorted(list(set(el_list)))
        mapping = {el: i for i, el in enumerate(unique_el)}
        elem_ids = torch.tensor([mapping[el] for el in el_list], device=c.device, dtype=torch.long)
        
    # Generate all symmetry-equivalent coordinates: [S, N, 3]
    eq_coords = cryst_generate_equivalent_atoms_impl(c, R_ops_t, t_ops_t)
    
    # Pairwise periodic differences: [S, N, N, 3]
    diff = eq_coords.unsqueeze(2) - c.unsqueeze(0).unsqueeze(1)
    diff = diff - torch.round(diff)
    
    # Distance squared: [S, N, N]
    dist_sq = torch.sum(diff ** 2, dim=-1)
    
    # Element matching mask: [N, N] -> [1, N, N]
    eq_mask = (elem_ids.unsqueeze(1) == elem_ids.unsqueeze(0))
    eq_mask = eq_mask.unsqueeze(0)
    
    # Matches must satisfy periodic distance tolerance and element type equivalence
    valid_matches = (dist_sq < tol_val ** 2) & eq_mask
    
    # An operation is a valid symmetry if every mapped atom has a match
    atom_has_match = torch.any(valid_matches, dim=2) # [S, N]
    op_is_symmetry = torch.all(atom_has_match, dim=1) # [S]
    
    return op_is_symmetry

def cryst_structure_factor_atoms_impl(coords, scattering_factors, hkl_indices):
    c = _to_tensor(coords).double()
    sf = _to_tensor(scattering_factors).double()
    hkl = _to_tensor(hkl_indices).double()
    
    if c.dim() == 1:
        c = c.unsqueeze(0)
    if hkl.dim() == 1:
        hkl = hkl.unsqueeze(0)
        
    # Dot product of Miller indices and fractional coordinates: [M, N]
    angles = 2.0 * math.pi * torch.matmul(hkl, c.t())
    
    # Complex exponentials: [M, N]
    phases = torch.complex(torch.cos(angles), torch.sin(angles))
    
    # Broadcast scattering factors
    if sf.dim() == 1:
        sf_bc = sf.unsqueeze(0)
    elif sf.dim() == 2 and sf.shape[1] == hkl.shape[0]:
        sf_bc = sf.t()
    else:
        sf_bc = sf.view(1, -1)
        
    # Sum over atoms to get structure factor for each reflection: [M]
    F = torch.sum(sf_bc * phases, dim=1)
    return F

def cryst_structure_factor_density_impl(density_map):
    rho = _to_tensor(density_map)
    # Perform Fast Fourier Transform (FFT) over all dimensions of the grid
    F = torch.fft.fftn(rho)
    return F
