# Thermal LBM in Dedekind — Convection and Heat Exchangers

This documentation describes the Thermal LBM solver in Dedekind: a
fully differentiable double-distribution solver for coupled
flow-heat problems.

## Mathematical Foundation

### Boussinesq Approximation

Incompressible flow with small thermal density anomaly:

```
∂u/∂t + (u·∇)u = -∇p/ρ₀ + ν∇²u - g·β·(T - T_ref)·ŷ
∂T/∂t + (u·∇)T = α·∇²T
∇·u = 0
```

with kinematic viscosity ν, thermal diffusivity α, thermal
expansion coefficient β, gravitational acceleration g, reference temperature T_ref.

### Double-Distribution LBM

- **Flow:** D2Q9 distribution f_i with BGK collision, relaxation time τ_u.
  ν = (τ_u - 0.5)/3 in lattice units.
- **Temperature:** D2Q5 distribution g_i (rest + 4 axial directions) as
  a passive scalar, relaxation time τ_T. α = (τ_T - 0.5)/3.
- **Coupling:** Boussinesq buoyancy as "shift-velocity" forcing in the
  flow equilibrium: u_eq_y = u_y + τ_u · F_y/ρ.

### Dimensionless Numbers

- **Rayleigh number:** Ra = g·β·ΔT·H³ / (ν·α)
  - Ra < 1708 (Ra_c): pure conduction, Nu = 1
  - Ra > 1708: convection rolls, Nu > 1 (Hopf bifurcation)
- **Prandtl number:** Pr = ν/α. For τ_u = τ_T, Pr = 1 (model default).
- **Nusselt number:** Nu = 1 + H/(α·ΔT) · ⟨u_y · (T - T_ref)⟩
  Measures the convective share of total heat transport relative to pure
  conduction.

## API

Activation: `use fluid_dynamics`

### Constructors

```dedekind
sim = lbm_thermal_simulation(nx, ny, tau_u, tau_T,
                              T_hot, T_cold, gravity_beta)
```

- `(nx, ny)`: grid size in lattice units.
- `tau_u, tau_T`: relaxation times (both > 0.5 for stability).
- `T_hot`: temperature at the lower wall (j=0).
- `T_cold`: temperature at the upper wall (j=ny-1).
- `gravity_beta`: product g·β. Controls the buoyancy strength.

With obstacle (Dirichlet-T):

```dedekind
sim = lbm_thermal_with_obstacle(nx, ny, tau_u, tau_T,
                                 T_hot, T_cold, gravity_beta,
                                 mask, T_obstacle)
```

- `mask`: soft mask (tensor (nx, ny) in [0, 1]) for obstacle cells.
- `T_obstacle`: fixed temperature of the obstacle (Dirichlet BC).

### Time Integration

```dedekind
thermal_step(sim)            // One time step
thermal_run(sim, n_steps)    // Multiple steps
```

### Fields & Diagnostics

```dedekind
T  = thermal_temperature(sim)   // (nx, ny) temperature field
u  = thermal_velocity(sim)      // (2, nx, ny) velocity field
Nu = thermal_nusselt(sim)       // scalar global Nusselt number
Ra = rayleigh_number(tau_u, tau_T, delta_T, gravity_beta, H)  // Helper
```

### Initialization with Perturbation

The default initial T distribution is linear (conductive). Float64 rounding noise
is often not enough to break symmetry — a deterministic perturbation
dramatically accelerates the onset of convection:

```dedekind
T_init = thermal_temperature(sim)
rows = linspace(0.0, nx - 1.0, nx)
cols = linspace(0.0, ny - 1.0, ny)
IX = outer(rows, 1.0 + 0.0 * cols)
IY = outer(1.0 + 0.0 * rows, cols)
pert = 0.05 * sin(2.0 * pi * IX / 25.0) * sin(pi * IY / (ny - 1.0))
thermal_set_temperature(sim, T_init + pert)
```

## Validation

- **Pure conduction (g·β = 0):** linear T profile, Nu = 1.000 exactly.
- **At g·β = 0:** velocity field remains strictly 0 (no drift).
- **Onset of convection:** Ra > 1708 produces Nu > 1, visible convection rolls.
- **Dirichlet obstacle:** T_obstacle is reached at the rod center (T ≈ T_obstacle).

See `tests/dedekind/thermal_lbm_test.ddk` (5 sub-tests).

## Examples

### Rayleigh-Bénard Convection

`examples/dedekind/engineering/lbm_rayleigh_benard.ddk` shows:

- Pure conduction (baseline): Nu = 1.000
- Convection at Ra ≈ 19440: Nu ≈ 1.81 (80 % additional heat transport)

### Heat Exchanger Geometry Study

`examples/dedekind/engineering/lbm_heat_exchanger.ddk` shows:

- Bénard convection cell with additional cooling rod (Dirichlet T_cold)
- Position of the rod (near hot wall, center, near cold wall) significantly
  influences the global Nu.
- Industrially relevant: reactor fuel rod arrangement, server room cooling,
  solar pond stratification, evaporator/condenser design.

## Differentiability

All input parameters (cy_position, r, T_obstacle, gravity_beta, τ_u, τ_T)
are PyTorch tensors and carry gradients. Optimization algorithms such as
Adam can navigate directly through hundreds of time steps to the optimal
geometry — without a separate adjoint solver.

Example pattern (Adam optimizes rod position):

```dedekind
fn nu_objective(params) {
    cy = params[0]
    M  = soft_cylinder_mask(nx, ny, cx_const, cy, r_const, 0.5)
    sim = lbm_thermal_with_obstacle(nx, ny, 0.6, 0.6,
                                     1.0, 0.0, 1e-4, M, 0.0)
    thermal_run(sim, 2000)
    return [-thermal_nusselt(sim)]   // negative → maximization
}

result = minimize(nu_objective, [25.0], "adam", 0.5, 30)
```

## Limitations & Extensions

- **2D only:** Currently only D2Q9/D2Q5; 3D (D3Q19/D3Q7) as a future
  extension possible.
- **Boussinesq only:** Strongly compressible thermal flows (e.g. shock waves)
  require a full energy model.
- **Adiabatic boundaries:** Currently only Dirichlet-T at obstacles. Neumann/
  adiabatic walls can be added as an extension.
