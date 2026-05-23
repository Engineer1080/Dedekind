# Differentiable CFD & Shape Optimization in Dedekind

This document describes the CFD module of Dedekind, with particular focus
on the differentiable shape parametrization for aerodynamic and
hydrodynamic optimization.

## Overview

Dedekind combines a complete Lattice-Boltzmann solver (D2Q9) with
PyTorch autograd. Shape parameters such as radius, semi-axes, or Fourier
coefficients are native tensors — the gradient of drag or lift values with respect to these
parameters falls out automatically from the forward pass. This eliminates the separate
adjoint solver required in OpenFOAM, SU2, or dolfin-adjoint.

Activation: `use fluid_dynamics`

## Shape Parametrizations

### 1. Soft Cylinder — `soft_cylinder_mask(nx, ny, cx, cy, r, alpha)`

Sigmoid mask around a circle with radius `r`. Differentiable in `cx, cy, r`.
Suitable for classical Karman flow and simple drag studies.

### 2. Soft Ellipse — `soft_ellipse_mask(nx, ny, cx, cy, a, b, alpha)`

Sigmoid mask around an ellipse with semi-axes `a` (x) and `b` (y). Differentiable
in `cx, cy, a, b`. With volume constraint `a·b = const` the
optimization reduces to a single aspect-ratio parameter — minimal search space,
fast Adam run.

Demo: `examples/dedekind/engineering/lbm_shape_optimization.ddk`. Reduces
drag by ~17 % compared to the circle through stretching in the flow direction.

### 3. Soft Airfoil — `soft_airfoil_mask(nx, ny, t, c, beta, x_start, x_end, y_center, alpha)`

NACA-like profile description with thickness (`t`), camber (`c`) and
trailing-edge profiling (`beta`) parameters. Suitable as a starting point for
classical wings.

### 4. Fourier Shape — `fourier_shape_mask(nx, ny, cx, cy, r0, a_coeffs, b_coeffs, alpha)`

Universal 2D topology description via Fourier series of the boundary contour:

```
r(θ) = r0 · (1 + Σ_{k=1..K} a_k·cos(k·θ) + b_k·sin(k·θ))
```

- `a_coeffs`, `b_coeffs`: 1D tensors or lists with K entries — one
  degree of freedom for each cos and sin mode.
- Differentiable in `cx, cy, r0` and every coefficient.
- Methodologically, this corresponds to "Class-Shape-Transformation" (CST), as it
  is established in the aerospace industry for aerodynamic shape optimization.

Smooth, arbitrarily complex topologies are already covered with K=4 harmonics
(8 parameters): teardrop, wing, tear-drop, bean shapes.

Demo: `examples/dedekind/engineering/lbm_fourier_airfoil_optimization.ddk`.
Adam differentiates through 150 LBM steps and reduces the drag by
~15–25 % by optimizing all eight coefficients simultaneously.

## Workflow for Shape Optimization

```dedekind
use fluid_dynamics

fn drag_of_shape(params) {
    a = [params[0], params[1], params[2], params[3]]
    b = [params[4], params[5], params[6], params[7]]
    M = fourier_shape_mask(80, 40, 25.0, 20.0, 4.0, a, b, 0.4)
    sim = lbm_simulation_full(80, 40, 0.7, M, 0.06)
    simulation_run(sim, 150, 0.06)
    F = simulation_get_drag_lift(sim)
    return [F[0]]
}

// Adam optimizer differentiates via autograd through the entire pipeline
x0     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
result = minimize(drag_of_shape, x0, "adam", 0.05, 20)
```

## Validation

- **Autograd vs. central finite differences**: relative error < 10⁻⁴
  (see `tests/dedekind/fluid_dynamics_test.ddk::test_autograd_validates_against_fd`).
- **Karman benchmark**: Strouhal number St ≈ 0.167 at Re=100 (literature 0.16–0.17).
- **MRT stability at Re=2000**: BGK diverges, MRT stays finite.

## Tests

- `tests/dedekind/fluid_dynamics_test.ddk` — 15 sub-tests for LBM core functions,
  unit-aware API, MRT, hard/soft bounce-back, autograd validation.
- `tests/dedekind/lbm_shape_opt_test.ddk` — ellipse mask + 1-parameter optimization.
- `tests/dedekind/lbm_fourier_shape_test.ddk` — Fourier mask + multi-parameter optimization,
  teardrop asymmetry verification, multi-parameter autograd vs. FD.
- `tests/dedekind/ns_ibm_test.ddk` — Chorin NS solver + IBM Brinkman penalization.

## Industrial Relevance

The Fourier shape parametrization is the established method for:

- **Aviation:** Class-Shape-Transformation in the aerospace industry for
  wing and body optimization.
- **Automotive:** Body drag minimization; flow-separation avoidance
  at side mirrors and tailgates.
- **Shipbuilding:** Hull profiles, stem contour, rudder blade design.
- **Wind energy:** Rotor blade profiles with variable spanwise stations.
- **Building physics:** Aerodynamics of high-rises, bridge piers under
  Kármán vortex resonance.

Dedekind offers this methodology natively in a single DSL: no coupling
between mesher, solver, and adjoint module, no script pipeline of
Python/MATLAB/OpenFOAM/SU2 — one file, one `minimize(...)` call.
