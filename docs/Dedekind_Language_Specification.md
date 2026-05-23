# Dedekind — A Modern Programming Language for Machine Learning and Scientific Computing

**Language Specification v3.0**  
Mario Michael Heinrich · github.com/Engineer1080  
Last revised: May 2026 · Tracks v0.6 (Physical Units), v0.7 (ODE), v0.8 (Probabilistic, PDE), v0.9 (Distributions, Integration), v0.9.1 (Documentation, Run-Examples), v0.9.2 (pi, e, CODATA), v0.9.3 (Uncertainty, Fitting), v0.9.4 (HMC, LaTeX-Export), v0.9.5 (Better Error Messages, Compile-time Units), v0.9.6 (Math Functions), v0.9.7 (Chemistry/Biology: mol, L, M, ppm), v0.9.8 (Convenience, Elements, File I/O, Network, JSON), v1.0.0 (Release), v1.0.1–v1.0.6 (Patch), v1.2.6 (Angle rad/deg, deg_to_rad, rad_to_deg), v1.2.7 (Dirichlet Distribution, dirichlet_function), v1.2.8 (Dedekind Cuts, Dedekind Rings, Riemann-Zeta, Riemann Sums), v1.2.9 (Absolute Value Bars, Solids of Revolution, Logical Operators), v1.3.0 (integrate_sym, Lagrange/Hamilton, Lotka-Volterra, Chemical Equilibrium), v1.3.1 (Medicine, Pharmacology, Epidemiology), v1.4.0 (Module System `use`, Seed/data_hash, DataFrame+CSV/Parquet/HDF5/NetCDF, Unit-aware Plots, `@units` Signatures with `fn f(x: [m]) -> [J]`, Dict Literals), v1.5.0 (benchmark/profile/time_block, JIT (torch.compile), SDE Solve (Euler-Maruyama, Milstein), Least Squares, Constrained Minimization, MILP, FEM Primitives mesh_unit_square/fem_assemble_*/fem_poisson_2d, `arange` int64), v1.6.0 (solve_sym/simplify_sym/series, CG/GMRES/BiCGSTAB + Jacobi/ILU Preconditioner, Export Notebook (HTML/MD), print_table (markdown/latex/csv/plain) with UncertainQuantity-±), v1.7.0 (Standard Library Modules physics/stats/chemistry/biology/math/ml via `use`, Custom Units `unit NAME = FACTOR[basis]` with Chaining & Compile-time Registration, Quantity Comparison Operators `<` `<=` `>` `>=` `==` `!=` with Auto-Conversion), v1.8.0 (Source Mapping for Runtime Errors — Tracebacks show `.ddk` lines + Code Excerpt; `pyimport <mod[.sub]> [as alias]` escape hatch to PyPI; `dedekind_exec` Helper in `compiler.py`), v1.8.1 (Purity Check: `print`/`plot`/`write_file`/`http_*`/I/O Built-ins in `jit`/`grad`/`fit`/`metropolis`/`hmc`/`sde_solve` arguments are transitively detected and blocked at compile time; Opt-out via `--no-purity-check`), v1.9.0 (Shape Annotations `Scalar`/`Vector[n]`/`Matrix[m,n]`/`Tensor[...]` with Symbolic Dimensions, bound and consistency-checked per call; `unwrap(x)` strips Quantity/UncertainQuantity wrapper for Hot-path Performance), v1.10.0 (PINN Primitive `partial(u, x, order=n)` for ∂u/∂x via Autograd, complementing `grad(fn, x)`; `fit()` hardening for PINN data lists with mixed shapes and collocation grad reset), v1.11.0 (`graph_laplacian(adj, normalized=False)` for combinatorial and normalized Laplacian matrix; dense/sparse input, directly usable in CG/GMRES/BiCGSTAB/EIGH), v1.12.0 (`Graph[N, E]` shape annotation for torch_geometric.data.Data-like objects; symbolic node/edge consistency; combinable with unit annotations on other arguments), v1.13.0 (Declarative MILP DSL via `Variable(name, lower, upper, integer)` and `optimize_milp(objective, constraints, sense)` with Operator Overloading and unit-aware constraints), v1.14.0 (Molecular Dynamics bridge `md_simulate_lj(...)` via OpenMM with dimension-validated force field parameters; new MD units nm/Angstrom/pm, amu, fs/ps/ns, and new dimension molar_energy with kJ/mol/kcal/mol/eV/Hartree), v1.15.0 (`LabeledTensor[lat, lon, time]` shape annotation for xarray.DataArray; `labeled_tensor(data, dims, coords, attrs)` for construction; axis name validation order-irrelevant), v1.16.0 (Bioinformatics Quick-Wins: `Sequence[DNA]/Sequence[RNA]/Sequence[Protein]` with Alphabet Validation, native bio built-ins gc_content/transcribe/translate/reverse_complement/k_mer_count, Cheminformatik via pyimport rdkit with smiles_descriptors/lipinski_rule_of_five), v1.17.0 (Language maturity: `try { ... } catch e { ... }` error handling, Python-style slicing `x[a:b]/x[:b]/x[a:]/x[::s]/x[a:b:s]/x[:]`), v1.18.0 (3D Geometric/Clifford Algebra G(3,0) native: `MultiVector` class with 8 components, constructors `scalar/vector/bivector/pseudoscalar/rotor`, operations `wedge/dot/reverse/grade/norm`, `rotate(v, R)` sandwich product), v1.19.0 (Multi-File Modules: `use foo.bar.baz` -> `modules/foo/bar/baz.ddk`; `pub fn` for export control with backward-compatible legacy mode), v1.20.0 (Generics / Type Parameters `fn name<T, U, ...>(args)` with polymorphic unit variables, auto-conversion on compatible dimension, enforced by the compiler — unlike Python's purely documentation-only `typing.TypeVar`), v1.21.0 (Quantum Computing Bridge: native QuantumCircuit class, pure Statevector simulator, Bell/GHZ/Grover convenience, VQE optimizer, Quantum Information utilities fidelity/entropy_von_neumann/schmidt_rank, Shape Annotations Qubit[N]/StateVec[N], units eV/meV/THz, optional Qiskit export), v2.0.0 (Release, Code Cleanup & Unification, Geosciences & Meteorology), v2.1.0 (Electrical Engineering & Control Theory)


---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Philosophy](#2-design-philosophy)
3. [Core Features](#3-core-features)
4. [Syntax Overview](#4-syntax-overview)
5. [Execution Modifiers](#5-execution-modifiers)
6. [Type System](#6-type-system)
7. [Built-in Algorithms](#7-built-in-algorithms)
8. [Parallelization](#8-parallelization)
9. [GPU/TPU Integration](#9-gputpu-integration)
10. [Standard Library](#10-standard-library)
11. [Use Cases](#11-use-cases)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [Technical Foundation](#13-technical-foundation)
14. [Conclusion](#14-conclusion)
15. [**Physical Units and Universal Constants (v0.6)**](#15-physical-units-and-universal-constants-v06) (incl. §15.7 ODE, §15.8 Probabilistic, §15.9 PDE, §15.10 Integration & Math v0.9, ..., §15.27 Quantum v1.21, §15.28 Geosciences v2.0, §15.29 EE & Control v2.1, §15.30 Robotics v2.1, §15.31 Space Physics v2.0, §15.32 Structural Mechanics v2.0, §15.33 Heat Transfer v2.0, §15.34 CFD v2.0, §15.35 DSP v2.0)

---

## 1. Introduction

Dedekind is a modern, high-performance programming language designed specifically for compute-intensive workloads in machine learning and scientific computing. Named after Richard Dedekind, whose work on real numbers (Dedekind cuts) and algebraic structures laid foundations for rigorous mathematical computation, the language embodies the principle of precise, structured transformation of data through parallel computation.

Unlike general-purpose languages retrofitted with parallel computing capabilities, Dedekind is built from the ground up with GPU/TPU acceleration and automatic parallelization as core features.

## 2. Design Philosophy

- **Parallel by Default**: Operations are automatically parallelized unless explicitly marked sequential.
- **Hardware Agnostic**: Write once, run efficiently on CPU, GPU, or TPU.
- **Readable Imperative Syntax**: Familiar syntax for easier adoption.
- **Smart Compilation**: Compiler makes intelligent decisions about execution strategy.
- **Explicit Control When Needed**: Developers can override defaults with simple modifiers (e.g. `.gpu()`, `.cpu()`).

## 3. Core Features

- Automatic Parallelization, Execution Modifiers (`.gpu()`, `.cpu()`), Built-in Algorithms
- **Ricci Calculus**: Native index notation (`A^ij * B_jk`) for Einstein summation.
- **Sparse Tensors**: `.sparse()` support for FEM/CFD simulations.
- **Fundamental Constants**: Native access to `c`, `G`, `h`, `k_B`, `k_e` as **Quantity** with SI units (see §15).
- **Physical Units**: Literals with units (`10[m]`, `5[m/s]`, `1.0[kg]`); arithmetic combines units; display simplified to J, N where applicable.
- **4D Rotational Math**: Native Quaternion support (`i`, `j`, `k` suffixes) and `.rotate()` method; unary minus supported (`-1.0 + 0i`).
- AOT Compilation (MLIR/LLVM) *(planned, see Roadmap Phase 16; current implementation is a text-only prototype)*, IDE Integration (v0.6).

## 4. Syntax Overview

Variables, functions (`fn name(args) { ... }`), control flow (`if`/`else`, `for`, `while`), and expression modifiers. **Absolute value bars:** `|expr|` is syntactic sugar for `abs(expr)`; e.g. `x = |-1|` yields 1. See implementation and examples in `examples/dedekind/`.

## 5. Execution Modifiers

Modifiers at the end of operation chains: `.gpu()`, `.cpu()`, `.single()`. Only one modifier per chain.

## 6. Type System

Type inference; primitive and collection types; first-class vectors/matrices and Quaternions.

## 7. Built-in Algorithms

Sorting, ML (convolution, pooling, softmax, relu), matrix operations (transpose, inverse, dot_product), FFT/ifft, signal processing.

## 8. Parallelization

Automatic parallelization; work-stealing scheduler; use `.single()` when order or debugging requires sequential execution.

## 9. GPU/TPU Integration

Compiler and runtime manage GPU/TPU utilization; `.gpu()` / `.cpu()` for explicit placement; PyTorch backend in current prototype.

## 10. Standard Library

Modules for ML, linalg, signal (FFT), visualization (`plot()`), scientific console (`print_latex(s)` for LaTeX-rendered output in Jupyter or any compatible IDE console), and runtime types (`Quantity`, `Quaternion`, `Dense`, `Sequential`).

## 11. Use Cases

ML training/inference, scientific computing, signal processing, physics simulations (with units and constants).

## 12. Implementation Roadmap

Phases 1–12 implemented in prototype (Python backend, Jupyter kernel, PyTorch runtime, Ricci, Sparse, AOT, Quaternion, **Physical Units v0.6**). See main README for detailed roadmap.

## 13. Technical Foundation

MLIR, LLVM, PyTorch; research in automatic differentiation, GPU compilation (Triton), type systems, work-stealing.

## 14. Conclusion

Dedekind aims to bridge productivity and performance for compute-intensive applications. This specification is updated to **v0.2** to reflect current prototype behaviour and **v0.6** language features (physical units and constants).

---

## 15. Physical Units and Universal Constants (v0.6)

This section documents the **physical units** and **universal constants** features introduced in Dedekind v0.6 (Option B: Unit Literals and Constants as Quantity).

### 15.1 Unit Literals

Values can carry SI units using bracket notation. **SI base units**: m (metre), kg (kilogram), s (second), A (ampere), K (kelvin), mol (mole), **cd (candela)**. Supplementary: **rad** (radian), **deg** (degree, 1 deg = π/180 rad; automatic conversion with rad), sr (steradiant). **Non-SI but supported**: **bar**, **atm** (pressure; 1 bar = 10⁵ Pa, 1 atm ≈ 101325 Pa); **g** (gram, 10⁻³ kg); **Bq** (Becquerel, activity = 1/s); **Sv** (Sievert, equivalent dose = J/kg).

```dedekind
distance = 10[m]
speed    = 5[m/s]
mass     = 1.0[kg]
freq     = 5.0e14[Hz]
charge   = 1.602e-19[C]
pressure = 101325[Pa]
p_bar    = 1.0[bar]
activity = 1000[Bq]
dose     = 0.01[Sv]
luminous_intensity = 1.0[cd]
illuminance = 500[lx]
```

- **Syntax**: `number [ unit ]` where `unit` is an identifier or product/quotient/power (e.g. `m`, `m/s`, `kg*m/s^2`, `m^2`, `cd*sr`, `bar`, `Bq`).
- **Semantics**: Such expressions are represented at runtime as `Quantity(value, unit)`.

### 15.2 Mathematical and Universal Constants as Quantity

**Mathematical constants** (dimensionless):

| Constant | Meaning           | Value (approx) |
|----------|-------------------|----------------|
| `pi`     | Kreiszahl π       | 3.14159…       |
| `e`      | Eulersche Zahl    | 2.71828…       |

**Fundamental physical constants** (CODATA 2018/2022) with SI dimensions:

| Constant     | Meaning                    | SI Unit        |
|--------------|----------------------------|----------------|
| `c`          | Speed of light              | m/s            |
| `G`          | Gravitational constant     | m³/(kg·s²)     |
| `h`          | Planck constant            | J·s            |
| `hbar`       | Reduced Planck (h/2π)      | J·s            |
| `k_B`        | Boltzmann constant         | J/K            |
| `k_e`        | Coulomb constant           | N·m²/C²        |
| `e_charge`   | Elementary charge          | C              |
| `epsilon_0`  | Vacuum permittivity         | F/m            |
| `mu_0`       | Vacuum permeability         | N/A²           |
| `m_e`        | Electron mass               | kg             |
| `m_p`        | Proton mass                 | kg             |
| `N_A`        | Avogadro constant           | 1/mol          |
| `R_gas`      | Universal gas constant      | J/(K·mol)      |
| `alpha`      | Fine-structure constant     | (dimensionless)|
| `sigma_SB`   | Stefan-Boltzmann constant   | W/(m²·K⁴)      |
| `F_faraday`  | Faraday constant            | C/mol          |

Example: `E = m * c^2` with `m = 1.0[kg]` yields a result in J (Joule); the runtime simplifies the displayed unit to **J** where applicable. Example with `pi`: `circumference = 2 * pi * r` with `r = 1.0[m]`.

### 15.3 Quantity Arithmetic

- **Addition / Subtraction**: Allowed when both operands have the **same unit**, or when both have **compatible units of the same dimension** (automatic conversion). Supported dimensions: **SI base** — length (m, cm, km, mm, dm), mass (kg, g, t, mg), time (s, min, h, ms), current (A, mA, kA, uA), temperature (K, mK), amount of substance (mol, mmol, kmol), luminous intensity (cd, mcd); **derived** — pressure (Pa, bar, atm), volume (L, mL, dm³, m³), energy (J, kJ, MJ, Wh, kWh), electric potential (V, mV, kV), frequency (Hz, kHz, MHz, GHz), charge (C, mC, uC), resistance (ohm, kohm, Mohm), power (W, kW, MW); **angle** — rad, deg. Example: `1[m] + 100[cm]` → `2.0[m]`; `90[deg] + (pi/2)*1[rad]` → `180[deg]`. The result unit is the unit of the **first operand**. **Conversion functions**: `deg_to_rad(x)`, `rad_to_deg(x)` for scalars, tensors, or Quantity. Incompatible units (e.g. `1[m] + 1[s]`) raise a compile-time or runtime error.
- **Multiplication / Division**: Units are combined (e.g. `m * m/s` → `m²/s`; `J/s` for power).
- **Power**: `Quantity ** exponent` is supported (e.g. `c^2`, `r^2`); the unit is raised to the given exponent (e.g. `(m/s)^2`).
- **Unary minus**: `-x` is supported for both `Quantity` and `Quaternion` (e.g. `-1.602e-19[C]`, `-1.0 + 0i`).

### 15.4 Unit Display Simplification

For readability, the runtime simplifies many compound units when displaying results:

- **SI base**: m, kg, s, A, K, mol, **cd**; supplementary: rad, sr.
- **Mechanics / energy**: **J** (Joule), **N** (Newton), **Pa** (Pascal), **W** (Watt).
- **Electricity / magnetism**: **V** (Volt), **F** (Farad), **ohm** (Ω), **S** (Siemens), **Wb** (Weber), **T** (Tesla), **H** (Henry).
- **Other derived**: **Hz** (Hertz, 1/s), **Gy** (Gray, J/kg, absorbed dose), **kat** (katal, mol/s).
- **Photometry**: **lm** (Lumen, cd·sr), **lx** (Lux, lm/m²).
- **Chemistry / pressure**: **M** (= mol/L), **bar**, **atm**; mass: **g**.
- **Radiation**: **Bq** (Becquerel, 1/s, activity); **Sv** (Sievert, J/kg, equivalent dose). Display: 1/s → Hz, J/kg → Gy when computed from expressions.

Examples: `(kg)*((m/s)^2)` → `[J]`; `N/m^2` → `[Pa]`; `J/s` → `[W]`; `cd*sr` → `[lm]`; literals `1[bar]`, `1000[Bq]`, `0.01[Sv]` stay as written. Internal unit representation remains exact for dimensional consistency.

### 15.5 Quaternion and Unary Minus

Complex numbers are represented as Quaternions (with `i` component; `j`, `k` zero when used as complex). As of v0.6:

- **Unary minus** on a Quaternion is supported via `__neg__` (e.g. `-1.0 + 0i`, `-(0.0 + 1.0i)`).
- This allows signal lists such as `[1.0+0i, 0.0+1.0i, -1.0+0i, 0.0-1.0i]` and FFT-based examples to run correctly.

### 15.6 Example: Universal Constants

```dedekind
fn main() {
    // E = m c²
    m = 1.0[kg]
    E = m * c^2
    print(E)   // e.g. 8.987…e+16[J]

    // Gravitation: F = G m1 m2 / r²
    m1 = 5.972e24[kg]
    m2 = 7.348e22[kg]
    r  = 3.844e8[m]
    F_gravity = G * m1 * m2 / r^2
    print(F_gravity)   // e.g. …[N]

    // Coulomb: F = k_e q1 q2 / r²
    q1 = 1.602e-19[C]
    q2 = -1.602e-19[C]
    r_atom = 5.29e-11[m]
    F_coulomb = k_e * q1 * q2 / r_atom^2
    print(F_coulomb)   // e.g. …[N]
}
main()
```

See `examples/dedekind/universal_constants.ddk` and `examples/dedekind/signal_physics.ddk` for full runnable examples.

### 15.7 Differentiable ODE Solvers

Dedekind provides a **differentiable ODE solver** for physics-informed ML and parameter identification:

- **`ode_solve(fun, y0, t, method="rk4")`**: Solves dy/dt = fun(t, y). `fun(t, y)` must return the time derivative (t scalar, y tensor); `y0` is the initial state; `t` is a 1D time grid. Returns a tensor of shape `(len(t), *y0.shape)`. Gradients flow through `y0` and through any parameters used inside `fun`.
- **`linspace(start, stop, steps)`**: Builds a 1D tensor of `steps` evenly spaced values from `start` to `stop` (for use as `t`).
- **Methods**: `"rk4"` (default, 4th-order Runge–Kutta) or `"euler"`.

Example: exponential decay dy/dt = -0.1*y; then `grad(final_state, y0)` gives d y(T)/d y0. See `examples/dedekind/differentiable_ode.ddk`.

### 15.8 Probabilistic Programming (v0.8)

Dedekind provides **first-class distributions** and **Bayesian inference** via `torch.distributions`:

- **Distributions**: `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`, `Dirichlet(alpha)` – return distribution objects. `Dirichlet(alpha)` is a multivariate distribution on the simplex (e.g. for topic modeling); `alpha` is a list or 1D tensor of concentration parameters.
- **sample(dist)** / **sample(dist, n)**: Draw one or `n` samples from a distribution.
- **log_prob(dist, value)**: Log-probability of `value` under `dist` (for inference).
- **metropolis(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size)**: Metropolis-Hastings MCMC. `log_prior_fn(theta)` and `log_likelihood_fn(data, theta)` return log-probs (tensor or scalar). Returns posterior samples of shape `(num_steps, *theta_shape)`.

Example: infer mean `theta` of Normal data with prior Normal(0,1); see `examples/dedekind/probabilistic.ddk`. Extended distributions (Exponential, Gamma, Beta, Poisson, Dirichlet): see `examples/dedekind/distributions_extended.ddk` and `examples/dedekind/dirichlet_distribution_function.ddk`.

### 15.9 Differentiable PDE Solvers (v0.8)

Dedekind provides **differentiable PDE solvers** for the heat equation \(u_t = k\,\Delta u\), built on finite differences and the differentiable `ode_solve` (RK4). Gradients flow through the initial condition `u0` and the diffusivity `k`, enabling inverse problems and physics-informed ML.

- **`pde_heat_1d(u0, x, t, k, bc="dirichlet")`**: 1D heat equation \(u_t = k\,u_{xx}\). `u0` is the initial condition (1D tensor, length = `len(x)`); `x` is the spatial grid (1D); `t` is the time grid (1D); `k` is the diffusivity (scalar or tensor). Returns a tensor of shape `(len(t), len(x))`. With `bc="dirichlet"`, boundary values from `u0` are held fixed.
- **`pde_heat_2d(u0, x, y, t, k, bc="dirichlet")`**: 2D heat equation \(u_t = k\,(u_{xx}+u_{yy})\). `u0` is the initial condition (2D tensor, shape `(nx, ny)`); `x`, `y` are 1D spatial grids; `t` is the time grid; `k` is the diffusivity. Returns a tensor of shape `(len(t), nx, ny)`.

Example: 1D heat with a spike initial condition; see `examples/dedekind/pde_heat.ddk`.

### 15.10 Numerical Integration and Math (v0.9, erweitert v0.9.6)

Dedekind provides **numerical integration** and **math functions** for scientific expressions. All math functions accept tensors or scalars (via `_to_tensor`), are element-wise and differentiable.

**Numerical integration:**

- **`integrate(f, a, b, n=100)`**: Numerically integrates \(f(x)\) from `a` to `b` using the trapezoidal rule with `n` points. `f` must accept a 1D tensor (the grid) and return a tensor of the same length; the integral is differentiable with respect to parameters in `f` when using autograd.
- **`riemann_sum(f, a, b, n=100, method="left"|"right"|"midpoint")`**: Riemann sum approximation of \(\int_a^b f(x)\,dx\). `method`: left endpoint, right endpoint, or midpoint (default). Differentiable when `f` is.
- **`volume_revolution_x(f, a, b, n=100)`**: Volume of revolution: rotate \(y=f(x)\) about the x-axis. \(V = \pi \int_a^b f(x)^2\,dx\) (disk method).
- **`volume_revolution_y(f, a, b, n=100)`**: Volume of revolution: rotate \(y=f(x)\) about the y-axis. \(V = 2\pi \int_a^b x\,f(x)\,dx\) (shell method). Valid for \(f(x)\ge 0\), \(a,b>0\).
- **`volume_revolution_vertical(f, a, b, x0, n=100)`**: Rotate about vertical axis \(x=x_0\). \(V = 2\pi \int_a^b |x-x_0|\,f(x)\,dx\).
- **`volume_revolution_horizontal(f, a, b, y0, n=100)`**: Rotate about horizontal axis \(y=y_0\). \(V = \pi \int_a^b (f(x)-y_0)^2\,dx\). \(f\) should lie entirely on one side of \(y_0\).
- **`pappus_volume_vertical(f, a, b, x0, n=100)`**: Pappus theorem: \(V = 2\pi\,R\,A\) with \(R=|\bar{x}-x_0|\), \(\bar{x}\) centroid.
- **`pappus_volume_horizontal(f, a, b, y0, n=100)`**: Pappus theorem: \(V = 2\pi\,R\,A\) with \(R=|\bar{y}-y_0|\), \(\bar{y}\) centroid.

**Trigonometry:** **`sin(x)`**, **`cos(x)`**, **`tan(x)`**.

**Exponential and Logarithm:** **`exp(x)`**, **`log(x)`** (natural logarithm), **`log10(x)`**.

**Square Root and Absolute Value:** **`sqrt(x)`**, **`abs(x)`**.

**Inverse Trigonometric Functions (Radians):** **`asin(x)`**, **`acos(x)`**, **`atan(x)`**, **`atan2(y, x)`** (angle of direction (x, y), range \((-\pi,\pi]\)).

**Hyperbolic Functions:** **`sinh(x)`**, **`cosh(x)`**, **`tanh(x)`**.

**Reductions:** **`min(x, dim=None)`**, **`max(x, dim=None)`** — minimum/maximum; **`argmin(x, dim=None)`**, **`argmax(x, dim=None)`** — index of min/max.

**Rounding:** **`round(x)`**, **`floor(x)`**, **`ceil(x)`**.

**Dirichlet Function:** **`dirichlet_function(x)`** — Dirichlet function D(x): returns 1 if x is rational (denominator ≤10000, tolerance 1e-6), else 0; element-wise for tensors. See `examples/dedekind/dirichlet_distribution_function.ddk`.

**Dedekind Cuts:** **`DedekindCut(x)`** — Dedekind cut representing real x as lower set A = {q ∈ Q : q < x}. **`dedekind_cut_from_rational(p, q)`** — cut for rational p/q. **`dedekind_cut_sqrt2()`** — cut for √2. Methods: `lower_set_contains(cut, q)`, `to_float()`; arithmetic `+`, `-`, `*`, comparisons. See `examples/dedekind/dedekind_cuts_rings.ddk`.

**Dedekind Rings:** **`DedekindRingZ()`** — ring of integers Z as Dedekind domain. **`ideal(n)`** — principal ideal (n) in Z. **`ideal_factor(i)`** — factor ideal into prime ideals; for Z: (n) = ∏ (p_i)^(e_i). **`DedekindIdeal`** has `.factor()`, `.norm()`, `*` (ideal multiplication). See `examples/dedekind/dedekind_cuts_rings.ddk`.

**Riemann Zeta:** **`zeta(s)`** — Riemann zeta function \(\zeta(s) = \sum_{n=1}^\infty 1/n^s\); uses scipy. For s=1 the series diverges (returns inf). See `examples/dedekind/riemann_zeta_sums.ddk`.

**Linear Algebra:** **`norm(x, p=None, dim=None)`** — vector or matrix norm (default p=2); **`det(A)`** — determinant; **`trace(A)`** — trace.

Examples: \(\int_0^1 x^2\,dx = 1/3\), \(\int_0^\pi \sin(x)\,dx = 2\); see `examples/dedekind/integration.ddk`. Full math showcase: `examples/dedekind/math_functions.ddk`.

### 15.11 Uncertainty Propagation and Fitting (v0.9.3)

**Uncertainty Propagation:**

- **`uncertain(value, std, unit="")`**: Creates an `UncertainQuantity` representing value ± std (Gaussian error propagation). Optional `unit` for physical quantities (e.g. `"m"`).
- **Arithmetic**: For +, -, *, /, ^ the standard deviation is propagated via the Gaussian approximation: e.g. for \(z = x + y\), \(\sigma_z^2 = \sigma_x^2 + \sigma_y^2\); for \(z = x \cdot y\), \((\sigma_z/z)^2 = (\sigma_x/x)^2 + (\sigma_y/y)^2\).
- **Display**: `repr` shows „value ± std [unit]“. See `examples/dedekind/uncertainty_propagation.ddk`.

**Fitting / Regression:**

- **`fit(loss_fn, params_init, data, method="gd"|"mcmc", lr=0.01, steps=500)`**: Minimizes `loss_fn(params, data)` with respect to `params`. `params_init` is the initial parameter tensor (or list); `data` is passed to `loss_fn`. With `method="gd"` (default), gradient descent is used (in-place updates); with `method="mcmc"`, Metropolis-Hastings is used (returns posterior samples). Returns the optimal parameters (tensor) for GD, or samples tensor for MCMC.
- Example: linear regression \(y = a\,x + b\); see `examples/dedekind/curve_fitting.ddk`.

### 15.12 Standard Library Modules and User-Defined Units (v1.7)

**Standard library modules** live in `modules/` as plain `.ddk` files and are loaded with `use NAME`. Built-ins (`sin`, `pi`, `ode_solve`, `michaelis_menten`, `c`, `R_gas`, …) remain globally available without any import; modules are strictly additive.

- **`use physics`** — `kinetic_energy`, `momentum`, `pendulum_period`, `spring_period`, `newton_gravity`, `escape_velocity`, `orbital_period`, `relativistic_gamma`/`relativistic_energy`, `coulomb_force`, `capacitor_energy`, `ohms_law_voltage`, `ideal_gas_pressure`/`ideal_gas_volume`, `rms_speed`, `blackbody_radiance`, `wave_speed`, `doppler_classical`, plus dimensionless numeric constants `C_NUM`, `G_NUM`, `K_B_NUM`, `HBAR_NUM`, `K_E_NUM`, `R_GAS_NUM`, `SIGMA_SB_NUM`.
- **`use stats`** — `z_score`, `cohens_d`, `pooled_std`, `hedges_g`, `glass_delta`, `standard_error_mean`, `ci_normal_mean`, `ci_proportion`, `r_squared`, `mse`/`rmse`/`mae`, `coefficient_of_variation`, `standardize`, `relative_error`.
- **`use chemistry`** — `pH_from_concentration`, `pOH_from_concentration`, `henderson_hasselbalch`, `ka_from_pka`/`pka_from_ka`, `dilution_volume`, `molarity`, `molality`, `mass_fraction`, `boyle_volume`, `charles_volume`, `gay_lussac_pressure`, `combined_gas_law_v2`, `van_der_waals_pressure`, `nernst_potential`/`nernst_potential_25C`, `faraday_mass`, `half_life_first_order`/`half_life_second_order`; dimensionless `R_GAS_NUM`, `F_FARADAY_NUM`.
- **`use biology`** — `exponential_growth`, `doubling_time`, `growth_rate_from_doubling`, `gompertz_growth`, `von_bertalanffy`, `population_change_logistic`, `carrying_capacity`, `kleibers_law`, `allometric_scaling`, `bmr_harris_benedict`, `bmi`, `hardy_weinberg_freq`, `fitness_selection_coefficient`, `mutation_drift_balance`, `r_naught_sir`, `herd_immunity_threshold`.
- **`use math`** — constants `PHI`, `TAU`; sequences `fibonacci`, `harmonic_sum`, `geometric_sum`; number theory `lcm`, `is_perfect_square`, `digital_root`; geometry `circle_area`, `circle_circumference`, `sphere_volume`, `sphere_surface`, `cylinder_volume`, `cone_volume`, `hypotenuse`, `law_of_cosines_c`; helpers `lerp`, `clamp_scalar`, `sigmoid`, `softplus`.
- **`use ml`** — activations `leaky_relu`, `elu`, `swish`, `gelu_approx`; losses `mse_loss`, `mae_loss`, `binary_crossentropy`; metrics `accuracy`, `precision_binary`, `recall_binary`, `f1_score`.
- **`use space`** — N-body simulation (`n_body_simulate`, `n_body_simulate_advanced`), Kepler equation solvers (`kepler_solve`), Keplerian-Cartesian conversions (`kepler_to_cartesian`, `kepler_to_cartesian_from_E`, `cartesian_to_kepler`).
- **`use signals`** — Unified electrical engineering, control theory, and digital signal processing.
  - **Electronics:** circuit builder (`circuit`), complex AC analysis (`phasor`), parallel resistance (`parallel_resistors`), RC time constant (`rc_time_constant`), RLC resonance (`rlc_resonance_freq`).
  - **DSP:** FIR/IIR filtering (`fir_filter`, `iir_filter`), biquad coefficient generators (`biquad_lowpass`, `biquad_highpass`, `biquad_bandpass`), frequency response (`freqz`), filter design (`butter`, `cheby1`).
  - **Control:** block-diagram primitives (`block_diagram`, `constant_block`, `gain_block`, `sum_block`, `product_block`, `saturation_block`, `integrator_block`), controllers (`pid_block`, `pid_block_saturated`), LTI plants (`transfer_function_block`, `state_space_block`, `state_space_block_with_d`), plus higher-level `state_space`, `pi_controller`, `bode_plot`.
- **`use structural`** — 2D meshes (`structural_mesh_2d`), static finite element solver (`structural_solve_2d`, `structural_solve_2d_params`), compliance calculations (`structural_compliance_2d`), Optimality Criteria topology optimization (`topo_opt_oc_2d`, `topo_opt_oc_2d_advanced`), and character-based visualization (`print_structural_topology_2d`).
- **`use thermal`** — 2D thermal meshes (`thermal_mesh_2d`), steady-state solver (`thermal_solve_2d`, `thermal_solve_2d_params`), transient thermal solver (`thermal_solve_transient_2d`, `thermal_solve_transient_2d_params`), thermal topology optimization (`topo_opt_thermal_oc_2d`, `topo_opt_thermal_oc_2d_advanced`), and character-based visualization (`print_thermal_topology_2d`).
- **`use fluid_dynamics`** — Lattice Boltzmann Method CFD solvers (`lbm_simulation`, `lbm_simulation_with_mask`), simulation iteration handlers (`simulation_step`, `simulation_run`), field extractors (`simulation_get_velocity`, `simulation_get_density`), force calculators (`simulation_get_drag_lift`, `simulation_get_drag_lift_for_mask`), dynamic obstacle modifier (`simulation_set_obstacle`), and soft masks (`soft_cylinder_mask`, `soft_airfoil_mask`, `add_wind_tunnel_walls`).
- **`use quantum`** — Quantum Computing bridge wrappers (`make_bell`, `make_ghz`, `make_grover`, `make_vqe_ansatz`, `simulate`, `sample_circuit`, `expectation`, `probs`, `state_fidelity`, `vn_entropy`).
- **`use robotics`** — Kinematic chains (`kinematic_chain`), revolute/prismatic joints (`add_revolute_joint`, `add_prismatic_joint`), forward kinematics solver (`forward_kinematics`), and end-effector extraction (`end_effector_pos`, `end_effector_rot`).
- **`use atomic`** — Differentiable Atomic & Molecular physics/chemistry, combining Molecular Dynamics (`molecular_lj_simulate`, `morse_potential`, etc.) and Crystallography / structural analysis (`cryst_symmetry_apply`, `cryst_structure_factor_atoms`, etc.).

**User-defined units** are declared at top level with the `unit` keyword (a *soft* keyword — `q.unit` member access and `unit="V"` kwargs remain valid):

```dedekind
unit Foot      = 0.3048[m]
unit Mile      = 1609.344[m]
unit eV        = 1.602176634e-19[J]
unit kcal      = 1000.0[calorie]      // chaining: calorie must be declared first
unit Bohr      = 5.29177210903e-11[m]
unit Darcy     = 9.869233e-13[m^2]    // permeability
```

Rules:

- The base unit (right-hand side bracket) must already be known to Dedekind: either an SI/derived base unit or a previously-declared user unit.
- The unit is registered in `DIMENSION_TO_BASE` under the dimension that owns the base unit. `Foot`, `Mile` join `length`; `eV`, `calorie`, `kcal` join `energy`; `Darcy` joins `permeability`.
- Arithmetic with built-in units in the same dimension is automatic: `1[Mile] + 1500[m]` ⇒ `1.93[Mile]` (result follows left-hand unit).
- Chains (`unit MicroInch = 1e-6[Inch]`) are resolved transitively, so the factor stored is `1e-6 * 0.0254` against the SI base `m`.
- Pre-pass in `check_units()` registers all `UnitDef` statements before any expression check; `_rebuild_additive_unit_sets()` mutates the shared `ADDITIVE_DIMENSION_UNIT_SETS` list in-place so the compile-time checker and the runtime stay consistent.

**Quantity comparison operators** (v1.7): `Quantity` supports `<`, `<=`, `>`, `>=`, `==`, `!=`, `__hash__`. Same-dimension operands (e.g. `cm` vs `m`) are converted to the dimension's base unit before comparison; mixed dimensions raise `ValueError`. Enables `if pressure < 1[atm] { ... }`.

Examples: `examples/dedekind/stdlib_modules_demo.ddk`, `examples/dedekind/user_defined_units.ddk`. Tests: `tests/dedekind/user_defined_units_test.ddk`, `tests/dedekind/stdlib_{physics,stats,chemistry,biology_math_ml}_test.ddk`.

### 15.13 Python Interop and Source-Mapped Tracebacks (v1.8)

**`pyimport`** — direct access to the PyPI ecosystem. Syntax:

```dedekind
pyimport scipy.special as sp
pyimport numpy as np
pyimport math                  // alias defaults to "math"
pyimport scipy.stats as st
```

The codegen emits `import MODULE as ALIAS` at the statement position; values are accessed as `alias.name(...)` and mix freely with Dedekind built-ins. `pyimport` is a *soft* keyword (only recognized at statement start; existing identifiers named `pyimport` remain valid). No dimension/shape inference is applied to returned values — the caller is responsible for unit consistency.

**Source-mapped runtime tracebacks** — every runtime exception originating from generated Python code is rewritten so the traceback points to the original `.ddk` file and line, including a snippet of the user's source. Example:

```
Traceback (most recent call last):
  File "demo.ddk", line 16, in <module>
    main()
  File "demo.ddk", line 12, in main
    s = outer(data)
  File "demo.ddk", line 7, in inner
    return arr[0] + arr[1] + arr[99]
IndexError: index 99 is out of bounds for dimension 0 with size 3
```

Implementation:

- The codegen writes `# ddk:<line>` markers before every statement (top-level, `if`/`else`, `while`, `for`, function body).
- `dedekind_exec(generated_code, ddk_file, exec_globals, ddk_source)` in `src/compiler/compiler.py` compiles with a virtual filename, runs the code, and on any `BaseException` rewrites `exc.args` to a translated multi-line traceback before re-raising the *original* exception type. So `except AssertionError`, `except ValueError`, `except IndexError` etc. continue to work — only the user-visible message is enriched.
- Frames inside the inlined runtime are shown as `<dedekind-runtime>` (so users see the call chain but not the implementation noise); frames from external libraries (scipy, torch) keep their real filenames.
- Used by `run_tests.py`, `run_examples.py`, `python -m compiler` and the Jupyter kernel.

Examples: `examples/dedekind/pyimport_demo.ddk`, `examples/dedekind/source_mapping_demo.ddk`. Test: `tests/dedekind/pyimport_test.ddk`.

### 15.14 Purity Check for Pure-Context Calls (v1.8.1)

A new compile-time pass `src/compiler/purity_check.py` rejects programs that pass impure functions into contexts that require purity. Concretely, the first (or first two) argument(s) of these built-ins must be pure:

| Built-in | Pure args | Reason |
|----------|-----------|--------|
| `jit(fn)` | `fn` | `torch.compile` traces a graph; side effects cause graph breaks |
| `grad(fn, x)` | `fn` | Autograd tape; side effects are recorded but not differentiable |
| `fit(loss_fn, ...)` | `loss_fn` | Called many times per training step |
| `metropolis(log_prior, log_likelihood, ...)` | both | Called thousands of times per chain |
| `hmc(log_prior, log_likelihood, ...)` | both | Same |
| `sde_solve(drift, diffusion, ...)` | both | Called at every Euler-Maruyama / Milstein step |

A function is *impure* if its body — transitively through any helper function it calls — invokes any of these built-ins:

```
print, plot, scatter, contour, print_latex, print_table,
write_file, read_file, file_exists,
http_get, http_post,
read_csv, write_csv, read_parquet, write_parquet,
read_hdf5, write_hdf5, read_netcdf,
export_notebook
```

The check is transitive via a function table: if `loss` calls `helper` and `helper` calls `print`, the error reports `"... ruft 'print()' ... in 'helper'"`. Cycle protection via a `visited` set. Calls via `pyimport`ed modules (e.g. `np.savetxt(...)`) are not analyzed — the caller is responsible.

Opt-out for special cases (debug builds, code migration):

- API: `compile_source(source, check_purity=False)`
- CLI: `python -m compiler datei.ddk --no-purity-check`

Example: `examples/dedekind/purity_check_demo.ddk`. Test: `tests/dedekind/purity_check_test.ddk`.

### 15.15 Shape Annotations and Quantity Stripping (v1.9)

**Shape annotations** declare tensor ranks and dimensions in function signatures, catching broadcasting bugs and shape mismatches that silently produce wrong results in plain NumPy/PyTorch:

```dedekind
fn dot_product(a: Vector[3], b: Vector[3]) -> Scalar { ... }
fn matvec(M: Matrix[2, 3], v: Vector[3]) -> Vector[2] { ... }
fn weighted_dot(x: Vector[N], w: Vector[N]) -> Scalar { ... }
fn forward(x: Tensor[batch, 28, 28]) -> Tensor[batch, 10] { ... }
```

Four type constructors:

| Type | Rank | Example |
|------|------|---------|
| `Scalar` | 0-D | `Scalar` |
| `Vector[n]` | 1-D | `Vector[3]`, `Vector[N]` |
| `Matrix[m, n]` | 2-D | `Matrix[2, 3]`, `Matrix[M, N]` |
| `Tensor[d1, d2, ...]` | N-D | `Tensor[batch, 28, 28]` |

Dimensions are either integer literals (`3`) or identifiers (`N`, `batch`). Identifiers are **symbolic dimensions** bound at call time: the first argument using `N` sets the value, subsequent arguments with the same `N` must match. Mismatch raises `ValueError` with the `.ddk` file and line (via the source-mapped traceback infrastructure from v1.8).

Implementation:

- Parser extends `_parse_signature_annotation()` to dispatch between `[unit]` (existing) and shape (new) annotations after `:` or `->`.
- AST: `FunctionDef` gains `arg_shapes: Optional[List[Optional[List]]]` and `return_shape: Optional[List]` fields. Each shape is a list mixing `int` (literal dim) and `str` (symbolic dim).
- Codegen emits at function entry:
  ```python
  _shape_env = {}
  _check_shape(arg1, [3], "fn", "arg1", _shape_env)
  _check_shape(arg2, ['N', 2], "fn", "arg2", _shape_env)
  ```
  and wraps each `return val` as `return _check_return_shape(val, [...], "fn", _shape_env)`.
- Runtime `_check_shape` calls `_shape_of(value)` which works on `torch.Tensor` (`.shape`), Python lists/tuples (recursive consistency check), and scalars (`()`); returns `None` for unrecognized types (no check).
- Unit annotations and shape annotations can be combined: unit-coercion runs first, then shape validation.

**Quantity stripping** for hot paths — new built-in `unwrap(x)` that strips unit/uncertainty wrappers, returning raw values:

```dedekind
fn pure_loss(params, data) {
    a = unwrap(params[0])   // Quantity(2.0, "m") -> 2.0
    b = unwrap(params[1])
    x = unwrap(data[0])
    y = unwrap(data[1])
    diff = a * x + b - y    // all-float arithmetic
    return diff * diff
}
fast_loss = jit(pure_loss)
```

Behavior of `unwrap`:

| Input | Output |
|-------|--------|
| `Quantity(value, unit)` | `value` (float) |
| `UncertainQuantity(value, std, unit)` | `value` (std discarded) |
| 0-D `torch.Tensor` | `.item()` |
| `list`/`tuple` | element-wise `unwrap` (recursive) |
| anything else | passthrough |

Rationale: the compile-time units checker has already validated dimensions; at runtime the wrapper adds overhead — invisible for a handful of calls, but significant in MCMC chains with 10 000+ samples or in `torch.compile`-d graphs where Python objects force graph breaks. Calling `unwrap()` at function entry gives the compiler clean Python floats / numpy arrays to optimize.

Examples: `examples/dedekind/shape_annotations_demo.ddk`, `examples/dedekind/quantity_stripping_demo.ddk`. Tests: `tests/dedekind/shape_annotations_test.ddk`, `tests/dedekind/quantity_stripping_test.ddk`.

### 15.16 Physics-Informed Neural Networks (v1.10)

PINNs train a neural network so that its output respects a differential equation in addition to (or instead of) data. Dedekind exposes the one primitive that was missing for this workflow:

```
partial(u, x, order=1)
```

Computes ∂u/∂x via `torch.autograd.grad`, where `u` is an already-evaluated tensor and `x` is a leaf tensor with `requires_grad=True` (typically via `.with_grad()`). Complementary to `grad(fn, x)`, which differentiates a *function* at a point.

Typical PINN pattern:

```dedekind
fn mlp_eval(params, x) {
    W1 = params.narrow(0, 0, 10).reshape(1, 10)
    b1 = params.narrow(0, 10, 10)
    W2 = params.narrow(0, 20, 10).reshape(10, 1)
    b2 = params.narrow(0, 30, 1)
    h = tanh(x @ W1 + b1)
    return h @ W2 + b2
}

fn pinn_loss(params, data) {
    x_bc = data[0]; u_bc = data[1]; x_coll = data[2]
    u_pred_bc = mlp_eval(params, x_bc)
    diff_bc = u_pred_bc - u_bc
    bc_loss = (diff_bc * diff_bc).sum()

    u_coll = mlp_eval(params, x_coll)
    u_x = partial(u_coll, x_coll)           // first-order: y'
    u_xx = partial(u_coll, x_coll, order=2) // second-order: y''
    residual = u_x + u_coll                  // example: y' + y = 0
    phys_loss = (residual * residual).mean()
    return bc_loss + phys_loss
}

x_coll = linspace(0.0, 1.0, 20).reshape(-1, 1).with_grad()
params_opt = fit(pinn_loss, random_vector(31) * 0.1, [x_bc, u_bc, x_coll],
                 method="gd", lr=0.02, steps=2000)
```

This trains a network to satisfy `y' + y = 0` with `y(0) = 1` — pure physics, no solution data. The analytical answer `exp(-x)` is matched to ~1% accuracy in 2000 steps with a single 10-neuron hidden layer.

Supporting infrastructure added in v1.10:

- `_to_tensor` now tolerates `torch.stack` failures on mixed-shape tensor lists (PINN `data=[x_data, u_data, x_coll]` has different shapes per element); returns the list unchanged in that case.
- `fit()` zeroes `.grad` on every `requires_grad` tensor in `data` between training steps, preventing memory accumulation on collocation tensors over long PINN runs.

What v1.10 **deliberately does not provide** (Stage 3 of the PINN roadmap, future releases):

- No `.with_physics_loss(pde_fn)` modifier. PINN loss balancing (NTK-based, self-adaptive) is an active research field; a naive built-in would over-promise.
- No automatic collocation sampling. Users choose `linspace`, `random_vector`, or Latin-hypercube; this is problem-specific.
- No spectral-bias-mitigation (SIREN, Fourier features). Important for high-frequency PDEs; can be implemented manually in `mlp_eval`.

Examples: `examples/dedekind/pinn_ode_demo.ddk` (`y' + y = 0`), `examples/dedekind/pinn_oscillator_demo.ddk` (`u'' + u = 0` on [0, π/2], demonstrates `order=2`). Test: `tests/dedekind/partial_test.ddk`.

### 15.17 Graph Laplacian and Spectral Methods (v1.11)

The graph Laplacian is the discrete equivalent of the continuous Laplace-Beltrami operator and the entry point to spectral methods on graphs (heat diffusion, clustering, dimensionality reduction, effective resistance, etc.).

```dedekind
L = graph_laplacian(adj)                    // combinatorial: L = D - A
L_sym = graph_laplacian(adj, normalized=true) // symmetric: I - D^{-1/2} A D^{-1/2}
```

Properties:

| Variant | Diagonal | Off-diagonal | Eigenvalues |
|---------|----------|--------------|-------------|
| combinatorial | `deg(i)` | `-A[i,j]` | `[0, 2 * max_deg]` |
| normalized sym | `1` if `deg(i) > 0` | `-A[i,j] / sqrt(deg(i)*deg(j))` | `[0, 2]` |

Input: dense `torch.Tensor` of shape `(N, N)`, sparse COO `torch.Tensor`, or nested Python list. Return follows input sparsity. The resulting Laplacian plugs directly into Dedekind's existing iterative solvers (`cg`, `gmres`, `bicgstab` from v1.6) and eigensolver (`eigh`) — no external graph library required for the core spectral pipeline.

**Heat diffusion on a graph** via implicit Euler:

```dedekind
M = I_mat + dt * L
for step in arange(num_steps) {
    res = cg(M, u, tol=1e-10)
    u = res["x"]
}
```

**Spectral partitioning** via the Fiedler vector (eigenvector of the second-smallest eigenvalue):

```dedekind
res = eigh(L)
eigvals = res[0]
eigvecs = res[1]
fiedler = eigvecs.narrow(1, 1, 1).reshape(-1)
// sign(fiedler[i]) labels the two clusters
```

Example: `examples/dedekind/graph_spectral_demo.ddk` (two-cluster graph with bridge edge, demonstrates both heat diffusion asymmetry and Fiedler partitioning). Test: `tests/dedekind/graph_laplacian_test.ddk`.

### 15.18 Graph Shape Annotations for GNNs (v1.12)

Extends the v1.9 shape-annotation system with a new `Graph[N, E]` type that wraps `torch_geometric.data.Data`-like objects:

```dedekind
pyimport torch_geometric.data as pyg_data
pyimport torch_geometric.nn as pyg_nn

fn coordination(g: Graph[N, E]) -> Scalar {
    return g.num_edges / g.num_nodes
}

fn pair_match(g1: Graph[N, E1], g2: Graph[N, E2]) -> Scalar {
    // both graphs must have the same number of nodes N
    return g1.num_nodes + g2.num_nodes
}

fn gcn_forward(g: Graph[N, E]) {
    conv = pyg_nn.GCNConv(1, 4)
    return conv.forward(g.x, g.edge_index)
}
```

Implementation:

- Parser: `Graph[N, E]` is added to `_SHAPE_TYPES`. `_parse_shape_annotation` returns `(kind, dims)` tuples; the codegen dispatches `_check_shape` (tensor kinds) vs. `_check_graph_shape` (graph kind).
- Runtime `_check_graph_shape(value, expected_dims, fn_name, arg_name, shape_env)`: reads `(num_nodes, num_edges)` via `value.num_nodes`/`value.num_edges` (PyG API) with a fallback to `value.x.shape[0]`/`value.edge_index.shape[1]`. Both dims must be `int` or symbolic. Symbolic dims bind into the same `_shape_env` as tensor dims, so a function can require multiple graphs to share a node-count.
- Validates at call time and at return; mismatch raises `ValueError` with source-mapped traceback (v1.8 infrastructure).

**The USP vs. plain PyG**: combinable with unit annotations on other arguments and return types:

```dedekind
fn molecular_mass(g: Graph[N, E]) -> [g/mol] {
    raw = g.x.sum().item()
    return Quantity(raw, "g/mol")
}

fn add_mass(m1: [g/mol], m2: [g/mol]) -> [g/mol] {
    return m1 + m2
}
```

PyTorch Geometric has no concept of physical units; mixing `eV` and `kcal/mol` on edge weights is a silent bug there. Dedekind catches it at compile time (via the existing units checker) and at runtime (via the Quantity arithmetic).

What v1.12 **deliberately does not provide** (Stage 3 of the graph roadmap):

- No native message-passing primitive. PyG covers 30+ Conv variants (GCN, GAT, GIN, SAGE, MPNN, EGNN, SchNet, DimeNet, …), three pooling strategies, and dozens of benchmark datasets — a multi-week reimplementation budget for a worse result. `pyimport torch_geometric.nn` is the honest answer for production GNN workflows.
- Dedekind's contribution to v1.12 is the annotation and unit layer **over** PyG, not a replacement.

Examples: `examples/dedekind/gnn_molecule_demo.ddk` (drug discovery: water molecule with atomic masses in `[g/mol]`, GCNConv), `examples/dedekind/gnn_materials_demo.ddk` (materials: 4-atom FCC unit cell with bond lengths in `[pm]` and binding energy in `[eV]`, GraphConv). Test: `tests/dedekind/graph_shape_test.ddk`.

### 15.19 Declarative MILP DSL with Units (v1.13)

A declarative DSL for (mixed-)integer linear programming. Users write constraints and objective in natural arithmetic notation; Dedekind builds the scipy.optimize.milp problem matrix and validates dimensions.

```dedekind
x      = Variable("strecke", lower=0[km])
trucks = Variable("trucks",  lower=1, integer=true)

cost = 2.5 * x + 1000 * trucks
res  = optimize_milp(cost, [
    x >= 500[km],
    trucks * 200[km] >= x
])
print(res["strecke"], res["trucks"], res["objective"])
// 500.0 km, 3 trucks, 4250 EUR
```

**Variable** parameters:

| Argument | Type | Meaning |
|----------|------|---------|
| `name`   | str  | Result key |
| `lower`  | number or Quantity, optional | Lower bound |
| `upper`  | number or Quantity, optional | Upper bound |
| `integer`| bool, default false | Force integer values |

**Operator overloads**:

- `Variable + Variable`, `Quantity * Variable`, `scalar * Variable` → `_MILPExpression`
- `_MILPExpression + _MILPExpression`, etc. → `_MILPExpression`
- `Variable >= Quantity`, `_MILPExpression <= number`, etc. → `_MILPConstraint`
- `Variable * Variable` raises `ValueError` (non-linear)

**optimize_milp** parameters:

| Argument | Type | Meaning |
|----------|------|---------|
| `objective` | `_MILPVariable` or `_MILPExpression` | Linear cost function |
| `constraints` | list of `_MILPConstraint`, optional | Inequality/equality constraints |
| `sense` | `"minimize"` (default) or `"maximize"` | Optimization direction |

Returns a `dict`:

```python
{
    "<var_name>": optimal_value,   # one entry per variable
    ...,
    "objective": optimal_objective,
    "status":    scipy_status_message,
}
```

**Unit-awareness as the USP** vs. Gurobi/cvxpy/pyomo/PuLP: every constraint runs through `_milp_units_compat` (which uses the existing v1.7 dimension system). `x >= 500[km]` with `x: Variable(lower=0[km])` validates; `x >= 500[kg]` raises `ValueError: MILP-Einheiten passen nicht in Constraint: [km] vs [kg]`. No other MILP toolkit catches dimensional errors before the solver runs.

The low-level `milp(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)` from v1.5 stays unchanged; the v1.13 DSL is an additive convenience layer.

Examples: `examples/dedekind/optimize_milp_demo.ddk` (Vehicle Routing, Production Mix, Energy Mix). Test: `tests/dedekind/optimize_milp_test.ddk`.

### 15.20 Molecular Dynamics Bridge to OpenMM (v1.14)

A typed Lennard-Jones MD wrapper around OpenMM with full unit validation of force-field parameters before the C++ kernel runs.

```dedekind
res = md_simulate_lj(
    n_particles=27,
    sigma=3.4[Angstrom],         // alternative input in Å
    epsilon=0.238[kcal/mol],     // or kJ/mol
    mass=39.948[amu],
    temperature=85[K],
    dt=1[fs],
    n_steps=200,
    box_size=2.0[nm],
    seed=42
)
// res["potential_energy"] : Quantity in [kJ/mol]
// res["kinetic_energy"]   : Quantity in [kJ/mol]
// res["temperature"]      : Quantity in [K]
// res["positions"]        : torch.Tensor (n_particles, 3) in nm
```

Parameters:

| Name | Type | Notes |
|------|------|-------|
| `n_particles` | int | |
| `sigma` | Quantity, length | nm, Å, pm, m |
| `epsilon` | Quantity, molar energy | kJ/mol, kcal/mol, eV/atom, Hartree/mol |
| `mass` | Quantity, mass | amu, Da, g, kg |
| `temperature` | Quantity in K | Langevin bath |
| `dt` | Quantity, time | fs, ps, ns |
| `n_steps` | int | integration steps |
| `box_size` | optional Quantity, length | cubic periodic box |
| `friction` | float | Langevin friction in 1/ps |
| `seed` | optional int | reproducible trajectory |

**The USP vs. raw `pyimport openmm`:** every input passes through the v1.7 dimension system before reaching the C++ kernel. `epsilon=0.238[eV]` raises `ValueError: epsilon=0.238[eV] hat falsche Dimension; erwartet kompatibel zu [kJ/mol] (molar_energy)` — in raw OpenMM, mixing eV with kcal/mol is a silent bug that costs hours to diagnose.

Supporting extensions to the dimension system in v1.14:

- **length**: added `nm`, `Angstrom`, `pm`, `fm` to the length dimension table.
- **mass**: added `amu`, `Da` (1.66053906660e-27 kg).
- **time**: added `fs`, `ps`, `ns`, `us`.
- **new dimension** `molar_energy`: `kJ/mol` (base), `kcal/mol` (4.184), `J/mol`, `eV/atom` (96.485), `Hartree/mol` (2625.5).

Initial positions are placed on a regular 3D grid with spacing ≥ 1.05·σ, then perturbed by 2% of grid spacing — avoids NaN energies from random overlaps (LJ-r⁻¹² explodes as r → 0) without losing thermalization stochasticity.

**What v1.14 deliberately does not provide:** no protein force fields (AMBER, CHARMM, OPLS), no implicit solvent, no REMD/free-energy/umbrella-sampling methods. For those, `pyimport openmm.app as omm_app` directly — Dedekind's role is the unit-aware bridge for the canonical LJ workflow, not a re-implementation of OpenMM.

Example: `examples/dedekind/md_lennard_jones_demo.ddk` (Argon cluster: 200 fs equilibration at 85 K, then 1 ps production at 150 K). Test: `tests/dedekind/md_simulate_lj_test.ddk`.

### 15.21 Labeled Tensors for Earth Science (v1.15)

A new shape-annotation kind for tensors whose axes have *names* rather than just sizes — the standard data model in climate, geosciences, and earth-system simulations. Built on top of `xarray.DataArray`.

```dedekind
pyimport numpy as np

fn temporal_mean(t: LabeledTensor[lat, lon, time]) -> LabeledTensor[lat, lon] {
    return t.mean(dim="time")
}

fn zonal_mean(t: LabeledTensor[lat, lon, time]) -> LabeledTensor[lat, time] {
    return t.mean(dim="lon")
}

T = labeled_tensor(raw_data,
    dims=["lat", "lon", "time"],
    coords={"lat": lats, "lon": lons, "time": times},
    attrs={"units": "K", "crs": "EPSG:4326"}
)
```

The annotation:

| Form | Meaning |
|------|---------|
| `LabeledTensor[a, b, c]` | DataArray with **exactly** the dim names `{a, b, c}` |

Order in the annotation is documentation only — xarray operates name-based, so validation is set-based. Mismatch raises `ValueError` with the specific missing/extra dims.

**The USP vs. raw xarray:** xarray has no type system. The classic climate bug `data.mean(axis=2)` instead of `data.mean(dim="time")` produces a silent wrong result; `LabeledTensor[...]` annotations catch this at the function boundary.

**`labeled_tensor` parameters:**

| Argument | Type | Notes |
|----------|------|-------|
| `data` | tensor / numpy / list | underlying values |
| `dims` | list of str | axis names, same order as `data.shape` |
| `coords` | dict[str → 1D array], optional | coordinate axes |
| `name` | str, optional | DataArray name |
| `attrs` | dict, optional | metadata (units, CRS, source, ...) |

**What v1.15 deliberately does not provide:** no re-implementation of xarray operations (regridding, interp_like, groupby_bins, rolling means, etc.). Users call those directly on the DataArray (`da.coarsen(time=12).mean()` etc.). No Dask-/distributed-aware annotations — chunked DataArrays work via `pyimport xarray`, but only dim-name validation runs. Dedekind's role is the annotation-and-type layer over xarray, not a competitor to it.

Example: `examples/dedekind/labeled_tensors_demo.ddk` (4×8×12 synthetic climate dataset: temporal mean, zonal mean, anomaly, `.sel`-slicing by coordinate value). Test: `tests/dedekind/labeled_tensors_test.ddk`.

### 15.22 Bioinformatics & Life Sciences (v1.16, updated v2.0)

Typed bioinformatics and life sciences primitives layered on top of the existing annotation system, with native sequence alignment, protein structure parsing, pharmacokinetics modeling, biochemical helpers, and database query integration.

```dedekind
fn dna_to_protein(dna: Sequence[DNA]) -> Sequence[Protein] {
    rna = transcribe(dna)
    return translate(rna)
}
protein = dna_to_protein("ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTG")
// "MALWMRLLPL" — first 10 AA of insulin signal peptide
```

**`Sequence[kind]`-Annotation:**

| Kind | Alphabet |
|------|----------|
| `DNA` | A, C, G, T, N (= unknown) |
| `RNA` | A, C, G, U, N |
| `Protein` | 20 standard amino acids + B, Z, X (ambiguity), `*` (stop) |

Mismatch raises `ValueError: Sequence[DNA]-Check in fn(seq): ungueltiges Zeichen 'U' an Position 3 (erlaubt: ACGNT).`

**Native bioinformatics and life science built-ins:**

| Built-in | Description / Returns |
|----------|---------|
| `gc_content(dna)` | float 0..1 |
| `reverse_complement(dna)` | str |
| `transcribe(dna)` | RNA str (T→U) |
| `translate(rna, stop_at_stop=True)` | protein str |
| `k_mer_count(seq, k)` | dict {k-mer: count} |
| `smith_waterman_alignment(seq1, seq2, match, mismatch, gap)` | Computes Smith-Waterman local alignment of two sequences (strings or 1D tensor/lists). Returns a dict with `score` (float), `aligned_seq1` (str), and `aligned_seq2` (str). |
| `protein_structure_parse(filepath)` | Parses a PDB or mmCIF file and returns a `DataFrame` with columns: `atom_name`, `res_name`, `chain_id`, `element`, and `x`, `y`, `z` coordinates (represented as `angstrom` quantities). |
| `two_compartment_pk(c0, k12, k21, ke, t)` | Differentiable two-compartment pharmacokinetics simulator. Computes central compartment concentration at time `t` (scalar or tensor) given initial concentration `c0`, and rate constants `k12`, `k21`, `ke`. |
| `concentration_to_pH(h_conc)` | Computes pH from Hydrogen ion concentration `h_conc` (can be a `Quantity` in `M`). Returns dimensionless float. |
| `pH_to_concentration(ph)` | Computes Hydrogen ion concentration (in `M`) from pH value. Returns dimensionless float/tensor. |

**Cheminformatics and Database Queries:**

| Built-in | Description / Returns |
|----------|---------|
| `smiles_molecular_weight(smiles)` | Returns the molecular weight of a SMILES string as a `Quantity` in `g/mol`. |
| `smiles_descriptors(smiles)` | dict with `mw` ([g/mol]), `logp`, `num_atoms`, `num_heavy_atoms`, `num_rings`, `num_aromatic_rings`, `hbd`, `hba`, `tpsa`, `num_rotatable_bonds` |
| `lipinski_rule_of_five(smiles)` | dict with `mw`, `logp`, `hbd`, `hba`, `checks` (four bool flags), `violations` (count), `passes` (bool) |
| `pubchem_get_molecular_formula(name)` | Fetches the molecular formula for a compound name from the PubChem REST API. |
| `chembl_get_ic50(target, compound)` | Fetches the IC50 value of a compound against a target from the ChEMBL database, returned as a `Quantity` in `nM`. |

If `rdkit` is not installed, Dedekind automatically falls back to an integrated pure-Python SMILES parser to compute the primary parameters (`mw`, `logp`, `hbd`, `hba`).

**Physical Units:**
Dedicated life sciences units are registered and fully supported:
- **`percent_wv`**: Percent weight-by-volume ($1\% \text{ w/v} = 10\text{ g/L}$).
- **`g/L`**: Grams per liter (concentration).
- **`mg/mL`**: Milligrams per milliliter (concentration).
- **`bar`**: Pressure unit ($1\text{ bar} = 100\,000\text{ Pa}$).
- **`atm`**: Standard atmosphere pressure ($1\text{ atm} = 101\,325\text{ Pa}$).
- **`angstrom`**: Length unit ($1\,\text{Å} = 10^{-10}\text{ m}$).

Example: `examples/dedekind/bioinformatics_demo.ddk`. Tests: `tests/dedekind/life_sciences_phase{1,2,3}_test.ddk`, `tests/dedekind/bioinformatics_test.ddk`.

### 15.23 try/catch and Slicing Syntax (v1.17)

Two small but long-overdue language features that turn Dedekind from a notebook-DSL into a proper scripting language.

**Exception handling:**

```dedekind
try {
    content = read_file("/maybe/exists.json")
    data = json_parse(content)
} catch e {
    print("Falling back to default:", e)
    data = {}
}
```

- New AST node `TryCatch(body, catch_var, handler)`; lexer keywords `try` and `catch`.
- Codegen emits standard Python `try: ... except Exception as e: ...`. DDK markers are emitted inside the catch block, so source-mapped tracebacks (v1.8) still resolve to the correct `.ddk` line.
- Nested blocks supported; each catch binds the exception to a freely-named variable.
- Not yet in v1.17: type filters (`catch e: ValueError`) and `finally` — both reachable, ask if needed.

**Python-style slicing:**

```dedekind
x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
head    = x[:5]            // [10..50]
tail    = x[5:]            // [60..100]
middle  = x[3:7]           // [40..70]
every2  = x[::2]           // [10, 30, 50, 70, 90]
window  = x[2:8:2]         // [30, 50, 70]
full    = x[:]             // copy
```

- New AST node `Slice(start, stop, step)`, each component optional (`None` = open bound).
- Parser extension in subscript parsing: after `[`, collects up to three `:`-separated components. A single component (no `:`) stays a regular `Subscript(value, index)`; one or more `:` produces a `Slice`.
- Codegen `visit_Slice` renders Python slice notation with empty components for open bounds.
- Works on lists, `torch.Tensor`, numpy arrays — anything supporting `__getitem__(slice)`.
- **PyTorch limitation:** negative step (`x[::-1]` for reverse) raises a runtime error. Use `torch.flip(x, [0])` instead.

Example: `examples/dedekind/try_catch_slicing_demo.ddk` (7 slice variants, `safe_divide` helper, file-read with fallback, nested try blocks). Test: `tests/dedekind/try_catch_slicing_test.ddk`.

### 15.24 Geometric / Clifford Algebra G(3,0) (v1.18)

A native implementation of 3D Euclidean Geometric Algebra — the unified framework that subsumes complex numbers, quaternions, vector calculus and bivector mechanics into a single algebraic system.

```dedekind
e1 = vector(1, 0, 0)
e2 = vector(0, 1, 0)
e3 = vector(0, 0, 1)

// Geometric product = inner + outer
print(e1 * e2)              // e12 (bivector — orthogonal vectors)
print(e1 * e1)              // 1 (basis vectors square to +1 in (3,0))

// Rotor for 90° rotation in the e1-e2 plane
R = rotor(1.5707963, 1, 0, 0)
print(rotate(e1, R))        // ~ e2
print(rotate(e2, R))        // ~ -e1
print(rotate(e3, R))        // e3 (axis-aligned, unchanged)

// Pseudoscalar squares to -1 — same algebraic role as imaginary unit
I = pseudoscalar(1)
print(I * I)                // -1
```

**Storage layout** (bit-pattern indexed):

| Index | Bits | Blade | Grade |
|-------|------|-------|-------|
| 0 | 000 | scalar | 0 |
| 1 | 001 | e1 | 1 |
| 2 | 010 | e2 | 1 |
| 3 | 011 | e12 | 2 |
| 4 | 100 | e3 | 1 |
| 5 | 101 | e13 | 2 |
| 6 | 110 | e23 | 2 |
| 7 | 111 | e123 | 3 |

**Geometric product** of basis blades `a` and `b`: result blade = `a XOR b` (identical factors annihilate in G(3,0)), sign from the number of swaps needed to put `b`'s indices in canonical order.

**API summary:**

| Constructor | Returns |
|-------------|---------|
| `scalar(s)` | scalar multivector |
| `vector(x, y, z)` | grade-1 multivector |
| `bivector(b12, b13, b23)` | grade-2 multivector |
| `pseudoscalar(s)` | grade-3 multivector |
| `multivector(s, e1, e2, e3, e12, e13, e23, e123)` | full multivector |
| `rotor(angle, b12, b13, b23)` | unit rotor `exp(-angle/2 * B)` |

| Operation | Description |
|-----------|-------------|
| `a + b`, `a - b`, `-a` | componentwise |
| `a * b` | geometric product |
| `a.wedge(b)` | outer (grade-raising) |
| `a.dot(b)` | inner (grade-reducing) |
| `a.reverse()` | reverse: sign `(-1)^(k(k-1)/2)` per grade k |
| `a.grade(n)` | extract grade-n part |
| `a.norm()` | `sqrt(<a * ~a>_0)` |
| `a.scalar_part()` | grade-0 component as float |
| `rotate(v, R)` | sandwich product `R v ~R` |

**What v1.18 deliberately does not provide:** no signatures beyond G(3,0). No spacetime algebra G(3,1) or G(1,3), no conformal geometric algebra G(4,1) with points/lines/circles/spheres as primitives. For those, `pyimport clifford` directly. Dedekind's contribution is the canonical 3D cases (robotics, computer graphics, classical physics) as typed native built-ins without an extra dependency.

Example: `examples/dedekind/geometric_algebra_demo.ddk`. Test: `tests/dedekind/geometric_algebra_test.ddk`.

### 15.25 Modules and Visibility (v1.19)

Two language features that turn `use` from a simple file include into a proper module system.

**Dotted module paths:**

```dedekind
use compiler_features.demo_geometry.area  // -> examples/dedekind/compiler_features/demo_geometry/area.ddk
```

`_resolve_module` splits the name on `.` and treats each segment as a directory level (`os.path.join(*parts) + ".ddk"`). This lets you organize larger codebases thematically instead of in a flat folder.

**Public/private functions:**

```dedekind
// examples/dedekind/compiler_features/demo_geometry/area.ddk

fn priv_pi() { return 3.14159265358979 }      // private
fn priv_half(x) { return x / 2.0 }            // private

pub fn circle_area(r) {
    return priv_pi() * r * r                  // private functions still callable within
}

pub fn triangle_area(base, height) {
    return priv_half(base * height)
}
```

After import, `circle_area` and `triangle_area` are visible; `priv_pi` and `priv_half` are not. The compile-time pass `_apply_module_visibility` renames every private function to `__ddk_<flat_modpath>_<name>` and rewrites all call sites inside the module accordingly. Attempting to call a private function from outside raises a runtime error (mangled name not defined).

**Backward compatibility:** modules that have **no** `pub` declarations run in *legacy mode* — every function stays public, matching pre-v1.19 behavior. The standard library modules (`physics`, `stats`, `chemistry`, `biology`, `math`, `ml`, etc.) continue to work unchanged. Adding a single `pub fn` to any of them flips the file into opt-in mode, where everything else becomes private.

**Constraint:** `use` statements must remain at the top level — they are expanded in the `_expand_uses` pre-pass before parsing function bodies. Inline `use` inside a function body is not supported.

Example: `examples/dedekind/compiler_features/multi_file_modules_demo.ddk` (uses the `compiler_features.demo_geometry.area` and `compiler_features.demo_geometry.volume` modules to show visibility, plus the legacy `math` module). Test: `tests/dedekind/multi_file_modules_test.ddk`.

### 15.26 Generics / Type Parameters (v1.20)

Function signatures may now declare explicit type parameters:

```dedekind
fn add_same<U>(a: [U], b: [U]) -> [U] {
    return a + b
}

fn pair<A, B>(a: [A], b: [B]) -> [B] { return b }
```

**Polymorphic unit variables.** A unit annotation `[U]` where `U` is declared in `<...>` is bound at the first argument's actual unit and checked for consistency on subsequent arguments and the return value. Same-dimension-different-scale arguments are auto-converted to the bound unit; dimensional mismatch raises `ValueError`.

```dedekind
add_same(2[m], 3[m])        // U = m         -> 5[m]
add_same(10[kg], 5[kg])     // U = kg        -> 15[kg]
add_same(2[m], 100[cm])     // U = m, auto-convert -> 3[m]
add_same(2[m], 3[kg])       // ValueError: U already bound to [m], here [kg]
```

**Shape parameters.** `<N>` with `Vector[N]`/`Matrix[M, N]`/`Tensor[batch, N, F]` makes the symbolic-dimension mechanism from §15.15 explicit. Behavior is unchanged: symbolic dims bind in a per-call `_shape_env` and enforce cross-argument consistency.

**Implementation.**

- Parser accepts optional `LT ID (COMMA ID)* GT` between the function name and the parameter list.
- AST: `FunctionDef.type_params: List[str]`.
- Codegen initializes a per-call `_unit_env = {}` when type params are declared. Argument-unit annotations matching a type param dispatch to `_check_param_unit(value, "U", fn_name, arg_name, _unit_env)`; other unit annotations keep the classic `_check_signature_unit` path. Same dispatch for the return value (`_check_return_param_unit` vs `_check_return_unit`).
- Runtime `_check_param_unit` binds the parameter to the actual unit on first occurrence, then for subsequent occurrences checks dimensional compatibility, auto-converts via `_coerce_to_expected_unit` if possible, or raises `ValueError` otherwise.
- Static units checker skips concrete-unit comparisons for type-parameter units (they are validated at runtime instead).

**Backward compatibility.** Functions without `<...>` are unaffected — `type_params` defaults to empty and all checks take the classic path.

**Difference from Python's `typing.TypeVar`.** Python's generics are documentary; the interpreter ignores them. Dedekind's type parameters are actively enforced by the compiler. Dimensionally inconsistent calls abort; dimensionally compatible-but-differently-scaled calls are auto-converted. This makes Dedekind's static-feeling type system actually meaningful at runtime.

Example: `examples/dedekind/generics_demo.ddk`. Test: `tests/dedekind/generics_test.ddk`.

### 15.27 Quantum Computing Bridge (v1.21)

Dedekind v1.21 provides a native, zero-dependency quantum computing layer. A Qiskit bridge is available for exporting to real hardware (`pyimport qiskit`), but all simulation runs within the Dedekind runtime.

#### 15.27.1 QuantumCircuit

```dedekind
qc = quantum_circuit(2)    // 2-qubit circuit
qc.h(0)                    // Hadamard on qubit 0
qc.cx(0, 1)               // CNOT: control=0, target=1
```

Gate methods: `.h(q)`, `.x(q)`, `.y(q)`, `.z(q)`, `.t(q)`, `.s(q)`, `.cx(ctrl, tgt)`, `.cz(ctrl, tgt)`, `.rx(θ, q)`, `.ry(θ, q)`, `.rz(θ, q)`, `.swap(q0, q1)`, `.measure(q)`, `.measure_all()`.

Properties: `.n_qubits`, `.n_gates()`, `.depth()`.

**Qubit convention:** qubit 0 is the least significant bit (LSB) of the state index. State |q_{n-1}...q_1 q_0> has integer index sum q_i * 2^i.

#### 15.27.2 Statevector Simulation

```dedekind
sv = statevec_sim(qc)           // returns list[complex], length 2^n
sv = statevec_sim(qc, 1000)    // 1000 shots -> dict[bitstring, count]
probs = statevec_probs(qc)      // probabilities |psi_i|^2
exp_zz = statevec_expectation(qc, "ZZ")   // <psi|ZZ|psi>
```

The Pauli-string observable must have exactly `n_qubits` characters from {I, X, Y, Z, H}.

#### 15.27.3 Convenience Constructors

```dedekind
phi_plus = bell_state(0)    // |Phi+> = (|00>+|11>)/sqrt(2); which in {0,1,2,3}
ghz = ghz_state(3)          // (|000>+|111>)/sqrt(2) for n qubits
grover = grover_circuit(3, 5)  // Grover search for |101> in 3-qubit space
```

#### 15.27.4 VQE (Variational Quantum Eigensolver)

```dedekind
// Hardware-efficient ansatz: Ry layers + linear CNOT chain
qc = vqe_circuit(2, 2, params)         // n_qubits=2, n_layers=2
terms = [[0.5, "ZI"], [-0.5, "IZ"]]   // Hamiltonian as (coeff, Pauli-string) pairs
e = vqe_energy(params, 2, 2, terms)   // <psi(theta)|H|psi(theta)>
```

#### 15.27.5 Quantum Information Utilities

```dedekind
f = fidelity(sv1, sv2)             // |<psi1|psi2>|^2
s = entropy_von_neumann(probs)     // -sum p_i log2 p_i (from probability list)
r = schmidt_rank(sv, n_a)         // Schmidt rank for A|B bipartition
```

#### 15.27.6 Physical Unit Validation

Quantum hardware parameters carry SI units:

```dedekind
freq = 5.1[GHz]
freq_ok = qubit_frequency_check(freq)  // Validates [GHz/MHz/THz], returns in GHz

t1 = 100.0[us]
t1_ok = coherence_time_check(t1)       // Validates [us/ns/ms/s]

gap = 10.5[meV]
gap_ok = energy_gap_check(gap)         // Validates [eV/meV/J]
```

New energy units: `eV` (1.602e-19 J), `meV` (1.602e-22 J), `MeV` (1.602e-13 J). New frequency unit: `THz` (1e12 Hz).

#### 15.27.7 Shape Annotations

```dedekind
fn apply_h(qc: Qubit[N]) -> Qubit[N] { qc.h(0); return qc }
fn check_fid(sv1: StateVec[2], sv2: StateVec[2]) { return fidelity(sv1, sv2) }
```

| Annotation | Validates | Symbolic dim? |
|-----------|-----------|---------------|
| `Qubit[N]` | `QuantumCircuit` with `n_qubits == N` | Yes |
| `Circuit[N, G]` | `QuantumCircuit` with `n_qubits == N` | Yes |
| `StateVec[N]` | `list[complex]` with length `2^N` | Yes |

#### 15.27.8 Qiskit Bridge (Optional)

```dedekind
pyimport qiskit as qk
qc = quantum_circuit(2)
qc.h(0).cx(0, 1)
qk_circuit = qc.to_qiskit()   // Requires: pip install qiskit
```

Examples: `quantum_bell.ddk`, `quantum_grover.ddk`, `quantum_vqe.ddk`, `quantum_units.ddk`, `quantum_shapes.ddk`. Tests: `quantum_circuit_test.ddk`, `quantum_units_test.ddk`.

### 15.28 Meteorology, Climatology & Geosciences (v2.0)

Dedekind provides native support for geoscientific, climatological, and meteorological calculations and physical quantities.

#### 15.28.1 Geophysical Units

A new unit of atmospheric pressure is supported:
- **`hPa` (hectopascal)**: Registered under the `pressure` dimension with a factor of `100.0` against the base unit `Pa` (Pascal).

#### 15.28.2 Geophysics Built-ins

The following geophysics functions are globally available:

| Function | Description | Return Unit |
|----------|-------------|-------------|
| `coriolis_parameter(latitude)` | Calculates the Coriolis parameter $f = 2\Omega\sin(\varphi)$. `latitude` can be a scalar, tensor, or Quantity (`[deg]` or `[rad]`). | `s^-1` |
| `saturated_vapor_pressure(T)` | Calculates the saturated vapor pressure of water vapor using the Clausius-Clapeyron Magnus formula. `T` is the air temperature (Kelvin). | `Pa` |
| `dew_point(T, RH)` | Calculates the dew point temperature from air temperature `T` (Kelvin) and relative humidity `RH` (0..100). | `K` |
| `seismic_wave_velocities(K, G, rho)` | Calculates the compression ($v_p$) and shear ($v_s$) wave velocities in a medium. `K` is Bulk modulus (Pa), `G` is Shear modulus (Pa), and `rho` is Density ($\text{kg}/\text{m}^3$ or $\text{g}/\text{cm}^3$). | List of Quantities `[v_p, v_s]` in `m/s` |

These functions are fully differentiable and compatible with Dedekind's autograd system (e.g. `jacobian` and `grad` operations).

Example standard geosciences calculations: Test: `tests/dedekind/geosciences_test.ddk`.

---

### 15.29 Electrical Engineering & Control Theory (v2.1)

Dedekind v2.1 introduces the concept of **Differentiable Engineering**, providing native integration of Electronic Circuit Simulation and Linear Time-Invariant (LTI) Control Theory directly into PyTorch's Autograd ecosystem.

Both circuit simulation and block-diagram control live in the unified
`use signals` module (along with DSP filters).

**1. Differentiable Circuit Simulation**

The `Circuit` primitive allows you to declaratively model an electrical network and solve for its DC node voltages and branch currents using Modified Nodal Analysis (MNA). Because the solver operates seamlessly on Autograd-enabled tensors, all component parameters can be optimized via `jacobian` or `minimize`.

```dedekind
use signals

c = circuit()
c.add_voltage_source("V1", 1, 0, 10.0[V])  // Node 1 to 0 (GND)
c.add_resistor("R1", 1, 2, 100.0[ohm])
c.add_resistor("R2", 2, 0, 100.0[ohm])

res = c.solve_dc()
v2 = res["v_2"]  // = 5.0[V]
```

**2. Block-Diagram Control**

Dedekind provides differentiable block-diagram primitives for closed-loop
control. Plants are expressed as `transfer_function_block` or
`state_space_block`; controllers (`pid_block`, `pid_block_saturated`) and
arithmetic blocks (`sum_block`, `gain_block`, `saturation_block`) compose
into a `block_diagram` that can be simulated and optimized end-to-end via
autograd.

```dedekind
use signals

// First-order plant: dx/dt = -x + u
plant = state_space_block(u, [[-1.0]], [[1.0]], [[1.0]], [[0.0]])
ref   = constant_block(1.0)
err   = sum_block([ref, plant], [1.0, -1.0])
ctrl  = pid_block(err, Kp, Ki, Kd)
```

Both models enforce Dedekind's strict unit rules for voltage, current, resistance, time, and states.

---

### 15.30 Differentiable Robotics (v2.1)

Dedekind v2.1 introduces native differentiable robotics modeling, enabling kinodynamic analysis and analytical Jacobian extraction of robotic manipulators directly within PyTorch's Autograd environment.

#### 15.30.1 Kinematic Chain Modeling (`use robotics`)

To construct a robotic system, Dedekind provides the `KinematicChain` class representing a serial-link manipulator defined by Denavit-Hartenberg (DH) parameters.

| Method / Built-in | Description |
|-------------------|-------------|
| `kinematic_chain()` | Instantiates and returns a new empty `KinematicChain` object. |
| `add_revolute_joint(d, a, alpha)` | Appends a revolute joint to the chain. Parameter `d` is joint offset (Quantity/scalar), `a` is link length (Quantity/scalar), and `alpha` is link twist (Quantity/scalar in radians/degrees). |
| `add_prismatic_joint(theta, a, alpha)` | Appends a prismatic joint to the chain. Parameter `theta` is joint angle, `a` is link length, and `alpha` is link twist. |
| `forward_kinematics(joint_vars)` | Computes the end-effector homogeneous transformation matrix $T$ (a $4 \times 4$ tensor) given a list/tensor of active joint variables (radians for revolute joints, meters for prismatic joints). |

The following global utility functions extract position and orientation from a homogeneous transformation matrix:
- **`end_effector_pos(T)`**: Extracts the 3D position vector $[x, y, z]^T$ as a tensor.
- **`end_effector_rot(T)`**: Extracts the $3\times3$ rotation matrix.

#### 15.30.2 Robotic Jacobian & Sensitivity via Autograd

Since forward kinematics propagates through Autograd-enabled tensors, users can extract the robotic Jacobian matrix analytically using Dedekind's built-in `jacobian` operator.

```dedekind
use robotics

arm = kinematic_chain()
arm.add_revolute_joint(0.0[m], 1.0[m], 0.0[rad])
arm.add_revolute_joint(0.0[m], 1.0[m], 0.0[rad])

fn fk_pos(q) {
    T = arm.forward_kinematics(q)
    return end_effector_pos(T)
}

// Joint angles (q1 = 45 deg, q2 = 45 deg)
q = [0.785398, 0.785398]

// Analytical Jacobian J(q) = d(pos)/d(q)
J = jacobian(fk_pos, q)
```

Example: `tests/dedekind/robotics_test.ddk`.

---

### 15.31 Differentiable Space Physics & Orbital Mechanics (v2.0)

Dedekind v2.0 supports orbital mechanics modeling and N-body physical simulation with full differentiability.

#### 15.31.1 Kepler's Equation Solver

Kepler's equation relates the mean anomaly $M$ and the eccentric anomaly $E$ for an elliptic orbit:
$$M = E - e\sin(E)$$

- **`kepler_solve(M, ecc)`**: Numerically solves Kepler's equation for eccentric anomaly $E$ given mean anomaly $M$ (scalar/tensor) and eccentricity $ecc$ (0 to 1). Fully differentiable.

#### 15.31.2 Coordinate Conversions

Dedekind supports conversion between Keplerian orbital elements and Cartesian state vectors (position and velocity vectors).

- **`kepler_to_cartesian(a, ecc, inc, Omega, omega, nu, mu)`**: Converts Keplerian orbital elements to Cartesian state. Returns a dictionary containing `"position"` ($[x, y, z]^T$ tensor) and `"velocity"` ($[v_x, v_y, v_z]^T$ tensor).
- **`cartesian_to_kepler(r, v, mu)`**: Converts Cartesian position `r` and velocity `v` vectors to Keplerian orbital elements. Returns a dictionary with keys `"a"`, `"ecc"`, `"inc"`, `"Omega"`, `"omega"`, and `"nu"`.

#### 15.31.3 Differentiable N-Body Simulation

- **`n_body_simulate(pos_init, vel_init, masses, dt, steps)`**: Runs a symplectic integration of $N$ point masses under mutual Newtonian gravitation. `pos_init` is a shape $[N, 3]$ tensor of initial positions, `vel_init` is a shape $[N, 3]$ tensor of initial velocities, `masses` is a shape $[N]$ tensor of masses, `dt` is time-step size, and `steps` is simulation step count. Returns a dictionary containing `"positions"` (tensor of shape $[steps, N, 3]$) and `"velocities"` (tensor of shape $[steps, N, 3]$).

```dedekind
use space

// Integrate orbital trajectory under gravity and backpropagate to optimize launch velocity
fn loss_fn(v_init_y) {
    pos_init = [[0.0, 0.0, 0.0], [7000000.0, 0.0, 0.0]]
    vel_init = [[0.0, 0.0, 0.0], [0.0, v_init_y, 0.0]]
    masses = [5.972e24, 1000.0]
    
    res = n_body_simulate(pos_init, vel_init, masses, 10.0, 5)
    final_pos = res["positions"][4][1] // step 5, satellite
    return final_pos[1] // Y coordinate
}

jac = jacobian(loss_fn, 7546.0)
```

Example: `tests/dedekind/space_test.ddk`.

---

### 15.32 Differentiable Structural Mechanics (v2.0)

Dedekind v2.0 integrates 2D static Finite Element Analysis (FEA) and topology optimization.

#### 15.32.1 Mesh and FE Solver

- **`structural_mesh_2d(nelx, nely)`**: Instantiates a 2D rectangular structural mesh of dimensions `nelx` by `nely`.
- **`structural_solve_2d(mesh, densities, loads, fixed_dofs)`**: Solves the static equilibrium equation $\mathbf{K}(\mathbf{\rho})\mathbf{U} = \mathbf{F}$ using the solid isotropic material with penalization (SIMP) method. Returns the displacement vector $\mathbf{U}$.
- **`structural_compliance_2d(mesh, densities, loads, fixed_dofs)`**: Computes the structural compliance (flexibility) $c = \mathbf{U}^T \mathbf{K} \mathbf{U} = \mathbf{F}^T \mathbf{U}$. Minimizing compliance maximizes structural stiffness.

#### 15.32.2 2D Truss Solver

- **`structural_solve_truss_2d(nodes, elements, E, A, loads, fixed_dofs)`**: Solves a 2D pin-jointed truss system under load. Returns the global joint displacement vector $\mathbf{U}$ of size `2*N`.
- **`structural_truss_stress_2d(nodes, elements, E, U)`**: Computes the axial stress $\sigma = \frac{E}{L} \mathbf{u}_e$ for each member. Returns a vector of member stresses of size `M`.

#### 15.32.3 Dynamic Modal Analysis

- **`structural_modal_2d(mesh, densities, fixed_dofs)`**: Computes the natural frequencies (in Hz) and mode shapes of a 2D Q4 grid structure. Returns a tuple `(frequencies, eigenvectors)`.
- **`structural_modal_2d_advanced(mesh, densities, fixed_dofs, rho, num_modes)`**: Advanced interface allowing a custom material density factor `rho` and specifies how many mode shapes to return.

#### 15.32.4 Section Capacity & Column Buckling

- **`concrete_beam_capacity(b, h, d, As, fprime_c, fy)`**: Calculates the nominal and LRFD design moment capacity of a rectangular reinforced concrete section under Ultimate Limit State (ULS) using ACI 318 criteria. Returns `(Mn, Md, eps_s, c, phi)`.
- **`concrete_beam_capacity_advanced(b, h, d, As, fprime_c, fy, Es)`**: Advanced capacity check supporting a custom steel elasticity modulus `Es`.
- **`steel_buckling_check(A, r, L, K, E, fy)`**: Performs steel column buckling checks according to AISC 360-16 LRFD and ASD methods. Returns `(Pn, Pd, Pa, Fe, Fcr, lambda)`.

#### 15.32.5 Topology Optimization

- **`topo_opt_oc_2d_advanced(mesh, loads, fixed_dofs, volfrac, max_steps, penalty, filter_radius)`**: Performs topology optimization using the Optimality Criteria (OC) method to distribute material within the mesh.
- **`print_structural_topology_2d(densities, nelx, nely, threshold=0.5)`**: Outputs an ASCII representation of the structure to print the density layout.

```dedekind
use structural

mesh = structural_mesh_2d(10, 4)
fixed_dofs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] // left end fixed
loads = linspace(0.0, 0.0, 110)
loads[101] = -1.0 // vertical load at tip

// Optimize layout for 50% volume fraction
opt_densities = topo_opt_oc_2d_advanced(mesh, loads, fixed_dofs, 0.5, 5, 3.0, 1.5)
print_structural_topology_2d(opt_densities, 10, 4)
```

Example: `tests/dedekind/structural_test.ddk`.

---

### 15.33 Differentiable Heat Transfer & Thermodynamics (v2.0)

Dedekind v2.0 provides finite element/finite difference solvers for 2D steady-state and transient heat equation simulation, coupled with thermal topology optimization.

#### 15.33.1 Steady-State and Transient Thermal Solvers

- **`thermal_mesh_2d(nelx, nely)`**: Creates a 2D thermal mesh grid.
- **`thermal_solve_2d(mesh, conductivities, heat_sources, fixed_nodes, fixed_temps)`**: Solves the steady-state heat equation $\nabla \cdot (k \nabla T) + Q = 0$ for nodal temperatures $T$.
- **`thermal_solve_transient_2d(mesh, conductivities, capacities, initial_temps, heat_sources, fixed_nodes, fixed_temps, dt, steps)`**: Solves the transient heat equation $\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$. Returns the nodal temperatures $T$ at the final simulation step.

#### 15.33.2 Thermal Topology Optimization

- **`topo_opt_thermal_oc_2d_advanced(mesh, heat_sources, fixed_nodes, volfrac, max_steps, penalty, filter_radius)`**: Optimizes the distribution of high-conductivity material to minimize the thermal compliance (average temperature) under a volume constraint.
- **`print_thermal_topology_2d(densities, nelx, nely, threshold=0.5)`**: Prints an ASCII map of thermal material densities.

```dedekind
use thermal

mesh = thermal_mesh_2d(10, 4)
fixed_nodes = [0.0] // bottom-left node fixed
fixed_temps = [100.0]
heat_sources = linspace(0.0, 0.0, 55)

conductivities = linspace(1.0, 1.0, 40)
capacities = linspace(1.0, 1.0, 40)
initial_temps = linspace(20.0, 20.0, 55)

// Simulate transient thermal diffusion
T_final = thermal_solve_transient_2d(mesh, conductivities, capacities, initial_temps, heat_sources, fixed_nodes, fixed_temps, 0.1, 5)
```

Example: `tests/dedekind/thermal_test.ddk`.

---

### 15.34 Computational Fluid Dynamics (v2.0)

Dedekind v2.0 integrates a 2D Lattice Boltzmann Method (LBM) CFD simulation engine with D2Q9 discretization and differentiable boundary/obstacle interaction.

#### 15.34.1 Simulation and Iteration Built-ins

- **`lbm_simulation(nx, ny, viscosity)`**: Initializes a D2Q9 LBM grid of shape `[nx, ny]` with a given kinematic viscosity.
- **`lbm_simulation_with_mask(nx, ny, viscosity, mask)`**: Initializes an LBM grid with a solid obstacle fraction mask tensor of shape `[nx, ny]`.
- **`simulation_step(sim, inflow_velocity)`**: Propagates the LBM state (stream and collide phases) by one time step with a given inflow boundary velocity.
- **`simulation_run(sim, steps, inflow_velocity)`**: Iterates the simulation for `steps` steps.
- **`simulation_get_density(sim)`**: Returns the current density field as a tensor of shape `[nx, ny]`.
- **`simulation_get_velocity(sim)`**: Returns the current velocity field as a tensor of shape `[2, nx, ny]`.
- **`simulation_get_drag_lift(sim)`**: Computes the integrated drag and lift force components $[F_x, F_y]$ acting on the obstacle mask.

#### 15.34.2 Differentiable Solid Mask Generators

To enable gradient-based design of aerodynamic shapes, masks are generated using soft, differentiable boundary functions:
- **`soft_cylinder_mask(nx, ny, cx, cy, r, sigma)`**: Generates a soft cylinder mask centered at `(cx, cy)` with radius `r` and boundary thickness `sigma`.
- **`soft_airfoil_mask(nx, ny, cx, cy, chord, angle, thickness)`**: Generates a soft NACA-style airfoil mask.

```dedekind
use fluid_dynamics

fn get_drag(r_arr) {
    r = r_arr[0]
    mask = soft_cylinder_mask(20, 10, 8.0, 5.0, r, 0.5)
    sim = lbm_simulation_with_mask(20, 10, 0.8, mask)
    simulation_run(sim, 4, 0.05)
    forces = simulation_get_drag_lift(sim)
    return [forces[0]] // drag force
}

// Compute analytical derivative of drag w.r.t cylinder radius
d_drag_d_r = jacobian(get_drag, [1.5])
```

Example: `tests/dedekind/fluid_dynamics_test.ddk`.

---

### 15.35 Digital Signal Processing (v2.0)

Dedekind v2.0 includes native Digital Signal Processing (DSP) primitives, enabling differentiable FIR/IIR filtering and filter coefficient design.

#### 15.35.1 Filtering Built-ins

- **`fir_filter(x, b)`**: Convolves 1D signal `x` (tensor/list) with FIR filter coefficients `b`.
- **`iir_filter(x, b, a)`**: Computes the output of an IIR filter with numerator coefficients `b` and denominator coefficients `a` on signal `x` using direct form II transposed representation.
- **`freqz(b, a, n_freqs)`**: Computes the complex frequency response $H(e^{j\omega})$ evaluated at `n_freqs` frequencies from $0$ to $\pi$. Returns a list `[omega, H]`.

#### 15.35.2 Differentiable Biquad & Filter Design Helpers

- **`biquad_lowpass(fc, Q, fs)`**: Designs a biquad lowpass filter. Returns a list `[b, a]` containing coefficients.
- **`biquad_highpass(fc, Q, fs)`**: Designs a biquad highpass filter. Returns `[b, a]`.
- **`biquad_bandpass(fc, Q, fs)`**: Designs a biquad bandpass filter. Returns `[b, a]`.
- **`butter(order, Wn)`**: Designs an N-th order digital Butterworth filter with normalized cutoff frequency `Wn`. Returns `[b, a]`.
- **`cheby1(order, rp, Wn)`**: Designs an N-th order digital Chebyshev Type I filter with peak-to-peak passband ripple `rp` (dB) and normalized cutoff frequency `Wn`. Returns `[b, a]`.

```dedekind
use signals

fn get_lowpass_b0(fc) {
    res = biquad_lowpass(fc, 0.707, 1.0)
    b = res[0]
    return b[0]
}

// Compute sensitivity of b0 coefficient w.r.t cutoff frequency
d_b0_d_fc = jacobian(get_lowpass_b0, [0.2])
```

Example: `tests/dedekind/dsp_test.ddk`.

---

### 15.36 Differentiable Atomic & Molecular Physics / Chemistry (use atomic)

Dedekind introduces support for differentiable molecular dynamics simulations and crystallography calculations under the unified `use atomic` module. All solvers, structure factor calculators, and geometry descriptors are fully differentiable, enabling analytical gradients to be computed automatically via PyTorch autograd.

#### 15.36.1 Velocity Verlet LJ Simulation Solvers

- **`molecular_lj_simulate(positions, velocities, masses, box_size, dt, steps)`**: Simulates a system of particles interacting via the Lennard-Jones potential using the Velocity Verlet algorithm. It applies periodic boundary conditions (PBC) with the minimum image convention. Returns a dictionary containing the histories of positions, velocities, potential energies, kinetic energies, and temperatures.
- **`molecular_lj_simulate_advanced(positions, velocities, masses, box_size, dt, steps, epsilon, sigma, cutoff, thermostat_tau, target_temp)`**: Advanced interface providing custom Lennard-Jones parameters ($\epsilon, \sigma$), cutoff radius, and NVT Berendsen thermostat velocity scaling parameters ($\tau$ and target temperature). A non-positive `thermostat_tau` disables velocity scaling.

#### 15.36.2 Potential Functions

- **`morse_potential(r, De, a, re)`**: Computes the differentiable Morse potential energy for a bond distance $r$:
  $$V(r) = D_e \left(1 - e^{-a(r - r_e)}\right)^2$$
  where $D_e$ is the well depth, $a$ controls the width of the potential, and $r_e$ is the equilibrium bond length.

#### 15.36.3 Molecular Geometry Descriptors

These descriptors are numerically stabilized with a small $\epsilon = 10^{-12}$ factor to prevent NaN or division-by-zero errors in backpropagation at boundaries.
- **`molecular_distance(pos1, pos2)`**: Calculates the Euclidean distance between two atomic positions `pos1` and `pos2`.
- **`molecular_angle(pos1, pos2, pos3)`**: Calculates the bond angle (in radians) formed by the three positions, with `pos2` as the central atom.
- **`molecular_dihedral(pos1, pos2, pos3, pos4)`**: Calculates the dihedral (torsional) angle (in radians) formed by four connected atomic positions.

#### 15.36.4 Crystallography & Structure Analysis

- **`cryst_symmetry_apply(coords, R, t)`**: Applies a Seitz matrix (rotation $R$, translation $t$) to fractional coordinates and wraps the coordinates periodically modulo 1.0.
- **`cryst_generate_equivalent_atoms(coords, R_ops, t_ops)`**: Vectorized application of a set of symmetry operations ($R_{ops}$, $t_{ops}$) to a set of coordinates.
- **`cryst_find_symmetries(coords, elements, R_ops, t_ops)`**: Evaluates which candidate symmetry operations are active for a given atomic structure within a default tolerance of $10^{-3}$.
- **`cryst_find_symmetries_advanced(coords, elements, R_ops, t_ops, tol)`**: Similar to `cryst_find_symmetries`, but with an explicit distance tolerance parameter `tol`.
- **`cryst_structure_factor_atoms(coords, scattering_factors, hkl_indices)`**: Computes the complex structure factor $F(hkl)$ directly from atomic coordinates, allowing differentiable crystal structure refinement.
- **`cryst_structure_factor_density(density_map)`**: Calculates structure factors from a 3D electron density grid using the Fast Fourier Transform (3D FFT).

```dedekind
use atomic

// Compute Morse potential and its derivative
De = 1.0
a = 1.0
re = 1.5

fn get_morse_pe(r_arr) {
    r = r_arr[0]
    return [morse_potential(r, De, a, re)]
}

// Jacobian at r = 1.0
jac = jacobian(get_morse_pe, [1.0])
```

Example: `tests/dedekind/molecular_test.ddk` and `tests/dedekind/crystallography_test.ddk`.

---

*This document is the Markdown source for the Language Specification. PDF can be generated e.g. with `pandoc Dedekind_Language_Specification.md -o Dedekind_Language_Specification_v0.2.pdf`.*

