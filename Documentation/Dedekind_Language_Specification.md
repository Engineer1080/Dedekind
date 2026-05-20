# Dedekind — A Modern Programming Language for Machine Learning and Graphics

**Language Specification v0.2**  
Mario Michael Heinrich · github.com/Engineer1080  
Draft: January 2026 · Updated for v0.6 (Physical Units), v0.7 (ODE), v0.8 (Probabilistic, PDE), v0.9 (Distributions, Integration), v0.9.1 (Dokumentation, Run-Examples), v0.9.2 (pi, e, CODATA), v0.9.3 (Uncertainty, Fitting), v0.9.4 (HMC, LaTeX-Export), v0.9.5 (Bessere Fehlermeldungen, Einheiten Compile-Zeit), v0.9.6 (Math-Funktionen), v0.9.7 (Chemie/Biologie: mol, L, M, ppm), v0.9.8 (Convenience, Elemente, Datei-I/O, Netzwerk, JSON), v1.0.0 (Release), v1.0.1–v1.0.6 (Patch), v1.2.6 (Winkel rad/deg, deg_to_rad, rad_to_deg), v1.2.7 (Dirichlet-Verteilung, dirichlet_function), v1.2.8 (Dedekind-Schnitte, Dedekind-Ringe, Riemann-Zeta, Riemann-Summen), v1.2.9 (Betragsstriche, Rotationskörper, logische Operatoren), v1.3.0 (integrate_sym, Lagrange/Hamilton, Lotka-Volterra, chemisches Gleichgewicht), v1.3.1 (Medizin, Pharmakologie, Epidemiologie), v1.4.0 (Modul-System `use`, Seed/data_hash, DataFrame+CSV/Parquet/HDF5/NetCDF, Unit-aware Plots, `@units`-Signaturen mit `fn f(x: [m]) -> [J]`, Dict-Literale), v1.5.0 (benchmark/profile/time_block, jit (torch.compile), sde_solve (Euler-Maruyama, Milstein), least_squares, minimize_constrained, milp, FEM-Primitiven mesh_unit_square/fem_assemble_*/fem_poisson_2d, `arange` int64), v1.6.0 (solve_sym/simplify_sym/series, cg/gmres/bicgstab + jacobi_/ilu_preconditioner, export_notebook (HTML/MD), print_table (markdown/latex/csv/plain) mit UncertainQuantity-±), v1.6.3 (Hardware-Beschleunigung & JIT-Optimierung (torch.compile), statischer Compile-Time Unit Checker, Runtime-Konsolidierung in ml_runtime.py)

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
15. [**Physical Units and Universal Constants (v0.6)**](#15-physical-units-and-universal-constants-v06) (incl. §15.7 ODE, §15.8 Probabilistic, §15.9 PDE, §15.10 Integration & Math v0.9, §15.11 Uncertainty & Fitting)
    - [15.12 LaTeX-Export & Unicode Console](#1512-latex-export--unicode-console)
    - [15.13 Chemistry & Biology Elements](#1513-chemistry--biology-elements)
    - [15.14 File I/O, Network & JSON APIs](#1514-file-io-network--json-apis)
    - [15.15 Medicine, Pharmacology & Epidemiology](#1515-medicine-pharmacology--epidemiology)
    - [15.16 Modul-System, Reproduzierbarkeit & DataFrames](#1516-modul-system-reproduzierbarkeit--dataframes)
    - [15.17 Benchmarking, JIT, SDEs & FEM](#1517-benchmarking-jit-sdes--fem)
    - [15.18 Symbolics, Sparse Krylov Solvers & Publishing](#1518-symbolics-sparse-krylov-solvers--publishing)
    - [15.19 Hardware-Beschleunigung & Unit Checking (v1.6.3)](#1519-hardware-beschleunigung--unit-checking-v163)


---

## 1. Introduction

Dedekind is a modern, high-performance programming language designed specifically for compute-intensive workloads in machine learning and graphics rendering. Named after Richard Dedekind, whose work on real numbers (Dedekind cuts) and algebraic structures laid foundations for rigorous mathematical computation, the language embodies the principle of precise, structured transformation of data through parallel computation.

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
- AOT Compilation (MLIR/LLVM), Dedekind Studio IDE (v0.6).

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

Modules for ML, linalg, signal (FFT), visualization (`plot()`), scientific console (`print_latex(s)` for LaTeX-rendered output in Dedekind Studio/Jupyter), and runtime types (`Quantity`, `Quaternion`, `Dense`, `Sequential`).

## 11. Use Cases

ML training/inference, graphics rendering, scientific computing, signal processing, physics simulations (with units and constants).

## 12. Implementation Roadmap

Phases 1–12 implemented in prototype (Python backend, Dedekind Studio, PyTorch runtime, Ricci, Sparse, AOT, Quaternion, **Physical Units v0.6**). See main README for detailed roadmap.

## 13. Technical Foundation

MLIR, LLVM, PyTorch; research in automatic differentiation, GPU compilation (Triton), type systems, work-stealing.

## 14. Conclusion

Dedekind aims to bridge productivity and performance for compute-intensive applications. This specification is updated to **v0.2** to reflect current prototype behaviour and **v0.6** language features (physical units and constants).

---

## 15. Physical Units and Universal Constants (v0.6)

This section documents the **physical units** and **universal constants** features introduced in Dedekind v0.6 (Option B: Einheiten-Literale und Konstanten als Quantity).

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

**Trigonometrie:** **`sin(x)`**, **`cos(x)`**, **`tan(x)`**.

**Exponential und Logarithmus:** **`exp(x)`**, **`log(x)`** (natürlicher Logarithmus), **`log10(x)`**.

**Wurzel und Betrag:** **`sqrt(x)`**, **`abs(x)`**.

**Arkusfunktionen (Bogenmaß):** **`asin(x)`**, **`acos(x)`**, **`atan(x)`**, **`atan2(y, x)`** (Winkel der Richtung (x,y), Wertebereich \((-\pi,\pi]\)).

**Hyperbelfunktionen:** **`sinh(x)`**, **`cosh(x)`**, **`tanh(x)`**.

**Reduktionen:** **`min(x, dim=None)`**, **`max(x, dim=None)`** — Minimum/Maximum; **`argmin(x, dim=None)`**, **`argmax(x, dim=None)`** — Index des Min/Max.

**Runden:** **`round(x)`**, **`floor(x)`**, **`ceil(x)`**.

**Dirichlet-Funktion:** **`dirichlet_function(x)`** — Dirichlet function D(x): returns 1 if x is rational (denominator ≤10000, tolerance 1e-6), else 0; element-wise for tensors. See `examples/dedekind/dirichlet_distribution_function.ddk`.

**Dedekind-Schnitte:** **`DedekindCut(x)`** — Dedekind cut representing real x as lower set A = {q ∈ Q : q < x}. **`dedekind_cut_from_rational(p, q)`** — cut for rational p/q. **`dedekind_cut_sqrt2()`** — cut for √2. Methods: `lower_set_contains(cut, q)`, `to_float()`; arithmetic `+`, `-`, `*`, comparisons. See `examples/dedekind/dedekind_cuts_rings.ddk`.

**Dedekind-Ringe:** **`DedekindRingZ()`** — ring of integers Z as Dedekind domain. **`ideal(n)`** — principal ideal (n) in Z. **`ideal_factor(i)`** — factor ideal into prime ideals; for Z: (n) = ∏ (p_i)^(e_i). **`DedekindIdeal`** has `.factor()`, `.norm()`, `*` (ideal multiplication). See `examples/dedekind/dedekind_cuts_rings.ddk`.

**Riemann-Zeta:** **`zeta(s)`** — Riemann zeta function \(\zeta(s) = \sum_{n=1}^\infty 1/n^s\); uses scipy. For s=1 the series diverges (returns inf). See `examples/dedekind/riemann_zeta_sums.ddk`.

**Lineare Algebra:** **`norm(x, p=None, dim=None)`** — Vektor- oder Matrixnorm (default p=2); **`det(A)`** — Determinante; **`trace(A)`** — Spur.

Examples: \(\int_0^1 x^2\,dx = 1/3\), \(\int_0^\pi \sin(x)\,dx = 2\); see `examples/dedekind/integration.ddk`. Full math showcase: `examples/dedekind/math_functions.ddk`.

### 15.11 Uncertainty Propagation and Fitting (v0.9.3)

**Uncertainty Propagation (Fehlerfortpflanzung):**

- **`uncertain(value, std, unit="")`**: Creates an `UncertainQuantity` representing value ± std (Gaussian error propagation). Optional `unit` for physical quantities (e.g. `"m"`).
- **Arithmetic**: For +, -, *, /, ^ the standard deviation is propagated via the Gaussian approximation: e.g. for \(z = x + y\), \(\sigma_z^2 = \sigma_x^2 + \sigma_y^2\); for \(z = x \cdot y\), \((\sigma_z/z)^2 = (\sigma_x/x)^2 + (\sigma_y/y)^2\).
- **Display**: `repr` shows „value ± std [unit]“. See `examples/dedekind/uncertainty_propagation.ddk`.

**Fitting / Regression:**

- **`fit(loss_fn, params_init, data, method="gd"|"mcmc", lr=0.01, steps=500)`**: Minimizes `loss_fn(params, data)` with respect to `params`. `params_init` is the initial parameter tensor (or list); `data` is passed to `loss_fn`. With `method="gd"` (default), gradient descent is used (in-place updates); with `method="mcmc"`, Metropolis-Hastings is used (returns posterior samples). Returns the optimal parameters (tensor) for GD, or samples tensor for MCMC.
- Example: linear regression \(y = a\,x + b\); see `examples/dedekind/curve_fitting.ddk`.

### 15.12 LaTeX-Export & Unicode Console

Dedekind prioritizes publication-quality output and scientific presentation. Math formulas can be automatically compiled into LaTeX notation or displayed as beautiful Unicode glyphs in compatible consoles.

#### Functions and Options

| Function / CLI Option | Signature / Usage | Description |
|:---|:---|:---|
| `export_to_latex(source)` | `export_to_latex(source: string) -> string` | Parses a Dedekind source code string and transpiles the mathematical equations into standard LaTeX math blocks. |
| `--latex` | CLI option: `python -m src.compiler.compiler <file.ddk> --latex` | Instructs the compiler to output a transpiled LaTeX representation of the code's mathematical expressions to standard output. |
| `print_latex(s)` | `print_latex(s: string) -> nil` | Renders mathematical LaTeX expressions directly in compatible scientific consoles (e.g., Dedekind Studio or Jupyter QtConsole) using Unicode characters and symbols. |

#### Code Example

```dedekind
fn main() {
    // Renders as a beautiful formula in the console
    print_latex("Integral von a bis b: \\int_a^b f(x) \\, dx")
    
    source_expr = "y = (sin(x)^2 + cos(x)^2) / sqrt(x^2 + y^2)"
    latex_out = export_to_latex(source_expr)
    print(latex_out) // y = \frac{\sin^{2}\left(x\right) + \cos^{2}\left(x\right)}{\sqrt{x^{2} + y^{2}}}
}
main()
```

---

### 15.13 Chemistry, Biology & Bioinformatics

Dedekind contains built-in domain-specific APIs for chemistry, biology, and bioinformatics, supporting molar mass calculations, chemical stoichiometry, sequence alignments, and structural parsing.

#### Extended Units & Chemical Conventions

*   **Pressure Units:** `bar` and `atm` are fully integrated into the physical units engine. They auto-convert to standard Pascals (`Pa`) during compile-time verification and runtime calculations:
    *   $1\text{ bar} = 10^5\text{ Pa}$
    *   $1\text{ atm} = 101325\text{ Pa}$
*   **Mass Concentration:** Supports `% w/v` (`percent_wv`), `g/L`, and `mg/mL`. These units are dimensionally consistent with $[kg/m^3]$ and automatically scale:
    *   $1\% \text{ w/v} = 10\text{ g/L} = 10\text{ mg/mL}$
*   **pH Calculations:** The pH value is calculated based on the logarithmic activity of free hydronium ions:
    *   $\text{pH} = -\log_{10}([H^+])$
    *   $[H^+] = 10^{-\text{pH}}$ (with concentration $[H^+]$ in `[M]` or `[mol/L]`).

#### Built-in Chemistry & Bioinformatics Functions

| Function | Signature | Description |
|:---|:---|:---|
| `atomic_mass(symbol)` | `atomic_mass(symbol: string) -> Quantity [g/mol]` | Returns the IUPAC standard atomic mass of a chemical element by its symbol. Over 50 common elements are supported. |
| `atomic_number(symbol)` | `atomic_number(symbol: string) -> int` | Returns the atomic number (number of protons) of the specified chemical element. |
| `concentration_to_pH(c)` | `concentration_to_pH(c: Quantity) -> float` | Converts hydronium ion concentration $[H^+]$ in $[M]$ (or $[mol/L]$) to pH value. |
| `pH_to_concentration(pH)` | `pH_to_concentration(pH: float) -> Quantity [M]` | Converts a pH value into hydronium ion concentration in $[M]$. |
| `balance_equation(reactants, products)` | `balance_equation(reactants: string, products: string) -> (list[int], list[int])` | Balances a chemical reaction equation using singular value decomposition (SVD) on the stoichiometry matrix. Returns lists of stoichiometric coefficients for reactants and products. |
| `smiles_molecular_weight(smiles)` | `smiles_molecular_weight(smiles: string) -> Quantity [g/mol]` | Computes the molecular weight of a molecule from its SMILES representation. |
| `lipinski_descriptors(smiles)` | `lipinski_descriptors(smiles: string) -> dict` | Evaluates Lipinski's Rule of Five. Returns a dictionary with keys `"molecular_weight"`, `"logp"`, `"hbd"`, `"hba"`, `"violations"`, and `"pass"`. |
| `pubchem_get_molecular_formula(name)` | `pubchem_get_molecular_formula(name: string) -> string` | Retrieves the chemical molecular formula for a compound name from PubChem. |
| `chembl_get_ic50(target, compound)` | `chembl_get_ic50(target: string, compound: string) -> Quantity [nM]` | Queries the ChEMBL database for the IC50 activity value of a compound against a specific target. |
| `smith_waterman_alignment(seq1, seq2, match_score, mismatch_penalty, gap_penalty)` | `smith_waterman_alignment(seq1: string \| Tensor, seq2: string \| Tensor, match_score: float, mismatch_penalty: float, gap_penalty: float) -> dict` | Computes local sequence alignment using PyTorch. Returns a dictionary with keys `"score"`, `"aligned_seq1"`, and `"aligned_seq2"`. |
| `protein_structure_parse(path_or_content)` | `protein_structure_parse(path_or_content: string) -> DataFrame` | Parses a PDB or mmCIF protein structure file (or content) into a unit-aware `DataFrame` with `"angstrom"` units for `"x"`, `"y"`, and `"z"` columns. |

#### Code Example

```dedekind
fn main() {
    // 1. Molare Massen bestimmen
    c_mass = atomic_mass("C")
    h_mass = atomic_mass("H")
    print(c_mass) // 12.011[g/mol]
    
    // 2. pH-Wert Berechnungen
    h_conc = 1.0e-3[M]
    ph_val = concentration_to_pH(h_conc)
    print(ph_val) // 3.0
    
    h_back = pH_to_concentration(7.0)
    print(h_back) // 1.0e-7[M]
    
    // 3. Stöchiometrie ausgleichen: Photosynthese
    // CO2 + H2O -> C6H12O6 + O2
    coeffs = balance_equation("CO2 + H2O", "C6H12O6 + O2")
    react_coeffs = coeffs[0]
    prod_coeffs  = coeffs[1]
    
    print(react_coeffs) // [6, 6]
    print(prod_coeffs)  // [1, 6]

    // 4. Chemoinformatics & Databases
    mw = smiles_molecular_weight("CCO")
    print(mw) // 46.068[g/mol]

    lip = lipinski_descriptors("CCO")
    print(lip["pass"]) // 1.0 (True)

    formula = pubchem_get_molecular_formula("aspirin")
    print(formula) // "C9H8O4"

    ic50 = chembl_get_ic50("COX-1", "aspirin")
    print(ic50) // e.g. 100000.0[nM]

    // 5. Sequence & Structural Biology
    align = smith_waterman_alignment("TGCATG", "GGCA", 2.0, -1.0, -1.0)
    print(align["score"]) // 6.0
    print(align["aligned_seq1"]) // "GCA"

    pdb_data = "ATOM      1  N   ALA A   1      11.111  22.222  33.333  1.00 15.00           N"
    df = protein_structure_parse(pdb_data)
    x_coords = df.column_with_unit("x")
    print(x_coords[0]) // 11.111[angstrom]
}
main()
```

---

### 15.14 File I/O, Network & JSON APIs

To facilitate building robust data pipelines, Dedekind provides built-in utilities for interacting with the filesystem, making HTTP requests, and parsing/generating JSON documents.

#### API Reference

| Function | Signature | Description |
|:---|:---|:---|
| `read_file(path)` | `read_file(path: string) -> string` | Reads the full text of a file in UTF-8 encoding. |
| `write_file(path, content)` | `write_file(path: string, content: string) -> nil` | Writes the specified string content to a file in UTF-8 encoding. |
| `file_exists(path)` | `file_exists(path: string) -> bool` | Checks if a file exists at the specified path. |
| `http_get(url)` | `http_get(url: string) -> string` | Performs a synchronous HTTP GET request and returns the response body as a string. |
| `http_post(url, data)` | `http_post(url: string, data: string \| dict \| list) -> string` | Performs a synchronous HTTP POST request, encoding dictionary/list data as JSON if necessary, and returns the response. |
| `json_parse(s)` | `json_parse(s: string) -> dict \| list \| primitive` | Parses a JSON-formatted string into native Dedekind types (Dict, List, etc.). |
| `json_stringify(obj)` | `json_stringify(obj: any) -> string` | Formats a Dedekind object (e.g., Dict, List, Quantity) into a standard JSON string. |

#### Code Example

```dedekind
fn main() {
    file_path = "dedekind_test_data.json"
    
    // Check, create, and read file
    if not file_exists(file_path) {
        config = {"api_url": "https://api.github.com", "version": 1.6}
        write_file(file_path, json_stringify(config))
    }
    
    content_str = read_file(file_path)
    parsed = json_parse(content_str)
    print(parsed["api_url"]) // https://api.github.com
    
    // Simple HTTP integration
    response = http_get("https://api.github.com/zen")
    print(response)
}
main()
```

---

### 15.15 Medicine, Pharmacology & Epidemiology

Dedekind exposes specialized, lightweight tools for mathematical biology, pharmacokinetic models, epidemiological simulations, and biostatistics.

#### Domain Functions

| Function | Signature | Description |
|:---|:---|:---|
| `hill_equation(dose, Emax, EC50, n)` | `hill_equation(dose: float, Emax: float, EC50: float, n: float) -> float` | Computes the pharmacodynamic response using the Hill equation: $E = \frac{E_{max} \cdot \text{dose}^n}{EC_{50}^n + \text{dose}^n}$. |
| `one_compartment_pk(C0, ke, t)` | `one_compartment_pk(C0: float, ke: float, t: Tensor) -> Tensor` | Evaluates a single-compartment intravenous bolus pharmacokinetic model: $C(t) = C_0 \cdot e^{-k_e \cdot t}$. |
| `two_compartment_pk(C0, k12, k21, ke, t)` | `two_compartment_pk(C0: float, k12: float, k21: float, ke: float, t: Tensor) -> Tensor` | Evaluates a two-compartment intravenous bolus pharmacokinetic model (with central and peripheral compartments) over a time grid. Fully differentiable. |
| `half_life(ke)` | `half_life(ke: float) -> float` | Returns the biological half-life calculated as $t_{1/2} = \frac{\ln(2)}{k_e}$. |
| `sir_model(S0, I0, R0, beta, gamma, t)` | `sir_model(S0: float, I0: float, R0: float, beta: float, gamma: float, t: Tensor) -> (Tensor, Tensor, Tensor)` | Simulates the classical SIR compartmental epidemic model over a time grid using Runge-Kutta numerical integration. Returns grids for Susceptible (S), Infectious (I), and Recovered (R). |
| `basic_reproduction_number(beta, gamma)` | `basic_reproduction_number(beta: float, gamma: float) -> float` | Computes the basic reproduction ratio $R_0 = \frac{\beta}{\gamma}$. |
| `confidence_interval(x, alpha)` | `confidence_interval(x: Tensor, alpha: float) -> (float, float)` | Computes the $1 - \alpha$ confidence interval for the mean of sample tensor `x` using a Student-t distribution. |
| `odds_ratio(a, b, c, d)` | `odds_ratio(a: int, b: int, c: int, d: int) -> float` | Calculates the odds ratio from a $2 \times 2$ contingency table with counts representing true/false exposure and outcome groups. |
| `sensitivity_specificity(TP, FN, FP, TN)` | `sensitivity_specificity(TP: int, FN: int, FP: int, TN: int) -> dict` | Returns a dictionary containing `"sensitivity"`, `"specificity"`, `"ppv"`, and `"npv"` based on classification counts. |

#### Code Example

```dedekind
fn main() {
    // 1. Pharmacology: PK Profile and Half-life
    ke = 0.15 // Elimination rate [1/h]
    t_halflife = half_life(ke)
    print(t_halflife) // ~4.62 hours
    
    t_grid = linspace(0.0, 24.0, 10)
    c_profile = one_compartment_pk(100.0, ke, t_grid)
    print(c_profile)

    c_profile_2comp = two_compartment_pk(100.0, 0.05, 0.02, ke, t_grid)
    print(c_profile_2comp)
    
    // 2. Epidemiology: SIR Simulation
    t_sir = linspace(0.0, 30.0, 30)
    sir_out = sir_model(0.99, 0.01, 0.0, 0.3, 0.1, t_sir)
    S = sir_out[0]
    I = sir_out[1]
    R = sir_out[2]
    
    r0 = basic_reproduction_number(0.3, 0.1)
    print(r0) // 3.0
    
    // 3. Biostatistics: Diagnostics
    metrics = sensitivity_specificity(95, 5, 10, 890)
    print(metrics["sensitivity"]) // 0.95
    print(metrics["specificity"]) // 0.9888
}
main()
```

---

### 15.16 Modul-System, Reproduzierbarkeit & DataFrames

To support structured modular programming, reproducible scientific studies, and structured data manipulation, Dedekind v1.4.0 introduced modules, global seeds, unit-aware DataFrames, and plotting integrations.

#### Keywords, Modifiers and Functions

##### `use` Module Loading
Modules are loaded dynamically using the `use <module_name>` statement. The compiler searches for module files in the current folder, a global `modules/` folder, or `examples/dedekind/`. It instantiates function scopes and constants under the loaded namespace.

##### `@units` Annotations
Functions can rigorously define physical unit signatures for parameters and return types:
```dedekind
@units
fn kinetic_energy(m: [kg], v: [m/s]) -> [J] {
    return 0.5 * m * v^2
}
```
At runtime, inputs are automatically converted to the declared units (e.g. `2000[g]` is converted to `2.0[kg]`), and the return expression is validated to guarantee dimensional correctness (in this case, $\text{kg} \cdot \text{m}^2/\text{s}^2 \equiv \text{J}$).

##### Reproducibility Built-ins
- `seed(n)`: Sets a uniform, reproducible seed across the random number generators of Python (`random`), NumPy, and PyTorch.
- `data_hash(x)`: Computes a deterministic SHA-256 hash of any data structure (tensor, list, dataframe, quantity, or dictionary) to verify data pipeline integrity.

##### DataFrames and Tabular I/O
- `DataFrame(data, units=None)`: Constructs a column-oriented table. The `data` argument is a dictionary mapping column keys to lists/tensors. `units` can specify a list or dictionary of expected column physical units.
- `read_csv(path, units=None, has_header=True)`: Parses CSV files. If headers contain brackets (e.g. `Time [s]`, `Temp [K]`), units are extracted automatically.
- `write_csv(path, df, include_units_in_header=True)`: Exports DataFrames to disk.
- Custom high-performance IO connectors for Parquet (`read_parquet`/`write_parquet`), HDF5 (`read_hdf5`/`write_hdf5`), and NetCDF (`read_netcdf`) are supported when respective optional libraries are present.

##### Unit-Aware Plotting
Built-in functions `plot(x, y, ...)`, `scatter(x, y, ...)`, and `contour(X, Y, Z, ...)` automatically check if input lists contain `Quantity` values. They dynamically extract unit dimensions and append them to axis labels (e.g. producing label `Time [s]`).

#### Code Example

```dedekind
use physics_helpers // loads modules/physics_helpers.ddk

@units
fn power_dissipated(v: [V], r: [ohm]) -> [W] {
    return (v^2) / r
}

fn main() {
    // 1. Seed & Hash
    seed(42)
    t_rand = random_matrix(3, 3)
    h_val = data_hash(t_rand)
    print(h_val)
    
    // 2. Unit-aware Functions
    p = power_dissipated(12[V], 4[ohm])
    print(p) // 36.0[W]
    
    // 3. DataFrame and CSV I/O
    df_data = {
        "Zeit": [0.0, 1.0, 2.0],
        "Geschwindigkeit": [0.0, 9.81, 19.62]
    }
    df = DataFrame(df_data, {"Zeit": [s], "Geschwindigkeit": [m/s]})
    
    write_csv("fall_data.csv", df)
    
    df_read = read_csv("fall_data.csv")
    print(df_read.column_with_unit("Geschwindigkeit"))
}
main()
```

---

### 15.17 Benchmarking, JIT, SDEs & FEM

Introduced in v1.5.0, these features expand Dedekind into high-performance computing, stochastics, PDE discretizations, and JIT compilation.

#### High-Performance Built-ins

##### Benchmarking & Profiling
- `benchmark(fn, n=10, warmup=2, label=None)`: Measures execution wall time of function `fn` over `n` iterations, reporting mean, standard deviation, and range.
- `profile(fn, label=None, top=5)`: Monitors call counts, CPU time, and peak memory allocations.
- `time_block(label, fn)`: Wraps an arbitrary block of execution or function call for ad-hoc time logging.
- `jit(fn)`: Wraps `fn` via `torch.compile` (using PyTorch's TorchInductor/Triton backend), translating standard PyTorch/tensor calls into optimized execution paths.

##### Stochastic Differential Equations (SDE)
- `sde_solve(drift, diffusion, y0, t, method="euler_maruyama", seed_value=None)`: Solves an Itô stochastic differential equation:
  $$dY = \mu(t, Y)\,dt + \sigma(t, Y)\,dW$$
  Methods: `"euler_maruyama"` (order 0.5) or `"milstein"` (order 1.0, utilizing central differences for the spatial derivative of the diffusion term).

##### Optimization & Mathematical Programming
- `least_squares(residuals, x0, jacobian=None, bounds=None, method="trf")`: Standard non-linear least-squares solver.
- `minimize_constrained(f, x0, constraints=None, bounds=None, method="SLSQP", tol=1e-8)`: General constrained minimization.
- `milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, integrality=None)`: Solves mixed-integer linear programs using SciPy's high-performance solvers.

##### Finite Element Method (FEM) 2D Poisson Solver
- `mesh_unit_square(n)`: Generates a structured triangular finite element mesh on $[0,1]^2$ with $(n+1)^2$ nodes. Returns a dictionary with keys `"nodes"`, `"elements"`, and `"boundary_nodes"`.
- `fem_assemble_stiffness(mesh)`: Assembles the global Galerkin stiffness matrix $K$ using linear Lagrangian basis functions.
- `fem_assemble_load(mesh, f_source)`: Assembles the global load/force vector $F$ using function `f_source(x, y)`.
- `fem_poisson_2d(mesh, f_source, dirichlet_value=0.0)`: Resolves the Poisson boundary value problem $-\Delta u = f$ with constant Dirichlet boundary condition $u|_{\partial \Omega} = u_{Dirichlet}$. Returns a node value tensor of shape $(n+1)^2$.

#### Code Example

```dedekind
fn drift_fn(t, y) { return -0.5 * y }
fn diff_fn(t, y) { return 0.2 * y }

fn compute_poisson() {
    mesh = mesh_unit_square(10)
    // -Δu = 1.0 in unit square, u = 0.0 on boundary
    u_sol = fem_poisson_2d(mesh, fn(x, y) => 1.0, 0.0)
    print(u_sol)
}

fn main() {
    // 1. SDE Simulation
    t = linspace(0.0, 5.0, 100)
    y_path = sde_solve(drift_fn, diff_fn, 1.0, t, "milstein", 42)
    print(y_path)
    
    // 2. Optimization: Non-linear Least Squares
    residuals = fn(x) => [x[0]**2 + x[1] - 11.0, x[0] + x[1]**2 - 7.0]
    opt_x = least_squares(residuals, [1.0, 1.0])
    print(opt_x) // [3.0, 2.0]
    
    // 3. JIT & Benchmarking
    fem_jit = jit(compute_poisson)
    benchmark(fem_jit, 5, 2, "FEM Poisson Solution")
}
main()
```

---

### 15.18 Symbolics, Sparse Krylov Solvers & Publishing

Added in v1.6.0, these systems provide symbolic solvers, advanced sparse iterative linear algebra, reproducible notebooks, and publication-ready reporting.

#### API Reference

##### Symbolic Computations (via SymPy)
- `solve_sym(equation, var)`: Symbolically solves algebraic equations or multi-variable systems (given as strings or equation lists) for `var`. Returns a list of solution strings (or dictionaries for systems).
- `simplify_sym(expr)`: Performs extensive algebraic, trigonometric, and exponential simplifications on a formula string (e.g. `"sin(x)^2 + cos(x)^2"` $\rightarrow$ `"1"`).
- `series(expr, var, x0=0, n=6)`: Computes the symbolical Taylor series expansion of `expr` with respect to `var` around point `x0` up to order `n` (returns expansion string without $O$-term).

##### Sparse Krylov Solvers & Preconditioners
Highly efficient solvers for massive linear systems $A x = b$:
- `cg(A, b, x0=None, tol=1e-8, max_iter=1000, preconditioner=None)`: Conjugate Gradient for symmetric positive-definite operators.
- `gmres(A, b, x0=None, tol=1e-8, max_iter=1000, restart=None, preconditioner=None)`: Generalized Minimal Residual for general linear systems.
- `bicgstab(A, b, x0=None, tol=1e-8, max_iter=1000, preconditioner=None)`: Biconjugate Gradient Stabilized.
- `jacobi_preconditioner(A)`: Computes a diagonal Jacobi preconditioner $M^{-1} = \text{diag}(A)^{-1}$ to accelerate convergence.
- `ilu_preconditioner(A, drop_tol=1e-4, fill_factor=10)`: Computes an Incomplete LU factorized preconditioner for general sparse matrices, reducing iteration count by 2 to 10 times.

##### Notebook & Publishing Tools
- `export_notebook(source_path, output_path=None, format="html", title=None, include_hash=True, capture_plots=True)`: Compiles and runs a `.ddk` script, captures standard output and generated plots, and bundles them into an HTML or Markdown report along with a SHA-256 hash of the code for full auditability.
- `print_table(rows, headers=None, format="markdown", precision=4, caption=None, label=None)`: Renders tabular arrays/DataFrames. Support formats: `"markdown"`, `"latex"` (generates standard professional publication `booktabs` blocks), `"csv"`, or `"plain"`. `UncertainQuantity` types are formatted as `val ± std [unit]` (or `$val \pm std\,[\mathrm{unit}]$` in LaTeX).

#### Code Example

```dedekind
fn main() {
    // 1. Symbolics
    sol = solve_sym("x^2 - 5*x + 6", "x")
    print(sol) // ["2", "3"]
    
    taylor = series("sin(x)", "x", 0, 5)
    print(taylor) // x - x^3/6
    
    // 2. Iterative Solvers
    A = [[4.0, 1.0], [1.0, 3.0]]
    b = [1.0, 2.0]
    M = jacobi_preconditioner(A)
    x = cg(A, b, preconditioner=M)
    print(x)
    
    // 3. Table Formatting
    val = uncertain(5.12, 0.03, "m/s")
    table_rows = [
        ["Messung 1", val],
        ["Messung 2", uncertain(4.98, 0.05, "m/s")]
    ]
    print_table(table_rows, ["ID", "Wert"], "latex")
}
main()
```

---

### 15.19 Hardware-Beschleunigung & Unit Checking (v1.6.3)

Version 1.6.3 completes hardware-native and compile-time semantic validation features.

#### Execution Modifiers
Dedekind compiles expressions using AST-level modifiers to control placement and acceleration strategy:
- `.gpu()`: Offloads computation to the PyTorch CUDA/Triton backend.
- `.cpu()`: Explicitly forces execution on the host CPU.
- `.single()`: Bypasses multi-threaded parallelization for serial debugging.
- `.sparse()`: Instructs the tensor engine to construct the backing tensor as a sparse Coordinate (COO) or Compressed Sparse Row (CSR) matrix.
- `.fast()`: Compiles the target block or expression using `torch.compile` / native Triton GPU kernel generators.

#### Static Compile-Time Unit Checker
The units analyzer (`units_checker.py`) runs as a static compiler pass. Key properties:
1. **Scope-based Environment Stack**: Track definitions, assignments, and scoping rules across nested blocks (`if`, `while`, `for`, `fn`).
2. **Canonical Dimensional Analysis**: Reduces compound units to canonical physical representations (e.g. `(m/s)*s` simplifies to `m`), detecting and throwing unit compilation errors (e.g. `1[m] + 1[s]`) at compile-time before any Python execution occurs.

#### Code Example

```dedekind
@units
fn fast_diffusion(t: [s], alpha: [m^2/s]) -> [m^2] {
    // Static compile-time units pass verifies: [s] * [m^2/s] -> [m^2]
    return t * alpha
}

fn main() {
    a = 10.0[m/s]
    b = 2.0[s]
    
    // Automatic inference matches dimensions
    distance = (a * b).fast() // Compiles via PyTorch Triton kernel generator
    print(distance) // 20.0[m]
}
main()
```

---

## Related Documentation and Artifacts

For more details on specific aspects of the Dedekind language, compiler architecture, and feature statuses, see:
- [Was_Dedekind_aktuell_kann.md](file:///c:/Users/Matrix/AntigravityProjects/Dedekind/Documentation/Was_Dedekind_aktuell_kann.md) - German language status and features checklist.
- [README.md](file:///c:/Users/Matrix/AntigravityProjects/Dedekind/README.md) - Root language introduction and "What's New" history.
- [implementation_plan.md](file:///C:/Users/Matrix/.gemini/antigravity/brain/e2d4c92b-811a-45a7-b24a-cd43b2aa2d4e/implementation_plan.md) - Documentation alignment plan details.
- [task.md](file:///C:/Users/Matrix/.gemini/antigravity/brain/e2d4c92b-811a-45a7-b24a-cd43b2aa2d4e/task.md) - Checklist of the alignment tasks.

---

*This document is the Markdown source for the Language Specification. PDF can be generated e.g. with `pandoc Dedekind_Language_Specification.md -o Dedekind_Language_Specification_v0.2.pdf`. See Documentation/README.md for build instructions.*
