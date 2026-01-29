# Fourier — A Modern Programming Language for Machine Learning and Graphics

**Language Specification v0.2**  
Mario Michael Heinrich · github.com/Engineer1080  
Draft: January 2026 · Updated for v0.6 (Physical Units), v0.7 (ODE), v0.8 (Probabilistic, PDE), v0.9 (Distributions, Integration), v0.9.1 (Dokumentation, Run-Examples), v0.9.2 (pi, e, CODATA), v0.9.3 (Uncertainty, Fitting), v0.9.4 (HMC, LaTeX-Export), v0.9.5 (Bessere Fehlermeldungen, Einheiten Compile-Zeit), v0.9.6 (Math-Funktionen: exp, log, sqrt, tan, abs, asin, acos, atan, atan2, sinh, cosh, tanh)

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
15. [**Physical Units and Universal Constants (v0.6)**](#15-physical-units-and-universal-constants-v06) (incl. §15.7 ODE, §15.8 Probabilistic, §15.9 PDE, §15.10 Integration & Math v0.9)

---

## 1. Introduction

Fourier is a modern, high-performance programming language designed specifically for compute-intensive workloads in machine learning and graphics rendering. Named after Joseph Fourier, whose mathematical transformations are fundamental to both signal processing and modern graphics, the language embodies the principle of efficient transformation of data through parallel computation.

Unlike general-purpose languages retrofitted with parallel computing capabilities, Fourier is built from the ground up with GPU/TPU acceleration and automatic parallelization as core features.

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
- AOT Compilation (MLIR/LLVM), Fourier Studio IDE (v0.6).

## 4. Syntax Overview

Variables, functions (`fn name(args) { ... }`), control flow (`if`/`else`, `for`, `while`), and expression modifiers. See implementation and examples in `examples/fourier/`.

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

Modules for ML, linalg, signal (FFT), visualization (`plot()`), and runtime types (`Quantity`, `Quaternion`, `Dense`, `Sequential`).

## 11. Use Cases

ML training/inference, graphics rendering, scientific computing, signal processing, physics simulations (with units and constants).

## 12. Implementation Roadmap

Phases 1–12 implemented in prototype (Python backend, Fourier Studio, PyTorch runtime, Ricci, Sparse, AOT, Quaternion, **Physical Units v0.6**). See main README for detailed roadmap.

## 13. Technical Foundation

MLIR, LLVM, PyTorch; research in automatic differentiation, GPU compilation (Triton), type systems, work-stealing.

## 14. Conclusion

Fourier aims to bridge productivity and performance for compute-intensive applications. This specification is updated to **v0.2** to reflect current prototype behaviour and **v0.6** language features (physical units and constants).

---

## 15. Physical Units and Universal Constants (v0.6)

This section documents the **physical units** and **universal constants** features introduced in Fourier v0.6 (Option B: Einheiten-Literale und Konstanten als Quantity).

### 15.1 Unit Literals

Values can carry SI units using bracket notation:

```fourier
distance = 10[m]
speed    = 5[m/s]
mass     = 1.0[kg]
freq     = 5.0e14[Hz]
charge   = 1.602e-19[C]
```

- **Syntax**: `number [ unit ]` where `unit` is an identifier or product/quotient/power (e.g. `m`, `m/s`, `kg*m/s^2`, `m^2`).
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

- **Addition / Subtraction**: Only allowed when both operands have the **same unit**; otherwise a runtime error is raised.
- **Multiplication / Division**: Units are combined (e.g. `m * m/s` → `m²/s`; `J/s` for power).
- **Power**: `Quantity ** exponent` is supported (e.g. `c^2`, `r^2`); the unit is raised to the given exponent (e.g. `(m/s)^2`).
- **Unary minus**: `-x` is supported for both `Quantity` and `Quaternion` (e.g. `-1.602e-19[C]`, `-1.0 + 0i`).

### 15.4 Unit Display Simplification

For readability, the runtime simplifies certain compound units when displaying results:

- **J (Joule)**: e.g. `(kg)*((m/s)^2)` or `(J*s)*(Hz)` → displayed as `[J]`.
- **N (Newton)**: e.g. gravitational or Coulomb force expressions → displayed as `[N]`.

Internal unit representation remains exact for dimensional consistency.

### 15.5 Quaternion and Unary Minus

Complex numbers are represented as Quaternions (with `i` component; `j`, `k` zero when used as complex). As of v0.6:

- **Unary minus** on a Quaternion is supported via `__neg__` (e.g. `-1.0 + 0i`, `-(0.0 + 1.0i)`).
- This allows signal lists such as `[1.0+0i, 0.0+1.0i, -1.0+0i, 0.0-1.0i]` and FFT-based examples to run correctly.

### 15.6 Example: Universal Constants

```fourier
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

See `examples/fourier/universal_constants.fourier` and `examples/fourier/signal_physics.fourier` for full runnable examples.

### 15.7 Differentiable ODE Solvers

Fourier provides a **differentiable ODE solver** for physics-informed ML and parameter identification:

- **`ode_solve(fun, y0, t, method="rk4")`**: Solves dy/dt = fun(t, y). `fun(t, y)` must return the time derivative (t scalar, y tensor); `y0` is the initial state; `t` is a 1D time grid. Returns a tensor of shape `(len(t), *y0.shape)`. Gradients flow through `y0` and through any parameters used inside `fun`.
- **`linspace(start, stop, steps)`**: Builds a 1D tensor of `steps` evenly spaced values from `start` to `stop` (for use as `t`).
- **Methods**: `"rk4"` (default, 4th-order Runge–Kutta) or `"euler"`.

Example: exponential decay dy/dt = -0.1*y; then `grad(final_state, y0)` gives d y(T)/d y0. See `examples/fourier/differentiable_ode.fourier`.

### 15.8 Probabilistic Programming (v0.8)

Fourier provides **first-class distributions** and **Bayesian inference** via `torch.distributions`:

- **Distributions**: `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)` – return distribution objects.
- **sample(dist)** / **sample(dist, n)**: Draw one or `n` samples from a distribution.
- **log_prob(dist, value)**: Log-probability of `value` under `dist` (for inference).
- **metropolis(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size)**: Metropolis-Hastings MCMC. `log_prior_fn(theta)` and `log_likelihood_fn(data, theta)` return log-probs (tensor or scalar). Returns posterior samples of shape `(num_steps, *theta_shape)`.

Example: infer mean `theta` of Normal data with prior Normal(0,1); see `examples/fourier/probabilistic.fourier`. Extended distributions (Exponential, Gamma, Beta, Poisson): see `examples/fourier/distributions_extended.fourier`.

### 15.9 Differentiable PDE Solvers (v0.8)

Fourier provides **differentiable PDE solvers** for the heat equation \(u_t = k\,\Delta u\), built on finite differences and the differentiable `ode_solve` (RK4). Gradients flow through the initial condition `u0` and the diffusivity `k`, enabling inverse problems and physics-informed ML.

- **`pde_heat_1d(u0, x, t, k, bc="dirichlet")`**: 1D heat equation \(u_t = k\,u_{xx}\). `u0` is the initial condition (1D tensor, length = `len(x)`); `x` is the spatial grid (1D); `t` is the time grid (1D); `k` is the diffusivity (scalar or tensor). Returns a tensor of shape `(len(t), len(x))`. With `bc="dirichlet"`, boundary values from `u0` are held fixed.
- **`pde_heat_2d(u0, x, y, t, k, bc="dirichlet")`**: 2D heat equation \(u_t = k\,(u_{xx}+u_{yy})\). `u0` is the initial condition (2D tensor, shape `(nx, ny)`); `x`, `y` are 1D spatial grids; `t` is the time grid; `k` is the diffusivity. Returns a tensor of shape `(len(t), nx, ny)`.

Example: 1D heat with a spike initial condition; see `examples/fourier/pde_heat.fourier`.

### 15.10 Numerical Integration and Math (v0.9, erweitert v0.9.6)

Fourier provides **numerical integration** and **math functions** for scientific expressions. All math functions accept tensors or scalars (via `_to_tensor`), are element-wise and differentiable.

**Numerical integration:**

- **`integrate(f, a, b, n=100)`**: Numerically integrates \(f(x)\) from `a` to `b` using the trapezoidal rule with `n` points. `f` must accept a 1D tensor (the grid) and return a tensor of the same length; the integral is differentiable with respect to parameters in `f` when using autograd.

**Trigonometrie:** **`sin(x)`**, **`cos(x)`**, **`tan(x)`**.

**Exponential und Logarithmus:** **`exp(x)`**, **`log(x)`** (natürlicher Logarithmus), **`log10(x)`**.

**Wurzel und Betrag:** **`sqrt(x)`**, **`abs(x)`**.

**Arkusfunktionen (Bogenmaß):** **`asin(x)`**, **`acos(x)`**, **`atan(x)`**, **`atan2(y, x)`** (Winkel der Richtung (x,y), Wertebereich \((-\pi,\pi]\)).

**Hyperbelfunktionen:** **`sinh(x)`**, **`cosh(x)`**, **`tanh(x)`**.

Examples: \(\int_0^1 x^2\,dx = 1/3\), \(\int_0^\pi \sin(x)\,dx = 2\); see `examples/fourier/integration.fourier`. Full math showcase: `examples/fourier/math_functions.fourier`.

### 15.11 Uncertainty Propagation and Fitting (v0.9.3)

**Uncertainty Propagation (Fehlerfortpflanzung):**

- **`uncertain(value, std, unit="")`**: Creates an `UncertainQuantity` representing value ± std (Gaussian error propagation). Optional `unit` for physical quantities (e.g. `"m"`).
- **Arithmetic**: For +, -, *, /, ^ the standard deviation is propagated via the Gaussian approximation: e.g. for \(z = x + y\), \(\sigma_z^2 = \sigma_x^2 + \sigma_y^2\); for \(z = x \cdot y\), \((\sigma_z/z)^2 = (\sigma_x/x)^2 + (\sigma_y/y)^2\).
- **Display**: `repr` shows „value ± std [unit]“. See `examples/fourier/uncertainty_propagation.fourier`.

**Fitting / Regression:**

- **`fit(loss_fn, params_init, data, method="gd"|"mcmc", lr=0.01, steps=500)`**: Minimizes `loss_fn(params, data)` with respect to `params`. `params_init` is the initial parameter tensor (or list); `data` is passed to `loss_fn`. With `method="gd"` (default), gradient descent is used (in-place updates); with `method="mcmc"`, Metropolis-Hastings is used (returns posterior samples). Returns the optimal parameters (tensor) for GD, or samples tensor for MCMC.
- Example: linear regression \(y = a\,x + b\); see `examples/fourier/curve_fitting.fourier`.

---

*This document is the Markdown source for the Language Specification. PDF can be generated e.g. with `pandoc Fourier_Language_Specification.md -o Fourier_Language_Specification_v0.2.pdf`. See Documentation/README.md for build instructions.*
