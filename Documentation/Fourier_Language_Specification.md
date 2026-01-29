# Fourier — A Modern Programming Language for Machine Learning and Graphics

**Language Specification v0.2**  
Mario Michael Heinrich · github.com/Engineer1080  
Draft: January 2026 · Updated for v0.6 (Physical Units) and v0.7 (Differentiable ODE)

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
15. [**Physical Units and Universal Constants (v0.6)**](#15-physical-units-and-universal-constants-v06) ← **NEW**

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

### 15.2 Universal Constants as Quantity

The following constants are predefined as **Quantity** values with correct SI dimensions:

| Constant | Meaning              | SI Unit           |
|----------|----------------------|-------------------|
| `c`      | Speed of light       | m/s               |
| `G`      | Gravitational const. | m³/(kg·s²)        |
| `h`      | Planck constant      | J·s               |
| `k_B`    | Boltzmann constant   | J/K               |
| `k_e`    | Coulomb constant     | N·m²/C²           |

Example: `E = m * c^2` with `m = 1.0[kg]` yields a result in J (Joule); the runtime simplifies the displayed unit to **J** where applicable.

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

---

*This document is the Markdown source for the Language Specification. PDF can be generated e.g. with `pandoc Fourier_Language_Specification.md -o Fourier_Language_Specification_v0.2.pdf`. See Documentation/README.md for build instructions.*
