# Roadmap

## 🗺️ Roadmap

### Phase 1: Foundation ✅
*   [x] Language Specification & Design
*   [x] Proof-of-Concept Compiler (Python Backend)
*   [x] IDE Integration (Jupyter Kernel)

### Phase 2: Core Development ✅
*   [x] Build-in Core Algorithms (FFT, Conv, Linalg)
*   [x] Robust Lexer & Parser (Windows Support, Unary Ops)
*   [x] IDE Terminal & Tab Support

### Phase 3: Hardware Acceleration ✅
*   [x] Integration with **PyTorch** for GPU execution.
*   [x] Implementation of `.gpu()` and `.cpu()` modifiers.

### Phase 4: Production (v0.2) ✅
*   [x] **Native Performance**: Integration with PyTorch Inductor via `.fast()` (`torch.compile`).
*   [x] **IDE Integration Upgrade**: Terminal and UI polish.

### Phase 5: Advanced Mathematics ✅
*   [x] **Autograd**: Native `grad()` operator for automatic differentiation.
*   [x] **Property Access**: Native `.shape` support for tensors and models.

### Phase 6: Tensor Contraction & Logic (v0.3) ✅
*   [x] **Einsum**: High-level elective tensor contraction syntax.
*   [x] **Complex/Quaternion**: Built-in support for rotational math.

### Phase 9: Ricci Calculus & Universal Constants ✅
*   [x] **Index Notation**: Support for `^` and `_` suffixes.
*   [x] **Auto-Einsum**: Lowering Ricci expressions to `torch.einsum`.
*   [x] **Physics Constants**: Native high-precision constants (`c`, `G`, etc.).

### Phase 10: Sparse Tensors & CFD ✅
*   [x] **Sparse API**: `.sparse()` modifier for COO/CSR formats.
*   [x] **Item Assignment**: `T[i][j] = val` for grid manipulation.
*   [x] **CFD Simulation**: Heat diffusion on 10,000 node grids.

### Phase 11: Quaternion & Rotational Math ✅
*   [x] **Hamilton Notation**: Support for `i`, `j`, `k` quaternion components.
*   [x] **Hamilton Product**: Native 4D arithmetic.
*   [x] **Robotics Support**: Native `.rotate(vector)` method.

### Phase 12: Physical Units & Constants (v0.6) ✅
*   [x] **Constants as Quantity**: `c`, `G`, `h`, `k_B`, `k_e` with SI units; `Quantity.__pow__` and unit simplification (J, N).
*   [x] **Unary Minus**: Codegen emits `-expr`; `Quantity` and `Quaternion` implement `__neg__` for correct behaviour in expressions like `-1.0[C]` and `-1.0 + 0i`.

### Phase 13: Differentiable ODE Solvers (v0.7) ✅
*   [x] **ode_solve**: Differentiable ODE solver dy/dt = fun(t,y) with RK4 (default) and Euler; gradients through y0 and parameters in fun.
*   [x] **linspace**: Time grid for ODE integration; integration with `grad()` for Physics-Informed ML.

### Phase 14: Probabilistic Programming (v0.8) ✅
*   [x] **Distributions**: `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)` (torch.distributions).
*   [x] **sample** / **log_prob**: Sampling and log density for Bayesian Inference.
*   [x] **metropolis**: Metropolis-Hastings MCMC for posterior sampling (log_prior, log_likelihood, data, init, steps).

### Phase 15: Differentiable PDE Solvers (v0.8) ✅
*   [x] **pde_heat_1d**: Differentiable 1D heat solver \(u_t = k\,u_{xx}\); finite differences + `ode_solve`; Dirichlet boundary conditions.
*   [x] **pde_heat_2d**: Differentiable 2D heat solver \(u_t = k\,(u_{xx}+u_{yy})\); gradients through `u0` and `k`.

### Phase 16: Module System, DataFrames, and Slices (v1.1 - v1.7) ✅
*   [x] **Module System**: `use` statement with dotted paths and `pub fn` export control.
*   [x] **Custom Units**: User-defined units `unit NAME = FACTOR[base]` with compile-time registration.
*   [x] **Data Science**: `DataFrame` with CSV/Parquet/HDF5/NetCDF loaders, print_table, and unit-aware plots.
*   [x] **Error Handling**: `try { ... } catch e { ... }` block structure and Python-style slicing.

### Phase 17: Generics and Advanced Physics/Math (v1.8 - v1.21) ✅
*   [x] **Generics**: Type parameters `fn name<T, U>(...)` with polymorphic unit variable enforcement.
*   [x] **3D Geometric Algebra**: Clifford/Geometric Algebra G(3,0) via `MultiVector`, rotor rotations, and sandwich products.
*   [x] **Quantum Computing**: Native `QuantumCircuit` class, pure Statevector simulator, and VQE optimizer.
*   [x] **Debugging**: Source-mapped tracebacks for runtime errors and transitive compiler purity checks.

### Phase 18: Differentiable Engineering (v2.2.0) ✅
*   [x] **Control Theory**: Block-diagram simulations (PID, transfer functions, saturation) with loop resolution and optimization.
*   [x] **Fluid Dynamics**: Vectorized Lattice Boltzmann Method (D2Q9) with continuous obstacle masks for drag/lift optimization.
*   [x] **Structural Mechanics**: 2D Finite Element simulation (Q4 bilinear elements) and topology optimization (SIMP/OC).
*   [x] **Heat Transfer**: Implicit backward Euler thermal diffusion and conductance topology optimization.
*   [x] **DSP & Electronics**: Biquad IIR/FIR filters and AC circuit MNA nodal network solver.

### Phase 19: Space Physics & Trajectory Optimization (v2.3.0) ✅
*   [x] **N-Body Simulation**: Native differentiable N-body integration (RK4) with softening parameters.
*   [x] **Orbital Mechanics**: Kepler equation solver (Newton-Raphson) and state-vector element coordinate transformations.
*   [x] **Trajectory Optimization**: Direct L-BFGS/GD-based trajectory optimization (e.g. Hohmann transfers).

### Phase 20: Standard Library Consolidation (v2.9.0) ✅
*   [x] **Atomic Consolidation**: Merged `molecular` (MD) and `crystallography` into unified `atomic.ddk`.
*   [x] **Geometry Cleanup**: Merged all area and volume geometry functions directly into `math.ddk`.
*   [x] **Legacy Removal**: Deleted deprecated example modules and folded helper utilities into standard modules.

### Phase 21: Native MLIR Backend & AOT Compilation (planned)
The current `src/dedekind/mlir_codegen.py` only emits MLIR-style text and `src/dedekind/aot_compiler.py` falls back to C++ stubs — there is no real MLIR/LLVM toolchain integration yet. This phase delivers a production-grade native backend.
*   [ ] **MLIR Dialect**: Promote the prototype text emitter to a real `dedekind` MLIR dialect (TableGen / Python bindings).
*   [ ] **MLIR Pipeline**: Wire up `mlir-opt` and `mlir-translate` so Dedekind -> MLIR -> LLVM IR -> object code runs end-to-end.
*   [ ] **Static Binary**: Replace the C++ stub generator in `aot_compiler.py` with real LLVM linking; produce standalone `.exe` / ELF without Python.
*   [ ] **Verification**: Execute the AOT-compiled binaries on Windows, Linux, and macOS in CI.

### Phase 22: Natural Units for Particle Physics (proposed)
Dedekind already ships `eV`, `keV`, `MeV`, `GeV` as energy units and CODATA constants (`hbar`, `c`, `m_e`, `m_p`, `alpha`). What is missing is first-class support for the high-energy-physics convention ℏ = c = 1, under which length, time, mass, momentum and cross sections collapse onto powers of energy. Two design variants are being considered; user feedback from the v3.x release cycle will decide which (or both) ship.

**Variant A — Bridge constants (`use particle`, library-only).** SI remains the single source of truth. The module exposes precomputed bridge quantities such as `hbar_c = 197.3269804[MeV*fm]`, particle masses in their natural representation (`m_e_natural = 0.51099895[MeV]`, `m_p_natural = 938.272[MeV]`, …), and explicit converters `to_natural(q)` / `from_natural(x, target_unit)`. The compile-time dimension checker is untouched: `1[m] + 1[MeV]` stays a compile error, formulas keep their explicit `c` and `hbar` factors. Low implementation cost, fully additive, reversible.

**Variant B — Scope-switched dimension system (`@natural_units` decorator).** Within a marked function the dimension checker is reparameterized so that ℏ = c = 1 holds at the type level: `[length] ≡ [time] ≡ [energy]⁻¹` and `[mass] ≡ [momentum] ≡ [energy]` become legal identities. QFT-style formulas read as in textbooks (no notational `c²`/`ℏ/c` noise), and the compiler enforces explicit coercion (`to_si(...)`) at scope boundaries. Higher implementation cost (checker extension, scope tracking, coercion rules, interaction with generics and solvers), and a non-trivial design commitment that is hard to roll back once users depend on it.

*   [ ] **Variant A**: ship `use particle` with bridge constants, particle masses, and `to_natural`/`from_natural`.
*   [ ] **Variant B**: optional `@natural_units` scope, gated on demonstrated user demand from the v3.x release.
*   [ ] **Examples**: Compton scattering, Mandelstam variables, Yukawa potential range, electroweak scale comparisons.

## 🔭 Beyond v1.0: Future Vision

Dedekind aims to become the "Standard Language for Nature's Laws." To achieve this, we are researching the native implementation of the following concepts:

1. **Differentiable ODE Solvers**: Implemented in v0.7: `ode_solve(fun, y0, t, method="rk4")` solves dy/dt = fun(t,y) with differentiable RK4 (or Euler); gradients flow through y0 and parameters in `fun`. Use with `grad()` for physics-informed ML. See `examples/dedekind/differentiable_ode.ddk`.
2. **Differentiable PDE Solvers**: Implemented in v0.8: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` solve the heat equation with finite differences and `ode_solve`; gradients through initial condition and diffusivity for inverse problems. See `examples/dedekind/pde_heat.ddk`.
3. **Physical Units**: Implemented at compile time in v0.9.5/v1.4.0: unit check at compile time with generic/type parameter bounds.
4. **Probabilistic Programming**: Implemented in v0.8: `Normal`, `Uniform`, `Bernoulli`; `sample(dist)`, `log_prob(dist, value)`; `metropolis`/`hmc` for Bayesian inference. See `examples/dedekind/probabilistic.ddk`. Future: more distributions, NUTS/VI, conditioning syntax.
5. **Symbolic Simplification**: Implemented in v1.6.0: `simplify_sym` SymPy bridge, as well as the compiler AST-to-AST pass in `simplify.py`.
