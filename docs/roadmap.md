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

### Phase 16: Native MLIR Backend & AOT Compilation (planned)
The current `src/dedekind/mlir_codegen.py` only emits MLIR-style text and `src/dedekind/aot_compiler.py` falls back to C++ stubs — there is no real MLIR/LLVM toolchain integration yet. This phase delivers a production-grade native backend.
*   [ ] **MLIR Dialect**: Promote the prototype text emitter to a real `dedekind` MLIR dialect (TableGen / Python bindings).
*   [ ] **MLIR Pipeline**: Wire up `mlir-opt` and `mlir-translate` so Dedekind -> MLIR -> LLVM IR -> object code runs end-to-end.
*   [ ] **Static Binary**: Replace the C++ stub generator in `aot_compiler.py` with real LLVM linking; produce standalone `.exe` / ELF without Python.
*   [ ] **Verification**: Execute the AOT-compiled binaries on Windows, Linux, and macOS in CI.

## 🔭 Beyond v1.0: Future Vision

Dedekind aims to become the "Standard Language for Nature's Laws." To achieve this, we are researching the native implementation of the following concepts:

1. **Differentiable ODE Solvers**: Implemented in v0.7: `ode_solve(fun, y0, t, method="rk4")` solves dy/dt = fun(t,y) with differentiable RK4 (or Euler); gradients flow through y0 and parameters in `fun`. Use with `grad()` for physics-informed ML. See `examples/dedekind/differentiable_ode.ddk`.
2. **Differentiable PDE Solvers**: Implemented in v0.8: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` solve the heat equation with finite differences and `ode_solve`; gradients through initial condition and diffusivity for inverse problems. See `examples/dedekind/pde_heat.ddk`.
3. **Physical Units**: Implemented at runtime: `10[m] / 2[s]` → `5[m/s]`, add/sub require same unit; future: compile-time unit checking.
4. **Probabilistic Programming**: Implemented in v0.8: `Normal`, `Uniform`, `Bernoulli`; `sample(dist)`, `log_prob(dist, value)`; `metropolis(log_prior, log_likelihood, data, init, steps)` for Bayesian inference. See `examples/dedekind/probabilistic.ddk`. Future: more distributions, NUTS/VI, conditioning syntax.
5. **Symbolic Simplification**: A compile-time algebraic engine that simplifies complex mathematical expressions before code generation to maximize efficiency.
