# Features for Scientists — Implementation Roadmap

**Dedekind Language**  
Draft: January 2026 · last roadmap status update: v1.17.0 (March 2027)

---

## Status Update (v1.17.0)

Since the original roadmap (January 2026), most of the planned phases have been delivered:

| Phase | Topic | Status | Delivered in |
|---|---|---|---|
| 1 | PDE solver (heat) | Done | v0.8; v1.x extended to Maxwell, Navier-Stokes, Burgers, reaction-diffusion, advection-diffusion |
| 1 | Extended distributions (Exponential, Gamma, Beta, Poisson) | Done | v0.9 |
| 1 | Numerical integration (`integrate`) | Done | v0.9 |
| 2 | Better error messages with line numbers | Done | v0.9.5 |
| 2 | **Source-mapped tracebacks** (beyond the roadmap) | Done | v1.8 |
| 3 | Uncertainty propagation (`uncertain()`) | Done | v0.9.3 |
| 3 | Compile-time unit check | Done | v0.9.5 |
| 4 | Fitting/regression (`fit()`) with GD/MCMC/HMC | Done | v0.9.3, v0.9.4 |
| 4 | LaTeX export (`export_to_latex`) | Done | v0.9.4 |
| 5 | Symbolic derivatives (`diff_sym`) | Done | v1.3 |
| 5 | **Beyond Phase 5**: `integrate_sym`, `solve_sym`, `simplify_sym`, `series` (SymPy bridge) | Done | v1.3 – v1.6 |

**Still open from the original roadmap:**
- **NUTS** (No-U-Turn Sampler) in addition to HMC — not implemented.
- **Variational Inference** (VI) — not implemented.

**Added since v1.7** (beyond the original roadmap): standard library modules, `pyimport`, purity check, shape annotations, PINN, graph methods, MILP DSL, MD bridge, labeled tensors, bioinformatics, try/catch, slicing. See `docs/README.md` and `Dedekind_Language_Specification.md` §15.12–§15.23.

---

## 1. Goal and Benefit

This roadmap prioritizes and plans **useful extensions** of the Dedekind language specifically for users in physics, chemistry, metrology, and related fields. Goals:

- **Prioritization**: clear order by effort, benefit, and dependencies.
- **Planability**: concrete phases with steps and success criteria.
- **Consistency**: connection to existing features (ODE, PDE, Quantity, probabilistic) and to the [Symbolic Simplification Roadmap](Symbolic_Simplification_Roadmap.md).

---

## 2. Feature Overview and Status

| Feature | Benefit | Effort | Status / Phase |
|--------|--------|---------|----------------|
| **PDE solver (differentiable)** | Heat, diffusion equation; PINNs often need ∇²u, ∂u/∂t. | High | **Implemented** (v0.9: `pde_heat_1d`, `pde_heat_2d`) |
| **More distributions** | Exponential, Gamma, Beta, Poisson for statistics, radiation, counting experiments. | Low | **Implemented** (v0.9: `Exponential`, `Gamma`, `Beta`, `Poisson`) |
| **Numerical integration** | `integrate(f, a, b)` for areas, expectations, normalization. | Low | **Implemented** (v0.9: `integrate(f, a, b, n)`; `sin`, `cos`) |
| **Better error messages** | Unit errors, dimension conflicts, "expected tensor, got Quantity" with line/context. | Medium | **Implemented** (v0.9.5: `CompileError` with line, parser/line in AST, runtime Quantity messages) |
| **Uncertainty propagation** | Error propagation: f(x ± Δx) → result with uncertainty; standard in metrology. | Medium | **Implemented** (v0.9.2: `uncertain(value, std)`, `UncertainQuantity`) |
| **Compile-time units** | `1[m] + 1[s]` → compile error instead of runtime error; fewer unit bugs. | Medium | **Implemented** (v0.9.5: `units_checker.py`, `compile_source(..., check_units=True)`, CLI `--no-units-check`) |
| **Fitting / regression** | `fit(model, data)` with gradient descent or MCMC; typical for curve fitting. | Medium | **Implemented** (v0.9.2: `fit(loss_fn, params_init, data, method="gd"|"mcmc")`) |
| **NUTS / VI** | More robust Bayesian inference (NUTS) or fast approximation (VI); Metropolis often slow. | Medium | Phase 4 (HMC done) |
| **LaTeX export of formulas** | Generate LaTeX from Dedekind expressions (for papers/notes). | Medium | **Implemented** (v0.9.4: `export_to_latex(source)`, CLI `--latex`) |
| **Symbolic derivatives** | `diff(expr, x)` returns a formula instead of numeric `grad()`; for papers, stability analysis. | Medium-high | Phase 5 |

---

## 3. Dependencies and Order

- **Phase 1** builds only on the existing runtime (no compiler changes needed).
- **Phase 2** improves compiler/runtime error output; helps all subsequent features.
- **Phase 3**: Uncertainty propagation uses `Quantity`-like types; compile-time units are a separate type/dimension system in the compiler.
- **Phase 4**: Fitting uses `grad()` and possibly `metropolis`; NUTS/VI extend the existing probabilistic API; LaTeX export uses the AST (possibly SymPy or a custom visitor).
- **Phase 5**: Symbolic derivatives can interact with symbolic simplification and possibly SymPy.

---

## 4. Implementation Phases

### Phase 1: Low Effort — More Distributions & Numerical Integration (v0.9)

**Goal**: Quick benefit for statistics and integration without compiler changes.

**Delivered**:

1. **More distributions** in `ml_runtime.py`: `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`; API like `Normal`/`Uniform`: `sample(dist)`, `sample(dist, n)`, `log_prob(dist, value)`.
2. **Numerical integration** in `ml_runtime.py`: `integrate(f, a, b, n=100)` with trapezoidal rule; differentiable when `f` accepts a tensor. Additionally `sin(x)`, `cos(x)` for expressions.
3. **Examples**: `examples/dedekind/distributions_extended.ddk`, `examples/dedekind/integration.ddk`.
4. **Documentation**: Language Spec §15.8 (extended distributions), §15.10 (integration & math); README "What's New in v0.9".

**Success criterion**: Met — both examples run without errors.

---

### Phase 2: Better Error Messages (v0.9.5)

**Goal**: Report unit errors, dimension conflicts, and type errors with line and context.

**Delivered**:

1. **Source positions**: AST nodes optionally carry `line`; the parser sets them on all constructs; the lexer reports the line per token.
2. **Compiler errors**: `CompileError(message, line=..., filepath=...)` with formatted output "file: line N: message"; the parser raises on expected tokens, invalid assignment targets, unexpected tokens.
3. **Runtime errors**: `Quantity`/`UncertainQuantity` on unit mismatch (+/-) with a clear message: "Unit error in addition: [m] vs [s]. For + and - both quantities must have the same unit."
4. **Pipeline**: `compile_source(source, filepath=..., check_units=...)`; `run_examples` passes `filepath`; the CLI catches `CompileError` and prints it formatted.

**Success criterion**: Met — typical errors produce a clear message with line number; all examples run.

---

### Phase 3: Uncertainty Propagation & Compile-time Units (estimated: 3–5 weeks)

**Goal**: Error propagation at runtime; unit checks before execution.

**3a) Uncertainty propagation**

1. **Type**: Extension of `Quantity` or a new type `UncertainQuantity(value, std)` or `value ± std` with propagation rules (Gaussian propagation for +, -, *, /, ^).
2. **API**: e.g. `x_with_err = uncertain(10.0, 0.5)` or a literal syntax; output "value ± std".
3. **Integration**: Combinable with existing units (value and std same unit); in `ml_runtime.py` and codegen (new function/built-in).

**3b) Compile-time units** (v0.9.5)

1. **Check before codegen**: `units_checker.py` — visitor over AST; for `+`/`-` the unit is inferred from literals, Quantity, and known constants; on mismatch (e.g. `1[m] + 1[s]`) a `CompileError` with line is raised. Unary minus (`0 - x`) allowed.
2. **API**: `compile_source(..., check_units=True)` (default); CLI `--no-units-check` to disable.
3. **Known constants**: `c`, `G`, `h`, `pi`, `e`, … with units in `KNOWN_UNITS`; identifiers are resolved as needed.

**Success criterion**: Met — `1[m] + 1[s]` produces a compiler error with line; all examples (including `universal_constants.ddk`) run.

---

### Phase 4: Fitting, NUTS/VI, LaTeX Export (estimated: 4–6 weeks)

**Goal**: Curve fitting, better Bayesian tools, and formula export for papers.

**4a) Fitting / regression** (v0.9.2, extended in v0.9.4)

1. **API**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=0.01, steps=500)` — minimizes `loss_fn(params, data)` via gradient descent, Metropolis-Hastings, or **HMC** (Hamiltonian Monte Carlo).
2. **Implementation**: in `ml_runtime.py`; GD with PyTorch backward; MCMC via `metropolis`; HMC with leapfrog integration and gradients. Examples: `curve_fitting.ddk`, `hmc_fitting.ddk`.

**4b) HMC** (v0.9.4), NUTS / VI optional

1. **HMC**: `hmc(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size=0.1, num_leapfrog=10)` — same API as `metropolis`; uses gradients for better proposals. Also usable as `fit(..., method="hmc")`.
2. **NUTS/VI**: optional later (Pyro/NumPyro or own implementation).

**4c) LaTeX export** (v0.9.4)

1. **AST → LaTeX**: visitor in `src/compiler/latex_export.py` — Literal, Identifier, BinaryOp (+, -, *, /, ^), Call (sin, cos, exp, log, sqrt, …), Quantity, Subscript, Lambda into LaTeX strings.
2. **API**: `export_to_latex(source_code)` in the `compiler` module; CLI: `python -m src.compiler.compiler <file.ddk> --latex`. Output: equations (assignments/returns) as LaTeX.
3. **Example**: `examples/dedekind/latex_demo.ddk`; output e.g. `E = m \cdot {c}^{2}`.

**Success criterion**: Met — `fit(..., method="hmc")` returns posterior samples; `export_to_latex(source)` produces readable LaTeX for typical formulas.

---

### Phase 5: Symbolic Derivatives (estimated: 4–8 weeks)

**Goal**: `diff(expr, x)` returns an expression (formula), not just a numeric `grad()`; for stability analysis, papers, simplified terms.

**Steps**:

1. **Option A — SymPy**: Dedekind AST → SymPy Expr → `sympy.diff(expr, x)` → back into Dedekind AST or directly LaTeX/code. Requires AST ↔ SymPy translator (similar to Symbolic Simplification Phase 4).
2. **Option B — Own implementation**: dedicated `symbolic_diff.py` module: visitor over AST, derivative rules for +, -, *, /, ^, `exp`, `log`, `sin`, `cos`, etc.; produces a new AST. No external dependency, but limited to implemented rules.
3. **API**: `diff(expr, var)` — `expr` can be a Dedekind expression (as a string or AST); returns simplified expression (or string/LaTeX).
4. **Integration**: Coordinate with Symbolic Simplification (simplified derivatives); optionally combine with LaTeX export.

**Success criterion**: For polynomial and simple transcendental expressions, `diff(expr, x)` returns the correct derivative as an expression; documentation and example.

---

## 5. Overview: Phases and Milestones

| Phase | Content | Estimated effort | Milestone |
|-------|--------|---------------------|-------------|
| 1 | More distributions, numerical integration | v0.9 | New stdlib functions running |
| 2 | Better error messages | v0.9.5 | Errors with line/context |
| 3 | Uncertainty propagation, compile-time units | v0.9.2 / v0.9.5 | Metrology & unit safety |
| 4 | Fitting, NUTS/VI, LaTeX export | v0.9.2–v0.9.4 | Regression, Bayesian, papers |
| 5 | Symbolic derivatives | 4–8 weeks | diff(expr, x) as a formula |

---

## 6. Risks and Options

| Risk | Mitigation |
|--------|------------|
| NUTS/VI increases dependencies (Pyro/NumPyro) | Optional as an extra; or lean custom implementation only for NUTS. |
| Compile-time units require a larger type system | Step by step: first only literals and simple binary ops; no full inference in the MVP. |
| Symbolic derivatives and SymPy path duplicated | Phase 5 first with custom implementation (Option B); SymPy optional later. |
| Fitting API too rigid | First version with `fit(loss_fn, params_init, data)`; extension with `model_fn(x, params)` in Phase 4. |

---

## 7. References and Next Steps

- **Existing roadmap**: [Symbolic_Simplification_Roadmap.md](Symbolic_Simplification_Roadmap.md) — units in simplification (Phase 5 there) coordinate with Phase 3 here.
- **Chemistry & biology**: [Chemistry_Biology_Roadmap.md](Chemistry_Biology_Roadmap.md) — units mol/L/M, examples (kinetics, dose-response, growth), convenience functions, docs "Dedekind for Chemistry & Biology".
- **Language Specification**: §15 Standard Library; §12 Implementation Roadmap; "Beyond v1.0".
- **Code base**: `src/compiler/ml_runtime.py` (stdlib), `src/compiler/codegen.py` (built-ins), `src/compiler/compiler.py` (pipeline), `src/compiler/parser.py` (AST, line info).
- **Next concrete step**: Phase 5 — symbolic derivatives (`diff(expr, x)` as a formula); optionally refine Phase 2/3 (e.g. column, IDE display).

---

*This document is the implementation roadmap for science-oriented features. The roadmap should be updated when the language spec or dependencies change.*
