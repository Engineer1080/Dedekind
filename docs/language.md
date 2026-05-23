# Dedekind Language Reference

Living catalogue of every Dedekind built-in, operator and runtime function. Generated from the original README feature list; the canonical specification is `docs/Dedekind_Language_Specification.md`.

## Core features

- **Ricci Calculus**: Native index notation (`A^ij * B_jk`) for Einstein summation.
- **Sparse Tensors**: Efficient `.sparse()` support for FEM/CFD simulations.
- **Fundamental Constants**: Mathematical `pi`, `e`; physical CODATA constants: `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — all as **Quantity** with SI units.
- **Physical Units**: SI base m, kg, s, A, K, mol, **cd** (candela); literals (`10[m]`, `5[m/s]`, `1.0[kg]`, `1[cd]`); **automatic conversion** for addition/subtraction of the same dimension — **SI base**: length (m, cm, km, mm, dm), mass (kg, g, t, mg), time (s, min, h, ms), current (A, mA, kA, uA), temperature (K, mK), amount of substance (mol, mmol, kmol), luminous intensity (cd, mcd); **derived**: pressure (Pa, bar, atm, hPa), volume (L, mL, dm³, m³), energy (J, kJ, MJ, Wh, kWh), voltage (V, mV, kV), frequency (Hz, kHz, MHz, GHz), charge (C, mC, uC), resistance (ohm, kohm, Mohm), power (W, kW, MW); **angle**: rad, deg. Result = unit of the first operand; e.g., `1[m] + 100[cm]` → `2[m]`, `90[deg] + (pi/2)*1[rad]` → `180[deg]`. `deg_to_rad(x)`, `rad_to_deg(x)` for conversion. Otherwise add/sub same unit; multiply/divide combine units; `^` for powers; display simplified (J, N, Pa, W, Hz, ...). **Chemistry**: mol, L, M (= mol/L), ppm, **bar**, **atm**, **g**; **Radioactivity**: **Bq**, **Sv**, Gy.
- **4D Rotational Math**: Native Quaternion support (`i`, `j`, `k` suffixes) and `.rotate()` method; unary minus supported (`-1.0 + 0i`).
- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` (RK4/Euler); gradients via `grad()` for physics-informed ML; `linspace(start, stop, steps)` for time grids.
- **Differentiable PDE Solvers**: `pde_heat_1d`, `pde_heat_2d` (heat); `pde_advection_1d`, `pde_advection_2d` (advection); `pde_wave_1d`, `pde_wave_2d` (wave); `pde_burgers_1d`, `pde_burgers_2d` (Burgers); `pde_reaction_diffusion_1d`, `pde_reaction_diffusion_2d`; `pde_advection_diffusion_1d`, `pde_advection_diffusion_2d`; `pde_maxwell_1d`, `pde_maxwell_2d` (Maxwell FDTD); `pde_navier_stokes_2d` (Navier-Stokes 2D incompressible, Chorin projection); finite differences + `ode_solve`; gradients through `u0` and parameters.
- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`, `Dirichlet(alpha)`; `sample(dist)`, `log_prob(dist, value)`; Bayesian inference via `metropolis(log_prior, log_likelihood, data, init, steps)`.
- **Numerical Integration**: `integrate(f, a, b, n)` — trapezoidal quadrature; differentiable when `f` accepts a tensor.
- **Math Functions**: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; `asin`, `acos`, `atan`, `atan2(y,x)`; `sinh`, `cosh`, `tanh` — element-wise, differentiable; Tensor or scalar. **Reductions**: `min(x)`, `max(x)`, `argmin(x)`, `argmax(x)` (optional `dim`). **Rounding**: `round(x)`, `floor(x)`, `ceil(x)`. **Linear algebra**: `norm(x)`, `det(A)`, `trace(A)`.
- **Uncertainty Propagation**: `uncertain(value, std)` or `UncertainQuantity` — Gaussian uncertainty propagation for +, -, *, /, ^; optionally with unit.
- **Fitting / Regression**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=0.01, steps=500)` — minimizes `loss_fn(params, data)` via Gradient Descent, Metropolis-Hastings or **HMC** (Hamiltonian Monte Carlo).
- **Eliminate paper-code drift (LaTeX from AST)**: `export_to_latex(source)` or CLI `--latex` — formulas, units, and measurement uncertainties in the manuscript are generated from the same AST that executes the simulation. This eliminates an entire class of bugs arising in manually typed methods and table sections. Domain nodes: **Ricci indices** (`A^ij * v^j` → `A^{ij}\, v^{j}` with Einstein convention, no `\cdot`), **`partial(u, x, order=2)`** → `\frac{\partial^2 u}{\partial x^2}`, **`pde_heat_2d`/`pde_wave_1d`/`pde_navier_stokes_2d`/...** → canonical PDE form, **`lagrange_ode_rhs(L)`** → Euler-Lagrange, **`hamilton_ode_rhs(H)`** → canonical equations. **Scientific console**: `print_latex(s)` renders LaTeX in the Jupyter or IDE console.
- **Reproducibility Report**: CLI `--reproducibility-report PATH` writes a paper appendix with the Git commit (including dirty flag), Python/torch/numpy/scipy versions, CUDA availability, SHA-256 of the source, detected RNG seeds, and the methods section as LaTeX — all from *one* source. Addresses **paper-code drift**, not the entire reproducibility crisis (data provenance, pre-registration, etc. remain unaffected). Demo: `reproducibility_demo.ddk`.
- **Better Error Messages**: Compiler errors with line numbers (`CompileError`); parser sets `line` in AST; runtime quantity messages with context.
- **Compile-time units**: `1[m] + 1[s]` → compiler error with line number; `compile_source(..., check_units=True)` (default), CLI `--no-units-check`.
- **File I/O**: `read_file(path)` (UTF-8 text), `write_file(path, content)`, `file_exists(path)`.
- **Network**: `http_get(url)`, `http_post(url, data)` (data as string or dict/list → JSON); response text UTF-8.
- **JSON**: `json_parse(s)` → object (access `obj["key"]`), `json_stringify(obj)` → string.
- **AOT Compilation** *(planned, see Roadmap Phase 16)*: Native binary generation via MLIR and LLVM. The current `aot_compiler.py` is a prototype that emits MLIR-style text and C++ stubs; the real toolchain integration is not yet wired up.
- **IDE Integration**: The **Dedekind Jupyter Kernel** (`src/dedekind_jupyter_kernel/`, installed via `python -m dedekind.install_kernel`) lets Dedekind run natively in Jupyter, JupyterLab, Spyder, and VS Code.

## Chemistry & biology

Dedekind supports chemical and biological applications with the same building blocks as physics and ML: units, ODE solvers, fitting, and uncertainty propagation.

- **Units**: Concentration in `[M]` (molarity), amount of substance in `[mol]`, volume in `[L]`, dilutions in `[ppm]`; `M` and `mol/L` are treated as equivalent (runtime and compile check).
- **Kinetics**: First-order reaction \(c(t) = c_0 e^{-kt}\) with `ode_solve` and units `[M]`, `[1/s]` — Example: `chemistry_kinetics.ddk`.
- **Dose-Response / Michaelis-Menten**: Hill equation or \(v = V_{\max}[S]/(K_M + [S])\); parameter fitting with `fit` (EC50, \(K_M\), \(V_{\max}\)) — Example: `dose_response.ddk`.
- **Growth**: Logistic growth \(dN/dt = r N (1 - N/K)\) with `ode_solve` — Example: `biology_growth.ddk`.
- **Convenience**: `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)` — callable in a single line.
- **Chemical Elements**: `atomic_mass("C")` → atomic mass in g/mol (Quantity); atomic number; IUPAC-like for H, C, N, O, S, P, Na, Cl, Fe, ... (approx. 50 elements). Example: Molar mass H₂O = 2*atomic_mass("H") + atomic_mass("O"); `chemistry_elements.ddk`.
- **Medicine, Pharmacology & Epidemiology**: `hill_equation`, `one_compartment_pk`, `half_life` (pharmacokinetics); `sir_model`, `basic_reproduction_number` (epidemiology); `confidence_interval`, `odds_ratio`, `sensitivity_specificity` (biostatistics) — Examples: `pharmacology_quickwins.ddk`, `epidemiology_sir.ddk`, `biostatistics_quickwins.ddk`.

Constants like `N_A`, `R_gas`, `F_faraday` are available as **Quantity** with SI units (`1/mol`, `J/(K*mol)`, `C/mol`). Detailed roadmap: `docs/Chemistry_Biology_Roadmap.md`.


## ML example

```dedekind
fn main() {
    // Define a Neural Network
    model = Sequential([
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    
    // Create data on GPU
    input = [[1.0, 2.0, 3.0]].gpu()
    
    // Run inference
    output = model.forward(input)
    print(output)
}
main()
```

```dedekind
fn main() {
    model = Sequential([
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    
    // Data is automatically processed as a tensor
    input = [[1.0, 2.0, 3.0]].gpu()
    
    print("Prediction:")
    print(model.forward(input))
}
main()
```


## Architecture

The project consists of two main parts:

1.  **Dedekind Compiler (`src/dedekind/`)**
    *   Implemented in Python; transpiles `.ddk` source into Python/PyTorch (future target: MLIR/LLVM via the AOT path).
    *   Used by the `dedekind` CLI and the Dedekind Jupyter Kernel.

2.  **Dedekind Jupyter Kernel (`src/dedekind_jupyter_kernel/`)**
    *   Installed via `python -m dedekind.install_kernel`. Provides a persistent execution context for Dedekind in Jupyter, JupyterLab, Spyder, and VS Code.


## Full example index

### Examples
Example programs are in `examples/dedekind/`, including:
- `hello.ddk` – basic I/O and tensors  
- `universal_constants.ddk` – physical constants and units (E = mc², gravitation, Coulomb)  
- `constants_extended.ddk` – mathematical `pi`, `e`; CODATA constants (`hbar`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday`)  
- `signal_physics.ddk` – complex numbers (Quaternions) and FFT  
- `differentiable_ode.ddk` – differentiable ODE solver with `ode_solve` and `grad`  
- `pde_heat.ddk` – differentiable PDE solver (1D/2D heat equation) with `pde_heat_1d` / `pde_heat_2d`  
- `distributions_extended.ddk` – Exponential, Gamma, Beta, Poisson; `sample`, `log_prob`
- `dirichlet_distribution_function.ddk` – Dirichlet distribution and Dirichlet function D(x)
- `dedekind_cuts_rings.ddk` – Dedekind cuts (construction of R from Q) and Dedekind rings (ideal factorization in Z)
- `riemann_zeta_sums.ddk` – Riemann zeta ζ(s) and Riemann sums (left, right, midpoint)
- `volume_revolution.ddk` – solids of revolution (sphere, cone, paraboloid)
- `abs_bars.ddk` – absolute value bars `|x|` = abs(x)  
- `integration.ddk` – numerical integration `integrate(f, a, b)` and `sin`/`cos`  
- `uncertainty_propagation.ddk` – `uncertain(value, std)`; Gaussian uncertainty propagation  
- `curve_fitting.ddk` – `fit(loss_fn, params_init, data)` for linear regression  
- `file_io_json.ddk` – file I/O (`read_file`, `write_file`, `file_exists`), JSON (`json_parse`, `json_stringify`), key access `obj["key"]`  
- `linear_regression.ddk` – quick win: `linear_regression(x, y)` → slope, intercept  
- `chemistry_kinetics.ddk` – first-order reaction with units [M], [1/s] and `ode_solve`  
- `chemistry_arrhenius.ddk` – quick win: `arrhenius(T, A, Ea)` (Arrhenius equation)  
- `chemistry_elements.ddk` – atomic mass `atomic_mass("C")` (g/mol), atomic number `atomic_number("C")`; molar mass H₂O, C₂H₆  
- `dose_response.ddk` – dose-response (EC50/Vmax/Km) with `michaelis_menten` and `fit`  
- `biology_growth.ddk` – logistic growth with `logistic_growth_dt`/`logistic` and `ode_solve`  
- `pharmacology_quickwins.ddk` – Hill equation, one-compartment PK, half-life  
- `epidemiology_sir.ddk` – SIR model, R₀  
- `biostatistics_quickwins.ddk` – confidence interval, odds ratio, sensitivity/specificity  
- `probabilistic.ddk` – distributions, sampling, and Bayesian inference with `metropolis`  
- `conditional_logic.ddk`, `basic_loops.ddk` – control flow  
- `mnist_classifier.ddk` – neural network with `Sequential`/`Dense`  

From the `src/` directory: `python -m compiler.compiler ../examples/dedekind/hello.ddk`

Test all examples at once (from project root): `python run_examples.py` — compiles and runs all `.ddk` files in `examples/dedekind`; options: `-q` (summary only), `-v` (verbose output), `--compile` (compile only), `--filter name` (only files with "name" in the filename).

## Further documentation

- **Language Specification**: `docs/Dedekind_Language_Specification.md` (v0.2; §15 Physical Units v0.6, §15.7 ODE v0.7, §15.8 Probabilistic v0.8, §15.9 PDE v0.8, §15.10 Integration & Math v0.9/v0.9.6; Chemistry/Biology v0.9.7; I/O/JSON v0.9.8; as of v1.0.10). PDF can be generated with `pandoc`.
- **Research & Architecture**: `docs/Dedekind_Research_and_Architecture.md` (includes §10 language features v0.6).
- **Symbolic Simplification**: `docs/Symbolic_Simplification_Roadmap.md` — implementation roadmap (phases, options, integration).
- **Features Roadmap**: `docs/Features_Implementation_Roadmap.md` — scientific features (Phase 1 completed: distributions, integration).
- **Chemistry & Biology**: `docs/Chemistry_Biology_Roadmap.md` — units mol/L/M/ppm, examples (kinetics, dose-response, growth), convenience functions.
