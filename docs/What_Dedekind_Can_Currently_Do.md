# What Dedekind Can Currently Do

**Status:** Based on code and changelogs (v2.2.0, May 2026). Dedekind is a **prototype** — the language is transpiled to Python and uses PyTorch/NumPy as runtime.

---

## Language Core

- **Syntax:** Imperative, C/JavaScript-like with blocks in `{}`, functions with `fn name(args) { ... }`, control flow `if`/`else`, `while`, `for ... in`. **Absolute-value bars:** `|x|` = `abs(x)` (e.g. `x = |-1|` → 1). **Logical operators:** `and`, `or`, `not`, `xor`, `nand`, `nor`, `xnor` (Python-like keywords; precedence: `or` < `xor` < `and`/`nand`/`nor`/`xnor` < `not`).
- **Types:** Dynamic type inference; primitive types, lists, vectors/matrices as tensors, quaternions; property access (e.g. `.shape`, `.gpu()`). **Matrix multiplication:** operator `@` (e.g. `A @ B`).
- **Execution modifiers:** `.gpu()`, `.cpu()`, `.single()` for hardware placement or sequential execution; `.sparse()` for sparse tensors; `.fast()` for MLIR/Inductor optimization (e.g. for models).
- **Error handling:** Compiler errors with line number (`CompileError`); compile-time unit check (`1[m] + 1[s]` → error); runtime messages with context. **Assert:** `assert(condition, message)` — raises AssertionError on a false condition; mini test runner `run_tests.py` for `tests/dedekind/*.ddk`.

---

## Mathematics

- **Constants:** `pi`, `e` (dimensionless).
- **Functions:** `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; arc and hyperbolic functions (`asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`) — element-wise, differentiable.
- **Reductions & rounding:** `min`, `max`, `argmin`, `argmax` (optional `dim`); `round`, `floor`, `ceil`.
- **Statistics:** `mean(x)`, `std(x)`, `var(x)`, `median(x)` (optional `dim`); `quantile(x, q)`, `percentile(x, p)`; `cov(x, y)`, `corrcoef(x, y)` (covariance/correlation, for 2D x: matrix); `skew(x)`, `kurtosis(x)` (skewness, kurtosis); `histogram(x, bins, range_lim)` (count per bin; range_lim optional (min, max)).
- **Linear algebra:** `norm(x)`, `det(A)`, `trace(A)`; `solve(A, b)` (Ax = b); `eigh(A)` (eigenvalues/eigenvectors symmetric); `eig(A)` (general); `svd(A)` (singular value decomposition); `lstsq(A, y)` (least squares); `cond(A)`, `rank(A)`, `pinv(A)` (condition, rank, pseudo-inverse); `expm(A)`, `logm(A)` (matrix exponential/logarithm). FFT (`fft`, `ifft`); matrix operations (transpose, inverse, dot_product). **Symbolic mathematics:** `diff_sym(expr, x)` (derivative), `integrate_sym(expr, var)` (indefinite integral), `solve_sym(eq, var)` (solving equations/systems of equations) — all via SymPy with a string interface. **Autograd:** `grad(f, x)` (gradient); `jacobian(f, x)` (Jacobian); `hessian(f, x)` (Hessian).
- **Numerics:** `interp(x, xp, fp)` (1D linear interpolation); `trapz(y, x)`, `simpson(y, x)` (trapezoidal/Simpson for discrete data); `riemann_sum(f, a, b, n, method="left"|"right"|"midpoint")` (Riemann sums for int f dx); `zeta(s)` (Riemann zeta ζ(s)=Σ 1/n^s, scipy); `volume_revolution_x`, `volume_revolution_y`; `volume_revolution_vertical(f,a,b,x0,n)` (axis x=x0), `volume_revolution_horizontal(f,a,b,y0,n)` (axis y=y0); `pappus_volume_vertical`, `pappus_volume_horizontal` (Pappus's theorem: V=2π·R·A); `root_bisect(f, a, b, tol)` (root via bisection); `newton(f, x0, tol)` (1D); `fsolve(f, x0)` (vector root via Newton); `minimize(f, x0, method="gd"|"lbfgs")` (multi-dimensional minimization). **Mathematical sequences:** `arange(n)` or `arange(start, stop, step)` (integer sequence); `arithmetic(a0, d, n)` (arithmetic: aₙ = a₀ + n·d); `geometric(a0, r, n)` (geometric: aₙ = a₀·rⁿ); `sequence(f, n)` (general: [f(0), f(1), …, f(n−1)]). **Additional algorithms:** `qr(A)`, `lu(A)` (QR/LU decomposition); `cholesky(A)`; `matrix_power(A, n)`; `kron(A, B)` (Kronecker product); `outer(a, b)` (outer product); `vander(x, n)` (Vandermonde matrix); `matrix_sqrt(A)` (matrix square root); `matrix_norm(A, ord)` (Frobenius, spectral norm, etc.); `cdist(X, Y, p)` (pairwise distances); `cross(a, b)` (3D cross product); `polyfit`, `polyval`; `unique`, `argsort`; `convolve1d`. **Signal & reductions:** `fftfreq`, `diff`, `cumsum`, `clip`, `shuffle`; `permutation(n)` (random permutation); `choice(a, size, replace)` (random sample); `autocorr(x, max_lag)`; `moving_mean(x, window)`. **Special functions:** `erf(x)`, `erfc(x)` (error function); `gamma(x)`, `lgamma(x)` (gamma/log-gamma); `bessel_j0(x)`, `bessel_j1(x)` (Bessel J₀, J₁); `bessel_j(n, x)` (Bessel Jₙ); `legendre(n, x)` (Legendre Pₙ); `hypergeom(a, b, c, z)` (₂F₁). **Number theory:** `gcd(a, b)`, `is_prime(n)`, `mod(a, m)`, `mod_inv(a, m)`, `mod_pow(base, exp, m)`; `dirichlet_function(x)` (D(x)=1 if x is rational with denominator ≤10000, else 0; element-wise for tensors). **Factorial:** postfix operator `n!` (e.g. `5!`, `n!`) or `factorial(n)`; example `factorial_test.ddk`. **Dedekind cuts:** `DedekindCut(x)` (real number as lower set A={q∈Q:q<x}); `dedekind_cut_from_rational(p,q)`, `dedekind_cut_sqrt2()`; `lower_set_contains(cut,q)`, `to_float()`. **Dedekind rings:** `DedekindRingZ()`, `ideal(n)`, `ideal_factor(i)`; `DedekindIdeal` with `.factor()`, `.norm()`, `*` (ideal multiplication). **Numerical integration:** `integrate(f, a, b, n)` (trapezoidal rule); differentiable.
- **Ricci notation:** Index notation `A^ij * B_jk` for Einstein sums (auto-einsum).
- **LaTeX export:** `export_to_latex(source)` or CLI `--latex` — formulas from Dedekind code as LaTeX; `print_latex(s)` displays formulas **only in the console** as Unicode (α, Δ, ∫, ½ etc.), no images in plots; future possibility: KaTeX/web (see docs/Console_KaTeX_Roadmap.md).

---

## Physics & Units

- **Unit literals:** SI base units m, kg, s, A, K, mol, cd; derived units (Pa, W, Hz, V, F, ohm, S, Wb, T, H, lm, lx, Gy, kat, M); chemistry: mol, L, M (= mol/L), ppm, bar, atm, g; radioactivity: Bq, Sv. **Angles:** rad, deg (SI supplement; automatic conversion). **Quick wins:** force (N, kN, MN), pressure (kPa, MPa), permeability (m², D, mD) for construction, materials, geology.
- **Arithmetic:** **Automatic conversion** in addition/subtraction for the same dimension; result in the unit of the first operand. Supported: **SI base** — length (m, cm, km, mm, dm), mass (kg, g, t, mg), time (s, min, h, ms), current (A, mA, kA, uA), temperature (K, mK), amount of substance (mol, mmol, kmol), luminous intensity (cd, mcd); **derived** — pressure (Pa, bar, atm), volume (L, mL, dm³, m³), energy (J, kJ, MJ, Wh, kWh), voltage (V, mV, kV), frequency (Hz, kHz, MHz, GHz), charge (C, mC, uC), resistance (ohm, kohm, Mohm), power (W, kW, MW); **angle** — rad, deg. Otherwise the same unit is required for +/-. Multiplication/division combines units; power with `^`. **Angle conversion:** `deg_to_rad(x)`, `rad_to_deg(x)` for scalar, tensor, or Quantity. **Display**: equal factors are combined (e.g. `m*m` → `m^2`, `m*m*m` → `m^3`); literals `1[m^2]`, `1[m^3]` are usable. Display is simplified (J, N, Pa, W, etc.).
- **Physical constants (CODATA):** `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — all as `Quantity` with SI units.
- **Differentiable ODE:** `ode_solve(fun, y0, t)` (RK4/Euler); `linspace(start, stop, steps)`; gradients via `grad()` for physics-informed ML. **Lagrange/Hamilton:** `lagrange_ode_rhs(L)` — right-hand side from Lagrangian L(q,v) for ode_solve; `hamilton_ode_rhs(H)` — from Hamiltonian H(q,p). **Lotka-Volterra:** `lotka_volterra(x0, y0, a, b, c, d, t)` — predator-prey model.
- **Differentiable PDE:** `pde_heat_1d`, `pde_heat_2d` (heat equation); `pde_advection_1d`, `pde_advection_2d` (advection); `pde_wave_1d`, `pde_wave_2d` (wave equation); `pde_burgers_1d`, `pde_burgers_2d` (Burgers); `pde_reaction_diffusion_1d`, `pde_reaction_diffusion_2d` (reaction-diffusion); `pde_advection_diffusion_1d`, `pde_advection_diffusion_2d` (advection-diffusion); `pde_maxwell_1d`, `pde_maxwell_2d` (Maxwell, FDTD); `pde_navier_stokes_2d(u0, v0, x, y, t, nu)` (Navier-Stokes 2D incompressible, Chorin projection); finite differences + `ode_solve`.
- **Quaternions:** Native support (`i`, `j`, `k` suffixes), `.rotate()`; unary minus (`-1.0 + 0i`).
- **Uncertainty propagation:** `uncertain(value, std)` or `UncertainQuantity` — Gaussian error propagation for +, -, *, /, ^; optional with unit.
- **Space physics & orbital mechanics (`use space`):** Differentiable N-body simulation (RK4), Kepler equation solver (Newton-Raphson), conversion of Kepler orbital elements to Cartesian coordinates (and vice versa, also directly from the eccentric anomaly $E$). Enables autograd-based trajectory optimization and orbit maneuver design (e.g. L-BFGS optimization of an orbital transfer).

- **Differential geometry:** `christoffel_symbols(g_func, x, h)`, `riemann_tensor(g_func, x, h)`, `covariant_derivative(T, g_func, x, h)` — Christoffel symbols, Riemann tensor, covariant derivative (numerical).

---

## Differentiable Engineering (v2.2)

- **Control engineering (`use signals`):** Differentiable control blocks for dynamic system modeling. Supports stateful blocks such as PID controllers (`PIDBlock`), transfer functions (`TransferFunctionBlock`), integrators (`IntegratorBlock`), and saturation elements (`SaturationBlock`). Provides automatic loop resolution for parameter-based optimization of closed control loops.
- **Fluid dynamics (`use fluid_dynamics`):** Fully differentiable CFD pipeline.
  - **LBM D2Q9:** BGK and MRT collision (`lbm_physical_mrt`, Lallemand-Luo 2000) for high Reynolds numbers; soft & hard (halfway) bounce-back boundary handling.
  - **Unit-aware API:** `lbm_physical(domain_x[m], domain_y[m], nx, inlet[m/s], nu[m²/s], rho[kg/m³])` returns drag/lift in `N/m`, pressure in `Pa`, velocity in `m/s` — Quantity consistency is checked at compile and runtime.
  - **Real MEM drag/lift:** Momentum-exchange method per Newton III; validated via Karman vortex benchmark (Strouhal number).
  - **Differentiable shape parametrization:**
    - `soft_cylinder_mask`, `soft_ellipse_mask` (2 parameters, volume constraint),
    - `soft_airfoil_mask` (NACA-like),
    - `fourier_shape_mask` (K harmonics, 2K free coefficients for cos/sin → complex topologies like class-shape transformation in industry),
    - all fully autograd-capable.
  - **End-to-end shape optimization:** Adam/L-BFGS differentiates through hundreds of LBM steps directly to the shape parameters — no separate adjoint solver needed. Validated vs. central finite differences (rel. error < 10⁻⁴).
  - **Immersed Boundary Method (IBM):** In the Chorin-Navier-Stokes solver (`pde_navier_stokes_2d`) — Brinkman penalization with force history for coupled FSI applications.
  - **Thermal LBM with Boussinesq buoyancy:** Double-distribution solver (`lbm_thermal_simulation`, `lbm_thermal_with_obstacle`) — D2Q9 for flow, D2Q5 for temperature, coupled via Boussinesq buoyancy. Heated Dirichlet obstacles, Rayleigh/Nusselt number diagnostics. Applications: Rayleigh-Bénard convection, heat exchanger design, climate research, reactor cooling.
- **Structural mechanics (`use structural`):** 2D finite-element elasticity solver (Q4 bilinear elements) for static mechanical stress analysis. Includes SIMP material density modeling, sensitivity convolution filters, and an integrated Optimality Criteria (OC) solver for topology optimization (e.g. bridge designs) including Unicode ASCII preview in the terminal.
- **Heat transfer (`use thermal`):** Steady and transient (backward Euler) finite-element heat diffusion on Q4 elements. Supports thermal SIMP density-conductance interpolation and OC topology optimization for heat-dissipating structures (heat sinks) with ASCII Unicode grid display.
- **Signal processing & AC networks:** Native FIR and IIR filter structures (biquads) with autograd; SPICE-like Modified Nodal Analysis (MNA) solver for AC voltages and currents including complex-valued phase angles.

---

## Stochastics & Fitting

- **Distributions:** `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma`, `Beta`, `Poisson`, `Dirichlet(alpha)`; `sample(dist)`, `log_prob(dist, value)`. **Binomial coefficient:** `binom(n, k)` (n choose k). **t-test:** `ttest_one_sample(x, mu0)`, `ttest_two_sample(x, y)` (Welch) — returns (t_statistic, p_value).
- **Bayesian inference:** `metropolis(log_prior, log_likelihood, data, init, steps)`; **HMC:** `hmc(...)` and `fit(..., method="hmc")`.
- **Fitting:** `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=..., steps=...)` — gradient descent, Metropolis-Hastings, or Hamiltonian Monte Carlo.

---

## Chemistry & Biology

- **Units:** Concentration in `[M]`, amount of substance in `[mol]`, volume in `[L]`, dilutions in `[ppm]`; mass concentration `[percent_wv]` (= g/100mL); M and mol/L are treated as the same (runtime and compile check). **pH:** `concentration_to_pH(c_M)`, `pH_to_concentration(pH)` for conversion [H⁺] ↔ pH.
- **Convenience:** `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)`. **Chemical equilibrium:** `chemical_equilibrium(K, n_A, n_B, n_C, n_D, A0, B0, C0, D0)` — law of mass action for aA + bB <-> cC + dD; returns (A_eq, B_eq, C_eq, D_eq).
- **Chemical elements:** `atomic_mass("C")` (g/mol), `atomic_number("C")`; about 50 elements (IUPAC-close); molar mass e.g. H₂O, C₂H₆.
- **Stoichiometry:** `balance_equation(reactants_str, products_str)` — coefficients for a balanced reaction equation (e.g. `"H2 + O2"`, `"H2O"` → ([2,1], [2])); uses linear algebra (null space via SVD).
- **Examples:** 1st-order kinetics, dose-response (EC50, Michaelis-Menten), logistic growth with `ode_solve` and `fit`.

---

## Medicine, Pharmacology & Epidemiology

- **Pharmacology:** `hill_equation(dose, Emax, EC50, n)` — Hill equation E = Emax·doseⁿ/(EC50ⁿ+doseⁿ); `one_compartment_pk(C0, ke, t)` — one-compartment C(t)=C0·e^(-ke·t); `half_life(ke)` — half-life t₁/₂ = ln(2)/ke.
- **Epidemiology:** `sir_model(S0, I0, R0, beta, gamma, t)` — SIR compartment model; `basic_reproduction_number(beta, gamma)` — R₀ = β/γ.
- **Biostatistics:** `confidence_interval(x, alpha)` — confidence interval for the mean (t-distribution); `odds_ratio(a, b, c, d)` — odds ratio from a 2×2 table; `sensitivity_specificity(TP, FN, FP, TN)` — sensitivity, specificity, PPV, NPV.
- **Examples:** `pharmacology_quickwins.ddk`, `epidemiology_sir.ddk`, `biostatistics_quickwins.ddk`.

---

## Quick Wins (underserved sciences)

- **Music:** `cents_to_ratio(cents)`, `ratio_to_cents(ratio)`, `equal_temperament(n, a4_hz)` — frequency ratios, equal-tempered tuning.
- **Economics:** `discount_factor(r, t, discrete)`, `cobb_douglas(K, L, alpha, A)`, `solow_rhs(K, s, delta, n, g, alpha)` — present value, Cobb-Douglas, Solow RHS.
- **Geology:** `darcy_velocity(K, grad_P, mu)` — Darcy's law v = -(K/μ)∇P.
- **Materials:** `johnson_mehl_avrami(t, k, n)`, `avrami_rate(t, k, n)` — JMAK phase transformation.
- **Examples:** `quickwins_units.ddk`, `music_intervals.ddk`, `economics_solow.ddk`, `geology_darcy.ddk`, `materials_jmak.ddk`.

---

## Machine Learning & Tensors

- **Models:** `Sequential([Dense(...), ...])`, `Dense(n, activation="relu"|"softmax"|…)`; forward pass with `model.forward(input)`; `.fast()` for torch.compile (MLIR/Inductor).
- **Tensors:** PyTorch backend; vectors/matrices, FFT, convolution, pooling; `.gpu()`, `.cpu()`; autograd (`grad()`), `.with_grad()`.
- **Sparse:** `.sparse()` for COO/CSR; item assignment `T[i][j] = val`. **Sparse CFD:** `sparse_laplacian_2d(N)` (5-point stencil), `sparse_diffusion_step(T, L, dt, alpha)`, `sparse_diffusion_simulate(T0, n_steps, dt, alpha)` for ∂T/∂t = α∇²T; example `cfd_sparse_sim.ddk`.

---

## I/O, Network & JSON

- **Files:** `read_file(path)` (UTF-8), `write_file(path, content)`, `file_exists(path)`.
- **Network:** `http_get(url)`, `http_post(url, data)` (data as string or dict/list → JSON).
- **JSON:** `json_parse(s)` → object (access `obj["key"]`), `json_stringify(obj)` → string.

---

## Quantum Computing (v1.21)

- **Native simulator:** `quantum_circuit(n)` creates a circuit; gates: `.h`, `.x`, `.y`, `.z`, `.cx`, `.cz`, `.rx`, `.ry`, `.rz`, `.t`, `.s`, `.swap`, `.measure`; `statevec_sim(qc)` → `list[complex]`; `statevec_sim(qc, shots)` → measurement dict; `statevec_expectation(qc, "ZZ")` → float.
- **Convenience:** `bell_state(which)`, `ghz_state(n)`, `grover_circuit(n, target)` usable in one line.
- **VQE:** `vqe_circuit(n, L, params)` hardware-efficient ansatz; `vqe_energy(params, n, L, terms)` with Pauli-string Hamiltonian.
- **Quantum information:** `fidelity(sv1, sv2)`, `entropy_von_neumann(probs)`, `schmidt_rank(sv, n_a)`.
- **Physical units:** `[GHz]`, `[THz]`, `[eV]`, `[meV]`, `[us]`, `[mK]` for quantum hardware parameters; `qubit_frequency_check`, `coherence_time_check`, `energy_gap_check`.
- **Shape annotations:** `Qubit[N]`, `StateVec[N]`, `Circuit[N,G]` in function signatures with symbolic dimensions.
- **Qiskit export:** `qc.to_qiskit()` (requires `pip install qiskit`).
- **Module:** `use quantum` with wrapper functions `simulate`, `sample_circuit`, `state_fidelity`, `vn_entropy`, etc.

---

## Tooling & IDE

- **Compiler:** Transpilation of `.ddk` to Python; CLI `python -m src.compiler.compiler <file.ddk>`; options `--latex`, `--no-units-check`.
- **Editor Integration:** Text editors (like VS Code or Spyder) can run Dedekind files using the Jupyter kernel, with syntax highlighting support for physical units and Ricci indices.
- **Jupyter kernel:** Dedekind in Jupyter/Spyder consoles; persistent context across cells.
- **Examples:** Over 49 `.ddk` examples in `examples/dedekind/`; `pde_navier_stokes.ddk` for Navier-Stokes 2D; `angle_units.ddk` for angles rad/deg; `sequences.ddk` for sequences; `stats_binom_ttest.ddk` for binom and t-test. Batch test with `run_examples.py` (-q, -v, --compile, --filter). **Tests:** `assert(condition, message)`; mini test runner `run_tests.py` for `tests/dedekind/*.ddk`.
- **Plots:** `plot(x, y, title=..., xscale="linear"|"log", yscale="log")`; `scatter(x, y)`; `contour(X, Y, Z, levels=...)`.

---

## What is (still) not, or only partially, supported

- **Symbolics:** Symbolic differentiation `diff_sym`, integration `integrate_sym`, and algebraic equation solvers `solve_sym` are available (via SymPy). Native symbolic simplifications in the compiler pass are limited to basic operations (constant folding such as `2*3` → `6`, identities such as `x+0` → `x`, `x*1` → `x`, `0*x` → `0`). No symbolic ODE solvers.
- **Types/modules:** No static typing (other than dimension and shape checks). No classic Python `import` (instead `use` for hierarchical Dedekind modules with visibility control via `pub fn`, plus `pyimport` for Python libraries).
- **Performance:** Native binaries (AOT via MLIR/LLVM) and standalone executables (`.exe`) are still experimental. In normal operation the AST is transpiled to Python/PyTorch code.
- **PDEs & geometries:** 1D/2D standard PDE solvers and structured 2D FEM (Q4 bilinear elements) are available. General unstructured 3D meshes or arbitrarily complex boundary conditions are not yet implemented natively.

---

*This document summarizes the implementation status from the README, language specification, Maturity Assessment, and changelogs.*
