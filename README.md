# Fourier Programming Language

![Version](https://img.shields.io/badge/Version-0.9.7-blue) ![Fourier Studio](https://img.shields.io/badge/Status-Prototype-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**Fourier** is a modern, high-performance programming language designed specifically for compute-intensive workloads in **Machine Learning** and **Graphics Rendering**.

Unlike general-purpose languages retrofitted with parallel computing capabilities, Fourier is built from the ground up with **GPU/TPU acceleration** and **Automatic Parallelization** as core features.

---

- **Ricci Calculus**: Native index notation (`A^ij * B_jk`) for Einstein summation.
- **Sparse Tensors**: Efficient `.sparse()` support for FEM/CFD simulations.
- **Fundamental Constants**: Mathematical `pi`, `e`; physical CODATA constants: `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — all as **Quantity** with SI units.
- **Physical Units**: Literals with units (`10[m]`, `5[m/s]`, `1.0[kg]`); add/sub require same unit; multiply/divide combine units; `^` for powers (`r^2`); display simplified to J, N where applicable. **Chemie**: mol, L, M (= mol/L), ppm; `0.1[M]`, `1[mol]`, `50[ppm]`; M und mol/L gelten als gleiche Einheit.
- **4D Rotational Math**: Native Quaternion support (`i`, `j`, `k` suffixes) and `.rotate()` method; unary minus supported (`-1.0 + 0i`).
- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` (RK4/Euler); gradients via `grad()` for physics-informed ML; `linspace(start, stop, steps)` for time grids.
- **Differentiable PDE Solvers**: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` for the heat equation \(u_t = k \cdot \Delta u\); finite differences + `ode_solve`; gradients through `u0` and `k`.
- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`; `sample(dist)`, `log_prob(dist, value)`; Bayesian inference via `metropolis(log_prior, log_likelihood, data, init, steps)`.
- **Numerical Integration**: `integrate(f, a, b, n)` — trapezoidal quadrature; differentiable when `f` accepts a tensor.
- **Math Functions**: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; `asin`, `acos`, `atan`, `atan2(y,x)`; `sinh`, `cosh`, `tanh` — element-wise, differentiable; Tensor or scalar. **Reductions**: `min(x)`, `max(x)`, `argmin(x)`, `argmax(x)` (optional `dim`). **Rounding**: `round(x)`, `floor(x)`, `ceil(x)`. **Linear algebra**: `norm(x)`, `det(A)`, `trace(A)`.
- **Uncertainty Propagation**: `uncertain(value, std)` bzw. `UncertainQuantity` — Gauß'sche Fehlerfortpflanzung für +, -, *, /, ^; optional mit Einheit.
- **Fitting / Regression**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=0.01, steps=500)` — minimiert `loss_fn(params, data)` via Gradient Descent, Metropolis-Hastings oder **HMC** (Hamiltonian Monte Carlo).
- **LaTeX-Export**: `export_to_latex(source)` bzw. CLI `--latex` — Formeln aus Fourier-Code als LaTeX (für Papers/Notizen).
- **Bessere Fehlermeldungen**: Compiler-Fehler mit Zeile (`CompileError`); Parser setzt `line` im AST; Runtime-Quantity-Meldungen mit Kontext.
- **Einheiten zur Compile-Zeit**: `1[m] + 1[s]` → Compiler-Fehler mit Zeile; `compile_source(..., check_units=True)` (Default), CLI `--no-units-check`.
- **AOT Compilation**: Truly native binary generation via MLIR and LLVM.
- **Modern IDE**: "Fourier Studio" – v0.9.7 with resizable terminal and file explorer.

### What's New in v0.9.7
- **Fourier für Chemie & Biologie**: Chemische Einheiten **mol**, **L**, **M** (= mol/L), **ppm** in Runtime und Compile-Check; M und mol/L gelten als gleiche Einheit. Einheiten-Literal `[1/s]` im Parser unterstützt (z. B. `0.05[1/s]`).
- **Beispiele**: `chemistry_kinetics.fourier` (Reaktion 1. Ordnung mit `ode_solve`, [M], [1/s]), `dose_response.fourier` (Dosis-Wirkung/EC50 mit `fit`), `biology_growth.fourier` (logistisches Wachstum mit `ode_solve`).
- **Dokumentation**: Abschnitt „Fourier für Chemie & Biologie“ im README; Verweis auf `Documentation/Chemistry_Biology_Roadmap.md`.

### What's New in v0.9.6
- **Grundlegende Math-Funktionen**: Erweiterung der Standard-Bibliothek um `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; Arkusfunktionen `asin`, `acos`, `atan`, `atan2(y,x)`; Hyperbelfunktionen `sinh`, `cosh`, `tanh`. Alle elementweise, differenzierbar; LaTeX-Export angepasst.
- **Reduktionen, Runden, Lineare Algebra**: `min(x)`, `max(x)`, `argmin(x)`, `argmax(x)` (optional `dim`); `round(x)`, `floor(x)`, `ceil(x)`; `norm(x)`, `det(A)`, `trace(A)`. Beispiel: `examples/fourier/math_functions.fourier`.

### What's New in v0.9.5
- **Phase 2 — Bessere Fehlermeldungen**: AST-Knoten tragen optional `line`; Parser wirft `CompileError(message, line, filepath)` bei erwartetem Token, ungültigem Zuweisungsziel, unerwartetem Token; Runtime-Quantity-Meldungen mit klarem Kontext („Einheitenfehler bei Addition: [m] vs [s] …“).
- **Phase 3b — Einheiten zur Compile-Zeit**: `units_checker.py` prüft vor Codegen: bei `+`/`-` müssen Einheiten übereinstimmen (soweit bekannt); `1[m] + 1[s]` → Compiler-Fehler mit Zeile; Unäres Minus erlaubt; CLI `--no-units-check` zum Abschalten.

### What's New in v0.9.4
- **HMC (Hamiltonian Monte Carlo)**: `hmc(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size, num_leapfrog)` und `fit(..., method="hmc")` — gradientenbasierte MCMC-Proposals. Beispiel: `hmc_fitting.fourier`.
- **LaTeX-Export**: `export_to_latex(source_code)` im Compiler; CLI: `python -m src.compiler.compiler <file.fourier> --latex`. Formeln (Zuweisungen, Returns) werden als LaTeX ausgegeben. Beispiel: `latex_demo.fourier`.

### What's New in v0.9.3
- **Version 0.9.3**: Release mit Uncertainty Propagation (`uncertain`, `UncertainQuantity`) und Fitting (`fit`); siehe §15.11 und Beispiele `uncertainty_propagation.fourier`, `curve_fitting.fourier`.

### What's New in v0.9.2
- **Version 0.9.2**: Mathematische Konstanten `pi`, `e`; erweiterte physikalische Konstanten (CODATA): `hbar`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday`. Beispiel: `constants_extended.fourier`; `integrate(f, 0, pi)` nutzt native `pi`.
- **Uncertainty Propagation**: `uncertain(value, std)` und `UncertainQuantity` — Fehlerfortpflanzung (Gauß) für +, -, *, /, ^; optional Einheit. Beispiel: `uncertainty_propagation.fourier`.
- **Fitting**: `fit(loss_fn, params_init, data, method="gd"|"mcmc", lr=0.01, steps=500)` — Kurvenanpassung via Gradient Descent oder MCMC. Beispiel: `curve_fitting.fourier`.

### What's New in v0.9.1
- **Version 0.9.1**: Run-Examples-Skript `run_examples.py` — alle `.fourier`-Beispiele automatisch kompilieren und ausführen (Optionen: `-q`, `-v`, `--compile`, `--filter`). Dokumentation ergänzt; Linter-Hinweise in `ml_runtime.py` behoben.

### What's New in v0.9
- **Release**: Differentiable PDE Solvers (`pde_heat_1d`, `pde_heat_2d`) als stabile Erweiterung.
- **Extended Distributions**: `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`; gleiche API wie `Normal`/`Uniform` (`sample`, `log_prob`). Example: `examples/fourier/distributions_extended.fourier`.
- **Numerical Integration**: `integrate(f, a, b, n)` — Trapezregel; differenzierbar wenn `f` Tensor akzeptiert; `sin(x)`, `cos(x)` für Ausdrücke. Example: `examples/fourier/integration.fourier`.

### What's New in v0.8
- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`; `sample(dist)` / `sample(dist, n)`; `log_prob(dist, value)`; Metropolis-Hastings `metropolis(log_prior, log_likelihood, data, init, steps)` for Bayesian inference. Example: `examples/fourier/probabilistic.fourier`.
- **Differentiable PDE Solvers**: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` for the heat equation; Dirichlet BC; gradients through initial condition and diffusivity. Example: `examples/fourier/pde_heat.fourier`.

### What's New in v0.7
- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` solves dy/dt = fun(t,y) with differentiable RK4 (default) or Euler; gradients flow through `y0` and parameters in `fun`. Use with `grad()` for physics-informed ML.
- **linspace**: `linspace(start, stop, steps)` builds a 1D time grid for ODE integration. Example: `examples/fourier/differentiable_ode.fourier`.

### What's New in v0.6
- **Physical Units (Option B)**: Constants `c`, `G`, `h`, `k_B`, `k_e` are now `Quantity` values with SI units; expressions like `m * c^2` and `G * m1 * m2 / r^2` yield results with correct dimensions; output simplified to **J** (Joule) and **N** (Newton) where applicable.
- **Quantity**: Full arithmetic including `__pow__` (e.g. `c^2`, `r^2`) and `__neg__`; unary minus for literals and Quaternions fixed in codegen.
- **Quaternion**: `__neg__` support so expressions such as `-1.0 + 0i` and signal lists with negative imaginary parts work correctly (e.g. `signal_physics.fourier`).

## 🧪 Fourier für Chemie & Biologie

Fourier unterstützt **chemische und biologische Anwendungen** mit denselben Bausteinen wie für Physik und ML: Einheiten, ODE-Löser, Fitting und Unsicherheitsfortpflanzung.

- **Einheiten**: Konzentration in `[M]` (Molarität), Stoffmenge in `[mol]`, Volumen in `[L]`, Verdünnungen in `[ppm]`; `M` und `mol/L` werden als gleich behandelt (Runtime und Compile-Check).
- **Kinetik**: Reaktion 1. Ordnung \(c(t) = c_0 e^{-kt}\) mit `ode_solve` und Einheiten `[M]`, `[1/s]` — Beispiel: `chemistry_kinetics.fourier`.
- **Dosis-Wirkung / Michaelis-Menten**: Hill-Gleichung oder \(v = V_{\max}[S]/(K_M + [S])\); Parameterfitting mit `fit` (EC50, \(K_M\), \(V_{\max}\)) — Beispiel: `dose_response.fourier`.
- **Wachstum**: Logistisches Wachstum \(dN/dt = r N (1 - N/K)\) mit `ode_solve` — Beispiel: `biology_growth.fourier`.

Konstanten wie `N_A`, `R_gas`, `F_faraday` sind als **Quantity** mit SI-Einheiten (`1/mol`, `J/(K*mol)`, `C/mol`) verfügbar. Ausführliche Roadmap: `Documentation/Chemistry_Biology_Roadmap.md`.

## 🧠 Machine Learning Example

```fourier
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

```fourier
fn main() {
    model = Sequential([
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    
    // Daten werden automatisch als Tensor verarbeitet
    input = [[1.0, 2.0, 3.0]].gpu()
    
    print("Prediction:")
    print(model.forward(input))
}
main()
```

## 🏗️ Architecture

The project consists of two main components:

1.  **Fourier Compiler Backend (`src/compiler`)**
    *   Implemented in Python (Prototype Phase).
    *   Transpiles Fourier source code (`.fourier`) into optimized high-performance Python/NumPy code (future target: MLIR/LLVM).
    *   Exposes a REST API (`src/server.py`) for the IDE.

2.  **Fourier Studio (`src/ui`)**
    *   A modern, dark-themed Integrated Development Environment.
    *   Built with **React**, **Vite**, and **Vanilla CSS**.
    *   Features a code editor, real-time console output, and syntax highlighting.

## 🛠️ Installation & Setup

### Prerequisites
*   **Python 3.10+**
*   **Node.js 18+** & `npm`

### 1. Clone the Repository
```bash
git clone https://github.com/Engineer1080/FourierLanguage.git
cd FourierLanguage
```

### 2. Setup Compiler Backend
The backend requires Python, Flask, and PyTorch (for tensor/ML runtime).

```bash
# Install dependencies (ein Befehl)
pip install -r requirements.txt

# Run the Backend Server
python src/server.py
```
*The server will start on `http://localhost:5000`.*

**Abhängigkeiten im Überblick:** `flask`, `flask-cors` (API), `torch` (PyTorch für Tensoren, FFT, ML), `matplotlib` (für `plot()`-Visualisierung).

### 3. Setup Fourier Studio (Native App)
Open a new terminal window for the frontend.

```bash
cd src/ui

# Install Node dependencies
npm install

# Run the Application (Electron + Vite)
npm run electron:dev
```
*This will launch the native Fourier Studio window.*

## 💻 Usage

1.  Open **Fourier Studio** in your browser.
2.  Write your Fourier code in the editor.
    ```fourier
    fn main() {
        print("Hello, Fourier!")
        
        // Automatic matrix operations
        data = [1.0, 2.0, 3.0, 4.0]
        result = data * 2.0
        
        print("Result:")
        print(result)
    }
    main()
    ```
3.  Click the **▶ Run Code** button.
4.  View the compilation result and program output in the terminal panel.

### Examples
Example programs are in `examples/fourier/`, including:
- `hello.fourier` – basic I/O and tensors  
- `universal_constants.fourier` – physical constants and units (E = mc², gravitation, Coulomb)  
- `constants_extended.fourier` – mathematical `pi`, `e`; CODATA constants (`hbar`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday`)  
- `signal_physics.fourier` – complex numbers (Quaternions) and FFT  
- `differentiable_ode.fourier` – differentiable ODE solver with `ode_solve` and `grad`  
- `pde_heat.fourier` – differentiable PDE solver (1D/2D heat equation) with `pde_heat_1d` / `pde_heat_2d`  
- `distributions_extended.fourier` – Exponential, Gamma, Beta, Poisson; `sample`, `log_prob`  
- `integration.fourier` – numerical integration `integrate(f, a, b)` and `sin`/`cos`  
- `uncertainty_propagation.fourier` – `uncertain(value, std)`; Gauß'sche Fehlerfortpflanzung  
- `curve_fitting.fourier` – `fit(loss_fn, params_init, data)` für lineare Regression  
- `chemistry_kinetics.fourier` – Reaktion 1. Ordnung mit Einheiten [M], [1/s] und `ode_solve`  
- `dose_response.fourier` – Dosis-Wirkung (EC50/Vmax/Km) mit `fit`  
- `biology_growth.fourier` – logistisches Wachstum mit `ode_solve`  
- `probabilistic.fourier` – distributions, sampling, and Bayesian inference with `metropolis`  
- `conditional_logic.fourier`, `basic_loops.fourier` – control flow  
- `mnist_classifier.fourier` – neural network with `Sequential`/`Dense`  

From the `src/` directory: `python -m compiler.compiler ../examples/fourier/hello.fourier`

**Alle Beispiele auf einmal testen** (aus Projektroot): `python run_examples.py` — kompiliert und führt alle `.fourier`-Dateien in `examples/fourier` aus; Optionen: `-q` (nur Zusammenfassung), `-v` (vollständige Ausgabe), `--compile` (nur kompilieren), `--filter name` (nur Dateien mit „name“ im Dateinamen).

## 🗺️ Roadmap

### Phase 1: Foundation ✅
*   [x] Language Specification & Design
*   [x] Proof-of-Concept Compiler (Python Backend)
*   [x] Fourier Studio IDE (React Frontend)

### Phase 2: Core Development ✅
*   [x] Build-in Core Algorithms (FFT, Conv, Linalg)
*   [x] Robust Lexer & Parser (Windows Support, Unary Ops)
*   [x] Resizable Studio Terminal & Tabs

### Phase 3: Hardware Acceleration ✅
*   [x] Integration with **PyTorch** for GPU execution.
*   [x] Implementation of `.gpu()` and `.cpu()` modifiers.

### Phase 4: Production (v0.2) ✅
*   [x] **Native Performance**: Integration with MLIR/Inductor via `.fast()`.
*   [x] **MLIR Prototype**: Fourier-Dialect IR generation.
*   [x] **Studio Upgrade**: Resizable terminal and UI polish.

### Phase 5: Advanced Mathematics ✅
*   [x] **Autograd**: Native `grad()` operator for automatic differentiation.
*   [x] **Property Access**: Native `.shape` support for tensors and models.

### Phase 6: Tensor Contraction & Logic (v0.3) ✅
*   [x] **Einsum**: High-level elective tensor contraction syntax.
*   [x] **Complex/Quaternion**: Built-in support for rotational math.

### Phase 8: AOT Compilation & LLVM Backend ✅
*   [x] **Static Binary**: Standalone `.exe` generation without Python.
*   [x] **MLIR Pipeline**: Fourier -> MLIR -> LLVM -> Binary.
*   [x] **Verification**: Native binary execution on Windows.

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
*   [x] **ode_solve**: Differenzierbarer ODE-Löser dy/dt = fun(t,y) mit RK4 (default) und Euler; Gradients durch y0 und Parameter in fun.
*   [x] **linspace**: Zeitgitter für ODE-Integration; Integration mit `grad()` für Physics-Informed ML.

### Phase 14: Probabilistic Programming (v0.8) ✅
*   [x] **Distributions**: `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)` (torch.distributions).
*   [x] **sample** / **log_prob**: Ziehen und Log-Dichte für Bayesian Inference.
*   [x] **metropolis**: Metropolis-Hastings MCMC für Posterior-Sampling (log_prior, log_likelihood, data, init, steps).

### Phase 15: Differentiable PDE Solvers (v0.8) ✅
*   [x] **pde_heat_1d**: Differenzierbarer 1D-Heat-Solver \(u_t = k\,u_{xx}\); Finite-Differenzen + `ode_solve`; Dirichlet-Randbedingungen.
*   [x] **pde_heat_2d**: Differenzierbarer 2D-Heat-Solver \(u_t = k\,(u_{xx}+u_{yy})\); Gradients durch `u0` und `k`.

## 🔭 Beyond v1.0: Future Vision

Fourier aims to become the "Standard Language for Nature's Laws." To achieve this, we are researching the native implementation of the following concepts:

1. **Differentiable ODE Solvers**: Implemented in v0.7: `ode_solve(fun, y0, t, method="rk4")` solves dy/dt = fun(t,y) with differentiable RK4 (or Euler); gradients flow through y0 and parameters in `fun`. Use with `grad()` for physics-informed ML. See `examples/fourier/differentiable_ode.fourier`.
2. **Differentiable PDE Solvers**: Implemented in v0.8: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` solve the heat equation with finite differences and `ode_solve`; gradients through initial condition and diffusivity for inverse problems. See `examples/fourier/pde_heat.fourier`.
3. **Physical Units**: Implemented at runtime: `10[m] / 2[s]` → `5[m/s]`, add/sub require same unit; future: compile-time unit checking.
4. **Probabilistic Programming**: Implemented in v0.8: `Normal`, `Uniform`, `Bernoulli`; `sample(dist)`, `log_prob(dist, value)`; `metropolis(log_prior, log_likelihood, data, init, steps)` for Bayesian inference. See `examples/fourier/probabilistic.fourier`. Future: more distributions, NUTS/VI, conditioning syntax.
5. **Symbolic Simplification**: A compile-time algebraic engine that simplifies complex mathematical expressions before code generation to maximize efficiency.

## 📚 Documentation

- **Language Specification**: `Documentation/Fourier_Language_Specification.md` (v0.2; §15 Physical Units v0.6, §15.7 ODE v0.7, §15.8 Probabilistic v0.8, §15.9 PDE v0.8, §15.10 Integration & Math v0.9/v0.9.6; Chemie/Biologie v0.9.7; Stand v0.9.7). PDF can be generated with `pandoc` (see `Documentation/README.md`).
- **Research & Architecture**: `Documentation/Fourier_Research_and_Architecture.md` (includes §10 Sprachfeatures v0.6).
- **Symbolic Simplification**: `Documentation/Symbolic_Simplification_Roadmap.md` — Implementierungs-Roadmap (Phasen, Optionen, Integration).
- **Features Roadmap**: `Documentation/Features_Implementation_Roadmap.md` — naturwissenschaftliche Features (Phase 1 abgeschlossen: Verteilungen, Integration).
- **Chemie & Biologie**: `Documentation/Chemistry_Biology_Roadmap.md` — Einheiten mol/L/M/ppm, Beispiele (Kinetik, Dosis-Wirkung, Wachstum), Convenience-Funktionen.
- Legacy PDFs (v0.1) remain in `Documentation/`; the Markdown sources are the up-to-date references.

## 📄 License

This project is licensed under the MIT License.
