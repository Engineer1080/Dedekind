# Dedekind Language Reference

Living catalogue of every Dedekind built-in, operator and runtime function. Generated from the original README feature list; the canonical specification is `Documentation/Dedekind_Language_Specification.md`.

## Core features

- **Ricci Calculus**: Native index notation (`A^ij * B_jk`) for Einstein summation.
- **Sparse Tensors**: Efficient `.sparse()` support for FEM/CFD simulations.
- **Fundamental Constants**: Mathematical `pi`, `e`; physical CODATA constants: `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — all as **Quantity** with SI units.
- **Physical Units**: SI base m, kg, s, A, K, mol, **cd** (candela); literals (`10[m]`, `5[m/s]`, `1.0[kg]`, `1[cd]`); **automatische Umrechnung** bei Addition/Subtraktion für gleiche Dimension — **SI-Basis**: Länge (m, cm, km, mm, dm), Masse (kg, g, t, mg), Zeit (s, min, h, ms), Strom (A, mA, kA, uA), Temperatur (K, mK), Stoffmenge (mol, mmol, kmol), Lichtstärke (cd, mcd); **abgeleitet**: Druck (Pa, bar, atm, hPa), Volumen (L, mL, dm³, m³), Energie (J, kJ, MJ, Wh, kWh), Spannung (V, mV, kV), Frequenz (Hz, kHz, MHz, GHz), Ladung (C, mC, uC), Widerstand (ohm, kohm, Mohm), Leistung (W, kW, MW); **Winkel**: rad, deg. Ergebnis = Einheit des ersten Operanden; z. B. `1[m] + 100[cm]` → `2[m]`, `90[deg] + (pi/2)*1[rad]` → `180[deg]`. `deg_to_rad(x)`, `rad_to_deg(x)` für Konvertierung. Sonst add/sub gleiche Einheit; multiply/divide kombinieren Einheiten; `^` für Potenzen; Anzeige vereinfacht (J, N, Pa, W, Hz, …). **Chemie**: mol, L, M (= mol/L), ppm, **bar**, **atm**, **g**; **Radioaktivität**: **Bq**, **Sv**, Gy.
- **4D Rotational Math**: Native Quaternion support (`i`, `j`, `k` suffixes) and `.rotate()` method; unary minus supported (`-1.0 + 0i`).
- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` (RK4/Euler); gradients via `grad()` for physics-informed ML; `linspace(start, stop, steps)` for time grids.
- **Differentiable PDE Solvers**: `pde_heat_1d`, `pde_heat_2d` (heat); `pde_advection_1d`, `pde_advection_2d` (advection); `pde_wave_1d`, `pde_wave_2d` (wave); `pde_burgers_1d`, `pde_burgers_2d` (Burgers); `pde_reaction_diffusion_1d`, `pde_reaction_diffusion_2d`; `pde_advection_diffusion_1d`, `pde_advection_diffusion_2d`; `pde_maxwell_1d`, `pde_maxwell_2d` (Maxwell FDTD); `pde_navier_stokes_2d` (Navier-Stokes 2D incompressible, Chorin projection); finite differences + `ode_solve`; gradients through `u0` and parameters.
- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`, `Dirichlet(alpha)`; `sample(dist)`, `log_prob(dist, value)`; Bayesian inference via `metropolis(log_prior, log_likelihood, data, init, steps)`.
- **Numerical Integration**: `integrate(f, a, b, n)` — trapezoidal quadrature; differentiable when `f` accepts a tensor.
- **Math Functions**: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; `asin`, `acos`, `atan`, `atan2(y,x)`; `sinh`, `cosh`, `tanh` — element-wise, differentiable; Tensor or scalar. **Reductions**: `min(x)`, `max(x)`, `argmin(x)`, `argmax(x)` (optional `dim`). **Rounding**: `round(x)`, `floor(x)`, `ceil(x)`. **Linear algebra**: `norm(x)`, `det(A)`, `trace(A)`.
- **Uncertainty Propagation**: `uncertain(value, std)` bzw. `UncertainQuantity` — Gauß'sche Fehlerfortpflanzung für +, -, *, /, ^; optional mit Einheit.
- **Fitting / Regression**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=0.01, steps=500)` — minimiert `loss_fn(params, data)` via Gradient Descent, Metropolis-Hastings oder **HMC** (Hamiltonian Monte Carlo).
- **Paper-Code-Drift eliminieren (LaTeX aus dem AST)**: `export_to_latex(source)` bzw. CLI `--latex` — Formeln, Einheiten und Messunsicherheiten im Manuskript werden aus demselben AST generiert, der die Simulation ausführt. Damit verschwindet eine ganze Bug-Klasse, die in händisch getippten Methoden- und Tabellenabschnitten entsteht. Domain-Knoten: **Ricci-Indizes** (`A^ij * v^j` → `A^{ij}\, v^{j}` mit Einstein-Konvention, kein `\cdot`), **`partial(u, x, order=2)`** → `\frac{\partial^2 u}{\partial x^2}`, **`pde_heat_2d`/`pde_wave_1d`/`pde_navier_stokes_2d`/...** → kanonische PDE-Form, **`lagrange_ode_rhs(L)`** → Euler-Lagrange, **`hamilton_ode_rhs(H)`** → kanonische Gleichungen. **Wissenschaftliche Konsole**: `print_latex(s)` rendert LaTeX in der Dedekind-Studio-/Jupyter-Konsole.
- **Reproducibility-Report**: CLI `--reproducibility-report PATH` schreibt einen Paper-Anhang mit Git-Commit (inkl. dirty-Flag), Python-/torch-/numpy-/scipy-Versionen, CUDA-Verfügbarkeit, SHA-256 der Quelle, erkannten RNG-Seeds und der Methods-Section als LaTeX — alles aus *einer* Quelle. Adressiert die **Paper-Code-Drift**, nicht die gesamte Reproduzierbarkeitskrise (Daten-Provenienz, Pre-Registration etc. bleiben unberührt). Demo: `reproducibility_demo.ddk`.
- **Bessere Fehlermeldungen**: Compiler-Fehler mit Zeile (`CompileError`); Parser setzt `line` im AST; Runtime-Quantity-Meldungen mit Kontext.
- **Einheiten zur Compile-Zeit**: `1[m] + 1[s]` → Compiler-Fehler mit Zeile; `compile_source(..., check_units=True)` (Default), CLI `--no-units-check`.
- **Datei-I/O**: `read_file(path)` (Text UTF-8), `write_file(path, content)`, `file_exists(path)`.
- **Netzwerk**: `http_get(url)`, `http_post(url, data)` (data String oder Dict/List als JSON); Antworttext UTF-8.
- **JSON**: `json_parse(s)` → Objekt (Dict/List; Zugriff `obj["key"]`), `json_stringify(obj)` → String.
- **AOT Compilation**: Truly native binary generation via MLIR and LLVM.
- **IDE**: **Dedekind Studio** ist ein Spyder-Fork (`DedekindStudio/`) mit **nativ Python und Dedekind**; siehe [Documentation/Dedekind_Studio_Spyder_Fork.md](Documentation/Dedekind_Studio_Spyder_Fork.md). Ein **Dedekind Jupyter Kernel** (`dedekind_jupyter_kernel/`) ermöglicht Dedekind in Jupyter/Spyder-Konsolen.

## Chemistry & biology

## 🧪 Dedekind für Chemie & Biologie

Dedekind unterstützt **chemische und biologische Anwendungen** mit denselben Bausteinen wie für Physik und ML: Einheiten, ODE-Löser, Fitting und Unsicherheitsfortpflanzung.

- **Einheiten**: Konzentration in `[M]` (Molarität), Stoffmenge in `[mol]`, Volumen in `[L]`, Verdünnungen in `[ppm]`; `M` und `mol/L` werden als gleich behandelt (Runtime und Compile-Check).
- **Kinetik**: Reaktion 1. Ordnung \(c(t) = c_0 e^{-kt}\) mit `ode_solve` und Einheiten `[M]`, `[1/s]` — Beispiel: `chemistry_kinetics.ddk`.
- **Dosis-Wirkung / Michaelis-Menten**: Hill-Gleichung oder \(v = V_{\max}[S]/(K_M + [S])\); Parameterfitting mit `fit` (EC50, \(K_M\), \(V_{\max}\)) — Beispiel: `dose_response.ddk`.
- **Wachstum**: Logistisches Wachstum \(dN/dt = r N (1 - N/K)\) mit `ode_solve` — Beispiel: `biology_growth.ddk`.
- **Convenience**: `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)` — einzeilig aufrufbar.
- **Chemische Elemente**: `atomic_mass("C")` → Atommasse in g/mol (Quantity); `atomic_number("C")` → Ordnungszahl; IUPAC-nah für H, C, N, O, S, P, Na, Cl, Fe, … (ca. 50 Elemente). Beispiel: Molare Masse H₂O = 2*atomic_mass("H") + atomic_mass("O"); `chemistry_elements.ddk`.
- **Medizin, Pharmakologie & Epidemiologie**: `hill_equation`, `one_compartment_pk`, `half_life` (Pharmakokinetik); `sir_model`, `basic_reproduction_number` (Epidemiologie); `confidence_interval`, `odds_ratio`, `sensitivity_specificity` (Biostatistik) — Beispiele: `pharmacology_quickwins.ddk`, `epidemiology_sir.ddk`, `biostatistics_quickwins.ddk`.

Konstanten wie `N_A`, `R_gas`, `F_faraday` sind als **Quantity** mit SI-Einheiten (`1/mol`, `J/(K*mol)`, `C/mol`) verfügbar. Ausführliche Roadmap: `Documentation/Chemistry_Biology_Roadmap.md`.


## ML example

## 🧠 Machine Learning Example

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
    
    // Daten werden automatisch als Tensor verarbeitet
    input = [[1.0, 2.0, 3.0]].gpu()
    
    print("Prediction:")
    print(model.forward(input))
}
main()
```


## Architecture

## 🏗️ Architecture

The project consists of two main parts:

1.  **Dedekind Compiler (`src/compiler`)**
    *   Implemented in Python (Prototype Phase).
    *   Transpiles Dedekind source code (`.ddk`) into optimized high-performance Python/NumPy code (future target: MLIR/LLVM).
    *   Used by the CLI, the Dedekind Jupyter Kernel, and Dedekind Studio.

2.  **Dedekind Studio (Spyder-Fork in `DedekindStudio/`)**
    *   Full IDE with **native Python** and **native Dedekind**: Editor, Konsolen (IPython + Dedekind-Kernel), Variable Explorer, Plots.
    *   Siehe [Documentation/Dedekind_Studio_Spyder_Fork.md](Documentation/Dedekind_Studio_Spyder_Fork.md).


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
- `dirichlet_distribution_function.ddk` – Dirichlet-Verteilung und Dirichlet-Funktion D(x)
- `dedekind_cuts_rings.ddk` – Dedekind-Schnitte (Konstruktion von R aus Q) und Dedekind-Ringe (Ideal-Faktorisierung in Z)
- `riemann_zeta_sums.ddk` – Riemann-Zeta ζ(s) und Riemann-Summen (links, rechts, Mittelpunkt)
- `volume_revolution.ddk` – Rotationskörper (Kugel, Kegel, Paraboloid)
- `abs_bars.ddk` – Betragsstriche `|x|` = abs(x)  
- `integration.ddk` – numerical integration `integrate(f, a, b)` and `sin`/`cos`  
- `uncertainty_propagation.ddk` – `uncertain(value, std)`; Gauß'sche Fehlerfortpflanzung  
- `curve_fitting.ddk` – `fit(loss_fn, params_init, data)` für lineare Regression  
- `file_io_json.ddk` – Datei-I/O (`read_file`, `write_file`, `file_exists`), JSON (`json_parse`, `json_stringify`), Schlüsselzugriff `obj["key"]`  
- `linear_regression.ddk` – Quick-Win: `linear_regression(x, y)` → Steigung, Achsenabschnitt  
- `chemistry_kinetics.ddk` – Reaktion 1. Ordnung mit Einheiten [M], [1/s] und `ode_solve`  
- `chemistry_arrhenius.ddk` – Quick-Win: `arrhenius(T, A, Ea)` (Arrhenius-Gleichung)  
- `chemistry_elements.ddk` – Atommasse `atomic_mass("C")` (g/mol), Ordnungszahl `atomic_number("C")`; Molare Masse H₂O, C₂H₆  
- `dose_response.ddk` – Dosis-Wirkung (EC50/Vmax/Km) mit `michaelis_menten` und `fit`  
- `biology_growth.ddk` – logistisches Wachstum mit `logistic_growth_dt`/`logistic` und `ode_solve`  
- `pharmacology_quickwins.ddk` – Hill-Gleichung, Ein-Kompartiment-PK, Halbwertszeit  
- `epidemiology_sir.ddk` – SIR-Modell, R₀  
- `biostatistics_quickwins.ddk` – Konfidenzintervall, Odds Ratio, Sensitivität/Spezifität  
- `probabilistic.ddk` – distributions, sampling, and Bayesian inference with `metropolis`  
- `conditional_logic.ddk`, `basic_loops.ddk` – control flow  
- `mnist_classifier.ddk` – neural network with `Sequential`/`Dense`  

From the `src/` directory: `python -m compiler.compiler ../examples/dedekind/hello.ddk`

**Alle Beispiele auf einmal testen** (aus Projektroot): `python run_examples.py` — kompiliert und führt alle `.ddk`-Dateien in `examples/dedekind` aus; Optionen: `-q` (nur Zusammenfassung), `-v` (vollständige Ausgabe), `--compile` (nur kompilieren), `--filter name` (nur Dateien mit „name“ im Dateinamen).

## Further documentation

## 📚 Documentation

- **Language Specification**: `Documentation/Dedekind_Language_Specification.md` (v0.2; §15 Physical Units v0.6, §15.7 ODE v0.7, §15.8 Probabilistic v0.8, §15.9 PDE v0.8, §15.10 Integration & Math v0.9/v0.9.6; Chemie/Biologie v0.9.7; I/O/JSON v0.9.8; Stand v1.0.10). PDF can be generated with `pandoc` (see `Documentation/README.md`).
- **Research & Architecture**: `Documentation/Dedekind_Research_and_Architecture.md` (includes §10 Sprachfeatures v0.6).
- **Symbolic Simplification**: `Documentation/Symbolic_Simplification_Roadmap.md` — Implementierungs-Roadmap (Phasen, Optionen, Integration).
- **Features Roadmap**: `Documentation/Features_Implementation_Roadmap.md` — naturwissenschaftliche Features (Phase 1 abgeschlossen: Verteilungen, Integration).
- **Chemie & Biologie**: `Documentation/Chemistry_Biology_Roadmap.md` — Einheiten mol/L/M/ppm, Beispiele (Kinetik, Dosis-Wirkung, Wachstum), Convenience-Funktionen.
- Legacy PDFs (v0.1) remain in `Documentation/`; the Markdown sources are the up-to-date references.

