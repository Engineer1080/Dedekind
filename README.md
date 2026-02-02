# Dedekind Programming Language

![Version](https://img.shields.io/badge/Version-1.1.9-blue) ![Dedekind Studio](https://img.shields.io/badge/Status-Prototype-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**Dedekind** is a modern, high-performance programming language designed specifically for compute-intensive workloads in **Machine Learning** and **Graphics Rendering**.

Unlike general-purpose languages retrofitted with parallel computing capabilities, Dedekind is built from the ground up with **GPU/TPU acceleration** and **Automatic Parallelization** as core features.

---

- **Ricci Calculus**: Native index notation (`A^ij * B_jk`) for Einstein summation.
- **Sparse Tensors**: Efficient `.sparse()` support for FEM/CFD simulations.
- **Fundamental Constants**: Mathematical `pi`, `e`; physical CODATA constants: `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — all as **Quantity** with SI units.
- **Physical Units**: SI base m, kg, s, A, K, mol, **cd** (candela); literals (`10[m]`, `5[m/s]`, `1.0[kg]`, `1[cd]`); **automatische Umrechnung** bei Addition/Subtraktion für gleiche Dimension — **SI-Basis**: Länge (m, cm, km, mm, dm), Masse (kg, g, t, mg), Zeit (s, min, h, ms), Strom (A, mA, kA, uA), Temperatur (K, mK), Stoffmenge (mol, mmol, kmol), Lichtstärke (cd, mcd); **abgeleitet**: Druck (Pa, bar, atm), Volumen (L, mL, dm³, m³), Energie (J, kJ, MJ, Wh, kWh), Spannung (V, mV, kV), Frequenz (Hz, kHz, MHz, GHz), Ladung (C, mC, uC), Widerstand (ohm, kohm, Mohm), Leistung (W, kW, MW). Ergebnis = Einheit des ersten Operanden; z. B. `1[m] + 100[cm]` → `2[m]`, `1[kJ] + 500[J]` → `1.5[kJ]`. Sonst add/sub gleiche Einheit; multiply/divide kombinieren Einheiten; `^` für Potenzen; Anzeige vereinfacht (J, N, Pa, W, Hz, …). **Chemie**: mol, L, M (= mol/L), ppm, **bar**, **atm**, **g**; **Radioaktivität**: **Bq**, **Sv**, Gy.
- **4D Rotational Math**: Native Quaternion support (`i`, `j`, `k` suffixes) and `.rotate()` method; unary minus supported (`-1.0 + 0i`).
- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` (RK4/Euler); gradients via `grad()` for physics-informed ML; `linspace(start, stop, steps)` for time grids.
- **Differentiable PDE Solvers**: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` for the heat equation \(u_t = k \cdot \Delta u\); finite differences + `ode_solve`; gradients through `u0` and `k`.
- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`; `sample(dist)`, `log_prob(dist, value)`; Bayesian inference via `metropolis(log_prior, log_likelihood, data, init, steps)`.
- **Numerical Integration**: `integrate(f, a, b, n)` — trapezoidal quadrature; differentiable when `f` accepts a tensor.
- **Math Functions**: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; `asin`, `acos`, `atan`, `atan2(y,x)`; `sinh`, `cosh`, `tanh` — element-wise, differentiable; Tensor or scalar. **Reductions**: `min(x)`, `max(x)`, `argmin(x)`, `argmax(x)` (optional `dim`). **Rounding**: `round(x)`, `floor(x)`, `ceil(x)`. **Linear algebra**: `norm(x)`, `det(A)`, `trace(A)`.
- **Uncertainty Propagation**: `uncertain(value, std)` bzw. `UncertainQuantity` — Gauß'sche Fehlerfortpflanzung für +, -, *, /, ^; optional mit Einheit.
- **Fitting / Regression**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=0.01, steps=500)` — minimiert `loss_fn(params, data)` via Gradient Descent, Metropolis-Hastings oder **HMC** (Hamiltonian Monte Carlo).
- **LaTeX-Export**: `export_to_latex(source)` bzw. CLI `--latex` — Formeln aus Dedekind-Code als LaTeX (für Papers/Notizen). **Wissenschaftliche Konsole**: `print_latex(s)` rendert LaTeX in der Dedekind-Studio-/Jupyter-Konsole (z. B. Formeln, griechische Buchstaben).
- **Bessere Fehlermeldungen**: Compiler-Fehler mit Zeile (`CompileError`); Parser setzt `line` im AST; Runtime-Quantity-Meldungen mit Kontext.
- **Einheiten zur Compile-Zeit**: `1[m] + 1[s]` → Compiler-Fehler mit Zeile; `compile_source(..., check_units=True)` (Default), CLI `--no-units-check`.
- **Datei-I/O**: `read_file(path)` (Text UTF-8), `write_file(path, content)`, `file_exists(path)`.
- **Netzwerk**: `http_get(url)`, `http_post(url, data)` (data String oder Dict/List als JSON); Antworttext UTF-8.
- **JSON**: `json_parse(s)` → Objekt (Dict/List; Zugriff `obj["key"]`), `json_stringify(obj)` → String.
- **AOT Compilation**: Truly native binary generation via MLIR and LLVM.
- **IDE**: **Dedekind Studio** ist ein Spyder-Fork (`DedekindStudio/`) mit **nativ Python und Dedekind**; siehe [Documentation/Dedekind_Studio_Spyder_Fork.md](Documentation/Dedekind_Studio_Spyder_Fork.md). Ein **Dedekind Jupyter Kernel** (`dedekind_jupyter_kernel/`) ermöglicht Dedekind in Jupyter/Spyder-Konsolen.

### What's New in v1.1.9
- **Patch:** `# type: ignore[reportMissingImports]` für numpy-Import in `balance_equation` (basedpyright).

### What's New in v1.1.8
- **Differentialgeometrie:** `christoffel_symbols(g_func, x, h)`, `riemann_tensor(g_func, x, h)`, `covariant_derivative(T, g_func, x, h)` – Christoffel-Symbole, Riemann-Tensor, kovariante Ableitung (numerisch).
- **Zahlentheorie:** `gcd(a, b)`, `is_prime(n)`, `mod(a, m)`, `mod_inv(a, m)`, `mod_pow(base, exp, m)` – ggT, Primzahltest, modulare Arithmetik.
- **Weitere Einheiten:** pH-Funktionen `concentration_to_pH(c_M)`, `pH_to_concentration(pH)`; Massenkonzentration `[percent_wv]` (= g/100mL).
- **Stöchiometrie:** `balance_equation(reactants_str, products_str)` – Reaktionsgleichungen ausbalancieren (z. B. H₂ + O₂ → H₂O).

### What's New in v1.1.7
- **Matrix-Operator @**: `A @ B` statt `matmul(A, B)` – gleiche Priorität wie * und /.
- **Spezialfunktionen**: `bessel_j0(x)`, `bessel_j1(x)` (Bessel J₀, J₁); `bessel_j(n, x)` (Bessel Jₙ); `legendre(n, x)` (Legendre Pₙ); `hypergeom(a, b, c, z)` (₂F₁). `bessel_j`, `legendre`, `hypergeom` erfordern scipy.

### What's New in v1.1.6
- **Symbolische Ableitung**: `diff_sym(expr, var)` – Ausdruck und Variable als Strings, Ableitung als String. Ohne externe Imports (nativ). Unterstützt: +, -, *, /, ^, sin, cos, tan, exp, log, sqrt.

### What's New in v1.1.5
- **Assert & Tests**: `assert(condition, message)`; Mini-Test-Runner `run_tests.py` für `tests/dedekind/*.ddk`.
- **Plots**: `scatter(x, y)`, `contour(X, Y, Z, levels)`; `plot(..., xscale="log", yscale="log")`.
- **Autograd**: `jacobian(f, x)`, `hessian(f, x)`.
- **Signal & Reduktionen**: `fftfreq(n, d)`, `diff(x, n, dim)`, `cumsum(x, dim)`, `clip(x, min_val, max_val)`, `shuffle(x, dim)`.

### What's New in v1.1.4
- **Statistik**: `cov(x, y)`, `corrcoef(x, y)`, `skew(x)`, `kurtosis(x)`, `histogram(x, bins, range_lim)`.
- **Algorithmen**: `qr(A)`, `cholesky(A)`; `polyfit(x, y, deg)`, `polyval(p, x)`; `unique(x)`, `argsort(x)`; `convolve1d(a, v, mode)`; `minimize_scalar(f, bounds)`, `newton(f, x0)`.

### What's New in v1.1.3
- **Numerik**: Neue Built-ins: `cond(A)`, `rank(A)`, `pinv(A)` (Kondition, Rang, Pseudo-Inverse); `expm(A)`, `logm(A)` (Matrix-Exponential/-Logarithmus); `interp(x, xp, fp)` (1D-lineare Interpolation); `trapz(y, x)` (Trapez-Integration für diskrete Daten); `root_bisect(f, a, b, tol)` (Nullstelle per Bisektion). Dokumentation und Beispiel `numerics_statistics.ddk` erweitert.

### What's New in v1.1.2
- **Einheiten-Anzeige**: Gleiche Faktoren werden in der Ausgabe zusammengefasst: `m*m` → `m^2`, `m*m*m` → `m^3`, `m^2*m` → `m^3`, `m*m*kg` → `m^2*kg`. Literale wie `1[m^3]`, `1[m^2]` sind nutzbar; `m^3` bei Volumen-Umrechnung (z. B. `1[m^3] + 500[L]` → `1.5[m^3]`).

### What's New in v1.1.1
- **Automatische Einheiten-Umrechnung**: Bei Addition und Subtraktion werden kompatible Einheiten derselben Dimension automatisch umgerechnet. **SI-Basis**: Länge (m, cm, km, mm, dm), Masse (kg, g, t, mg), Zeit (s, min, h, ms), Strom (A, mA, kA, uA), Temperatur (K, mK), Stoffmenge (mol, mmol, kmol), Lichtstärke (cd, mcd). **Abgeleitet**: Druck (Pa, bar, atm), Volumen (L, mL, dm³, m³), Energie (J, kJ, MJ, Wh, kWh), Spannung (V, mV, kV), Frequenz (Hz, kHz, MHz, GHz), Ladung (C, mC, uC), Widerstand (ohm, kohm, Mohm), Leistung (W, kW, MW). Ergebnis-Einheit = Einheit des ersten Operanden. Gilt für `Quantity` und `UncertainQuantity`. Compile-Zeit-Check erlaubt gleiche Dimension; inkompatible Einheiten → CompileError. Beispiel: `examples/dedekind/length_units_conversion.ddk`.

### What's New in v1.1.0
- **Konsole**: Reihenfolge von `print_latex()` und `print()` korrigiert – beide schreiben in denselben stdout-Puffer, Ausgabe erscheint in der richtigen Reihenfolge.
- **LaTeX**: `\texttt{...}` in der Unicode-Konvertierung unterstützt; Methodennamen wie `.sparse()` in cfd_sparse_sim.ddk semantisch als Code gekennzeichnet.
- **Beispiele**: Führendes `\n` in allen `print("\n...")` entfernt – keine leeren Zeilen mehr vor Meldungen wie „Uncertainty propagation aktiv.“

### What's New in v1.0.10
- **Wissenschaftliche Konsole**: `print_latex(s)` rendert LaTeX in der Dedekind-Studio-/Jupyter-Konsole (QtConsole).
- **Dedekind Studio Branding**: Neues Taskleisten- und Fenster-Icon (`dedekind_app_icon.svg`, „D“ in Dedekind-Grün); utils, mainwindow, restart und Editor-Fenster nutzen es.
- **Beispiel**: `chemistry_units_radiation.ddk` Ausgabe auf ASCII umgestellt (Radioaktivitaet, Aktivitaet) für konsistente Darstellung.

### What's New in v1.0.9
- **Einheiten**: Chemische Einheiten **bar**, **atm**, **g** und Radioaktivität **Bq** (Becquerel), **Sv** (Sievert); Beispiel `chemistry_units_radiation.ddk`; Language Spec und Chemistry_Biology_Roadmap ergänzt.
- **SI-Einheiten**: **Candela (cd)** und viele Vereinfachungen (Pa, W, Hz, V, F, ohm, S, Wb, T, H, lm, lx, Gy, kat, M).
- **Dedekind Studio**: Beim Start werden **alle** Beispiele aus `assets/dedekind_examples` als Tabs geladen; `chemistry_units_radiation.ddk` in Assets aufgenommen.

### What's New in v1.0.8
- **Release 1.0.8**: Versionserhöhung und Veröffentlichung der Änderungen.

### What's New in v1.0.7
- **Dedekind Studio Fenstertitel**: Python-Version aus der Fensterüberschrift entfernt – Titel ist nun „Dedekind Studio &lt;Version&gt;“.
- **Dedekind Studio Splash-Screen**: Neuer Untertitel „Scientific Dedekind Development Environment“ und „by Mario Michael Heinrich“ (ohne Python-Erwähnung).
- **Dedekind Studio Oberfläche**: Immer grünes Theme in Dedekind Studio; Variable Explorer und leere Plot-Fläche nutzen die grüne Palettenfarbe; EmptyMessageWidget-Hintergrund explizit gesetzt.
- **Beispiele & Restore**: Beispiele nur aus Assets laden (Fallback entfernt); beim Wiederherstellen der Sitzung nur existierende Dateien öffnen, damit wissenschaftliche Beispiele nach Entfernen alter Hello-World-Dateien erscheinen.
- **FFT-Beispiel & Plot**: `scientific_fft_spectrum.ddk` auf ASCII-Kommentare umgestellt (Encoding); Runtime: komplexe Plot-Werte werden vor dem Zeichnen in reell umgewandelt (UserWarning behoben).
- **Basedpyright**: `# type: ignore[reportMissingImports]` für Laufzeitabhängigkeiten in `spyder/__init__.py` (packaging, qtpy, spyder_kernels).

### What's New in v1.0.6
- **Wissenschaftliche Plot-Beispiele**: Sechs neue Beispiele (`scientific_wave_superposition.ddk`, `scientific_damped_oscillator.ddk`, `scientific_arrhenius_plot.ddk`, `scientific_gravitational_potential.ddk`, `scientific_ricci_plot.ddk`, `scientific_fft_spectrum.ddk`) mit Plots für Wissenschaftler – nutzen Dedekind-Features wie `pi`, `sin`/`cos`/`exp`, Einheiten, Ricci-Notation, `fft()`, `arrhenius()`, `plot()`.
- **Dedekind Studio Start**: Beim Start (ohne gespeicherte Sitzung) werden die **wissenschaftlichen Plot-Beispiele** als Tabs geladen; die bisherigen Hello-World-Dateien (`welcome_dedekind.ddk`, `hello.ddk` in den Assets) wurden entfernt.
- **Dedekind Studio Fokus**: Pylint, Profiler und Debugger (Python-spezifisch) sind als **deprecated** gekennzeichnet und werden in Dedekind Studio **nicht mehr geladen** – sie erscheinen weder in den Layouts noch unter View > Panes; Layout-Wiederherstellung blendet sie nicht ein.

### What's New in v1.0.5
- **Dedekind Studio**: `plot()` zeigt Abbildungen in der **Plots-Pane** (oben rechts); Kernel sendet display_data. Warnung „Unknown message type: comm_open“ behoben; Hinweis „Figures are displayed…“ in Dedekind Studio unterdrückt.

### What's New in v1.0.4
- **Dedekind Studio (.ddk)**: Syntax-Highlighting für **Einheiten** (z. B. `10[m]`, `[kg]`) und **Ricci-Indizes** (`A^ij`, `B_jk`) – eigene Farben (grün/teal).

### What's New in v1.0.3
- **Compiler**: ML-Runtime wird eingebunden, wenn Programme Runtime-Built-ins nutzen (z. B. `integrate`, `sin`, `arrhenius`, `uncertain`) – alle 36 Beispiele laufen.
- **Dedekind Studio**: PyTorch-Backend wird beim Start geprüft und bei Bedarf aus Projektroot installiert; Spyder-Update-Benachrichtigungen deaktiviert (Fork).

### What's New in v1.0.2
- **Umbenennung**: Fourier → Dedekind (Sprache, IDE, Kernel, Dateiendung `.ddk`).

### What's New in v1.0.1
- **Patch**: Bugfixes und kleine Verbesserungen (Dedekind Studio / Kernel).

### What's New in v1.0.0
- **Release**: Erste stabile Version 1.0. **Dedekind Studio** (Spyder-Fork) und **Dedekind Jupyter Kernel**; Sprache und Tooling als 1.0. (Die Umbenennung Fourier → Dedekind war bereits v0.9.9.)

### What's New in v0.9.8
- **Convenience (Chemie/Biologie)**: `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)` — einzeilig aufrufbar; Beispiele `dose_response.ddk`, `biology_growth.ddk`, `chemistry_arrhenius.ddk`, `linear_regression.ddk`.
- **Chemische Elemente**: `atomic_mass("C")` (g/mol), `atomic_number("C")`; ca. 50 Elemente (IUPAC-nah); Molare Masse z. B. 2*atomic_mass("H")+atomic_mass("O"); Beispiel `chemistry_elements.ddk`.
- **Datei-I/O, Netzwerk, JSON**: `read_file(path)`, `write_file(path, content)`, `file_exists(path)`; `http_get(url)`, `http_post(url, data)`; `json_parse(s)` → Objekt (Zugriff `obj["key"]`), `json_stringify(obj)`; Beispiel `file_io_json.ddk`.
- **Dokumentation**: Maturity_Assessment (Mathematik, Physik, Informatik, Biologie, Chemie); Chemistry_Biology_Roadmap Phase 2 (Convenience, Elemente) abgeschlossen.

### What's New in v0.9.7
- **Dedekind für Chemie & Biologie**: Chemische Einheiten **mol**, **L**, **M** (= mol/L), **ppm** in Runtime und Compile-Check; M und mol/L gelten als gleiche Einheit. Einheiten-Literal `[1/s]` im Parser unterstützt (z. B. `0.05[1/s]`).
- **Beispiele**: `chemistry_kinetics.ddk` (Reaktion 1. Ordnung mit `ode_solve`, [M], [1/s]), `dose_response.ddk` (Dosis-Wirkung/EC50 mit `fit`), `biology_growth.ddk` (logistisches Wachstum mit `ode_solve`).
- **Dokumentation**: Abschnitt „Dedekind für Chemie & Biologie“ im README; Verweis auf `Documentation/Chemistry_Biology_Roadmap.md`.

### What's New in v0.9.6
- **Grundlegende Math-Funktionen**: Erweiterung der Standard-Bibliothek um `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; Arkusfunktionen `asin`, `acos`, `atan`, `atan2(y,x)`; Hyperbelfunktionen `sinh`, `cosh`, `tanh`. Alle elementweise, differenzierbar; LaTeX-Export angepasst.
- **Reduktionen, Runden, Lineare Algebra**: `min(x)`, `max(x)`, `argmin(x)`, `argmax(x)` (optional `dim`); `round(x)`, `floor(x)`, `ceil(x)`; `norm(x)`, `det(A)`, `trace(A)`. Beispiel: `examples/dedekind/math_functions.ddk`.

### What's New in v0.9.5
- **Phase 2 — Bessere Fehlermeldungen**: AST-Knoten tragen optional `line`; Parser wirft `CompileError(message, line, filepath)` bei erwartetem Token, ungültigem Zuweisungsziel, unerwartetem Token; Runtime-Quantity-Meldungen mit klarem Kontext („Einheitenfehler bei Addition: [m] vs [s] …“).
- **Phase 3b — Einheiten zur Compile-Zeit**: `units_checker.py` prüft vor Codegen: bei `+`/`-` müssen Einheiten übereinstimmen (soweit bekannt); `1[m] + 1[s]` → Compiler-Fehler mit Zeile; Unäres Minus erlaubt; CLI `--no-units-check` zum Abschalten.

### What's New in v0.9.4
- **HMC (Hamiltonian Monte Carlo)**: `hmc(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size, num_leapfrog)` und `fit(..., method="hmc")` — gradientenbasierte MCMC-Proposals. Beispiel: `hmc_fitting.ddk`.
- **LaTeX-Export**: `export_to_latex(source_code)` im Compiler; CLI: `python -m src.compiler.compiler <file.ddk> --latex`. Formeln (Zuweisungen, Returns) werden als LaTeX ausgegeben. Beispiel: `latex_demo.ddk`.

### What's New in v0.9.3
- **Version 0.9.3**: Release mit Uncertainty Propagation (`uncertain`, `UncertainQuantity`) und Fitting (`fit`); siehe §15.11 und Beispiele `uncertainty_propagation.ddk`, `curve_fitting.ddk`.

### What's New in v0.9.2
- **Version 0.9.2**: Mathematische Konstanten `pi`, `e`; erweiterte physikalische Konstanten (CODATA): `hbar`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday`. Beispiel: `constants_extended.ddk`; `integrate(f, 0, pi)` nutzt native `pi`.
- **Uncertainty Propagation**: `uncertain(value, std)` und `UncertainQuantity` — Fehlerfortpflanzung (Gauß) für +, -, *, /, ^; optional Einheit. Beispiel: `uncertainty_propagation.ddk`.
- **Fitting**: `fit(loss_fn, params_init, data, method="gd"|"mcmc", lr=0.01, steps=500)` — Kurvenanpassung via Gradient Descent oder MCMC. Beispiel: `curve_fitting.ddk`.

### What's New in v0.9.1
- **Version 0.9.1**: Run-Examples-Skript `run_examples.py` — alle `.ddk`-Beispiele automatisch kompilieren und ausführen (Optionen: `-q`, `-v`, `--compile`, `--filter`). Dokumentation ergänzt; Linter-Hinweise in `ml_runtime.py` behoben.

### What's New in v0.9
- **Release**: Differentiable PDE Solvers (`pde_heat_1d`, `pde_heat_2d`) als stabile Erweiterung.
- **Extended Distributions**: `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`; gleiche API wie `Normal`/`Uniform` (`sample`, `log_prob`). Example: `examples/dedekind/distributions_extended.ddk`.
- **Numerical Integration**: `integrate(f, a, b, n)` — Trapezregel; differenzierbar wenn `f` Tensor akzeptiert; `sin(x)`, `cos(x)` für Ausdrücke. Example: `examples/dedekind/integration.ddk`.

### What's New in v0.8
- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`; `sample(dist)` / `sample(dist, n)`; `log_prob(dist, value)`; Metropolis-Hastings `metropolis(log_prior, log_likelihood, data, init, steps)` for Bayesian inference. Example: `examples/dedekind/probabilistic.ddk`.
- **Differentiable PDE Solvers**: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` for the heat equation; Dirichlet BC; gradients through initial condition and diffusivity. Example: `examples/dedekind/pde_heat.ddk`.

### What's New in v0.7
- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` solves dy/dt = fun(t,y) with differentiable RK4 (default) or Euler; gradients flow through `y0` and parameters in `fun`. Use with `grad()` for physics-informed ML.
- **linspace**: `linspace(start, stop, steps)` builds a 1D time grid for ODE integration. Example: `examples/dedekind/differentiable_ode.ddk`.

### What's New in v0.6
- **Physical Units (Option B)**: Constants `c`, `G`, `h`, `k_B`, `k_e` are now `Quantity` values with SI units; expressions like `m * c^2` and `G * m1 * m2 / r^2` yield results with correct dimensions; output simplified to **J** (Joule) and **N** (Newton) where applicable.
- **Quantity**: Full arithmetic including `__pow__` (e.g. `c^2`, `r^2`) and `__neg__`; unary minus for literals and Quaternions fixed in codegen.
- **Quaternion**: `__neg__` support so expressions such as `-1.0 + 0i` and signal lists with negative imaginary parts work correctly (e.g. `signal_physics.ddk`).

## 🧪 Dedekind für Chemie & Biologie

Dedekind unterstützt **chemische und biologische Anwendungen** mit denselben Bausteinen wie für Physik und ML: Einheiten, ODE-Löser, Fitting und Unsicherheitsfortpflanzung.

- **Einheiten**: Konzentration in `[M]` (Molarität), Stoffmenge in `[mol]`, Volumen in `[L]`, Verdünnungen in `[ppm]`; `M` und `mol/L` werden als gleich behandelt (Runtime und Compile-Check).
- **Kinetik**: Reaktion 1. Ordnung \(c(t) = c_0 e^{-kt}\) mit `ode_solve` und Einheiten `[M]`, `[1/s]` — Beispiel: `chemistry_kinetics.ddk`.
- **Dosis-Wirkung / Michaelis-Menten**: Hill-Gleichung oder \(v = V_{\max}[S]/(K_M + [S])\); Parameterfitting mit `fit` (EC50, \(K_M\), \(V_{\max}\)) — Beispiel: `dose_response.ddk`.
- **Wachstum**: Logistisches Wachstum \(dN/dt = r N (1 - N/K)\) mit `ode_solve` — Beispiel: `biology_growth.ddk`.
- **Convenience**: `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)` — einzeilig aufrufbar.
- **Chemische Elemente**: `atomic_mass("C")` → Atommasse in g/mol (Quantity); `atomic_number("C")` → Ordnungszahl; IUPAC-nah für H, C, N, O, S, P, Na, Cl, Fe, … (ca. 50 Elemente). Beispiel: Molare Masse H₂O = 2*atomic_mass("H") + atomic_mass("O"); `chemistry_elements.ddk`.

Konstanten wie `N_A`, `R_gas`, `F_faraday` sind als **Quantity** mit SI-Einheiten (`1/mol`, `J/(K*mol)`, `C/mol`) verfügbar. Ausführliche Roadmap: `Documentation/Chemistry_Biology_Roadmap.md`.

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

## 🏗️ Architecture

The project consists of two main parts:

1.  **Dedekind Compiler (`src/compiler`)**
    *   Implemented in Python (Prototype Phase).
    *   Transpiles Dedekind source code (`.ddk`) into optimized high-performance Python/NumPy code (future target: MLIR/LLVM).
    *   Used by the CLI, the Dedekind Jupyter Kernel, and Dedekind Studio.

2.  **Dedekind Studio (Spyder-Fork in `DedekindStudio/`)**
    *   Full IDE with **native Python** and **native Dedekind**: Editor, Konsolen (IPython + Dedekind-Kernel), Variable Explorer, Plots.
    *   Siehe [Documentation/Dedekind_Studio_Spyder_Fork.md](Documentation/Dedekind_Studio_Spyder_Fork.md).

## 🛠️ Installation & Setup

### Prerequisites
*   **Python 3.10+**

### 1. Clone the Repository
```bash
git clone https://github.com/Engineer1080/DedekindLanguage.git
cd DedekindLanguage
```

### 2. Compiler & Runtime
```bash
pip install -r requirements.txt
```
**Abhängigkeiten:** `torch` (PyTorch für Tensoren, FFT, ML), `matplotlib` (für `plot()`-Visualisierung), `ipykernel` (für Dedekind Jupyter Kernel).

### 3. Dedekind Studio starten (Spyder-Fork)
Aus dem Projektroot:

```bash
start_dedekind_studio.bat
```
(Unter Windows; unter Linux/macOS: `cd DedekindStudio && python bootstrap.py`.)

Beim ersten Start werden ggf. Spyder-Abhängigkeiten (PyQt5, qtpy, …) aus `DedekindStudio/requirements-dedekind-studio.txt` installiert.

## 💻 Usage

1.  **Dedekind Studio** starten (siehe oben). Im Editor `.ddk`-Dateien öffnen, in der Konsole „Dedekind“ als Kernel wählen oder Python nutzen.
2.  Code ausführen: `.ddk`-Datei mit Run/F5 ausführen oder Dedekind-Code in der Dedekind-Konsole eingeben.
3.  **CLI** (ohne IDE): `python -m src.compiler.compiler examples/dedekind/hello.ddk`
4.  **Jupyter/Spyder** (ohne Fork): Dedekind-Kernel installieren (`jupyter kernelspec install dedekind_jupyter_kernel/kernelspec`), dann Kernel „Dedekind“ wählen.

### Examples
Example programs are in `examples/dedekind/`, including:
- `hello.ddk` – basic I/O and tensors  
- `universal_constants.ddk` – physical constants and units (E = mc², gravitation, Coulomb)  
- `constants_extended.ddk` – mathematical `pi`, `e`; CODATA constants (`hbar`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday`)  
- `signal_physics.ddk` – complex numbers (Quaternions) and FFT  
- `differentiable_ode.ddk` – differentiable ODE solver with `ode_solve` and `grad`  
- `pde_heat.ddk` – differentiable PDE solver (1D/2D heat equation) with `pde_heat_1d` / `pde_heat_2d`  
- `distributions_extended.ddk` – Exponential, Gamma, Beta, Poisson; `sample`, `log_prob`  
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
- `probabilistic.ddk` – distributions, sampling, and Bayesian inference with `metropolis`  
- `conditional_logic.ddk`, `basic_loops.ddk` – control flow  
- `mnist_classifier.ddk` – neural network with `Sequential`/`Dense`  

From the `src/` directory: `python -m compiler.compiler ../examples/dedekind/hello.ddk`

**Alle Beispiele auf einmal testen** (aus Projektroot): `python run_examples.py` — kompiliert und führt alle `.ddk`-Dateien in `examples/dedekind` aus; Optionen: `-q` (nur Zusammenfassung), `-v` (vollständige Ausgabe), `--compile` (nur kompilieren), `--filter name` (nur Dateien mit „name“ im Dateinamen).

## 🗺️ Roadmap

### Phase 1: Foundation ✅
*   [x] Language Specification & Design
*   [x] Proof-of-Concept Compiler (Python Backend)
*   [x] Dedekind Studio (Spyder-Fork)

### Phase 2: Core Development ✅
*   [x] Build-in Core Algorithms (FFT, Conv, Linalg)
*   [x] Robust Lexer & Parser (Windows Support, Unary Ops)
*   [x] Resizable Studio Terminal & Tabs

### Phase 3: Hardware Acceleration ✅
*   [x] Integration with **PyTorch** for GPU execution.
*   [x] Implementation of `.gpu()` and `.cpu()` modifiers.

### Phase 4: Production (v0.2) ✅
*   [x] **Native Performance**: Integration with MLIR/Inductor via `.fast()`.
*   [x] **MLIR Prototype**: Dedekind-Dialect IR generation.
*   [x] **Studio Upgrade**: Resizable terminal and UI polish.

### Phase 5: Advanced Mathematics ✅
*   [x] **Autograd**: Native `grad()` operator for automatic differentiation.
*   [x] **Property Access**: Native `.shape` support for tensors and models.

### Phase 6: Tensor Contraction & Logic (v0.3) ✅
*   [x] **Einsum**: High-level elective tensor contraction syntax.
*   [x] **Complex/Quaternion**: Built-in support for rotational math.

### Phase 8: AOT Compilation & LLVM Backend ✅
*   [x] **Static Binary**: Standalone `.exe` generation without Python.
*   [x] **MLIR Pipeline**: Dedekind -> MLIR -> LLVM -> Binary.
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

Dedekind aims to become the "Standard Language for Nature's Laws." To achieve this, we are researching the native implementation of the following concepts:

1. **Differentiable ODE Solvers**: Implemented in v0.7: `ode_solve(fun, y0, t, method="rk4")` solves dy/dt = fun(t,y) with differentiable RK4 (or Euler); gradients flow through y0 and parameters in `fun`. Use with `grad()` for physics-informed ML. See `examples/dedekind/differentiable_ode.ddk`.
2. **Differentiable PDE Solvers**: Implemented in v0.8: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` solve the heat equation with finite differences and `ode_solve`; gradients through initial condition and diffusivity for inverse problems. See `examples/dedekind/pde_heat.ddk`.
3. **Physical Units**: Implemented at runtime: `10[m] / 2[s]` → `5[m/s]`, add/sub require same unit; future: compile-time unit checking.
4. **Probabilistic Programming**: Implemented in v0.8: `Normal`, `Uniform`, `Bernoulli`; `sample(dist)`, `log_prob(dist, value)`; `metropolis(log_prior, log_likelihood, data, init, steps)` for Bayesian inference. See `examples/dedekind/probabilistic.ddk`. Future: more distributions, NUTS/VI, conditioning syntax.
5. **Symbolic Simplification**: A compile-time algebraic engine that simplifies complex mathematical expressions before code generation to maximize efficiency.

## 📚 Documentation

- **Language Specification**: `Documentation/Dedekind_Language_Specification.md` (v0.2; §15 Physical Units v0.6, §15.7 ODE v0.7, §15.8 Probabilistic v0.8, §15.9 PDE v0.8, §15.10 Integration & Math v0.9/v0.9.6; Chemie/Biologie v0.9.7; I/O/JSON v0.9.8; Stand v1.0.10). PDF can be generated with `pandoc` (see `Documentation/README.md`).
- **Research & Architecture**: `Documentation/Dedekind_Research_and_Architecture.md` (includes §10 Sprachfeatures v0.6).
- **Symbolic Simplification**: `Documentation/Symbolic_Simplification_Roadmap.md` — Implementierungs-Roadmap (Phasen, Optionen, Integration).
- **Features Roadmap**: `Documentation/Features_Implementation_Roadmap.md` — naturwissenschaftliche Features (Phase 1 abgeschlossen: Verteilungen, Integration).
- **Chemie & Biologie**: `Documentation/Chemistry_Biology_Roadmap.md` — Einheiten mol/L/M/ppm, Beispiele (Kinetik, Dosis-Wirkung, Wachstum), Convenience-Funktionen.
- Legacy PDFs (v0.1) remain in `Documentation/`; the Markdown sources are the up-to-date references.

## 📄 License

This project is licensed under the MIT License.
