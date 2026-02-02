# Dedekind Documentation

This folder contains the **source** and **generated** documentation for the Dedekind language.

## Contents

| File | Description |
|------|-------------|
| **Dedekind_Language_Specification.md** | Language Specification (Markdown source, v0.2; §15 Physical Units v0.6, §15.7 ODE v0.7, §15.8 Probabilistic v0.8, §15.9 PDE v0.8, §15.10 Integration & Math v0.9/v0.9.6; Chemie/Biologie v0.9.7; I/O/JSON v0.9.8; Stand v1.1.8) |
| **Dedekind_Research_and_Architecture.md** | Research foundation & architecture (Markdown source; §10 Sprachfeatures v0.6) |
| **Symbolic_Simplification_Roadmap.md** | Implementierungs-Roadmap für Symbolic Simplification (Phasen, Optionen, Integration) |
| **Features_Implementation_Roadmap.md** | Implementierungs-Roadmap für naturwissenschaftliche Features (Verteilungen, Integration, Einheiten Compile-Zeit, NUTS/VI, Fitting, LaTeX, symbolische Ableitungen) |
| **Chemistry_Biology_Roadmap.md** | Roadmap für Chemie und Biologie (Einheiten mol/L/M, Beispiele Kinetik/Dosis-Wirkung/Wachstum, Convenience-Funktionen, Doku „Dedekind für Chemie & Biologie“) |
| **Commercialization_Options.md** | Potenzielle Kommerzialisierungsoptionen (Beratung, Support, Lizenzen, SaaS, Förderung, Phasierung, Risiken) |
| **IDE_Studio_Roadmap.md** | Dedekind in bestehenden IDEs (VS Code, Jupyter) + Dedekind Studio als kommerzielle Wissenschaftler-IDE (Einheiten, Plots, Postgres, LaTeX, lokale KI) |
| **Maturity_Assessment.md** | Ausgereiftheit von Dedekind für Mathematik, Physik, Informatik, Biologie und Chemie (Stand v0.9.8; Stärken, Lücken, Roadmap) |
| **Dedekind_Language_Specification_v0.1.pdf** | Legacy PDF (v0.1); for current spec use the Markdown or generate v0.2 PDF below |
| **Dedekind_Research_Papers_and_Architecture.pdf** | Legacy PDF; for current content use the Markdown or generate PDF below |

## Generating PDFs from Markdown

The Markdown files (`.md`) are the **canonical sources**. To produce updated PDFs:

### Option 1: Pandoc (recommended)

Install [pandoc](https://pandoc.org/) and a LaTeX engine (e.g. MiKTeX, TeX Live), then run from this folder:

```bash
# Language Specification
pandoc Dedekind_Language_Specification.md -o Dedekind_Language_Specification_v0.2.pdf --toc

# Research & Architecture
pandoc Dedekind_Research_and_Architecture.md -o Dedekind_Research_and_Architecture.pdf --toc
```

### Option 2: Other tools

- **VS Code**: Use an extension such as "Markdown PDF" to export the open `.md` file to PDF.
- **Online**: Paste the Markdown into a service that converts Markdown to PDF (e.g. markdown-to-pdf converters).
- **Typora / other editors**: Open the `.md` file and export to PDF from the application.


## What changed in v1.1.8 (documented here)

- **Version 1.1.8**: Dokumentation und Versionsangaben vereinheitlicht; Changelog konsolidiert.

## What changed in v1.1.7 (documented here)

- **Version 1.1.7**: **Matrix-Operator @**: A @ B statt matmul(A, B); gleiche Priorität wie * und /. **Spezialfunktionen**: bessel_j0(x), bessel_j1(x); bessel_j(n, x); legendre(n, x); hypergeom(a, b, c, z). bessel_j, legendre, hypergeom erfordern scipy.

## What changed in v1.1.6 (documented here)

- **Version 1.1.6**: **Symbolische Ableitung**: diff_sym(expr, var) – Ausdruck und Variable als Strings; Ableitung als String. Ohne re/typing (nativ). Unterstützt: +, -, *, /, ^, sin, cos, tan, exp, log, sqrt.

## What changed in v1.1.5 (documented here)

- **Version 1.1.5**: **Assert & Tests**: assert(condition, message); Mini-Test-Runner run_tests.py für tests/dedekind/*.ddk. **Plots**: scatter, contour; plot mit xscale/yscale (log). **Autograd**: jacobian(f, x), hessian(f, x). **Signal & Reduktionen**: fftfreq, diff, cumsum, clip, shuffle.

## What changed in v1.1.4 (documented here)

- **Version 1.1.4**: **Statistik**: cov, corrcoef, skew, kurtosis, histogram. **Algorithmen**: qr, cholesky, polyfit, polyval, unique, argsort, convolve1d, minimize_scalar, newton.

## What changed in v1.1.3 (documented here)

- **Version 1.1.3**: **Numerik**: Neue Built-ins `cond`, `rank`, `pinv`, `expm`, `logm`, `interp`, `trapz`, `root_bisect`; Dokumentation und Beispiel erweitert.

## What changed in v1.1.2 (documented here)

- **Version 1.1.2**: **Einheiten-Anzeige**: Gleiche Faktoren werden zusammengefasst (`m*m` → `m^2`, `m*m*m` → `m^3`, `m^2*m` → `m^3`, `m*m*kg` → `m^2*kg`). Literale `1[m^2]`, `1[m^3]` nutzbar; `m^3` bei Volumen-Umrechnung (z. B. `1[m^3] + 500[L]` → `1.5[m^3]`).

## What changed in v1.1.1 (documented here)

- **Version 1.1.1**: **Automatische Einheiten-Umrechnung** bei Addition und Subtraktion: SI-Basis (Länge, Masse, Zeit, Strom, Temperatur, Stoffmenge, Lichtstärke) und abgeleitet (Druck, Volumen, Energie, Spannung, Frequenz, Ladung, Widerstand, Leistung). Z. B. `1[m] + 100[cm]` → `2.0[m]`, `1[kJ] + 500[J]` → `1.5[kJ]`. Ergebnis-Einheit = erste Operand-Einheit. Gilt für `Quantity` und `UncertainQuantity`. Compile-Zeit-Check erlaubt gleiche Dimension; inkompatible Einheiten → CompileError. Beispiel: `examples/dedekind/length_units_conversion.ddk`.

## What changed in v1.1.0 (documented here)

- **Version 1.1.0**: Konsole: Reihenfolge von `print_latex()` und `print()` korrigiert (beide schreiben in denselben stdout-Puffer). LaTeX-Konvertierung: `\texttt{...}` unterstützt (z. B. `.sparse()` als Code in cfd_sparse_sim.ddk). Beispiele: führendes `\n` in allen `print("\n...")` entfernt – saubere Ausgabe ohne leere Zeilen.

## What changed in v1.0.10 (documented here)

- **Version 1.0.10**: Wissenschaftliche Konsole: print_latex(s) für LaTeX-Rendering. Dedekind Studio: neues App-Icon (dedekind_app_icon.svg). chemistry_units_radiation.ddk Ausgabe auf ASCII.

## What changed in v1.0.9 (documented here)

- **Version 1.0.9**: Chemische Einheiten bar, atm, g; Radioaktivität Bq, Sv; Beispiel chemistry_units_radiation.ddk. Candela (cd) und viele SI-Vereinfachungen. Dedekind Studio: Alle Beispiele beim Start laden; chemistry_units_radiation.ddk in Assets.

## What changed in v1.0.8 (documented here)

- **Version 1.0.8**: Versionserhöhung und Veröffentlichung der Änderungen.

## What changed in v1.0.7 (documented here)

- **Version 1.0.7**: Dedekind Studio Fenstertitel ohne Python-Version; Splash-Screen-Untertitel „Scientific Dedekind Development Environment“ / „by Mario Michael Heinrich“. Immer grünes Theme; Variable Explorer und leere Plot-Fläche mit grüner Palettenfarbe. Beispiele nur aus Assets; Sitzungswiederherstellung nur für existierende Dateien. FFT-Beispiel ASCII-Kommentare; Runtime: komplexe Plot-Werte vor Zeichnen in reell. Basedpyright: type-ignore für Laufzeitabhängigkeiten in spyder/__init__.py.

## What changed in v1.0.6 (documented here)

- **Version 1.0.6**: Wissenschaftliche Plot-Beispiele (`scientific_*.ddk`): Wellen-Superposition, gedämpfter Oszillator, Arrhenius-Plot, Gravitationspotential, Ricci-Plot, FFT-Spektrum – mit Dedekind-Features (pi, sin/cos/exp, Einheiten, Ricci-Notation, fft, arrhenius, plot). Dedekind Studio lädt beim Start diese Beispiele als Tabs; Hello-World-Dateien (welcome_dedekind.ddk, hello.ddk in den Assets) entfernt. Pylint, Profiler und Debugger (Python-spezifisch) als deprecated markiert und in Dedekind Studio nicht mehr geladen – sie erscheinen nicht in Layouts, View > Panes oder bei Layout-Wiederherstellung.

## What changed in v1.0.5 (documented here)

- **Version 1.0.5**: Dedekind Studio: `plot()` zeigt Abbildungen in der Plots-Pane; Kernel sendet display_data. Warnung „Unknown message type: comm_open“ behoben; Hinweis „Figures are displayed…“ unterdrückt.

## What changed in v1.0.4 (documented here)

- **Version 1.0.4**: Dedekind Studio: Syntax-Highlighting für Einheiten (z. B. `10[m]`, `[kg]`) und Ricci-Indizes (`A^ij`, `B_jk`) mit eigenen Farben.

## What changed in v1.0.3 (documented here)

- **Version 1.0.3**: Compiler: ML-Runtime-Einbindung bei Runtime-Built-ins (alle 36 Beispiele laufen). Dedekind Studio: PyTorch-Backend beim Start, Spyder-Update-Check deaktiviert (Fork).

## What changed in v1.0.2 (documented here)

- **Version 1.0.2**: Patch release (Dedekind Studio 1.0.2). Umbenennung Fourier → Dedekind (Sprache, IDE, Kernel, Dateiendung `.ddk`).

## What changed in v1.0.1 (documented here)

- **Version 1.0.1**: Patch release. Bugfixes und kleine Verbesserungen (Dedekind Studio / Kernel); keine neuen Features.

## What changed in v1.0.0 (documented here)

- **Version 1.0.0**: Erste stabile Release-Version 1.0. Dedekind Studio (Spyder-Fork) und Dedekind Jupyter Kernel; Sprache und Tooling als 1.0. Die Umbenennung Fourier → Dedekind war bereits v0.9.9.

## What changed in v0.9.8 (documented here)

- **Version 0.9.8**: Convenience-Funktionen (Michaelis-Menten, logistisches Wachstum, Arrhenius, lineare Regression); chemische Elemente (`atomic_mass`, `atomic_number`, ca. 50 Elemente); **Datei-I/O** (`read_file`, `write_file`, `file_exists`), **Netzwerk** (`http_get`, `http_post`), **JSON** (`json_parse`, `json_stringify`; Zugriff `obj["key"]`). Beispiele: `file_io_json.ddk`, `chemistry_elements.ddk`, `linear_regression.ddk`, `chemistry_arrhenius.ddk`. Maturity_Assessment und Chemistry_Biology_Roadmap ergänzt.

## What changed in v0.9.7 (documented here)

- **Version 0.9.7**: Dedekind für Chemie & Biologie — chemische Einheiten mol, L, M (= mol/L), ppm in Runtime und Compile-Check; M und mol/L gelten als gleiche Einheit. Einheiten-Literal `[1/s]` im Parser. Beispiele: `chemistry_kinetics.ddk`, `dose_response.ddk`, `biology_growth.ddk`. Abschnitt im README; Roadmap: **Chemistry_Biology_Roadmap.md**.

## What changed in v0.9.6 (documented here)

- **Version 0.9.6**: Grundlegende Math-Funktionen — `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; Arkusfunktionen `asin`, `acos`, `atan`, `atan2(y,x)`; Hyperbelfunktionen `sinh`, `cosh`, `tanh`. Siehe **Dedekind_Language_Specification.md** §15.10 und `examples/dedekind/math_functions.ddk`.

## What changed in v0.9.5 (documented here)

- **Version 0.9.5**: Bessere Fehlermeldungen (Phase 2) — AST-Knoten mit `line`; Parser wirft `CompileError(message, line, filepath)`; Runtime-Quantity-Meldungen mit Kontext. Einheiten zur Compile-Zeit (Phase 3b) — `1[m] + 1[s]` → Compiler-Fehler; `units_checker.py`, `compile_source(..., check_units=True)`, CLI `--no-units-check`.

## What changed in v0.9.4 (documented here)

- **Version 0.9.4**: HMC (Hamiltonian Monte Carlo) — `hmc(...)` und `fit(..., method="hmc")`; LaTeX-Export — `export_to_latex(source)` bzw. CLI `--latex`. Siehe **Dedekind_Language_Specification.md** und `examples/dedekind/hmc_fitting.ddk`, `examples/dedekind/latex_demo.ddk`.

## What changed in v0.9.3 (documented here)

- **Version 0.9.3**: Uncertainty Propagation (`uncertain`, `UncertainQuantity`) und Fitting (`fit`). Siehe **Dedekind_Language_Specification.md** §15.11 und `examples/dedekind/uncertainty_propagation.ddk`, `examples/dedekind/curve_fitting.ddk`.

## What changed in v0.9.2 (documented here)

- **Extended Constants**: Mathematical `pi`, `e`; physical CODATA constants: `hbar`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday`. See **Dedekind_Language_Specification.md** §15.2 and `examples/dedekind/constants_extended.ddk`.

## What changed in v0.9.1 (documented here)

- **Run-Examples-Skript**: `run_examples.py` im Projektroot kompiliert und führt alle `.ddk`-Dateien in `examples/dedekind` aus; Optionen `-q`, `-v`, `--compile`, `--filter`. Siehe Haupt-README unter „Alle Beispiele auf einmal testen“.

## What changed in v0.9 (documented here)

- **Extended Distributions**: `Exponential`, `Gamma`, `Beta`, `Poisson`; `sample(dist)`, `log_prob(dist, value)`. See **Dedekind_Language_Specification.md** §15.8 and `examples/dedekind/distributions_extended.ddk`.
- **Numerical Integration**: `integrate(f, a, b, n)`; `sin(x)`, `cos(x)`. See **Dedekind_Language_Specification.md** §15.10 and `examples/dedekind/integration.ddk`.

## What changed in v0.8 (documented here)

- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`; `sample(dist)` / `sample(dist, n)`; `log_prob(dist, value)`; Metropolis-Hastings `metropolis(log_prior, log_likelihood, data, init, steps)` for Bayesian inference. See **Dedekind_Language_Specification.md** §15.8 and `examples/dedekind/probabilistic.ddk`.
- **Differentiable PDE Solvers**: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` for the heat equation; finite differences + `ode_solve`; Dirichlet BC; gradients through `u0` and `k`. See **Dedekind_Language_Specification.md** §15.9 and `examples/dedekind/pde_heat.ddk`.

## What changed in v0.7 (documented here)

- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` (RK4/Euler); gradients through `grad()` for physics-informed ML; `linspace(start, stop, steps)` for time grids. See **Dedekind_Language_Specification.md** §15.7 and `examples/dedekind/differentiable_ode.ddk`.

## What changed in v0.6 (documented here)

- **Physical Units**: Literals with units (`1.0[kg]`, `5.0e14[Hz]`), constants `c`, `G`, `h`, `k_B`, `k_e` as Quantity with SI units, arithmetic and `^` for powers, display simplification (J, N).
- **Quantity & Quaternion**: Full arithmetic including `__pow__` and `__neg__`; unary minus in codegen for expressions like `-1.0[C]` and `-1.0 + 0i`.

See **Dedekind_Language_Specification.md** §15 and **Dedekind_Research_and_Architecture.md** §10 for full details.
