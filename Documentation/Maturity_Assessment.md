# Dedekind — Ausgereiftheit für Mathematik, Physik, Informatik, Biologie und Chemie

**Dedekind Language**  
Draft: January 2026 · Stand: v1.7.0 (Prototyp)

---

## Übersicht

Dedekind ist aktuell ein **Prototyp (v1.7.0)**. Die folgende Einschätzung bezieht sich auf den **aktuellen Implementierungsstand** und nennt Lücken sowie Roadmap-Punkte.

| Domäne       | Ausgereiftheit (kurz) | Nutzbar für …                          | Wichtige Lücken / Roadmap          |
|-------------|------------------------|----------------------------------------|------------------------------------|
| **Mathematik** | Gut nutzbar          | Analysis, LA, Statistik, Integration  | Symbolik, Differentiale, mehr LA  |
| **Physik**    | Gut nutzbar          | Mechanik, E&M, Thermodynamik, ODE/PDE | Weitere PDE, mehr Konstanten       |
| **Informatik**| Grundlegend          | ML-Pipeline, Algos, Kontrollfluss      | Typen, Module, Tooling, Performance|
| **Biologie** | Grundlegend          | Wachstum, Fitting, Unsicherheit, `logistic` | Lotka-Volterra, weitere Modelle     |
| **Chemie**    | Grundlegend          | Kinetik, Konzentration, Einheiten, Convenience | Gleichgewichte, weitere Einheiten   |

---

## 1. Mathematik

### Was vorhanden ist

- **Konstanten**: `pi`, `e` als `Quantity` (dimensionslos).
- **Elementare Funktionen**: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; Arkus- und Hyperbelfunktionen (`asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`); elementweise, differenzierbar.
- **Reduktionen & Runden**: `min`, `max`, `argmin`, `argmax` (optional `dim`); `round`, `floor`, `ceil`.
- **Lineare Algebra**: `norm(x)`, `det(A)`, `trace(A)`; `eigh`, `eig`, `svd`, `qr`, `lstsq`, `cond`, `rank`, `pinv`; Tensoren, Matrizen, `.shape`; FFT (`fft`, `ifft`). **Numerik**: `integrate`, `trapz`, `simpson`, `riemann_sum`; `zeta(s)` (Riemann-Zeta); `volume_revolution_x`, `volume_revolution_y`, `volume_revolution_vertical`, `volume_revolution_horizontal`; `pappus_volume_vertical`, `pappus_volume_horizontal` (Satz von Pappus).
- **Numerische Integration**: `integrate(f, a, b, n)` (Trapezregel); differenzierbar, wenn `f` Tensor akzeptiert.
- **Statistik & Stochastik**: Verteilungen `Normal`, `Uniform`, `Bernoulli`, `Exponential`, `Gamma`, `Beta`, `Poisson`; `sample`, `log_prob`; MCMC (`metropolis`, `hmc`); Fitting `fit(..., method="gd"|"mcmc"|"hmc")`.
- **Tensor-Notation**: Ricci-ähnliche Indexnotation (`A^ij * B_jk`) für Einstein-Summen.
- **LaTeX-Export**: `export_to_latex(source)` — Formeln aus Dedekind-Code als LaTeX.

### Lücken / Roadmap

- **Symbolische Mathematik**: `diff_sym(expr, var)` liefert Ableitung als String; keine allgemeine symbolische Vereinfachung. Geplant: Phase 5 (Symbolic Simplification).
- **LA**: Umfangreiche API vorhanden; spezielle Domänen (z. B. strukturierte Matrizen) optional.
- **Differentialgeometrie**: Ricci-Notation, `christoffel_symbols`, `riemann_tensor`, `covariant_derivative` (numerisch) vorhanden.
- **Zahlentheorie**: `gcd`, `is_prime`, `mod`, `mod_inv`, `mod_pow`; `binom(n, k)`; Fakultät `n!`; `dirichlet_function(x)` (D(x)=1 wenn x rational, sonst 0). **Dedekind-Schnitte**: `DedekindCut`, `dedekind_cut_from_rational`, `dedekind_cut_sqrt2` (Konstruktion der reellen Zahlen aus Q). **Dedekind-Ringe**: `DedekindRingZ`, `ideal(n)`, `ideal_factor` (Z mit eindeutiger Ideal-Faktorisierung).

**Fazit Mathematik**: Für **numerische Analysis, lineare Algebra, Statistik, Integration und Stochastik** gut nutzbar; für **symbolische und höhere mathematische Themen** noch Lücken. **Ausgereiftheit: gut nutzbar.**

---

## 2. Physik

### Was vorhanden ist

- **Einheiten**: `Quantity`, Literale mit Einheiten (`10[m]`, `5[m/s]`, `1.0[kg]`); **automatische Umrechnung** bei Addition/Subtraktion für gleiche Dimension (Länge, Masse, Zeit, Druck, Winkel rad/deg, …); Multiplikation/Division kombiniert Einheiten; `^` für Potenzen; Anzeige vereinfacht (J, N, Pa, M).
- **Compile-Zeit-Check**: `1[m] + 1[s]` → Compiler-Fehler mit Zeile; `units_checker.py`, CLI `--no-units-check`.
- **Konstanten (CODATA)**: `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — alle als `Quantity` mit SI-Einheiten.
- **Differenzierbare ODE**: `ode_solve(fun, y0, t)` (RK4/Euler); `grad()` durch `y0` und Parameter; `linspace(start, stop, steps)`.
- **Differenzierbare PDE**: `pde_heat_1d`, `pde_heat_2d` (Wärmeleitung); `pde_advection_1d`, `pde_advection_2d` (Advektion); `pde_wave_1d`, `pde_wave_2d` (Wellengleichung); `pde_burgers_1d`, `pde_burgers_2d` (Burgers); `pde_reaction_diffusion_1d`, `pde_reaction_diffusion_2d`; `pde_advection_diffusion_1d`, `pde_advection_diffusion_2d`; `pde_maxwell_1d`, `pde_maxwell_2d` (Maxwell FDTD); Finite Differenzen + `ode_solve`; Gradients durch `u0`, Parameter.
- **Quaternionen**: Native Unterstützung (`i`, `j`, `k`), `.rotate()`; für Rotationen und Signalverarbeitung.
- **Uncertainty**: `uncertain(value, std)`, `UncertainQuantity` — Gauß’sche Fehlerfortpflanzung; optional mit Einheit.
- **Beispiele**: `universal_constants.ddk`, `physical_units.ddk`, `differentiable_ode.ddk`, `pde_heat.ddk`, `pde_advection.ddk`, `pde_wave.ddk`, `pde_burgers.ddk`, `pde_reaction_diffusion.ddk`, `pde_advection_diffusion.ddk`, `pde_maxwell.ddk`, `relativity_physics.ddk`, `signal_physics.ddk`, `quantum_rotations.ddk`.

### Lücken / Roadmap

- **PDE**: Wärmeleitung, Advektion, Wellengleichung, Burgers, Reaktions-Diffusion, Advektions-Diffusion, Maxwell (FDTD 1D/2D) und Navier-Stokes 2D (Chorin-Projektion) als Standard-API vorhanden.
- **Lagrange/Hamilton**: `lagrange_ode_rhs(L)`, `hamilton_ode_rhs(H)` – RHS für ode_solve aus L(q,v) bzw. H(q,p).
- **Feldtheorie**: Keine speziellen Primitiven.

**Fazit Physik**: Für **Mechanik, E&M (über Konstanten und Einheiten), Thermodynamik, ODE/PDE (Wärme), Unsicherheit und Rotationen** gut nutzbar. **Ausgereiftheit: gut nutzbar.**

---

## 3. Informatik

### Was vorhanden ist

- **Sprachkern**: Funktionen (`fn`), Zuweisungen, Kontrollfluss (`if`, `while`, `for … in`), Listen/Tensoren, Subscript, Member-Access (`.shape`, `.gpu`, `.sparse`).
- **ML-Runtime**: `Sequential`, `Dense`, `compile_model`, Forward-Pass; `.gpu()`, `.cpu()`; Autograd (`grad()`), `.with_grad()`.
- **Tensoren**: PyTorch-Backend; Matrizen, Vektoren, FFT, Faltung, Pooling; Sparse (`.sparse()`), Item-Assignment.
- **Algorithmen**: `sort`, `quicksort`; `fit` (GD, MCMC, HMC); MCMC (`metropolis`, `hmc`).
- **Beispiele**: `hello.ddk`, `basic_loops.ddk`, `conditional_logic.ddk`, `algo_showcase.ddk`, `autograd_showcase.ddk`, `mnist_classifier.ddk`, `matrix_gpu.ddk`, `cfd_sparse_sim.ddk`.

### Lücken / Roadmap

- **Typsystem**: Kein statisches Typing; nur Laufzeit- und Einheiten-Check.
- **Module/Imports**: Kein `import`; alles in einer Datei oder über Compiler-Pipeline.
- **Standardbibliothek (vorhanden)**: Datei-I/O (`read_file`, `write_file`, `file_exists`), Netzwerk (`http_get`, `http_post`), JSON (`json_parse`, `json_stringify`); Zugriff `obj["key"]`. Beispiel: `file_io_json.ddk`.
- **Tooling**: Dedekind Studio (IDE) vorhanden; Debugger, Profiler, Test-Runner nicht als Teil der Sprache.
- **Performance**: AOT/MLIR/LLVM als Ziel; aktuell Transpilation zu Python/PyTorch; native Binaries experimentell.

**Fazit Informatik**: Für **ML-Pipelines, numerische Algorithmen und Kontrollfluss** grundlegend nutzbar; für **große Softwareprojekte, Typen, Module und Tooling** noch nicht ausgereift. **Ausgereiftheit: grundlegend.**

---

## 4. Biologie

### Was vorhanden ist

- **Dynamische Systeme**: `ode_solve` für beliebige ODEs; Convenience: `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)` für logistisches Wachstum.
- **Fitting**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc")` — Parameter aus Daten (z. B. Wachstumsrate, Kapazität).
- **Statistik & Unsicherheit**: Verteilungen, `uncertain(value, std)`, MCMC/HMC für Bayesian Inference.
- **Beispiel**: `biology_growth.ddk` — logistisches Wachstum mit `ode_solve`.

### Lücken / Roadmap (Chemistry_Biology_Roadmap)

- **Weitere Modelle**: `lotka_volterra(x0, y0, a, b, c, d, t)` – Räuber-Beute-Modell vorhanden.
- **Einheiten**: Physikalische Einheiten (inkl. chemische) nutzbar; keine biologischen Konventionen (z. B. Zellzahl, Verdünnungsstufen) als eigene Einheiten.
- **Dokumentation**: Abschnitt „Dedekind für Chemie & Biologie“ im README; Chemistry_Biology_Roadmap.

**Fazit Biologie**: Für **Wachstumsmodelle, Fitting und Unsicherheit** grundlegend nutzbar; `logistic` vorhanden; weitere Modelle (Lotka-Volterra) geplant. **Ausgereiftheit: grundlegend.**

---

## 5. Chemie

### Was vorhanden ist

- **Einheiten**: mol, L, M (= mol/L), ppm; `0.1[M]`, `1[mol]`, `50[ppm]`; M und mol/L gelten als gleiche Einheit (Runtime und Compile-Check).
- **Konstanten**: `N_A`, `R_gas`, `F_faraday` als `Quantity` mit SI-Einheiten (`1/mol`, `J/(K*mol)`, `C/mol`).
- **Kinetik**: Reaktion 1. Ordnung \(c(t) = c_0 e^{-kt}\) mit `ode_solve` oder analytisch mit `exp`; Einheiten [M], [1/s].
- **Dosis-Wirkung / Michaelis-Menten**: `michaelis_menten(S, Vmax, Km)`; \(v = V_{\max}[S]/(K_M + [S])\) über `fit` an Daten (EC50, \(K_M\), \(V_{\max}\)).
- **Uncertainty**: `uncertain(value, std)` mit Einheit für Fehlerfortpflanzung.
- **Convenience**: `michaelis_menten`, `arrhenius`, `linear_regression`; `atomic_mass`, `atomic_number`; `balance_equation` (Stöchiometrie).
- **Beispiele**: `chemistry_kinetics.ddk`, `dose_response.ddk`, `chemistry_elements.ddk`; `universal_constants.ddk` (inkl. chemischer Konstanten).

### Lücken / Roadmap (Chemistry_Biology_Roadmap)

- **Gleichgewichte**: `chemical_equilibrium(K, n_A, n_B, n_C, n_D, A0, B0, C0, D0)` – Massenwirkungsgesetz für aA + bB <-> cC + dD.
- **Weitere Einheiten**: bar, atm, g, pH-Funktionen, % w/v vorhanden; weitere domänenspezifische Einheiten optional.

**Fazit Chemie**: Für **Kinetik, Konzentrationen, Einheiten, Dosis-Wirkung, Fitting, Stöchiometrie und Elemente** grundlegend nutzbar; Gleichgewichte fehlen. **Ausgereiftheit: grundlegend.**

---

## 6. Zusammenfassung und Priorisierung

| Domäne       | Reife      | Stärken                                      | Nächste Schritte (aus Roadmaps)           |
|-------------|------------|----------------------------------------------|-------------------------------------------|
| **Mathematik** | Gut       | Analysis, LA, Statistik, Integration, FFT   | Symbolik, ggf. mehr LA-Primitive          |
| **Physik**    | Gut       | Einheiten, ODE/PDE, Konstanten, Uncertainty | Weitere PDE, ggf. Lagrange/Hamilton       |
| **Informatik**| Grundlegend| ML, Tensoren, Kontrollfluss, Algos           | Typen, Module, Tooling, AOT-Reife          |
| **Biologie**  | Grundlegend| ODE, Fitting, Unsicherheit, `logistic`      | Lotka-Volterra, weitere Modelle          |
| **Chemie**    | Grundlegend| Einheiten mol/L/M/ppm, Kinetik, Fitting, Convenience | Gleichgewichte, weitere Einheiten        |

**Gesamtbewertung**: Dedekind ist für **Mathematik und Physik** bereits **gut nutzbar** (numerische und einheitenbewusste Anwendungen). Für **Informatik** ist die Basis gelegt, aber Typen, Module und Tooling fehlen für „ausgereift“. Für **Biologie und Chemie** sind die **Grundbausteine** (Einheiten, ODE, Fitting, Unsicherheit) sowie **Convenience-Funktionen** (`logistic`, `michaelis_menten`, `arrhenius`, `balance_equation`, Elemente) da; **Gleichgewichte und weitere domänenspezifische Modelle** würden die Ausgereiftheit weiter erhöhen.
