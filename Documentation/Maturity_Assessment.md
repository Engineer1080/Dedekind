# Dedekind — Ausgereiftheit für Mathematik, Physik, Informatik, Biologie und Chemie

**Dedekind Language**  
Draft: January 2026 · Stand: v0.9.8 (Prototyp)

---

## Übersicht

Dedekind ist aktuell ein **Prototyp (v0.9.9)**. Die folgende Einschätzung bezieht sich auf den **aktuellen Implementierungsstand** und nennt Lücken sowie Roadmap-Punkte.

| Domäne       | Ausgereiftheit (kurz) | Nutzbar für …                          | Wichtige Lücken / Roadmap          |
|-------------|------------------------|----------------------------------------|------------------------------------|
| **Mathematik** | Gut nutzbar          | Analysis, LA, Statistik, Integration  | Symbolik, Differentiale, mehr LA  |
| **Physik**    | Gut nutzbar          | Mechanik, E&M, Thermodynamik, ODE/PDE | Weitere PDE, mehr Konstanten       |
| **Informatik**| Grundlegend          | ML-Pipeline, Algos, Kontrollfluss      | Typen, Module, Tooling, Performance|
| **Biologie** | Grundlegend          | Wachstum, Fitting, Unsicherheit       | Convenience-Funktionen, Ökosystem  |
| **Chemie**    | Grundlegend          | Kinetik, Konzentration, Einheiten      | Convenience, weitere Einheiten      |

---

## 1. Mathematik

### Was vorhanden ist

- **Konstanten**: `pi`, `e` als `Quantity` (dimensionslos).
- **Elementare Funktionen**: `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; Arkus- und Hyperbelfunktionen (`asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`); elementweise, differenzierbar.
- **Reduktionen & Runden**: `min`, `max`, `argmin`, `argmax` (optional `dim`); `round`, `floor`, `ceil`.
- **Lineare Algebra**: `norm(x)`, `det(A)`, `trace(A)`; Tensoren, Matrizen, `.shape`; FFT (`fft`, `ifft`).
- **Numerische Integration**: `integrate(f, a, b, n)` (Trapezregel); differenzierbar, wenn `f` Tensor akzeptiert.
- **Statistik & Stochastik**: Verteilungen `Normal`, `Uniform`, `Bernoulli`, `Exponential`, `Gamma`, `Beta`, `Poisson`; `sample`, `log_prob`; MCMC (`metropolis`, `hmc`); Fitting `fit(..., method="gd"|"mcmc"|"hmc")`.
- **Tensor-Notation**: Ricci-ähnliche Indexnotation (`A^ij * B_jk`) für Einstein-Summen.
- **LaTeX-Export**: `export_to_latex(source)` — Formeln aus Dedekind-Code als LaTeX.

### Lücken / Roadmap

- **Symbolische Mathematik**: Kein `diff(expr, x)` als Formel; nur numerisches `grad()`. Geplant in Features-Roadmap Phase 5 (Symbolic Simplification).
- **Weitere LA**: Kein eingebautes Eigenwert/Eigenvektor, SVD, QR; über Tensoren/Backend möglich, aber keine Dedekind-API.
- **Differentialgeometrie**: Ricci-Notation vorhanden; keine kovariante Ableitung oder Riemann-Tensor als Bibliothek.
- **Zahlentheorie / Diskrete Mathematik**: Keine speziellen Primitiven.

**Fazit Mathematik**: Für **numerische Analysis, lineare Algebra, Statistik, Integration und Stochastik** gut nutzbar; für **symbolische und höhere mathematische Themen** noch Lücken. **Ausgereiftheit: gut nutzbar.**

---

## 2. Physik

### Was vorhanden ist

- **Einheiten**: `Quantity`, Literale mit Einheiten (`10[m]`, `5[m/s]`, `1.0[kg]`); Addition/Subtraktion nur bei gleicher Einheit; Multiplikation/Division kombiniert Einheiten; `^` für Potenzen; Anzeige vereinfacht (J, N, M).
- **Compile-Zeit-Check**: `1[m] + 1[s]` → Compiler-Fehler mit Zeile; `units_checker.py`, CLI `--no-units-check`.
- **Konstanten (CODATA)**: `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — alle als `Quantity` mit SI-Einheiten.
- **Differenzierbare ODE**: `ode_solve(fun, y0, t)` (RK4/Euler); `grad()` durch `y0` und Parameter; `linspace(start, stop, steps)`.
- **Differenzierbare PDE**: `pde_heat_1d(u0, x, t, k)`, `pde_heat_2d(u0, x, y, t, k)` für Wärmeleitung; Finite Differenzen + `ode_solve`; Gradients durch `u0`, `k`.
- **Quaternionen**: Native Unterstützung (`i`, `j`, `k`), `.rotate()`; für Rotationen und Signalverarbeitung.
- **Uncertainty**: `uncertain(value, std)`, `UncertainQuantity` — Gauß’sche Fehlerfortpflanzung; optional mit Einheit.
- **Beispiele**: `universal_constants.ddk`, `physical_units.ddk`, `differentiable_ode.ddk`, `pde_heat.ddk`, `relativity_physics.ddk`, `signal_physics.ddk`, `quantum_rotations.ddk`.

### Lücken / Roadmap

- **Weitere PDE**: Nur Wärmeleitung 1D/2D; keine Wellen-, Advektions- oder Maxwell-Gleichungen als Standard-API.
- **Lagrange/Hamilton**: Keine eingebaute Formulierung; mit `ode_solve` und eigener RHS modellierbar.
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
- **Standardbibliothek**: **Datei-I/O** (`read_file`, `write_file`, `file_exists`), **Netzwerk** (`http_get`, `http_post`), **JSON** (`json_parse`, `json_stringify`); Zugriff auf geparste Objekte z. B. `obj["key"]`. Beispiel: `file_io_json.ddk`.
- **Tooling**: Dedekind Studio (IDE) vorhanden; Debugger, Profiler, Test-Runner nicht als Teil der Sprache.
- **Performance**: AOT/MLIR/LLVM als Ziel; aktuell Transpilation zu Python/PyTorch; native Binaries experimentell.

**Fazit Informatik**: Für **ML-Pipelines, numerische Algorithmen und Kontrollfluss** grundlegend nutzbar; für **große Softwareprojekte, Typen, Module und Tooling** noch nicht ausgereift. **Ausgereiftheit: grundlegend.**

---

## 4. Biologie

### Was vorhanden ist

- **Dynamische Systeme**: `ode_solve` für beliebige ODEs (z. B. logistisches Wachstum \(dN/dt = r N (1 - N/K)\)).
- **Fitting**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc")` — Parameter aus Daten (z. B. Wachstumsrate, Kapazität).
- **Statistik & Unsicherheit**: Verteilungen, `uncertain(value, std)`, MCMC/HMC für Bayesian Inference.
- **Beispiel**: `biology_growth.ddk` — logistisches Wachstum mit `ode_solve`.

### Lücken / Roadmap (Chemistry_Biology_Roadmap)

- **Convenience-Funktionen**: `logistic_growth(N, r, K)` bzw. analytische Lösung `logistic(t, r, K, N0)` noch nicht als eingebaute API; mit `ode_solve` und eigener RHS umsetzbar.
- **Weitere Modelle**: Keine vordefinierten Ökologie-/Populationsmodelle (z. B. Lotka-Volterra) als Standard-API.
- **Einheiten**: Physikalische Einheiten (inkl. chemische) nutzbar; keine biologischen Konventionen (z. B. Zellzahl, Verdünnungsstufen) als eigene Einheiten.
- **Dokumentation**: Abschnitt „Dedekind für Chemie & Biologie“ im README; Roadmap mit Phase 2 (Convenience).

**Fazit Biologie**: Für **Wachstumsmodelle, Fitting und Unsicherheit** grundlegend nutzbar; Convenience und domänenspezifische Modelle sind geplant. **Ausgereiftheit: grundlegend.**

---

## 5. Chemie

### Was vorhanden ist

- **Einheiten**: mol, L, M (= mol/L), ppm; `0.1[M]`, `1[mol]`, `50[ppm]`; M und mol/L gelten als gleiche Einheit (Runtime und Compile-Check).
- **Konstanten**: `N_A`, `R_gas`, `F_faraday` als `Quantity` mit SI-Einheiten (`1/mol`, `J/(K*mol)`, `C/mol`).
- **Kinetik**: Reaktion 1. Ordnung \(c(t) = c_0 e^{-kt}\) mit `ode_solve` oder analytisch mit `exp`; Einheiten [M], [1/s].
- **Dosis-Wirkung / Michaelis-Menten**: \(v = V_{\max}[S]/(K_M + [S])\) über `fit` an Daten (EC50, \(K_M\), \(V_{\max}\)).
- **Uncertainty**: `uncertain(value, std)` mit Einheit für Fehlerfortpflanzung.
- **Beispiele**: `chemistry_kinetics.ddk`, `dose_response.ddk`; `universal_constants.ddk` (inkl. chemischer Konstanten).

### Lücken / Roadmap (Chemistry_Biology_Roadmap)

- **Convenience-Funktionen**: `michaelis_menten(S, Vmax, Km)`, `arrhenius(T, A, Ea)` noch nicht als eingebaute API; Phase 2 geplant.
- **Weitere Einheiten**: bar, atm, pH-Hinweis (pH = -log10([H+])), % w/v optional — Phase 3.
- **Gleichgewichte**: Keine vordefinierten Gleichgewichts- oder Titrationsmodelle als Standard-API.
- **Elemente**: `atomic_mass("C")` (g/mol), `atomic_number("C")`; ca. 50 Elemente (IUPAC-nah); Molare Masse z. B. 2*atomic_mass("H")+atomic_mass("O"). Beispiel: `chemistry_elements.ddk`.
- **Stöchiometrie**: Keine dedizierte Reaktionsgleichungs- oder Reaktionsnetzwerk-API (Gleichungen ausbalancieren o. Ä.).

**Fazit Chemie**: Für **Kinetik 1. Ordnung, Konzentrationen, Einheiten, Dosis-Wirkung und Fitting** grundlegend nutzbar; Convenience und weitere Einheiten sind in der Roadmap. **Ausgereiftheit: grundlegend.**

---

## 6. Zusammenfassung und Priorisierung

| Domäne       | Reife      | Stärken                                      | Nächste Schritte (aus Roadmaps)           |
|-------------|------------|----------------------------------------------|-------------------------------------------|
| **Mathematik** | Gut       | Analysis, LA, Statistik, Integration, FFT   | Symbolik, ggf. mehr LA-Primitive          |
| **Physik**    | Gut       | Einheiten, ODE/PDE, Konstanten, Uncertainty | Weitere PDE, ggf. Lagrange/Hamilton       |
| **Informatik**| Grundlegend| ML, Tensoren, Kontrollfluss, Algos           | Typen, Module, Tooling, AOT-Reife          |
| **Biologie**  | Grundlegend| ODE, Fitting, Unsicherheit                  | `logistic_growth`, weitere Modelle        |
| **Chemie**    | Grundlegend| Einheiten mol/L/M/ppm, Kinetik, Fitting      | `michaelis_menten`, `arrhenius`, bar/atm  |

**Gesamtbewertung**: Dedekind ist für **Mathematik und Physik** bereits **gut nutzbar** (numerische und einheitenbewusste Anwendungen). Für **Informatik** ist die Basis gelegt, aber Typen, Module und Tooling fehlen für „ausgereift“. Für **Biologie und Chemie** sind die **Grundbausteine** (Einheiten, ODE, Fitting, Unsicherheit) da; **Convenience-Funktionen und weitere Einheiten** würden die Ausgereiftheit in diesen Domänen deutlich erhöhen (Chemistry_Biology_Roadmap Phase 2/3).
