# Dedekind Programming Language

![Version](https://img.shields.io/badge/Version-1.17.0-blue) ![Dedekind Studio](https://img.shields.io/badge/Status-Prototype-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**Dedekind** is a modern, high-performance programming language designed specifically for compute-intensive workloads in **Machine Learning** and **Graphics Rendering**.

Unlike general-purpose languages retrofitted with parallel computing capabilities, Dedekind is built from the ground up with **GPU/TPU acceleration** and **Automatic Parallelization** as core features.

---

- **Ricci Calculus**: Native index notation (`A^ij * B_jk`) for Einstein summation.
- **Sparse Tensors**: Efficient `.sparse()` support for FEM/CFD simulations.
- **Fundamental Constants**: Mathematical `pi`, `e`; physical CODATA constants: `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` — all as **Quantity** with SI units.
- **Physical Units**: SI base m, kg, s, A, K, mol, **cd** (candela); literals (`10[m]`, `5[m/s]`, `1.0[kg]`, `1[cd]`); **automatische Umrechnung** bei Addition/Subtraktion für gleiche Dimension — **SI-Basis**: Länge (m, cm, km, mm, dm), Masse (kg, g, t, mg), Zeit (s, min, h, ms), Strom (A, mA, kA, uA), Temperatur (K, mK), Stoffmenge (mol, mmol, kmol), Lichtstärke (cd, mcd); **abgeleitet**: Druck (Pa, bar, atm), Volumen (L, mL, dm³, m³), Energie (J, kJ, MJ, Wh, kWh), Spannung (V, mV, kV), Frequenz (Hz, kHz, MHz, GHz), Ladung (C, mC, uC), Widerstand (ohm, kohm, Mohm), Leistung (W, kW, MW); **Winkel**: rad, deg. Ergebnis = Einheit des ersten Operanden; z. B. `1[m] + 100[cm]` → `2[m]`, `90[deg] + (pi/2)*1[rad]` → `180[deg]`. `deg_to_rad(x)`, `rad_to_deg(x)` für Konvertierung. Sonst add/sub gleiche Einheit; multiply/divide kombinieren Einheiten; `^` für Potenzen; Anzeige vereinfacht (J, N, Pa, W, Hz, …). **Chemie**: mol, L, M (= mol/L), ppm, **bar**, **atm**, **g**; **Radioaktivität**: **Bq**, **Sv**, Gy.
- **4D Rotational Math**: Native Quaternion support (`i`, `j`, `k` suffixes) and `.rotate()` method; unary minus supported (`-1.0 + 0i`).
- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` (RK4/Euler); gradients via `grad()` for physics-informed ML; `linspace(start, stop, steps)` for time grids.
- **Differentiable PDE Solvers**: `pde_heat_1d`, `pde_heat_2d` (heat); `pde_advection_1d`, `pde_advection_2d` (advection); `pde_wave_1d`, `pde_wave_2d` (wave); `pde_burgers_1d`, `pde_burgers_2d` (Burgers); `pde_reaction_diffusion_1d`, `pde_reaction_diffusion_2d`; `pde_advection_diffusion_1d`, `pde_advection_diffusion_2d`; `pde_maxwell_1d`, `pde_maxwell_2d` (Maxwell FDTD); `pde_navier_stokes_2d` (Navier-Stokes 2D incompressible, Chorin projection); finite differences + `ode_solve`; gradients through `u0` and parameters.
- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`, `Dirichlet(alpha)`; `sample(dist)`, `log_prob(dist, value)`; Bayesian inference via `metropolis(log_prior, log_likelihood, data, init, steps)`.
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

### What's New in v1.17.0

- **`try { ... } catch e { ... }` — natives Error-Handling.** Bisher konnten Forscher Exceptions nicht im `.ddk`-Code abfangen. Jetzt:
  ```dedekind
  try {
      content = read_file("/maybe/exists.json")
      return json_parse(content)
  } catch e {
      print("Datei nicht lesbar:", e)
      return {}
  }
  ```
  Verschachtelbar. Codegen emittiert standardmaessiges Python `try: ... except Exception as e: ...`. Damit ist Dedekind von einem Skript-DSL zu einer echten Anwendungs-Sprache geworden — defensive Programmierung und Fallback-Pfade sind ab v1.17 erstklassig.
- **Python-Style Slicing-Syntax `x[a:b]`, `x[:b]`, `x[a:]`, `x[::s]`, `x[a:b:s]`, `x[:]`.** Vorher musste man `x.narrow(0, a, b-a)` schreiben — jetzt:
  ```dedekind
  x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  head = x[:5]            // [10..50]
  tail = x[5:]            // [60..100]
  every2nd = x[::2]       // [10, 30, 50, 70, 90]
  middle = x[2:8:2]       // [30, 50, 70]
  ```
  Funktioniert auf Listen, Tensoren, allem was `__getitem__` mit `slice` versteht. Neuer `Slice(start, stop, step)`-AST-Knoten; jede Komponente ist optional.
- **Hinweis: PyTorch-Limitierung.** Negativer Step (`x[::-1]` fuer Reverse) wird von PyTorch nicht unterstuetzt — nutze `torch.flip(x, [0])`. Slice mit positivem Step funktioniert unveraendert.
- Beispiel: `try_catch_slicing_demo.ddk`. Test: `try_catch_slicing_test.ddk` (18 Asserts: 7 Slice-Varianten, try/catch flat und verschachtelt, kombinierte try+Slice). 43/43 Tests gruen, 99/99 Beispiele kompilieren.

### What's New in v1.16.0

- **`Sequence[DNA]` / `Sequence[RNA]` / `Sequence[Protein]`-Shape-Annotation:** Typsichere Sequenz-Validierung. Akzeptiert genau die Zeichen des jeweiligen Alphabets — sonst `ValueError: Sequence[DNA]-Check in fn(seq): ungueltiges Zeichen 'U' an Position 3 (erlaubt: ACGNT)`. Verhindert klassische Bioinformatik-Bugs wie RNA-statt-DNA-Verwechselung, die in Python/Biopython still falsche GC-Counts produzieren.
- **Native Bio-Built-ins** (kein `pyimport` noetig):
  - `gc_content(dna)` — Anteil G+C (0..1)
  - `reverse_complement(dna)` — DNA reverse-complement
  - `transcribe(dna)` — DNA → RNA (T → U)
  - `translate(rna, stop_at_stop=true)` — RNA → Protein via Standard-Codon-Tabelle
  - `k_mer_count(seq, k)` — alle ueberlappenden k-Mere mit Counts
- **Cheminformatik via `pyimport rdkit`:**
  - `smiles_descriptors(smiles)` liefert Dict mit `mw` ([g/mol]), `logp`, `num_atoms`, `num_heavy_atoms`, `num_rings`, `num_aromatic_rings`, `hbd`, `hba`, `tpsa`, `num_rotatable_bonds`. MW kommt als `Quantity` zurueck — direkt nutzbar in unit-aware Berechnungen.
  - `lipinski_rule_of_five(smiles)` prueft die vier Drug-Likeness-Kriterien (MW≤500, LogP≤5, HBD≤5, HBA≤10) und liefert `{checks, violations, passes}`.
- **Zentrales Dogma als Beispiel:**
  ```dedekind
  fn dna_to_protein(dna: Sequence[DNA]) -> Sequence[Protein] {
      rna = transcribe(dna)
      return translate(rna)
  }
  protein = dna_to_protein("ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTG")
  // "MALWMRLLPL" (Insulin-Signalpeptid-Anfang)
  ```
- **Bewusst NICHT geliefert:** kein Sequence-Alignment (Smith-Waterman, Needleman-Wunsch) als nativen Built-in — Forscher rufen `pyimport Bio.Align as aln` direkt auf. Kein PDB-/Strukturen-Parsing — `pyimport Bio.PDB`. Keine phylogenetischen Baeume. Dedekinds Rolle: Typsicherheit + Quick-Wins, nicht Replacement von Biopython/rdkit.
- Beispiel: `bioinformatics_demo.ddk` (DNA-Pipeline, k-Mer-Analyse, Aspirin/Coffein/Ibuprofen SMILES + Lipinski). Test: `bioinformatics_test.ddk` (18 Asserts). 42/42 Tests gruen, 98/98 Beispiele kompilieren.

### What's New in v1.15.0

- **`LabeledTensor[lat, lon, time]`-Shape-Annotation:** Tensoren mit Achsen-NAMEN statt Groessen — fuer Klima-, Geo- und Earth-Science-Workflows. Validiert zur Laufzeit, dass ein xarray.DataArray genau diese Dimension-Namen traegt (Reihenfolge irrelevant, weil xarray namens-basiert operiert):
  ```dedekind
  fn temperature_anomaly(t: LabeledTensor[lat, lon, time]) -> LabeledTensor[lat, lon, time] {
      return t - t.mean(dim="time")
  }
  fn zonal_mean(t: LabeledTensor[lat, lon, time]) -> LabeledTensor[lat, time] {
      return t.mean(dim="lon")
  }
  ```
  Fehlt eine Dimension oder ist eine zusaetzliche da: `ValueError: LabeledTensor-Shape-Mismatch ... fehlende Dimensionen: ['time']; zusaetzliche Dimensionen: ['level']`.
- **`labeled_tensor(data, dims, coords, name, attrs)`-Built-in:** erzeugt ein `xarray.DataArray` direkt in Dedekind. Akzeptiert Tensoren, numpy-Arrays oder Listen; haengt Achsen-Namen, Koordinaten und Meta-Attribute (units, CRS, source, ...) an:
  ```dedekind
  T = labeled_tensor(raw_data,
      dims=["lat", "lon", "time"],
      coords={"lat": lats, "lon": lons, "time": times},
      attrs={"units": "K", "crs": "EPSG:4326"}
  )
  ```
- **Der USP gegenueber rohem xarray:** xarray hat selbst kein Typsystem — `data_array.mean(axis=2)` statt `dim="time"` ist ein klassischer Bug in der Klimaforschung. `LabeledTensor[...]` erzwingt die richtigen Achsen-Namen am Funktions-Eingang und Return.
- **Bewusst NICHT geliefert:** kein Re-Implementieren von xarray-Operationen (regridding, interp_like, groupby_bins, etc.) — wer das braucht, ruft sie direkt ueber `da.regrid(...)` auf. Dedekinds Rolle ist die Annotations-Schicht. **Kein Dask-/distributed support** — zarr-/Dask-Backed-DataArrays funktionieren via `pyimport xarray` weiterhin, aber wir validieren nur die dim-Namen, nicht Chunk-Topologie.
- Beispiel: `labeled_tensors_demo.ddk` (4 x 8 x 12 Klima-Datensatz: temporal mean, zonal mean, Anomalie, `.sel`-Slicing nach Koordinate). Test: `labeled_tensors_test.ddk` (9 Asserts: Shape, Dim-Namen, Reihenfolge-Insensitivitaet, Coords, Attrs). 41/41 Tests gruen, 97/97 Beispiele kompilieren.

### What's New in v1.14.0

- **Molekulardynamik via OpenMM-Bruecke:** Neue Built-in `md_simulate_lj(n_particles, sigma, epsilon, mass, temperature, dt, n_steps, box_size, friction, seed)` startet eine Lennard-Jones NVT-Simulation (Langevin-Integrator) auf dem OpenMM-C++-Kernel — **mit erzwungener Dimensionssicherheit** vor dem Aufruf.
  ```dedekind
  res = md_simulate_lj(
      n_particles=27,
      sigma=3.4[Angstrom],         // alternative Eingabe in A
      epsilon=0.238[kcal/mol],     // oder kJ/mol
      mass=39.948[amu],
      temperature=85[K],
      dt=1[fs],
      n_steps=200,
      box_size=2.0[nm],
      seed=42
  )
  // res["potential_energy"] : Quantity in [kJ/mol]
  // res["kinetic_energy"]   : Quantity in [kJ/mol]
  // res["temperature"]      : Quantity in [K]
  // res["positions"]        : torch.Tensor (n_particles, 3) in nm
  ```
- **USP gegenueber rohem `pyimport openmm`:** Dedekind validiert die Dimensionen ALLER Force-Field-Parameter, bevor der C++-Kernel laeuft. `epsilon=0.238[eV]` wirft `ValueError: epsilon=0.238[eV] hat falsche Dimension; erwartet kompatibel zu [kJ/mol] (molar_energy)`. In rohem OpenMM ist eV-vs-kcal/mol-Verwechselung ein stiller Bug.
- **Neue MD-Einheiten im Dimensionssystem:**
  - Laenge: `nm`, `Angstrom`, `pm`, `fm`
  - Masse: `amu`, `Da` (Dalton)
  - Zeit: `fs`, `ps`, `ns`, `us`
  - Neue Dimension `molar_energy`: `kJ/mol` (Basis), `kcal/mol` (4.184), `J/mol`, `eV/atom`, `Hartree/mol`
- **Gitter-Initialisierung statt Random-Placement:** vermeidet NaN-Energien durch Teilchen-Ueberlappungen (LJ-r⁻¹² explodiert bei r → 0). Leichte Stoerung gegen perfekte Gitter-Symmetrie.
- **Bewusst NICHT geliefert:** keine Protein-Force-Fields (AMBER, CHARMM, OPLS), kein implicit solvent, keine REMD/free-energy-Methoden. Wer das braucht: `pyimport openmm.app as omm_app` — wir bauen die einfache LJ-Routine als Brueckenkopf, nicht als Replacement von OpenMM.
- Beispiel: `md_lennard_jones_demo.ddk` (Argon-Cluster, Equilibrierung + Produktion). Test: `md_simulate_lj_test.ddk` (8 Asserts: Shape, Energien, Temperatur, Angstrom/kcal-Alternativeingabe). 40/40 Tests gruen, 96/96 Beispiele kompilieren.

### What's New in v1.13.0

- **Operations Research mit deklarativer MILP-DSL:** Neue Built-ins `Variable(name, lower, upper, integer)` und `optimize_milp(objective, constraints, sense)`. Constraints werden ueber Operator-Overloading direkt geschrieben — keine A_ub/b_ub-Matrizen mehr selber bauen. Beispiel:
  ```dedekind
  x = Variable("strecke", lower=0[km])
  trucks = Variable("trucks", lower=1, integer=true)
  cost = 2.5 * x + 1000 * trucks
  res = optimize_milp(cost, [
      x >= 500[km],
      trucks * 200[km] >= x
  ])
  // res = {strecke: 500.0, trucks: 3.0, objective: 4250.0, status: "..."}
  ```
- **Unit-Awareness in Constraints — der USP:** `x >= 500[km]` mit `x: Variable(lower=0[km])` passt; `x >= 500[kg]` wirft `ValueError: MILP-Einheiten passen nicht in Constraint: [km] vs [kg]`. Keine andere MILP-Bibliothek (Gurobi, cvxpy, pyomo, PuLP) hat das.
- **Operator-Overloading:** `_MILPVariable` ueberlaedt `+`, `-`, `*`, `/`, `>=`, `<=` mit Linearitaets-Check (nichtlineares `x * y` wirft sofort `ValueError`). Variablen sind Identitaets-hashbar, damit Coefficient-Dicts intern funktionieren.
- **Drei Demos in `optimize_milp_demo.ddk`:** Vehicle Routing (Strecke + Truck-Anzahl), Produktions-Mix (max Profit unter Ressourcen-Constraints), Energie-Mix (billigste Quelle zuerst unter Bedarfs-Constraint).
- **Bestehender `milp(c, A_ub, b_ub, ...)`-Aufruf (v1.5) bleibt unveraendert** — die DSL ist eine Komfort-Schicht, kein Replacement.
- Test: `optimize_milp_test.ddk` (13 Asserts: LP-Min/Max, Integer-Variablen, Vehicle Routing mit km, Energie-Mix mit kW). 39/39 Tests gruen, 95/95 Beispiele kompilieren.

### What's New in v1.12.0

- **`Graph[N, E]` als Shape-Annotation:** Funktionssignaturen erkennen jetzt einen Graph-Typ ueber den bereits in v1.9 etablierten Annotations-Mechanismus:
  ```dedekind
  pyimport torch_geometric.data as pyg_data
  fn coordination(g: Graph[N, E]) -> Scalar { return g.num_edges / g.num_nodes }
  fn pair_match(g1: Graph[N, E1], g2: Graph[N, E2]) -> Scalar { ... }
  ```
  Validiert zur Laufzeit, dass das uebergebene Objekt `num_nodes`/`num_edges`-Attribute hat (typisch `torch_geometric.data.Data`) und dass die Dimensionen passen. Symbolische Dimensionen werden gebunden und konsistent gehalten — zwei Graphen mit `Graph[N, ...]` und `Graph[N, ...]` muessen die gleiche Knotenzahl haben.
- **Kombinierbar mit Unit-Annotationen:** Knoten-Features in `[g/mol]`, Kanten-Energien in `[eV]`, Bindungslaengen in `[pm]` — die dimensionale Sicherheit, die `torch_geometric` strukturell nicht hat. Beispiel:
  ```dedekind
  fn molecular_mass(g: Graph[N, E]) -> [g/mol] { ... }
  fn add_mass(m1: [g/mol], m2: [g/mol]) -> [g/mol] { ... }
  ```
- **Zwei GNN-Demos via `pyimport torch_geometric`:**
  - `gnn_molecule_demo.ddk` — Wirkstoffdomaene: H2O-Molekuel mit atomarer Masse in `[g/mol]` auf Knoten, GCNConv-Forward.
  - `gnn_materials_demo.ddk` — Materialdomaene: 4-Atom-FCC-Einheitszelle mit Bindungslaengen in `[pm]` und Bindungsenergien in `[eV]`, GraphConv.
- **Bewusst NICHT geliefert** (Stufe 3 der Graph-Roadmap): Kein natives Message-Passing. Die ehrliche Antwort fuer produktive GNNs bleibt `pyimport torch_geometric` — dort sind 30+ Conv-Varianten, Pooling-Strategien und Benchmark-Datasets. Dedekinds Rolle: Unit-Awareness, Shape-Annotation, Source-Mapping ueber den PyG-Aufruf legen.
- Test: `graph_shape_test.ddk` (5 Asserts: Knoten-/Kanten-Counts, symbolische N-Konsistenz ueber zwei Graphen). 38/38 Tests gruen, 94/94 Beispiele kompilieren.

### What's New in v1.11.0

- **`graph_laplacian(adj, normalized=False)` — Spektrale Graph-Methoden:** Neue Built-in fuer die diskrete Laplace-Matrix eines Graphen. Akzeptiert dichte Matrizen, sparse `torch.Tensor` und verschachtelte Listen als Adjazenz; gibt sparse zurueck wenn die Eingabe sparse ist, sonst dicht.
  - **Kombinatorisch** (Default): `L = D - A`  (Zeilensummen = 0, alle Eigenwerte >= 0).
  - **Normalisiert symmetrisch**: `L_sym = I - D^{-1/2} A D^{-1/2}`  (Eigenwerte in `[0, 2]`).
  - Direkt einsetzbar in `cg`, `gmres`, `bicgstab` und `eigh` aus den vorhandenen v1.6-Solvern — kein zusaetzliches Framework noetig.
- **Demo `graph_spectral_demo.ddk`:** zwei klassische Anwendungen auf einem 8-Knoten-Zwei-Cluster-Graphen:
  - **Heat-Diffusion** ueber implizites Euler-Verfahren: `cg(I + dt*L, u_prev)`. Zeigt deutlich asymmetrische Diffusion ueber die schmale Brueckenkante.
  - **Spektrale Partitionierung** via Fiedler-Vektor (zweitkleinster Eigenwert): trennt die zwei Cluster sauber an der Bruecke (Vorzeichen-Aufteilung).
- **Bewusst NICHT geliefert** (Stufe 2+3 der Graph-Roadmap):
  - Kein `Graph[N, E]`-Shape-Typ. Kommt in v1.12 als Wrapper um `torch_geometric.data` mit Unit-Annotationen — das ist der echte USP gegenueber PyG.
  - Kein natives Message-Passing. Anti-Muster: ein halb-fertiges PyG-Rebuild waere schlechter als `pyimport torch_geometric`. Wenn nativ, dann mit dem ganzen Forschungs-Budget (Stufe 3, mehrere Wochen).
- Test: `graph_laplacian_test.ddk` (Pfad-Graph, Zwei-Cluster, normalisierte Variante, Fiedler-Partitionierung). 37/37 Tests gruen, 92/92 Beispiele kompilieren.

### What's New in v1.10.0

- **`partial(u, x, order=n)` — Physics-Informed Neural Networks (PINNs):** Neue Built-in fuer die Ableitung von Netz-Outputs nach Netz-Inputs via Autograd. Anders als das bestehende `grad(fn, x)` (das eine *Funktion* an einer Stelle differenziert) arbeitet `partial` mit bereits berechneten Tensoren: `u = net(x)` wird ausgewertet, dann `du_dx = partial(u, x)`. Damit lassen sich PDE-Residuen (u_t - α·u_xx, u_x + u, ...) direkt in der Loss-Funktion ausdruecken. Funktioniert mit `create_graph=True` und `retain_graph=True`, sodass `fit()` ueber das Residuum optimieren kann — das ist genau der Mechanismus, der Forschern in PyTorch sonst wochenlange Custom-Loop-Implementierung kostet.
- **`fit()`-Patch fuer PINNs:** `_to_tensor` faengt nun gemischte Tensor-Listen ab (PINN-Daten + Collocation-Tensoren mit unterschiedlichen Shapes); `fit()` zeroed pro Schritt die `.grad`-Akkumulatoren aller `requires_grad`-Tensoren in `data` (verhindert Speicher-Leak bei langen MCMC/PINN-Trainings).
- **Zwei PINN-Beispiele:**
  - `pinn_ode_demo.ddk` — 1. Ordnung ODE `y' + y = 0, y(0)=1`. PINN lernt exp(-x) in 2000 Schritten mit ~1% Fehler ohne je einen Loesungswert zu sehen.
  - `pinn_oscillator_demo.ddk` — 2. Ordnung ODE `u'' + u = 0, u(0)=1, u'(0)=0` auf [0, π/2]. Demonstriert `partial(u, x, order=2)`. Fehler < 1% in 5000 Schritten.
- **Beide Beispiele transparent zu Hyperparameter-Limits:** Auf groesseren Intervallen oder mit groesseren Frequenzen scheitern naive PINNs — das ist Stage 3 der Roadmap (NTK-basierte Loss-Balancierung, adaptives Sampling, Fourier-Features), nicht v1.10. Wir liefern bewusst kein „magisches" `.with_physics_loss(pde)`, das bei Forschern Erwartungen weckt, die wir nicht halten koennen.
- Test: `partial_test.ddk` (17 Asserts: x², sin, kubisch, 2-D-Eingang); 36/36 Tests gruen, 91/91 Beispiele kompilieren.

### What's New in v1.9.0

- **Shape-Annotationen fuer Tensoren:** Funktionssignaturen erlauben jetzt explizite Tensor-Shapes — als statische Garantie fuer Forschungs-Code, der sonst still unter Broadcasting leidet:
  ```dedekind
  fn dot_product(a: Vector[3], b: Vector[3]) -> Scalar { ... }
  fn matvec(M: Matrix[2, 3], v: Vector[3]) -> Vector[2] { ... }
  fn weighted_dot(x: Vector[N], w: Vector[N]) -> Scalar { ... }  // symbolisches N
  fn forward(x: Tensor[batch, 28, 28]) -> Tensor[batch, 10] { ... }
  ```
  Vier Typkonstruktoren: `Scalar` (0-D), `Vector[n]` (1-D), `Matrix[m, n]` (2-D), `Tensor[d1, d2, ...]` (N-D). Dimensionen sind Integer-Literale **oder** Identifier (symbolisch, an den Caller gebunden). Symbolische Dims werden beim ersten Auftreten gebunden und auf Konsistenz geprueft — `weighted_dot(Vector[3], Vector[2])` wirft `ValueError: Symbolische Shape-Dimension 'N' in weighted_dot(w): bereits als 3 gebunden, hier 2.` Funktioniert auf Listen, Tuples und `torch.Tensor`. Return-Shape wird automatisch nach jeder `return`-Anweisung geprueft. Erkennt fehlerhafte Broadcasts und Form-Mismatches, die in reinem NumPy/PyTorch still falsche Ergebnisse produzieren.
- **`unwrap(x)` — Quantity-Stripping fuer Hot-Paths:** Neue Built-in entfernt Einheits-Wrapper am Eingang von pure-context-Funktionen, damit `jit`/`grad`/`fit`/`metropolis`/`hmc`/`sde_solve` mit nackten Floats arbeiten koennen:
  ```dedekind
  fn pure_loss(params, data) {
      a = unwrap(params[0])   // Quantity(2.0, "m") -> 2.0
      b = unwrap(params[1])
      x = unwrap(data[0])
      diff = a * x + b - unwrap(data[1])
      return diff * diff
  }
  ```
  Verarbeitet `Quantity`, `UncertainQuantity` (std verworfen), 0-d `torch.Tensor` (via `.item()`), Listen/Tuples (elementweise) und beliebige andere Werte (passthrough). Die Compile-Zeit-Einheitenpruefung hat die Dimensionen bereits validiert; zur Laufzeit erzeugt der nackte Float keinen Wrapper-Overhead — wichtig bei 10 000+ Iterationen in MCMC-Loops oder JIT-kompilierten Graphen, wo `torch.compile` Python-Objekte als Graph-Break behandelt.
- Beispiele: `shape_annotations_demo.ddk`, `quantity_stripping_demo.ddk`. Tests: `shape_annotations_test.ddk`, `quantity_stripping_test.ddk` (35/35 gruen; alle 89 Beispiele kompilieren).

### What's New in v1.8.1

- **Purity-Check fuer pure-context-Aufrufe:** Neuer Compile-Zeit-Pass `purity_check.py`. Funktionen, die an `jit(fn)`, `grad(fn, x)`, `fit(loss, ...)`, `metropolis(log_prior, log_likelihood, ...)`, `hmc(...)` oder `sde_solve(drift, diffusion, ...)` uebergeben werden, duerfen **keine I/O-/Konsolen-Built-ins** mehr aufrufen — sonst `CompileError` mit Datei + Zeile.
- Blockierte Built-ins (transitive Erkennung): `print`, `plot`, `scatter`, `contour`, `print_latex`, `print_table`, `write_file`, `read_file`, `file_exists`, `http_get`, `http_post`, `read_csv`/`write_csv`, `read_parquet`/`write_parquet`, `read_hdf5`/`write_hdf5`, `read_netcdf`, `export_notebook`.
- Transitive Aufloesung: ruft `loss` eine Hilfsfunktion `helper` auf und `helper` ruft `print`, wird der gesamte Pfad gemeldet (`"... ruft 'print()' ... in 'helper'"`).
- Opt-Out: `compile_source(..., check_purity=False)` bzw. CLI `python -m compiler datei.ddk --no-purity-check`.
- Verhindert eine ganze Bug-Klasse: stille `torch.compile`-Graph-Breaks, Tape-Recording-Anomalien in Autograd, mehrfache Datei-Writes in MCMC-Loops mit 10 000 Samples.
- Beispiel: `purity_check_demo.ddk`. Test: `purity_check_test.ddk` (33/33 gruen).

### What's New in v1.8.0

- **Source-Mapping fuer Runtime-Fehler:** Wenn der generierte Python-Code (oder eine darunter liegende NumPy/SciPy/Torch-Funktion) zur Laufzeit eine Exception wirft, zeigt der Traceback nun direkt die Original-Zeilen aus der `.ddk`-Datei inkl. Code-Auszug:
  ```
  Traceback (most recent call last):
    File "demo.ddk", line 16, in <module>
      main()
    File "demo.ddk", line 12, in main
      s = outer(data)
    File "demo.ddk", line 7, in inner
      return arr[0] + arr[1] + arr[99]
  IndexError: index 99 is out of bounds for dimension 0 with size 3
  ```
  Implementiert via `# ddk:<line>`-Marker im generierten Code + neuer Helper `dedekind_exec(generated_code, ddk_file, ddk_source)` in `src/compiler/compiler.py`. Der Exception-Typ bleibt erhalten (`except AssertionError:` etc. funktioniert weiter), nur die Nachricht wird mit dem zurueckgemappten Traceback ersetzt. Frames innerhalb der inlinierten Runtime werden als `<dedekind-runtime>` markiert; externe Library-Frames (scipy, torch) zeigen weiterhin ihre echten Pfade fuer Stack-Analyse. Genutzt in `run_tests.py`, `run_examples.py`, `compiler.py` CLI und dem Jupyter-Kernel.
- **`pyimport` — Python-Fluchtluke ins PyPI-Oekosystem:** Neue Top-Level-Syntax `pyimport scipy.special as sp` (oder `pyimport math` mit Auto-Alias = letztes Segment). Beliebige Python-Module sind als `alias.name(...)` aufrufbar, mischen sich mit Dedekind-Built-ins. Beispiel:
  ```dedekind
  pyimport scipy.stats as st
  pyimport numpy as np
  p = st.norm.cdf(1.96)
  arr_mean = np.mean(np.array([1.0, 2.0, 3.0]))
  ```
  `pyimport` ist Soft-Keyword (bestehender Code mit `pyimport` als Variablenname bleibt gueltig). Codegen emittiert `import MODULE as ALIAS` an der Stelle des Statements. Damit ist Dedekind ab v1.8 nicht mehr „walled garden": jedes PyPI-Paket (`rdkit`, `astropy`, `qiskit`, `polars`, ...) ist sofort verfuegbar.
- **`_dedekind_assert`-Fix:** Korrekte Behandlung von numpy-Skalaren (`numpy.bool_`, `numpy.float64`). Vorher kollidierte der Tensor-Check (`numel()`) mit numpy-Werten aus `pyimport`-Aufrufen.
- Beispiele: `pyimport_demo.ddk`, `source_mapping_demo.ddk`. Tests: `pyimport_test.ddk` (32/32 gruen; alle 86 Beispiele kompilieren).

### What's New in v1.7.0

- **Standardbibliothek aus echten Modulen:** Sechs kuratierte `.ddk`-Module unter `modules/` ergänzen die globalen Built-ins um Convenience-Funktionen. **Wichtig:** Alle bestehenden Built-ins (`sin`, `ode_solve`, `michaelis_menten`, `pi`, `c`, ...) bleiben **weiterhin global** verfügbar – `use` ist rein additiv und macht weder Imports verpflichtend noch verschwindet etwas hinter Namespaces.
  - `use physics` — `kinetic_energy`, `pendulum_period`, `escape_velocity`, `orbital_period`, `relativistic_gamma`, `coulomb_force`, `ideal_gas_pressure`, `rms_speed`, `blackbody_radiance`, `doppler_classical`, … (+ dimensionslose Numerik-Konstanten `C_NUM`, `G_NUM`, `K_B_NUM`, `R_GAS_NUM`, `SIGMA_SB_NUM`)
  - `use stats` — `z_score`, `cohens_d`, `pooled_std`, `hedges_g`, `glass_delta`, `r_squared`, `mse`/`rmse`/`mae`, `standard_error_mean`, `ci_normal_mean`, `ci_proportion`, `coefficient_of_variation`, `standardize`
  - `use chemistry` — `pH_from_concentration`, `henderson_hasselbalch`, `ka_from_pka`, `dilution_volume`, `boyle_volume`, `charles_volume`, `combined_gas_law_v2`, `van_der_waals_pressure`, `nernst_potential`, `faraday_mass`, `half_life_first_order`/`half_life_second_order`
  - `use biology` — `exponential_growth`, `doubling_time`, `gompertz_growth`, `von_bertalanffy`, `kleibers_law`, `allometric_scaling`, `bmr_harris_benedict`, `bmi`, `hardy_weinberg_freq`, `mutation_drift_balance`, `r_naught_sir`, `herd_immunity_threshold`
  - `use math` — `PHI`, `TAU`, `fibonacci`, `harmonic_sum`, `geometric_sum`, `lcm`, `is_perfect_square`, `digital_root`, `circle_area`, `sphere_volume`, `hypotenuse`, `law_of_cosines_c`, `lerp`, `clamp_scalar`, `sigmoid`, `softplus`
  - `use ml` — `leaky_relu`, `elu`, `swish`, `gelu_approx`, `mse_loss`, `mae_loss`, `binary_crossentropy`, `accuracy`, `precision_binary`, `recall_binary`, `f1_score`
- **Benutzerdefinierte Einheiten:** `unit NAME = FAKTOR[basis]` registriert eine neue Einheit zur Compile- und Laufzeit. Beispiele: `unit Foot = 0.3048[m]`, `unit Mile = 1609.344[m]`, `unit eV = 1.602176634e-19[J]`, `unit kcal = 4184.0[J]`. Verkettung erlaubt (`unit MicroInch = 1e-6[Inch]`), wird transitiv aufgelöst. Built-in und User-Units mischen sich mit automatischer Umrechnung (`1[Mile] + 1500[m]` ergibt `1.93[Mile]`). `unit` ist ein **Soft-Keyword** – bestehender Code mit `q.unit`, `unit="V"`-Kwarg etc. bleibt unverändert gültig.
- **Quantity-Vergleichsoperatoren:** `Quantity` unterstützt jetzt `<`, `<=`, `>`, `>=`, `==`, `!=` mit automatischer Einheiten-Konvertierung bei gleicher Dimension. `10[cm] < 1[m]` → `True`. Macht Quantity-Werte in `if`/`while`-Bedingungen direkt nutzbar.
- Beispiele: `stdlib_modules_demo.ddk` (alle 6 Module), `user_defined_units.ddk` (Foot, Mile, AU, ly, Bohr, eV, kcal, uDarcy). Tests: `stdlib_physics_test.ddk`, `stdlib_stats_test.ddk`, `stdlib_chemistry_test.ddk`, `stdlib_biology_math_ml_test.ddk`, `user_defined_units_test.ddk` (31/31 grün; alle 84 Beispiele kompilieren).

### What's New in v1.6.0

- **Tiefere Symbolik:** `solve_sym(equation, var)` löst Gleichungen (auch Systeme mit Listen) symbolisch via SymPy; `"x^2 - 5*x + 6"` → `["2", "3"]`. `simplify_sym(expr)` vereinfacht Ausdrücke (`"sin(x)^2 + cos(x)^2"` → `"1"`). `series(expr, var, x0, n)` liefert Taylor-Entwicklungen. Ergänzt `diff_sym`/`integrate_sym`.
- **Sparse iterative Solver:** `cg(A, b)`, `gmres(A, b)`, `bicgstab(A, b)` als Krylov-Solver für große (sparse oder dichte) lineare Systeme, mit Iterations-Callback. `jacobi_preconditioner(A)` und `ilu_preconditioner(A)` als `M=`-Argument für 2–10×-Speedup. Dichte Matrix, sparse Tensor und scipy.sparse-Matrizen werden alle akzeptiert (intern auf CSR-float64 normalisiert).
- **Reproducible-Notebook-Export:** `export_notebook(source_path, output_path, format="html"|"md", title, include_hash=True)` führt eine `.ddk`-Datei aus und bündelt Quellcode, Stdout-Output, alle generierten Plots (Base64-PNG) und SHA-256-Hash zu einer Standalone-Datei. Re-Entry-Guard verhindert Endlosrekursion, wenn die Quelldatei sich selbst exportiert.
- **Paper-Mode-Tabellen:** `print_table(rows, headers, format="markdown"|"latex"|"csv"|"plain", precision, caption, label)` erzeugt Tabellen in vier Formaten; LaTeX nutzt Booktabs (`\toprule`/`\midrule`/`\bottomrule`). `UncertainQuantity` wird automatisch als `val ± std [unit]` formatiert (in LaTeX: `$val \pm std\,[\mathrm{unit}]$`), `Quantity` als `val [unit]`. Akzeptiert `DataFrame` direkt; Einheiten aus `df.units` landen in Header.
- Beispiele: `symbolic_solve_series.ddk`, `sparse_iterative_solvers.ddk`, `notebook_export_demo.ddk`, `paper_table_demo.ddk`. Tests: 26/26 grün; alle 82 Beispiele kompilieren.

### What's New in v1.5.0

- **Benchmarking & Profiling als Built-ins:** `benchmark(fn, n=10, warmup=2, label="...")` misst Wandzeit über n Wiederholungen (Mittelwert ± Std, Min/Max); `profile(fn)` liefert zusätzlich Peak-Speicher (`tracemalloc`) und Top-Funktionen (`cProfile`); `time_block(label, fn)` für Ad-hoc-Messungen. Beispiel: `bm = benchmark(work, n=50)`.
- **JIT-Backend:** `jit(fn)` wrappt eine Funktion mit `torch.compile` (TorchInductor) wenn verfügbar, fällt sonst auf das Original zurück. Realistischer Zwischenschritt Richtung AOT; nutzt denselben Compiler-Stack wie reines PyTorch.
- **SDE-Solver:** `sde_solve(drift, diffusion, y0, t, method="euler_maruyama"|"milstein", seed_value=None)` für Itô-SDEs `dY = drift(t,Y) dt + diffusion(t,Y) dW`. Euler-Maruyama (Ordnung 0.5) und Milstein (Ordnung 1, mit numerischer Ableitung der Diffusion).
- **Erweiterte Optimierung:** `least_squares(residuals, x0, jacobian=None, bounds=None, method="trf")` für nichtlineare Kleinste-Quadrate (mit float32-stabiler Default-Schrittweite); `minimize_constrained(f, x0, constraints=[{"type":"ineq","fun":g}], bounds=...)` für SLSQP/trust-constr/COBYLA; `milp(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality)` für (gemischt-)ganzzahlige LPs.
- **FEM-Primitiven:** `mesh_unit_square(n)` erzeugt strukturiertes Dreiecksgitter mit Knoten/Elementen/Rand; `fem_assemble_stiffness(mesh)`, `fem_assemble_load(mesh, f)` für lineare Galerkin-Assemblierung; `fem_poisson_2d(mesh, f, dirichlet_value=0)` löst -Δu=f mit Dirichlet-Randwert.
- **`arange` für Indexierung:** `arange(n)` und `arange(start, stop)` liefern jetzt int64 (vorher float32); macht `for i in arange(N) { x[i] = ... }` direkt nutzbar. Float-Schritt-Variante (`arange(0, 10, 0.5)`) bleibt float32. Beispiel: `v1_5_features_showcase.ddk`. Tests: `benchmark_profile_test.ddk`, `jit_test.ddk`, `sde_solve_test.ddk`, `optimization_test.ddk`, `fem_test.ddk`.

### What's New in v1.4.0

- **Modul-System:** `use mymodule` lädt `modules/mymodule.ddk` (oder dieselbe Verzeichnis-Datei) und stellt deren Funktionen/Konstanten zur Verfügung. Beispiel: `use mathlib` → `square`, `cube`, `PHI`. Suchpfade: aktuelles Verzeichnis, `modules/`, `examples/dedekind/`, CWD.
- **Reproduzierbarkeit:** `seed(n)` setzt deterministischen Seed in `random`, NumPy und PyTorch. `data_hash(x)` liefert SHA-256-Digest beliebiger Eingaben (Tensor, Liste, Dict, DataFrame, Zahl, String) für reproduzierbare Pipelines.
- **DataFrames + tabular I/O:** Leichte spaltenorientierte `DataFrame`-Klasse mit Einheiten pro Spalte; `read_csv(path)` parst Header der Form `name [unit]` automatisch; `write_csv(path, df)`. Optional: `read_parquet`/`write_parquet` (pyarrow), `read_hdf5`/`write_hdf5` (h5py), `read_netcdf` (netCDF4).
- **Unit-aware Plots:** `plot()`, `scatter()`, `contour()` erkennen Listen von `Quantity`-Werten, extrahieren Zahlenwerte und ergänzen Einheiten automatisch in den Achsenbeschriftungen (`Zeit [s]`, `Temperatur [K]`).
- **`@units`-Signaturen:** Funktionen können Argument- und Return-Einheiten deklarieren: `fn kinetic_energy(m: [kg], v: [m/s]) -> [J] { ... }`. Eingaben werden automatisch in die deklarierte Einheit umgerechnet (z. B. `2000[g]` → `2[kg]`); Return-Wert wird dimensional geprüft (z. B. `kg*m²/s² == J`).
- **Dict-Literale:** `{"key": value, "k2": v2}` als Ausdruck (z. B. für `DataFrame`-Konstruktion oder `json_stringify`).
- Beispiel: `v1_4_features_showcase.ddk`. Tests: `use_module_test.ddk`, `seed_reproducibility_test.ddk`, `dataframe_csv_test.ddk`, `signature_units_test.ddk`, `unit_plot_test.ddk`.

### What's New in v1.3.1

- **Medizin, Pharmakologie & Epidemiologie:** `hill_equation`, `one_compartment_pk`, `half_life` (Pharmakokinetik); `sir_model`, `basic_reproduction_number` (Epidemiologie); `confidence_interval`, `odds_ratio`, `sensitivity_specificity` (Biostatistik). Beispiele: `pharmacology_quickwins.ddk`, `epidemiology_sir.ddk`, `biostatistics_quickwins.ddk`.

### What's New in v1.3.0

- **Unbestimmte Integrale:** `integrate_sym(expr, var)` – symbolische Integration; nutzt SymPy. Beispiel: `integrate_sym_demo.ddk`.
- **Lagrange/Hamilton:** `lagrange_ode_rhs(L)`, `hamilton_ode_rhs(H)` – RHS für ode_solve aus L(q,v) bzw. H(q,p). Beispiel: `lagrange_hamilton.ddk`.
- **Lotka-Volterra:** `lotka_volterra(x0, y0, a, b, c, d, t)` – Räuber-Beute-Modell. Beispiel: `lotka_volterra.ddk`.
- **Chemisches Gleichgewicht:** `chemical_equilibrium(K, n_A, n_B, n_C, n_D, A0, B0, C0, D0)` – Massenwirkungsgesetz. Beispiel: `chemical_equilibrium.ddk`.

### What's New in v1.2.9

- **Betragsstriche:** `|expr|` = syntaktischer Zucker für `abs(expr)`; z. B. `x = |-1|` → 1. Beispiel: `abs_bars.ddk`.
- **Rotationskörper:** `volume_revolution_x`, `volume_revolution_y`, `volume_revolution_vertical`, `volume_revolution_horizontal`, `pappus_volume_vertical`, `pappus_volume_horizontal`. Beispiel: `volume_revolution.ddk`.
- **Logische Operatoren:** `and`, `or`, `not`, `xor`, `nand`, `nor`, `xnor` als Keywords. Beispiel: `logical_operators.ddk`.

### What's New in v1.2.8

- **Dedekind-Schnitte:** `DedekindCut(x)` – Konstruktion der reellen Zahlen aus Q; `dedekind_cut_from_rational(p,q)`, `dedekind_cut_sqrt2()`; `lower_set_contains(cut,q)`, `to_float()`; Arithmetik und Vergleiche.
- **Dedekind-Ringe:** `DedekindRingZ()`, `ideal(n)`, `ideal_factor(i)` – Z mit eindeutiger Ideal-Faktorisierung; `DedekindIdeal` mit `.factor()`, `.norm()`, `*`.
- **Riemann-Zeta-Funktion:** `zeta(s)` – ζ(s)=Σ 1/n^s (scipy); ζ(2)=π²/6, ζ(4)=π⁴/90.
- **Riemann-Summen:** `riemann_sum(f, a, b, n, method="left"|"right"|"midpoint")` – Approximation von ∫f dx. Beispiele: `dedekind_cuts_rings.ddk`, `riemann_zeta_sums.ddk`.

### What's New in v1.2.7

- **Dirichlet-Verteilung:** `Dirichlet(alpha)` – multivariate Verteilung auf dem Simplex (z. B. Topic-Modeling); `alpha` als Liste oder 1D-Tensor; `sample(dist)`, `log_prob(dist, value)` wie bei anderen Verteilungen.
- **Dirichlet-Funktion:** `dirichlet_function(x)` – D(x)=1 wenn x rational (Nenner ≤10000, Toleranz 1e-6), sonst 0; elementweise für Skalar oder Tensor. Beispiel: `dirichlet_distribution_function.ddk`.

### What's New in v1.2.6

- **Winkel als native Einheiten:** `rad` und `deg` mit automatischer Umrechnung bei Addition/Subtraktion (z. B. `90[deg] + (pi/2)*1[rad]` → `180[deg]`). Konvertierungsfunktionen: `deg_to_rad(x)`, `rad_to_deg(x)` für Skalar, Tensor oder Quantity. Beispiel: `angle_units.ddk`.

### What's New in v1.2.5

- **Quick Wins (vernachlässigte Wissenschaften):** Einheiten kN, MPa, MN, kPa, D, mD (Bau, Werkstoffe, Geologie). Musik: `cents_to_ratio`, `ratio_to_cents`, `equal_temperament`. Ökonomie: `discount_factor`, `cobb_douglas`, `solow_rhs`. Geologie: `darcy_velocity`. Werkstoffe: `johnson_mehl_avrami`, `avrami_rate`. Beispiele: `quickwins_units.ddk`, `music_intervals.ddk`, `economics_solow.ddk`, `geology_darcy.ddk`, `materials_jmak.ddk`. Parser: Wissenschaftliche Notation (1e5, 1e-12) korrekt geparst.

### What's New in v1.2.4
- **Maxwell-Gleichungen:** `pde_maxwell_1d(E0, B0, x, t, c_light)` (ebene Welle E_y, B_z); `pde_maxwell_2d(Ez0, Hx0, Hy0, x, y, t, c_light)` (TM-Mode E_z, H_x, H_y); FDTD mit zentralen Differenzen; Beispiel `pde_maxwell.ddk`.

### What's New in v1.2.3
- **Reaktions-Diffusion:** `pde_reaction_diffusion_1d(u0, x, t, D, r, reaction="fisher")` (Fisher-KPP u_t = D∇²u + r·u·(1-u)); `pde_reaction_diffusion_2d(u0, v0, x, y, t, Du, Dv, a, b)` (Gray-Scott, Turing-Muster); Beispiel `pde_reaction_diffusion.ddk`.
- **Advektions-Diffusion:** `pde_advection_diffusion_1d(u0, x, t, v, D)`, `pde_advection_diffusion_2d(u0, x, y, t, vx, vy, D)` für u_t + v·∇u = D∇²u; Upwind + zentrale Differenzen; Beispiel `pde_advection_diffusion.ddk`.

### What's New in v1.2.2
- **Wellengleichung:** `pde_wave_1d(u0, x, t, c, v0)`, `pde_wave_2d(u0, x, y, t, c, v0)` für u_tt = c²∇²u; Reduktion auf System 1. Ordnung; zentrale Differenzen; periodische oder dirichlet Randbedingungen; Beispiel `pde_wave.ddk`.
- **Burgers 2D:** `pde_burgers_2d(u0, x, y, t, nu)` für u_t + u·∇u = ν∇²u; Upwind für Advektion, zentrale Differenzen für Diffusion; periodische oder dirichlet Randbedingungen; Beispiel `pde_burgers.ddk` um 2D-Simulation und Konturplots erweitert.

### What's New in v1.2.1
- **Advektion:** `pde_advection_1d(u0, x, t, v)`, `pde_advection_2d(u0, x, y, t, vx, vy)` für u_t + v·∇u = 0; Upwind-Schema, periodische Ränder; Beispiel `pde_advection.ddk`.

### What's New in v1.2.0
- **Sparse CFD:** `sparse_laplacian_2d(N)`, `sparse_diffusion_step(T, L, dt, alpha)`, `sparse_diffusion_simulate(T0, n_steps, dt, alpha)` für echte 2D-Wärmediffusion ∂T/∂t = α∇²T.
- **Beispiel cfd_sparse_sim.ddk:** Echte Simulation (50×50 Gitter, 100 Zeitschritte, Konturplots) statt Platzhalter.
- **Compiler-Fix:** `tensor * 0` wird nur noch zu `0` vereinfacht, wenn beide Operanden Literale sind (Fix für `random_matrix(N,N)*0.0`).
- **Postfix-Fakultät:** Operator `n!` (z. B. `5!`, `n!`); AST PostfixFactorial, Lexer, Parser, Runtime `factorial(n)`; Beispiel `factorial_test.ddk`.

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
- **Medizin, Pharmakologie & Epidemiologie**: `hill_equation`, `one_compartment_pk`, `half_life` (Pharmakokinetik); `sir_model`, `basic_reproduction_number` (Epidemiologie); `confidence_interval`, `odds_ratio`, `sensitivity_specificity` (Biostatistik) — Beispiele: `pharmacology_quickwins.ddk`, `epidemiology_sir.ddk`, `biostatistics_quickwins.ddk`.

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
