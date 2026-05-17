# Was Dedekind aktuell kann

**Stand:** Basierend auf Code und Changelogs (v1.7.0, Mai 2026). Dedekind ist ein **Prototyp** – die Sprache wird nach Python transpiliert und nutzt PyTorch/NumPy als Laufzeit.

---

## Sprachkern

- **Syntax:** Imperativ, C/JavaScript-ähnlich mit Blöcken in `{}`, Funktionen mit `fn name(args) { ... }`, Kontrollfluss `if`/`else`, `while`, `for ... in`. **Betragsstriche:** `|x|` = `abs(x)` (z. B. `x = |-1|` → 1). **Logische Operatoren:** `and`, `or`, `not`, `xor`, `nand`, `nor`, `xnor` (Python-ähnliche Keywords; Präzedenz: `or` < `xor` < `and`/`nand`/`nor`/`xnor` < `not`).
- **Typen:** Dynamische Typinferenz; primitive Typen, Listen, Vektoren/Matrizen als Tensoren, Quaternionen; Property-Zugriff (z. B. `.shape`, `.gpu()`). **Matrix-Multiplikation:** Operator `@` (z. B. `A @ B`).
- **Ausführungsmodifikatoren:** `.gpu()`, `.cpu()`, `.single()` für Hardware-Platzierung bzw. sequentielle Ausführung; `.sparse()` für Sparse-Tensoren; `.fast()` für MLIR/Inductor-Optimierung (z. B. bei Modellen).
- **Fehlerbehandlung:** Compiler-Fehler mit Zeilennummer (`CompileError`); Einheiten-Check zur Compile-Zeit (`1[m] + 1[s]` → Fehler); Runtime-Meldungen mit Kontext. **Assert:** `assert(condition, message)` – löst AssertionError bei falscher Bedingung; Mini-Test-Runner `run_tests.py` für `tests/dedekind/*.ddk`.

---

## Mathematik

- **Konstanten:** `pi`, `e` (dimensionslos).
- **Funktionen:** `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; Arkus- und Hyperbelfunktionen (`asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`) – elementweise, differenzierbar.
- **Reduktionen & Runden:** `min`, `max`, `argmin`, `argmax` (optional `dim`); `round`, `floor`, `ceil`.
- **Statistik:** `mean(x)`, `std(x)`, `var(x)`, `median(x)` (optional `dim`); `quantile(x, q)`, `percentile(x, p)`; `cov(x, y)`, `corrcoef(x, y)` (Kovarianz/Korrelation, bei 2D x: Matrix); `skew(x)`, `kurtosis(x)` (Schiefe, Kurtosis); `histogram(x, bins, range_lim)` (Zählung in Klassen; range_lim optional (min, max)).
- **Lineare Algebra:** `norm(x)`, `det(A)`, `trace(A)`; `solve(A, b)` (Ax = b); `eigh(A)` (Eigenwerte/-vektoren symmetrisch); `eig(A)` (allgemein); `svd(A)` (Singulärwertzerlegung); `lstsq(A, y)` (Least Squares); `cond(A)`, `rank(A)`, `pinv(A)` (Kondition, Rang, Pseudo-Inverse); `expm(A)`, `logm(A)` (Matrix-Exponential/-Logarithmus). FFT (`fft`, `ifft`); Matrix-Operationen (transpose, inverse, dot_product). **Symbolische Ableitung:** `diff_sym(expr, x)` (expr, x als Strings; Rückgabe Ableitung als String). **Unbestimmte Integrale:** `integrate_sym(expr, var)` – symbolische Integration ∫ expr d(var); Stammfunktion als String; nutzt SymPy. **Autograd:** `grad(f, x)` (Gradient); `jacobian(f, x)` (Jacobi-Matrix); `hessian(f, x)` (Hesse-Matrix).
- **Numerik:** `interp(x, xp, fp)` (1D-lineare Interpolation); `trapz(y, x)`, `simpson(y, x)` (Trapez/Simpson für diskrete Daten); `riemann_sum(f, a, b, n, method="left"|"right"|"midpoint")` (Riemann-Summen für int f dx); `zeta(s)` (Riemann-Zeta ζ(s)=Σ 1/n^s, scipy); `volume_revolution_x`, `volume_revolution_y`; `volume_revolution_vertical(f,a,b,x0,n)` (Achse x=x0), `volume_revolution_horizontal(f,a,b,y0,n)` (Achse y=y0); `pappus_volume_vertical`, `pappus_volume_horizontal` (Satz von Pappus: V=2π·R·A); `root_bisect(f, a, b, tol)` (Nullstelle Bisektion); `newton(f, x0, tol)` (1D); `fsolve(f, x0)` (Vektor-Nullstelle Newton); `minimize(f, x0, method="gd"|"lbfgs")` (mehrdimensionale Minimierung). **Mathematische Folgen:** `arange(n)` bzw. `arange(start, stop, step)` (Integer-Folge); `arithmetic(a0, d, n)` (arithmetisch: aₙ = a₀ + n·d); `geometric(a0, r, n)` (geometrisch: aₙ = a₀·rⁿ); `sequence(f, n)` (allgemein: [f(0), f(1), …, f(n−1)]). **Weitere Algorithmen:** `qr(A)`, `lu(A)` (QR/LU-Zerlegung); `cholesky(A)`; `matrix_power(A, n)`; `kron(A, B)` (Kronecker-Produkt); `outer(a, b)` (äußeres Produkt); `vander(x, n)` (Vandermonde-Matrix); `matrix_sqrt(A)` (Matrix-Quadratwurzel); `matrix_norm(A, ord)` (Frobenius, Spektralnorm etc.); `cdist(X, Y, p)` (paarweise Abstände); `cross(a, b)` (3D-Kreuzprodukt); `polyfit`, `polyval`; `unique`, `argsort`; `convolve1d`. **Signal & Reduktionen:** `fftfreq`, `diff`, `cumsum`, `clip`, `shuffle`; `permutation(n)` (Zufallspermutation); `choice(a, size, replace)` (Zufallsstichprobe); `autocorr(x, max_lag)`; `moving_mean(x, window)`. **Spezialfunktionen:** `erf(x)`, `erfc(x)` (Fehlerfunktion); `gamma(x)`, `lgamma(x)` (Gamma/Log-Gamma); `bessel_j0(x)`, `bessel_j1(x)` (Bessel J₀, J₁); `bessel_j(n, x)` (Bessel Jₙ); `legendre(n, x)` (Legendre Pₙ); `hypergeom(a, b, c, z)` (₂F₁). **Zahlentheorie:** `gcd(a, b)`, `is_prime(n)`, `mod(a, m)`, `mod_inv(a, m)`, `mod_pow(base, exp, m)`; `dirichlet_function(x)` (D(x)=1 wenn x rational mit Nenner ≤10000, sonst 0; elementweise für Tensoren). **Fakultät:** Postfix-Operator `n!` (z. B. `5!`, `n!`) bzw. `factorial(n)`; Beispiel `factorial_test.ddk`. **Dedekind-Schnitte:** `DedekindCut(x)` (reelle Zahl als untere Menge A={q∈Q:q<x}); `dedekind_cut_from_rational(p,q)`, `dedekind_cut_sqrt2()`; `lower_set_contains(cut,q)`, `to_float()`. **Dedekind-Ringe:** `DedekindRingZ()`, `ideal(n)`, `ideal_factor(i)`; `DedekindIdeal` mit `.factor()`, `.norm()`, `*` (Ideal-Multiplikation). **Numerische Integration:** `integrate(f, a, b, n)` (Trapezregel); differenzierbar.
- **Ricci-Notation:** Indexnotation `A^ij * B_jk` für Einstein-Summen (Auto-Einsum).
- **LaTeX-Export:** `export_to_latex(source)` bzw. CLI `--latex` – Formeln aus Dedekind-Code als LaTeX; `print_latex(s)` zeigt Formeln **nur in der Konsole** als Unicode (α, Δ, ∫, ½ etc.), keine Bilder in Plots; zukünftig möglich: KaTeX/Web (siehe Documentation/Console_KaTeX_Roadmap.md).

---

## Physik & Einheiten

- **Einheiten-Literale:** SI-Basiseinheiten m, kg, s, A, K, mol, cd; abgeleitete Einheiten (Pa, W, Hz, V, F, ohm, S, Wb, T, H, lm, lx, Gy, kat, M); Chemie: mol, L, M (= mol/L), ppm, bar, atm, g; Radioaktivität: Bq, Sv. **Winkel:** rad, deg (SI-Ergänzung; automatische Umrechnung). **Quick Wins:** Kraft (N, kN, MN), Druck (kPa, MPa), Durchlässigkeit (m², D, mD) für Bau, Werkstoffe, Geologie.
- **Arithmetik:** **Automatische Umrechnung** bei Addition/Subtraktion für gleiche Dimension; Ergebnis in der Einheit des ersten Operanden. Unterstützt: **SI-Basis** — Länge (m, cm, km, mm, dm), Masse (kg, g, t, mg), Zeit (s, min, h, ms), Strom (A, mA, kA, uA), Temperatur (K, mK), Stoffmenge (mol, mmol, kmol), Lichtstärke (cd, mcd); **abgeleitet** — Druck (Pa, bar, atm), Volumen (L, mL, dm³, m³), Energie (J, kJ, MJ, Wh, kWh), Spannung (V, mV, kV), Frequenz (Hz, kHz, MHz, GHz), Ladung (C, mC, uC), Widerstand (ohm, kohm, Mohm), Leistung (W, kW, MW); **Winkel** — rad, deg. Sonst gleiche Einheit für +/-. Multiplikation/Division kombiniert Einheiten; Potenz mit `^`. **Winkel-Konvertierung:** `deg_to_rad(x)`, `rad_to_deg(x)` für Skalar, Tensor oder Quantity. **Anzeige**: Gleiche Faktoren werden zusammengefasst (z. B. `m*m` → `m^2`, `m*m*m` → `m^3`); Literale `1[m^2]`, `1[m^3]` nutzbar. Anzeige vereinfacht (J, N, Pa, W usw.).
- **Physikalische Konstanten (CODATA):** `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` – alle als `Quantity` mit SI-Einheiten.
- **Differenzierbare ODE:** `ode_solve(fun, y0, t)` (RK4/Euler); `linspace(start, stop, steps)`; Gradients durch `grad()` für Physics-Informed ML. **Lagrange/Hamilton:** `lagrange_ode_rhs(L)` – rechte Seite aus Lagrangefunktion L(q,v) für ode_solve; `hamilton_ode_rhs(H)` – aus Hamiltonfunktion H(q,p). **Lotka-Volterra:** `lotka_volterra(x0, y0, a, b, c, d, t)` – Räuber-Beute-Modell.
- **Differenzierbare PDE:** `pde_heat_1d`, `pde_heat_2d` (Wärmeleitung); `pde_advection_1d`, `pde_advection_2d` (Advektion); `pde_wave_1d`, `pde_wave_2d` (Wellengleichung); `pde_burgers_1d`, `pde_burgers_2d` (Burgers); `pde_reaction_diffusion_1d`, `pde_reaction_diffusion_2d` (Reaktions-Diffusion); `pde_advection_diffusion_1d`, `pde_advection_diffusion_2d` (Advektions-Diffusion); `pde_maxwell_1d`, `pde_maxwell_2d` (Maxwell, FDTD); `pde_navier_stokes_2d(u0, v0, x, y, t, nu)` (Navier-Stokes 2D inkompressibel, Chorin-Projektion); Finite Differenzen + `ode_solve`.
- **Quaternionen:** Native Unterstützung (`i`, `j`, `k`-Suffixe), `.rotate()`; unäres Minus (`-1.0 + 0i`).
- **Uncertainty Propagation:** `uncertain(value, std)` bzw. `UncertainQuantity` – Gauß’sche Fehlerfortpflanzung für +, -, *, /, ^; optional mit Einheit.

- **Differentialgeometrie:** `christoffel_symbols(g_func, x, h)`, `riemann_tensor(g_func, x, h)`, `covariant_derivative(T, g_func, x, h)` – Christoffel-Symbole, Riemann-Tensor, kovariante Ableitung (numerisch).

---

## Stochastik & Fitting

- **Verteilungen:** `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma`, `Beta`, `Poisson`, `Dirichlet(alpha)`; `sample(dist)`, `log_prob(dist, value)`. **Binomialkoeffizient:** `binom(n, k)` (n über k). **t-Test:** `ttest_one_sample(x, mu0)`, `ttest_two_sample(x, y)` (Welch) — Rückgabe (t_statistik, p_value).
- **Bayesian Inference:** `metropolis(log_prior, log_likelihood, data, init, steps)`; **HMC:** `hmc(...)` und `fit(..., method="hmc")`.
- **Fitting:** `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=..., steps=...)` – Gradient Descent, Metropolis-Hastings oder Hamiltonian Monte Carlo.

---

## Chemie & Biologie

- **Einheiten:** Konzentration in `[M]`, Stoffmenge in `[mol]`, Volumen in `[L]`, Verdünnungen in `[ppm]`; Massenkonzentration `[percent_wv]` (= g/100mL); M und mol/L gelten als gleich (Runtime und Compile-Check). **pH:** `concentration_to_pH(c_M)`, `pH_to_concentration(pH)` für Umrechnung [H⁺] ↔ pH.
- **Convenience:** `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)`. **Chemisches Gleichgewicht:** `chemical_equilibrium(K, n_A, n_B, n_C, n_D, A0, B0, C0, D0)` – Massenwirkungsgesetz für aA + bB <-> cC + dD; Rückgabe (A_eq, B_eq, C_eq, D_eq).
- **Chemische Elemente:** `atomic_mass("C")` (g/mol), `atomic_number("C")`; ca. 50 Elemente (IUPAC-nah); molare Masse z. B. H₂O, C₂H₆.
- **Stöchiometrie:** `balance_equation(reactants_str, products_str)` – Koeffizienten für ausgeglichene Reaktionsgleichung (z. B. `"H2 + O2"`, `"H2O"` → ([2,1], [2])); nutzt lineare Algebra (Nullraum via SVD).
- **Beispiele:** Kinetik 1. Ordnung, Dosis-Wirkung (EC50, Michaelis-Menten), logistisches Wachstum mit `ode_solve` und `fit`.

---

## Medizin, Pharmakologie & Epidemiologie

- **Pharmakologie:** `hill_equation(dose, Emax, EC50, n)` – Hill-Gleichung E = Emax·doseⁿ/(EC50ⁿ+doseⁿ); `one_compartment_pk(C0, ke, t)` – Ein-Kompartiment C(t)=C0·e^(-ke·t); `half_life(ke)` – Halbwertszeit t₁/₂ = ln(2)/ke.
- **Epidemiologie:** `sir_model(S0, I0, R0, beta, gamma, t)` – SIR-Kompartimentmodell; `basic_reproduction_number(beta, gamma)` – R₀ = β/γ.
- **Biostatistik:** `confidence_interval(x, alpha)` – Konfidenzintervall für Mittelwert (t-Verteilung); `odds_ratio(a, b, c, d)` – Odds Ratio aus 2×2-Tabelle; `sensitivity_specificity(TP, FN, FP, TN)` – Sensitivität, Spezifität, PPV, NPV.
- **Beispiele:** `pharmacology_quickwins.ddk`, `epidemiology_sir.ddk`, `biostatistics_quickwins.ddk`.

---

## Quick Wins (vernachlässigte Wissenschaften)

- **Musik:** `cents_to_ratio(cents)`, `ratio_to_cents(ratio)`, `equal_temperament(n, a4_hz)` – Frequenzverhältnisse, gleichstufige Stimmung.
- **Ökonomie:** `discount_factor(r, t, discrete)`, `cobb_douglas(K, L, alpha, A)`, `solow_rhs(K, s, delta, n, g, alpha)` – Barwert, Cobb-Douglas, Solow RHS.
- **Geologie:** `darcy_velocity(K, grad_P, mu)` – Darcy-Gesetz v = -(K/μ)∇P.
- **Werkstoffe:** `johnson_mehl_avrami(t, k, n)`, `avrami_rate(t, k, n)` – JMAK-Phasenumwandlung.
- **Beispiele:** `quickwins_units.ddk`, `music_intervals.ddk`, `economics_solow.ddk`, `geology_darcy.ddk`, `materials_jmak.ddk`.

---

## Machine Learning & Tensoren

- **Modelle:** `Sequential([Dense(...), ...])`, `Dense(n, activation="relu"|"softmax"|…)`; Forward-Pass mit `model.forward(input)`; `.fast()` für torch.compile (MLIR/Inductor).
- **Tensoren:** PyTorch-Backend; Vektoren/Matrizen, FFT, Faltung, Pooling; `.gpu()`, `.cpu()`; Autograd (`grad()`), `.with_grad()`.
- **Sparse:** `.sparse()` für COO/CSR; Item-Assignment `T[i][j] = val`. **Sparse CFD:** `sparse_laplacian_2d(N)` (5-Punkt-Stencil), `sparse_diffusion_step(T, L, dt, alpha)`, `sparse_diffusion_simulate(T0, n_steps, dt, alpha)` für ∂T/∂t = α∇²T; Beispiel `cfd_sparse_sim.ddk`.

---

## I/O, Netzwerk & JSON

- **Dateien:** `read_file(path)` (UTF-8), `write_file(path, content)`, `file_exists(path)`.
- **Netzwerk:** `http_get(url)`, `http_post(url, data)` (data als String oder Dict/List → JSON).
- **JSON:** `json_parse(s)` → Objekt (Zugriff `obj["key"]`), `json_stringify(obj)` → String.

---

## Tooling & IDE

- **Compiler:** Transpilation von `.ddk` nach Python; CLI `python -m src.compiler.compiler <file.ddk>`; Optionen `--latex`, `--no-units-check`.
- **Dedekind Studio:** Spyder-Fork mit nativem Dedekind- und Python-Kernel; Editor mit Syntax-Highlighting (Einheiten, Ricci-Indizes); Plots-Pane für `plot()`; wissenschaftliche Beispiele als Tabs; Fenster-/Taskleisten-Icon (Windows: .ico für scharfe Darstellung).
- **Jupyter-Kernel:** Dedekind in Jupyter/Spyder-Konsolen; persistenter Kontext über Zellen.
- **Beispiele:** Über 49 `.ddk`-Beispiele in `examples/dedekind/`; `pde_navier_stokes.ddk` für Navier-Stokes 2D; `angle_units.ddk` für Winkel rad/deg; `sequences.ddk` für Folgen; `stats_binom_ttest.ddk` für binom und t-Test. Batch-Test mit `run_examples.py` (-q, -v, --compile, --filter). **Tests:** `assert(condition, message)`; Mini-Test-Runner `run_tests.py` für `tests/dedekind/*.ddk`.
- **Plots:** `plot(x, y, title=..., xscale="linear"|"log", yscale="log")`; `scatter(x, y)`; `contour(X, Y, Z, levels=...)`.

---

## Was (noch) nicht oder nur eingeschränkt

- **Symbolik:** Symbolische Ableitung `diff_sym(expr, x)` und unbestimmte Integrale `integrate_sym(expr, var)` sind vorhanden (siehe Mathematik). **Symbolische Vereinfachung** (Compiler-Pass): Konstantenfaltung (`2*3` → `6`), `x+0` → `x`, `x*1` → `x`, `0*x` → `0`, `x-0` → `x`, `x^0` → `1`, `x^1` → `x`. Keine allgemeine Gleichungslöser.
- **Typen/Module:** Kein statisches Typing; kein `import` – alles in einer Datei oder über Compiler-Pipeline.
- **Performance:** Ziel AOT/MLIR/LLVM; aktuell Transpilation zu Python/PyTorch; native Binaries experimentell (z. B. `.exe`-Erzeugung).
- **Weitere PDE:** Wärmeleitung, Advektion, Wellengleichung, Burgers, Reaktions-Diffusion, Advektions-Diffusion und Maxwell (FDTD 1D/2D) als Standard-API vorhanden.

---

*Dieser Text fasst den Implementierungsstand aus README, Language Specification, Maturity Assessment und Changelogs zusammen.*
