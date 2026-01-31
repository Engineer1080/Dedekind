# Was Dedekind aktuell kann

**Stand:** Basierend auf Code und Changelogs (v1.1.0, Januar 2026). Dedekind ist ein **Prototyp** – die Sprache wird nach Python transpiliert und nutzt PyTorch/NumPy als Laufzeit.

---

## Sprachkern

- **Syntax:** Imperativ, C/JavaScript-ähnlich mit Blöcken in `{}`, Funktionen mit `fn name(args) { ... }`, Kontrollfluss `if`/`else`, `while`, `for ... in`.
- **Typen:** Dynamische Typinferenz; primitive Typen, Listen, Vektoren/Matrizen als Tensoren, Quaternionen; Property-Zugriff (z. B. `.shape`, `.gpu()`).
- **Ausführungsmodifikatoren:** `.gpu()`, `.cpu()`, `.single()` für Hardware-Platzierung bzw. sequentielle Ausführung; `.sparse()` für Sparse-Tensoren; `.fast()` für MLIR/Inductor-Optimierung (z. B. bei Modellen).
- **Fehlerbehandlung:** Compiler-Fehler mit Zeilennummer (`CompileError`); Einheiten-Check zur Compile-Zeit (`1[m] + 1[s]` → Fehler); Runtime-Meldungen mit Kontext.

---

## Mathematik

- **Konstanten:** `pi`, `e` (dimensionslos).
- **Funktionen:** `sin`, `cos`, `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; Arkus- und Hyperbelfunktionen (`asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`) – elementweise, differenzierbar.
- **Reduktionen & Runden:** `min`, `max`, `argmin`, `argmax` (optional `dim`); `round`, `floor`, `ceil`.
- **Lineare Algebra:** `norm(x)`, `det(A)`, `trace(A)`; FFT (`fft`, `ifft`); Matrix-Operationen (transpose, inverse, dot_product).
- **Numerische Integration:** `integrate(f, a, b, n)` (Trapezregel); differenzierbar, wenn `f` Tensoren akzeptiert.
- **Ricci-Notation:** Indexnotation `A^ij * B_jk` für Einstein-Summen (Auto-Einsum).
- **LaTeX-Export:** `export_to_latex(source)` bzw. CLI `--latex` – Formeln aus Dedekind-Code als LaTeX; `print_latex(s)` zeigt Formeln **nur in der Konsole** als Unicode (α, Δ, ∫, ½ etc.), keine Bilder in Plots; zukünftig möglich: KaTeX/Web (siehe Documentation/Console_KaTeX_Roadmap.md).

---

## Physik & Einheiten

- **Einheiten-Literale:** SI-Basiseinheiten m, kg, s, A, K, mol, cd; abgeleitete Einheiten (Pa, W, Hz, V, F, ohm, S, Wb, T, H, lm, lx, Gy, kat, M); Chemie: mol, L, M (= mol/L), ppm, bar, atm, g; Radioaktivität: Bq, Sv.
- **Arithmetik:** Addition/Subtraktion nur bei gleicher Einheit; Multiplikation/Division kombiniert Einheiten; Potenz mit `^`; Anzeige vereinfacht (J, N, Pa, W usw.).
- **Physikalische Konstanten (CODATA):** `c`, `G`, `h`, `hbar`, `k_B`, `k_e`, `e_charge`, `epsilon_0`, `mu_0`, `m_e`, `m_p`, `N_A`, `R_gas`, `alpha`, `sigma_SB`, `F_faraday` – alle als `Quantity` mit SI-Einheiten.
- **Differenzierbare ODE:** `ode_solve(fun, y0, t)` (RK4/Euler); `linspace(start, stop, steps)`; Gradients durch `grad()` für Physics-Informed ML.
- **Differenzierbare PDE:** `pde_heat_1d(u0, x, t, k)`, `pde_heat_2d(u0, x, y, t, k)` für die Wärmeleitungsgleichung; Finite Differenzen + `ode_solve`; Gradients durch `u0` und `k`.
- **Quaternionen:** Native Unterstützung (`i`, `j`, `k`-Suffixe), `.rotate()`; unäres Minus (`-1.0 + 0i`).
- **Uncertainty Propagation:** `uncertain(value, std)` bzw. `UncertainQuantity` – Gauß’sche Fehlerfortpflanzung für +, -, *, /, ^; optional mit Einheit.

---

## Stochastik & Fitting

- **Verteilungen:** `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`, `Exponential(rate)`, `Gamma`, `Beta`, `Poisson`; `sample(dist)`, `log_prob(dist, value)`.
- **Bayesian Inference:** `metropolis(log_prior, log_likelihood, data, init, steps)`; **HMC:** `hmc(...)` und `fit(..., method="hmc")`.
- **Fitting:** `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=..., steps=...)` – Gradient Descent, Metropolis-Hastings oder Hamiltonian Monte Carlo.

---

## Chemie & Biologie

- **Einheiten:** Konzentration in `[M]`, Stoffmenge in `[mol]`, Volumen in `[L]`, Verdünnungen in `[ppm]`; M und mol/L gelten als gleich (Runtime und Compile-Check).
- **Convenience:** `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)`.
- **Chemische Elemente:** `atomic_mass("C")` (g/mol), `atomic_number("C")`; ca. 50 Elemente (IUPAC-nah); molare Masse z. B. H₂O, C₂H₆.
- **Beispiele:** Kinetik 1. Ordnung, Dosis-Wirkung (EC50, Michaelis-Menten), logistisches Wachstum mit `ode_solve` und `fit`.

---

## Machine Learning & Tensoren

- **Modelle:** `Sequential([Dense(...), ...])`, `Dense(n, activation="relu"|"softmax"|…)`; Forward-Pass mit `model.forward(input)`; `.fast()` für torch.compile (MLIR/Inductor).
- **Tensoren:** PyTorch-Backend; Vektoren/Matrizen, FFT, Faltung, Pooling; `.gpu()`, `.cpu()`; Autograd (`grad()`), `.with_grad()`.
- **Sparse:** `.sparse()` für COO/CSR; Item-Assignment `T[i][j] = val`; Anwendung z. B. CFD/FEM-Simulationen.

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
- **Beispiele:** Über 40 `.ddk`-Beispiele in `examples/dedekind/`; Batch-Test mit `run_examples.py` (-q, -v, --compile, --filter).

---

## Was (noch) nicht oder nur eingeschränkt

- **Symbolik:** Kein symbolisches `diff(expr, x)`; nur numerisches `grad()`. LaTeX-Export für Formeln vorhanden.
- **Typen/Module:** Kein statisches Typing; kein `import` – alles in einer Datei oder über Compiler-Pipeline.
- **Performance:** Ziel AOT/MLIR/LLVM; aktuell Transpilation zu Python/PyTorch; native Binaries experimentell (z. B. `.exe`-Erzeugung).
- **Weitere PDE:** Nur Wärmeleitung 1D/2D; keine Wellen-, Advektions- oder Maxwell-Gleichungen als Standard-API.

---

*Dieser Text fasst den Implementierungsstand aus README, Language Specification, Maturity Assessment und Changelogs zusammen.*
