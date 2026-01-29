# Features für Naturwissenschaftler — Implementierungs-Roadmap

**Fourier Language**  
Draft: January 2026

---

## 1. Ziel und Nutzen

Diese Roadmap priorisiert und plant **sinnvolle Erweiterungen** der Fourier-Sprache speziell für Nutzer aus Physik, Chemie, Messtechnik und verwandten Fächern. Ziele:

- **Priorisierung**: Klare Reihenfolge nach Aufwand, Nutzen und Abhängigkeiten.
- **Planbarkeit**: Konkrete Phasen mit Schritten und Erfolgskriterien.
- **Konsistenz**: Anschluss an bestehende Features (ODE, PDE, Quantity, Probabilistic) und an die [Symbolic Simplification Roadmap](Symbolic_Simplification_Roadmap.md).

---

## 2. Feature-Übersicht und Status

| Feature | Nutzen | Aufwand | Status / Phase |
|--------|--------|---------|----------------|
| **PDE-Solver (differenzierbar)** | Wärme-, Diffusionsgleichung; PINNs brauchen oft ∇²u, ∂u/∂t. | Hoch | ✅ **Implementiert** (v0.9: `pde_heat_1d`, `pde_heat_2d`) |
| **Mehr Verteilungen** | Exponential, Gamma, Beta, Poisson für Statistik, Strahlung, Zählexperimente. | Gering | ✅ **Implementiert** (v0.9: `Exponential`, `Gamma`, `Beta`, `Poisson`) |
| **Numerische Integration** | `integrate(f, a, b)` für Flächen, Erwartungswerte, Normalisierung. | Gering | ✅ **Implementiert** (v0.9: `integrate(f, a, b, n)`; `sin`, `cos`) |
| **Bessere Fehlermeldungen** | Einheitenfehler, Dimensionskonflikte, „expected tensor, got Quantity“ mit Zeile/Kontext. | Mittel | ✅ **Implementiert** (v0.9.5: `CompileError` mit Zeile, Parser/Zeile im AST, Runtime-Quantity-Meldungen) |
| **Uncertainty-Propagation** | Fehlerfortpflanzung: f(x ± Δx) → Ergebnis mit Unsicherheit; Standard in Messtechnik. | Mittel | ✅ **Implementiert** (v0.9.2: `uncertain(value, std)`, `UncertainQuantity`) |
| **Einheiten zur Compile-Zeit** | `1[m] + 1[s]` → Fehler beim Kompilieren statt zur Laufzeit; weniger Unit-Bugs. | Mittel | ✅ **Implementiert** (v0.9.5: `units_checker.py`, `compile_source(..., check_units=True)`, CLI `--no-units-check`) |
| **Fitting / Regression** | `fit(model, data)` mit Gradient Descent oder MCMC; typisch für Kurvenanpassung. | Mittel | ✅ **Implementiert** (v0.9.2: `fit(loss_fn, params_init, data, method="gd"|"mcmc")`) |
| **NUTS / VI** | Robusteres Bayesian Inference (NUTS) oder schnelle Approximation (VI); Metropolis oft langsam. | Mittel | Phase 4 (HMC ✅) |
| **LaTeX-Export von Formeln** | Aus Fourier-Ausdrücken LaTeX erzeugen (für Papers/Notizen). | Mittel | ✅ **Implementiert** (v0.9.4: `export_to_latex(source)`, CLI `--latex`) |
| **Symbolische Ableitungen** | `diff(expr, x)` liefert Formel statt numerisches `grad()`; für Paper, Stabilitätsanalyse. | Mittel–hoch | Phase 5 |

---

## 3. Abhängigkeiten und Reihenfolge

- **Phase 1** baut nur auf der bestehenden Runtime auf (keine Compiler-Änderungen nötig).
- **Phase 2** verbessert Compiler/Runtime-Fehlerausgabe; hilft allen nachfolgenden Features.
- **Phase 3**: Uncertainty-Propagation nutzt `Quantity`-ähnliche Typen; Compile-Time-Units sind ein eigenes Typ-/Dimensionssystem im Compiler.
- **Phase 4**: Fitting nutzt `grad()` und ggf. `metropolis`; NUTS/VI erweitern die bestehende Probabilistic-API; LaTeX-Export nutzt AST (evtl. SymPy oder eigener Visitor).
- **Phase 5**: Symbolische Ableitungen können mit Symbolic Simplification und ggf. SymPy zusammenspielen.

---

## 4. Implementierungs-Phasen

### Phase 1: Geringer Aufwand — Mehr Verteilungen & Numerische Integration ✅ (v0.9)

**Ziel**: Schneller Nutzen für Statistik und Integration ohne Compiler-Änderungen.

**Umgesetzt**:

1. **Mehr Verteilungen** in `ml_runtime.py`: `Exponential(rate)`, `Gamma(concentration, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`; API wie `Normal`/`Uniform`: `sample(dist)`, `sample(dist, n)`, `log_prob(dist, value)`.
2. **Numerische Integration** in `ml_runtime.py`: `integrate(f, a, b, n=100)` mit Trapezregel; differenzierbar, wenn `f` Tensor akzeptiert. Zusätzlich `sin(x)`, `cos(x)` für Ausdrücke.
3. **Beispiele**: `examples/fourier/distributions_extended.fourier`, `examples/fourier/integration.fourier`.
4. **Dokumentation**: Language Spec §15.8 (erweiterte Verteilungen), §15.10 (Integration & Math); README „What’s New in v0.9“.

**Erfolgskriterium**: Erfüllt — beide Beispiele laufen fehlerfrei.

---

### Phase 2: Bessere Fehlermeldungen ✅ (v0.9.5)

**Ziel**: Einheitenfehler, Dimensionskonflikte und Typfehler mit Zeile und Kontext melden.

**Umgesetzt**:

1. **Quelltext-Positionen**: AST-Knoten tragen optional `line` (Zeile); Parser setzt sie bei allen Konstrukten; Lexer liefert Zeile pro Token.
2. **Compiler-Fehler**: `CompileError(message, line=..., filepath=...)` mit formatierter Ausgabe „Datei: Zeile N: Meldung“; Parser wirft bei erwartetem Token, ungültigem Zuweisungsziel, unerwartetem Token.
3. **Runtime-Fehler**: `Quantity`/`UncertainQuantity` bei Einheiten-Mismatch (+/-) mit klarer Meldung: „Einheitenfehler bei Addition: [m] vs [s]. Für + und - müssen beide Größen die gleiche Einheit haben.“
4. **Pipeline**: `compile_source(source, filepath=..., check_units=...)`; `run_examples` übergibt `filepath`; CLI fängt `CompileError` und gibt sie formatiert aus.

**Erfolgskriterium**: Erfüllt — typische Fehler liefern verständliche Meldung mit Zeilenangabe; alle Beispiele laufen.

---

### Phase 3: Uncertainty-Propagation & Einheiten Compile-Zeit (geschätzt: 3–5 Wochen)

**Ziel**: Fehlerfortpflanzung zur Laufzeit; Unit-Checks vor der Ausführung.

**3a) Uncertainty-Propagation**

1. **Typ**: Erweiterung von `Quantity` oder neuer Typ `UncertainQuantity(value, std)` bzw. `value ± std` mit Propagationsregeln (Gauß’sche Fortpflanzung für +, -, *, /, ^).
2. **API**: z. B. `x_with_err = uncertain(10.0, 0.5)` oder Literal-Syntax; Ausgabe „value ± std“.
3. **Integration**: Mit bestehenden Einheiten kombinierbar (value und std gleiche Einheit); in `ml_runtime.py` und Codegen (neue Funktion/Built-in).

**3b) Einheiten zur Compile-Zeit** ✅ (v0.9.5)

1. **Check vor Codegen**: `units_checker.py` — Visitor über AST; für `+`/`-` wird Einheit aus Literal, Quantity und bekannten Konstanten inferiert; bei Mismatch (z. B. `1[m] + 1[s]`) wird `CompileError` mit Zeile geworfen. Unäres Minus (`0 - x`) erlaubt.
2. **API**: `compile_source(..., check_units=True)` (Default); CLI `--no-units-check` zum Abschalten.
3. **Bekannte Konstanten**: `c`, `G`, `h`, `pi`, `e`, … mit Einheiten in `KNOWN_UNITS`; Identifier werden bei Bedarf aufgelöst.

**Erfolgskriterium**: Erfüllt — `1[m] + 1[s]` führt zu Compiler-Fehler mit Zeile; alle Beispiele (inkl. `universal_constants.fourier`) laufen.

---

### Phase 4: Fitting, NUTS/VI, LaTeX-Export (geschätzt: 4–6 Wochen)

**Ziel**: Kurvenanpassung, bessere Bayesian-Tools und Formel-Export für Papers.

**4a) Fitting / Regression** ✅ (v0.9.2, erweitert v0.9.4)

1. **API**: `fit(loss_fn, params_init, data, method="gd"|"mcmc"|"hmc", lr=0.01, steps=500)` — minimiert `loss_fn(params, data)` via Gradient Descent, Metropolis-Hastings oder **HMC** (Hamiltonian Monte Carlo).
2. **Implementierung**: In `ml_runtime.py`; GD mit PyTorch backward; MCMC via `metropolis`; HMC mit Leapfrog-Integration und Gradienten. Beispiele: `curve_fitting.fourier`, `hmc_fitting.fourier`.

**4b) HMC** ✅ (v0.9.4), NUTS / VI optional

1. **HMC**: `hmc(log_prior_fn, log_likelihood_fn, data, init_theta, num_steps, step_size=0.1, num_leapfrog=10)` — gleiche API wie `metropolis`; nutzt Gradienten für bessere Proposals. Auch als `fit(..., method="hmc")` nutzbar.
2. **NUTS/VI**: Optional später (Pyro/NumPyro oder Eigenbau).

**4c) LaTeX-Export** ✅ (v0.9.4)

1. **AST → LaTeX**: Visitor in `src/compiler/latex_export.py` — Literal, Identifier, BinaryOp (+, -, *, /, ^), Call (sin, cos, exp, log, sqrt, …), Quantity, Subscript, Lambda in LaTeX-Strings.
2. **API**: `export_to_latex(source_code)` im Modul `compiler`; CLI: `python -m src.compiler.compiler <file.fourier> --latex`. Ausgabe: Gleichungen (Zuweisungen/Returns) als LaTeX.
3. **Beispiel**: `examples/fourier/latex_demo.fourier`; Ausgabe z. B. `E = m \cdot {c}^{2}`.

**Erfolgskriterium**: Erfüllt — `fit(..., method="hmc")` liefert Posterior-Samples; `export_to_latex(source)` erzeugt lesbaren LaTeX für typische Formeln.

---

### Phase 5: Symbolische Ableitungen (geschätzt: 4–8 Wochen)

**Ziel**: `diff(expr, x)` liefert einen Ausdruck (Formel), nicht nur numerisches `grad()`; für Stabilitätsanalyse, Paper, vereinfachte Terme.

**Schritte**:

1. **Option A — SymPy**: Fourier-AST → SymPy-Expr → `sympy.diff(expr, x)` → zurück in Fourier-AST oder direkt LaTeX/Code. Erfordert AST ↔ SymPy-Übersetzer (ähnlich Symbolic Simplification Phase 4).
2. **Option B — Eigenbau**: Eigenes Modul `symbolic_diff.py`: Visitor über AST, Ableitungsregeln für +, -, *, /, ^, `exp`, `log`, `sin`, `cos` etc.; Ausgabe neuer AST. Keine externe Abhängigkeit, aber begrenzt auf implementierte Regeln.
3. **API**: `diff(expr, var)` — `expr` kann Fourier-Ausdruck (als String oder AST); Rückgabe vereinfachter Ausdruck (oder String/LaTeX).
4. **Integration**: Mit Symbolic Simplification abstimmen (vereinfachte Ableitungen); optional mit LaTeX-Export kombinieren.

**Erfolgskriterium**: Für polynomielle und einfache transzendente Ausdrücke liefert `diff(expr, x)` die korrekte Ableitung als Ausdruck; Dokumentation und Beispiel.

---

## 5. Übersicht: Phasen und Meilensteine

| Phase | Inhalt | Geschätzter Aufwand | Meilenstein |
|-------|--------|---------------------|-------------|
| 1 | Mehr Verteilungen, Numerische Integration | ✅ v0.9 | Neue Stdlib-Funktionen lauffähig |
| 2 | Bessere Fehlermeldungen | ✅ v0.9.5 | Fehler mit Zeile/Kontext |
| 3 | Uncertainty-Propagation, Einheiten Compile-Zeit | ✅ v0.9.2 / v0.9.5 | Messtechnik & Unit-Safety |
| 4 | Fitting, NUTS/VI, LaTeX-Export | ✅ v0.9.2–v0.9.4 | Regression, Bayesian, Papers |
| 5 | Symbolische Ableitungen | 4–8 Wochen | diff(expr, x) als Formel |

---

## 6. Risiken und Optionen

| Risiko | Mitigation |
|--------|------------|
| NUTS/VI erhöht Abhängigkeiten (Pyro/NumPyro) | Optional als Extra; oder schlanker Eigenbau nur für NUTS. |
| Compile-Time-Units erfordern größeres Typ-System | Schrittweise: zuerst nur Literale und einfache Binärops; keine vollständige Inferenz im MVP. |
| Symbolische Ableitungen und SymPy-Pfad doppelt | Phase 5 zuerst mit Eigenbau (Option B); SymPy optional später. |
| Fitting-API zu starr | Erste Version mit `fit(loss_fn, params_init, data)`; Erweiterung mit `model_fn(x, params)` in Phase 4. |

---

## 7. Referenzen und nächste Schritte

- **Bestehende Roadmap**: [Symbolic_Simplification_Roadmap.md](Symbolic_Simplification_Roadmap.md) — Einheiten in Vereinfachung (Phase 5 dort) mit Phase 3 hier abstimmen.
- **Chemie & Biologie**: [Chemistry_Biology_Roadmap.md](Chemistry_Biology_Roadmap.md) — Einheiten mol/L/M, Beispiele (Kinetik, Dosis-Wirkung, Wachstum), Convenience-Funktionen, Doku „Fourier für Chemie & Biologie“.
- **Language Specification**: §15 Standard Library; §12 Implementation Roadmap; „Beyond v1.0“.
- **Codebasis**: `src/compiler/ml_runtime.py` (Stdlib), `src/compiler/codegen.py` (Built-ins), `src/compiler/compiler.py` (Pipeline), `src/compiler/parser.py` (AST, Zeileninfo).
- **Nächster konkreter Schritt**: Phase 5 — Symbolische Ableitungen (`diff(expr, x)` als Formel); optional Phase 2/3 verfeinern (z. B. Spalte, IDE-Anzeige).

---

*Dieses Dokument ist die Implementierungs-Roadmap für naturwissenschaftliche Features. Bei Änderungen an der Language Spec oder neuen Abhängigkeiten sollte die Roadmap angepasst werden.*
