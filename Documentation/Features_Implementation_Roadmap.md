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
| **Bessere Fehlermeldungen** | Einheitenfehler, Dimensionskonflikte, „expected tensor, got Quantity“ mit Zeile/Kontext. | Mittel | Phase 2 |
| **Uncertainty-Propagation** | Fehlerfortpflanzung: f(x ± Δx) → Ergebnis mit Unsicherheit; Standard in Messtechnik. | Mittel | Phase 3 |
| **Einheiten zur Compile-Zeit** | `1[m] + 1[s]` → Fehler beim Kompilieren statt zur Laufzeit; weniger Unit-Bugs. | Mittel | Phase 3 |
| **Fitting / Regression** | `fit(model, data)` mit Gradient Descent oder MCMC; typisch für Kurvenanpassung. | Mittel | Phase 4 |
| **NUTS / VI** | Robusteres Bayesian Inference (NUTS) oder schnelle Approximation (VI); Metropolis oft langsam. | Mittel | Phase 4 |
| **LaTeX-Export von Formeln** | Aus Fourier-Ausdrücken LaTeX erzeugen (für Papers/Notizen). | Mittel | Phase 4 |
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

### Phase 2: Bessere Fehlermeldungen (geschätzt: 2–3 Wochen)

**Ziel**: Einheitenfehler, Dimensionskonflikte und Typfehler mit Zeile und Kontext melden.

**Schritte**:

1. **Quelltext-Positionen**: Sicherstellen, dass AST-Knoten Zeilen-/Spalteninformation tragen (Parser/lexer bereits vorhanden?); bei Fehlern diese ausgeben.
2. **Runtime-Fehler**: Bei `Quantity`-Arithmetik (z. B. Addition verschiedener Einheiten) Fehlermeldung mit erwarteter vs. tatsächlicher Einheit und, wenn möglich, Zeile des Aufrufs.
3. **Typ-/Dimensions-Hinweise**: „expected tensor of shape (n,), got Quantity“ oder „expected same unit for +, got [m] vs [s]“.
4. **Compiler-Fehler**: Bei unbekannten Funktionen, falscher Argumentanzahl oder falschem Typ klare Meldung inkl. Zeile.
5. **Optional**: Kurzer Abschnitt in der Language Spec („Error Reporting“) und ggf. IDE-Anzeige (Fehler im Editor markieren).

**Erfolgskriterium**: Typische Fehler (Unit-Mismatch, falsche Argumente) liefern eine verständliche Meldung mit Zeilenangabe; keine Regression bei bestehenden Programmen.

---

### Phase 3: Uncertainty-Propagation & Einheiten Compile-Zeit (geschätzt: 3–5 Wochen)

**Ziel**: Fehlerfortpflanzung zur Laufzeit; Unit-Checks vor der Ausführung.

**3a) Uncertainty-Propagation**

1. **Typ**: Erweiterung von `Quantity` oder neuer Typ `UncertainQuantity(value, std)` bzw. `value ± std` mit Propagationsregeln (Gauß’sche Fortpflanzung für +, -, *, /, ^).
2. **API**: z. B. `x_with_err = uncertain(10.0, 0.5)` oder Literal-Syntax; Ausgabe „value ± std“.
3. **Integration**: Mit bestehenden Einheiten kombinierbar (value und std gleiche Einheit); in `ml_runtime.py` und Codegen (neue Funktion/Built-in).

**3b) Einheiten zur Compile-Zeit**

1. **Dimensionen im AST**: Einheiten als Typ/Dimension führen (z. B. `[m]`, `[s]`, `[m/s]`); bei Literalen und Variablen typisiert.
2. **Check vor Codegen**: Für jede Operation (+, -, *, /, ^) prüfen, ob Dimensionen zusammenpassen; bei `1[m] + 1[s]` Compiler-Fehler mit Zeile.
3. **Option**: Zunächst opt-in (z. B. `--units-check`), später Default.
4. **Abhängigkeit**: Kann mit Symbolic Simplification Roadmap Phase 5 (Einheiten in Vereinfachung) abgestimmt werden.

**Erfolgskriterium**: Uncertainty-Propagation liefert für einfache Ausdrücke korrekte Fehlerbalken; Compile-Time-Units lehnen inkonsistente Einheiten-Arithmetik mit klarer Fehlermeldung ab.

---

### Phase 4: Fitting, NUTS/VI, LaTeX-Export (geschätzt: 4–6 Wochen)

**Ziel**: Kurvenanpassung, bessere Bayesian-Tools und Formel-Export für Papers.

**4a) Fitting / Regression**

1. **API**: `fit(loss_fn, params_init, data, method="gd"|"mcmc")` — `loss_fn(params, data)` minimieren via Gradient Descent (nutzt `grad()`) oder MCMC (nutzt `metropolis`).
2. **Einfache Variante**: `fit(model_fn, x, y)` mit MSE-Verlust; `model_fn(params, x)` z. B. lineares Modell oder benutzerdefinierte Funktion.
3. **Implementierung**: In `ml_runtime.py`; Optimizer (SGD/Adam) über PyTorch; Dokumentation und Beispiel `curve_fitting.fourier`.

**4b) NUTS / VI**

1. **NUTS**: Nutzer von Pyro/NumPyro-ähnlicher API — z. B. `nuts(log_prior, log_likelihood, data, init_theta, num_samples)`; Implementierung über PyTorch-Implementierung von NUTS oder externe Bibliothek (optional).
2. **VI**: Variational Inference — `vi(model, guide, data, num_steps)`; aufwändiger; optional oder als zweite Priorität nach NUTS.
3. **Dokumentation**: Language Spec §15.8 erweitern; Beispiel für NUTS vs. Metropolis.

**4c) LaTeX-Export**

1. **AST → LaTeX**: Visitor über Fourier-AST, der unterstützte Ausdrücke (Literal, Identifier, +, -, *, /, ^, bekannte Funktionen) in LaTeX-Strings übersetzt.
2. **API**: `latex(expr)` als Built-in oder Compiler-Option `--latex expr`; Ausgabe in Konsole oder Datei.
3. **Einschränkung**: Zunächst nur mathematische Ausdrücke (kein vollständiges Programm); ausreichend für Formeln in Papers.

**Erfolgskriterium**: `fit(...)` konvergiert für einfache Modelle; NUTS liefert Posterior-Samples; `latex(expr)` erzeugt lesbaren LaTeX-Code für typische Ausdrücke.

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
| 2 | Bessere Fehlermeldungen | 2–3 Wochen | Fehler mit Zeile/Kontext |
| 3 | Uncertainty-Propagation, Einheiten Compile-Zeit | 3–5 Wochen | Messtechnik & Unit-Safety |
| 4 | Fitting, NUTS/VI, LaTeX-Export | 4–6 Wochen | Regression, Bayesian, Papers |
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
- **Language Specification**: §15 Standard Library; §12 Implementation Roadmap; „Beyond v1.0“.
- **Codebasis**: `src/compiler/ml_runtime.py` (Stdlib), `src/compiler/codegen.py` (Built-ins), `src/compiler/compiler.py` (Pipeline), `src/compiler/parser.py` (AST, Zeileninfo).
- **Nächster konkreter Schritt**: Phase 2 — Bessere Fehlermeldungen (Zeile/Kontext, Einheiten-/Typfehler) im Compiler und in der Runtime.

---

*Dieses Dokument ist die Implementierungs-Roadmap für naturwissenschaftliche Features. Bei Änderungen an der Language Spec oder neuen Abhängigkeiten sollte die Roadmap angepasst werden.*
