# Dedekind für Chemie und Biologie — Roadmap

**Dedekind Language**  
Draft: January 2026 · letzter Status-Update: v1.17.0 (März 2027)

---

## Status-Update (v1.17.0)

Diese Roadmap ist **inhaltlich weitgehend abgearbeitet**:

| Phase | Thema | Status | Geliefert in |
|---|---|---|---|
| 1 | mol, L, M (= mol/L), ppm als Einheiten + Compile-Zeit-Check | ✅ erledigt | v0.9.7, v1.7 |
| 1 | Beispiele `chemistry_kinetics.ddk`, `dose_response.ddk`, `biology_growth.ddk` | ✅ erledigt | v0.9.7 ff. |
| 2 | `michaelis_menten`, `logistic`, `arrhenius`, `linear_regression` als Convenience-Built-ins | ✅ erledigt | v0.9.8 |
| 2 | `atomic_mass`, `atomic_number` für ~50 Elemente | ✅ erledigt | v0.9.8 |
| 3 | bar, atm, g, Bq, Sv, Gy als Einheiten | ✅ erledigt | v1.x |
| 3 | pH-Funktionen / % w/v | ✅ erledigt | v1.0 |
| 4 | Tutorials + Blog-Posts | 📋 nicht prioritisiert (in-repo-Beispiele decken den Pfad) |
| Bonus | Bioinformatik-Quick-Wins (`Sequence[DNA/RNA/Protein]`, `gc_content`, `transcribe`, `translate`, `reverse_complement`, `k_mer_count`) | ✅ erledigt | v1.16 |
| Bonus | Cheminformatik via `pyimport rdkit` (`smiles_descriptors`, `lipinski_rule_of_five`) | ✅ erledigt | v1.16 |
| Bonus | Molekulardynamik via OpenMM (`md_simulate_lj` mit Unit-Validierung) | ✅ erledigt | v1.14 |
| Bonus | Standardbibliothek-Module `use chemistry`, `use biology` | ✅ erledigt | v1.7 |

**Status:** Diese Roadmap ist effektiv geschlossen — die ursprünglichen Phasen 1–3 sind fertig, Phase 4 (Outreach) ist eine Marketing-Frage, kein Engineering. Für aktuelle Chemie-/Biologie-Features siehe `Dedekind_Language_Specification.md` §15.12 (Standardbibliothek) und §15.22 (Bioinformatik).

---

## 1. Ziel und Nutzen

Diese Roadmap priorisiert **Erweiterungen und Sichtbarkeit** der Dedekind-Sprache speziell für **Chemiker** und **Biologen**. Mathematik und Physik sind bereits gut abgedeckt; hier geht es um:

- **Einheiten und Konventionen**: mol, L, M, ppm, bar — die „Sprache“ von Labor und Lehrbuch.
- **Beispiele und Narrative**: Kinetik, Gleichgewicht, Dosis-Wirkung, Wachstum — zeigen, dass Dedekind auch für Chemie/Biologie gedacht ist.
- **Convenience**: Benannte Modelle (z. B. Michaelis-Menten, logistisches Wachstum) und ggf. Wrapper (lineare Regression) für typische Auswertungen.
- **Dokumentation**: Ein klarer Abschnitt „Dedekind für Chemie & Biologie“ und Verweise auf passende Beispiele.

**Abhängigkeit**: Baut auf der [Features Implementation Roadmap](Features_Implementation_Roadmap.md) und der bestehenden Runtime auf (ODE, `fit`, Einheiten, Uncertainty, Verteilungen).

---

## 2. Was bereits nutzbar ist (Basis)

| Bedarf (Chemie/Biologie) | In Dedekind bereits vorhanden |
|--------------------------|------------------------------|
| Thermodynamik / Statistik | `R_gas`, `N_A`, `k_B`, Verteilungen, `fit()`, MCMC/HMC |
| Kinetik / Dynamik | `ode_solve`, `exp`, `log`, Einheiten |
| Kurvenanpassung | `fit(loss_fn, ..., method="gd"\|"mcmc"\|"hmc")` |
| Fehler & Unsicherheit | `uncertain(value, std)`, Einheiten-Check zur Compile-Zeit |
| Spektren / Signale | `fft`, `integrate`, Tensoren, `plot()` |
| Konzentrationen, SI | `Quantity`, `[m]`, `[s]`, `[kg]` — erweiterbar auf mol, L, M |

Damit sind viele chemische und biologische Rechnungen (Kinetik, Fitting, Fehlerfortpflanzung) **ohne neue Kern-Features** möglich; fehlen vor allem **Einheiten-Konventionen**, **Beispiele** und **Sichtbarkeit**.

---

## 3. Feature-Übersicht und Phasen

| Feature | Nutzen | Aufwand | Phase |
|--------|--------|---------|--------|
| **Einheiten mol, L, M, ppm** | Konzentrationen, Stoffmengen, Verdünnungen; Compile-Check für mol+L. | Gering–Mittel | 1 |
| **Beispiele Chemie/Biologie** | Kinetik 1. Ordnung, Michaelis-Menten, Dosis-Wirkung, logistisches Wachstum. | Gering | 1 |
| **Doku „Dedekind für Chemie & Biologie“** | README/Spec-Abschnitt + Verweise auf Beispiele; klare Zielgruppe. | Gering | 1 |
| **Convenience-Funktionen** | `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`. | Gering | 2 ✅ |
| **Linear-Regression-Wrapper** | `linear_regression(x, y)` → [Steigung, Achsenabschnitt] (intern `fit`). | Gering | 2 ✅ |
| **Chemische Elemente** | `atomic_mass("C")` (g/mol), `atomic_number("C")`; IUPAC-nah, ca. 50 Elemente; Molare Masse z. B. 2*atomic_mass("H")+atomic_mass("O"). | Gering | 2 ✅ |
| **Weitere Einheiten** | bar, atm, g; Radioaktivität Bq, Sv, Gy. pH-Hinweis (pH = -log10([H+]); % w/v optional. | Gering | 3 ✅ (bar, atm, g, Bq, Sv) |
| **Tutorial / Blog** | „Kinetik in Dedekind“, „EC50-Fitting mit Dedekind“ (optional). | Mittel | 4 |

---

## 4. Abhängigkeiten und Reihenfolge

- **Phase 1** baut nur auf der bestehenden Runtime auf: Einheiten-Strings erweitern (mol, L, M, ppm), Beispiele schreiben, Doku-Abschnitt ergänzen. Keine neuen Compiler-Features nötig.
- **Phase 2** fügt schlanke Hilfsfunktionen in `ml_runtime.py` hinzu (Michaelis-Menten, logistisches Wachstum, ggf. Arrhenius; optional `linear_regression`).
- **Phase 3** erweitert Einheiten (bar, atm) und dokumentiert Konventionen (pH, %).
- **Phase 4** ist optional: Tutorials, Blogposts, weitere Beispiele.

Abstimmung mit [Features_Implementation_Roadmap](Features_Implementation_Roadmap.md): Einheiten mol/L/M sind konsistent mit dem bestehenden Unit-System (Quantity, Compile-Check); neue Funktionen nutzen nur bestehende Primitiven (`fit`, `ode_solve`, `exp`, …).

---

## 5. Implementierungs-Phasen (Detail)

### Phase 1: Einheiten, Beispiele, Doku (geschätzt: 1–2 Wochen)

**Ziel**: Chemiker und Biologen erkennen Dedekind als geeignet (Einheiten, Beispiele, klare Doku).

**Schritte**:

1. **Einheiten mol, L, M, ppm**
   - In `_unit_simplify` (bzw. Anzeige) und in `units_checker.py` (KNOWN_UNITS) die Konventionen für **mol**, **L**, **M** (= mol/L) und **ppm** unterstützen.
   - Literale: z. B. `0.1[M]`, `1[mol]`, `50[ppm]`; M = mol/L konsistent mit bestehender Einheiten-Arithmetik (Multiplikation/Division).
   - Optional: Compile-Check erweitern, sodass z. B. mol + L nur in sinnvollen Kombinationen erlaubt ist (oder zunächst nur Laufzeit wie bisher).

2. **Beispiele**
   - **`chemistry_kinetics.ddk`**: Reaktion 1. Ordnung \(c(t) = c_0 e^{-kt}\); `ode_solve` oder analytisch mit `exp`; Einheiten [M], [1/s].
   - **`chemistry_michaelis_menten.ddk`** oder **`dose_response.ddk`**: \(v = V_{\max} [S]/(K_M + [S])\); Fitting mit `fit` an künstliche oder typische Daten; Ausgabe EC50/IC50 oder \(K_M\), \(V_{\max}\).
   - **`biology_growth.ddk`**: Logistisches Wachstum \(dN/dt = r N (1 - N/K)\) mit `ode_solve`; oder exponentielle Phase mit `exp`.

3. **Dokumentation**
   - **README**: Abschnitt „Dedekind für Chemie & Biologie“ (kurz: Einheiten, Fitting, ODE, Uncertainty; Link zu den neuen Beispielen).
   - **Language Spec**: Kurzer Unterabschnitt unter §15 (z. B. §15.12 „Units and models for chemistry and biology“) mit mol, L, M, ppm und Verweis auf Kinetik/Growth-Beispiele.

**Erfolgskriterium**: (1) Konzentrationen in [M] oder [mol/L] laufen; (2) mindestens zwei neue Beispiele (z. B. Kinetik + Dosis-Wirkung oder Wachstum) laufen; (3) README und Spec enthalten den neuen Abschnitt mit Links.

---

### Phase 2: Convenience-Funktionen (geschätzt: 1 Woche) ✅

**Ziel**: Typische Modelle als benannte Funktionen — weniger Boilerplate, bessere Lesbarkeit.

**Umgesetzt**:

1. **Runtime-Funktionen in `ml_runtime.py`**
   - **`michaelis_menten(S, Vmax, Km)`**: \(v = V_{\max} S / (K_M + S)\); S, Vmax, Km als Tensoren/Skalare; differenzierbar.
   - **`logistic(t, r, K, N0)`**: Analytische Lösung \(N(t) = K / (1 + (K/N_0 - 1) e^{-rt})\).
   - **`logistic_growth_dt(N, r, K)`**: \(dN/dt = r N (1 - N/K)\) als RHS für `ode_solve` (z. B. `fn rhs(t, y) { return [logistic_growth_dt(y[0], r, K)] }`).
   - **`arrhenius(T, A, Ea)`**: \(k = A e^{-E_a/(R T)}\); R = `R_gas.value` (J/(K·mol)).
   - **`linear_regression(x, y)`**: Rückgabe `[slope, intercept]` via `fit(..., method="gd")`.

2. **Beispiele**: `dose_response.ddk` nutzt `michaelis_menten`; `biology_growth.ddk` nutzt `logistic_growth_dt` und `logistic`; `chemistry_arrhenius.ddk`, `linear_regression.ddk` neu.

**Erfolgskriterium**: Erfüllt — Michaelis-Menten, logistisches Wachstum, Arrhenius und lineare Regression sind einzeilig aufrufbar; Beispiele laufen.

**Phase 2 ergänzt: Chemische Elemente** ✅

- **`atomic_mass(symbol)`**: Atommasse in g/mol (Quantity); IUPAC-nah für ca. 50 Elemente (H, C, N, O, S, P, Na, Cl, Fe, Cu, Zn, …).
- **`atomic_number(symbol)`**: Ordnungszahl (int).
- **Nutzen**: Molare Masse z. B. H₂O = `2*atomic_mass("H") + atomic_mass("O")`; Ethan C₂H₆ = `2*atomic_mass("C") + 6*atomic_mass("H")`.
- **Beispiel**: `chemistry_elements.ddk`.

---

### Phase 3: Weitere Einheiten und Konventionen (geschätzt: 3–5 Tage)

**Ziel**: Druck (bar, atm), pH-Hinweis, ggf. % (w/v) — für Vollständigkeit in der Doku.

**Schritte**:

1. **Einheiten**: **bar**, **atm** in Anzeige/Konventionen (z. B. in `_unit_simplify` oder Doku); Umrechnung 1 bar = 10^5 Pa, 1 atm = 101325 Pa falls Konstanten genutzt werden.
2. **Dokumentation**: Kurzer Hinweis **pH**: „pH = -log10([H+])“; in Dedekind: `pH = -log10(H_plus)` mit `H_plus` in [M]. Kein neuer Typ nötig.
3. **Optional**: **%** (w/v, v/v) als Einheit oder in Doku als Konvention (z. B. 1 % = 10 g/L bei w/v).

**Erfolgskriterium**: bar/atm in Doku und ggf. Runtime erwähnt; pH in Spec erklärt; keine Regression.

---

### Phase 4: Tutorials und Sichtbarkeit (optional, geschätzt: 1–2 Wochen)

**Ziel**: Außerhalb des Repos sichtbar machen (Tutorials, Blog, ggf. Konferenz/Workshop).

**Schritte**:

1. **Tutorial „Kinetik 1. Ordnung in Dedekind“**: Schritt-für-Schritt von ODE bis Fitting; Zielgruppe Chemie-Studierende oder Lehrkräfte.
2. **Tutorial „EC50 / Dosis-Wirkung in Dedekind“**: Daten → `fit` mit Michaelis-Menten/Hill → EC50/IC50; Zielgruppe Biologie/Pharmakologie.
3. **Blogpost oder Projektseite**: „Dedekind für Chemie & Biologie“ mit Kurzfassung und Links zu den Beispielen.
4. **Optional**: Vortrag/Poster (z. B. Fachbereich Chemie/Biologie, OSS-Projekt).

**Erfolgskriterium**: Mindestens ein Tutorial oder Blogpost veröffentlicht; Verlinkung von README oder Projektseite.

---

## 6. Risiken und Optionen

| Risiko | Mitigation |
|--------|------------|
| Einheiten mol/L/M erfordern Erweiterung des Unit-Parsers | Zunächst nur als String-Konvention (z. B. `[M]` = mol/L); keine neue Grammatik nötig. |
| Zu viele domänenspezifische Funktionen | Nur wenige, sehr verbreitete Modelle (Michaelis-Menten, logistisch); Rest in Beispielen als Formel. |
| Zielgruppe findet Dedekind nicht | Phase 1 Doku + Beispiele; Phase 4 Tutorial/Blog; Suchbegriffe „Dedekind Sprache Chemie Kinetik“ etc. |

---

## 7. Referenzen und nächste Schritte

- **Features Implementation Roadmap**: [Features_Implementation_Roadmap.md](Features_Implementation_Roadmap.md) — Einheiten, Fitting, ODE, Uncertainty sind die Basis; diese Roadmap erweitert nur Konventionen und Beispiele.
- **Language Specification**: §15 Standard Library (Physical Units, ODE, Fitting, Math); neuer Abschnitt §15.12 oder Ergänzung in §15.2 für mol, L, M.
- **Codebasis**: `src/compiler/ml_runtime.py` (Quantity, _unit_simplify, fit, ode_solve); `src/compiler/units_checker.py` (KNOWN_UNITS); `examples/dedekind/`.
- **Nächster konkreter Schritt**: Phase 1 — Einheiten mol/L/M in Runtime/Doku, zwei Beispiele (Kinetik + Dosis-Wirkung oder Wachstum), README-Abschnitt „Dedekind für Chemie & Biologie“.

---

*Dieses Dokument ist die Roadmap für Chemie- und Biologie-spezifische Erweiterungen von Dedekind. Es baut auf der Features Implementation Roadmap und der bestehenden Sprache auf.*
