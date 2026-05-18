# Dedekind für vernachlässigte Wissenschaften — Roadmap

**Dedekind Language**  
Draft: Januar 2026 · letzter Status-Update: v1.17.0 (März 2027)

---

## Status-Update (v1.17.0)

Diese Roadmap ist **weitgehend ausgeliefert**:

| Phase | Thema | Status | Geliefert in |
|---|---|---|---|
| 1 | Chemie/Biologie-Quick-Wins (siehe Chemistry_Biology_Roadmap) | ✅ erledigt | v0.9.7 – v1.16 |
| 2 | Maschinenbau-Einheiten (kN, MPa, MN, kPa) | ✅ erledigt | v1.x |
| 2 | Musik-Convenience (`cents_to_ratio`, `ratio_to_cents`, `equal_temperament`) | ✅ erledigt | v1.x |
| 2 | Ökonomie-Convenience (`discount_factor`, `cobb_douglas`, `solow_rhs`) | ✅ erledigt | v1.x |
| 2 | Geologie (Darcy-Einheit, `darcy_velocity`) | ✅ erledigt | v1.x |
| 2 | Materialwissenschaft (`johnson_mehl_avrami`, `avrami_rate`) | ✅ erledigt | v1.x |
| 3 | Geologie-/Materials-/Structural-Beispiele | ✅ erledigt | `geology_darcy.ddk`, `materials_jmak.ddk` etc. |
| 4 | Ökonomie- und Musik-Beispiele | ✅ erledigt | `economics_solow.ddk`, `music_intervals.ddk` |
| 5 (optional) | `nernst`, `van_t_hoff`, `rate_order`; `fick_diffusion_1d`, `beam_eigenmodes`, `truss_solve` | 📋 nicht prioritisiert — bei konkreter Forscher-Nachfrage nachreichbar |
| 5 (optional) | Benutzerdefinierte Einheiten `unit Darcy = 9.87e-13[m^2]` | ✅ erledigt | v1.7 (`unit X = N[base]`) |
| 6 (optional) | Formale Linguistik, Archäologie | 📋 nicht angefangen — als langfristig markiert; ohne konkreten Anwender wenig Sinn |
| Bonus | **Molekulardynamik via OpenMM** (`md_simulate_lj`) — über die Roadmap hinausgehend | ✅ erledigt | v1.14 |
| Bonus | **Bioinformatik/Cheminformatik** (`Sequence[DNA]`, `smiles_descriptors`) | ✅ erledigt | v1.16 |
| Bonus | **Earth-Science Labeled Tensors** (`LabeledTensor[lat, lon, time]`) | ✅ erledigt | v1.15 |

**Status:** Diese Roadmap ist effektiv geschlossen für Phasen 1–4. Phase 5 (zusätzliche Spezial-Funktionen) und Phase 6 (sehr nischige Disziplinen) bleiben **on-demand** — nicht von sich aus interessant, aber realisierbar wenn ein konkreter Forscher danach fragt.

---

## 1. Ziel und Motivation

R und Python dominieren die datengetriebene Wissenschaft (Statistik, ML, Bioinformatik, Physik). Viele Disziplinen nutzen jedoch weiterhin **Excel, Origin, Mathematica, Maple** oder **domänenspezifische Tools** — oft ohne Einheiten-Check, ohne einheitliche Notation und ohne die Vorteile einer wissenschaftlichen Programmiersprache.

**Strategie:** Dedekind gezielt für Wissenschaften positionieren, die von R/Python **unterversorgt** sind. Dedekinds Stärken (Einheiten zur Compile-Zeit, Ricci-Notation, PDE-Suite, kompakte Syntax, LaTeX-Export) passen gut zu Domänen, in denen **physikalische Korrektheit**, **formale Notation** und **Lehrbarkeit** zählen.

**Prototyp-Phase:** Dedekind ist aktuell ein privater Prototyp. **Größere Kern-Änderungen** sind noch möglich — neue Einheiten, domänenspezifische Built-ins, ggf. Sprachkonstrukte. Diese Roadmap schlägt sowohl **Beispiele/Doku** als auch **neue Kern-Features** vor.

---

## 2. Vernachlässigte Domänen — Übersicht

| Domäne | R/Python-Nutzung | Typische Tools heute | Dedekind-Fit | Priorität |
|--------|------------------|----------------------|--------------|-----------|
| **Chemie** | Python wächst (RDKit), R selten | Excel, Origin, Mathematica | mol, L, M, Kinetik, Stöchiometrie | 1 (bereits Roadmap) |
| **Geologie / Geowissenschaften** | Nischenhaft | Petrel, ArcGIS, MATLAB | Einheiten, PDE (Diffusion, Advektion) | 2 |
| **Werkstoffwissenschaften** | Nischenhaft | COMSOL, Abaqus, Origin | PDE, Einheiten, Arrhenius | 2 |
| **Bauingenieurwesen / Statik** | Nischenhaft | SAP2000, ETABS, Excel | Einheiten, lineare Algebra, ODE | 2 |
| **Theoretische Ökonomie** | R/Python für Empirie; Theorie oft Mathematica | Mathematica, Maple | ODE, dynamische Modelle, Fitting | 3 |
| **Musiktheorie / Musikologie** | Nischenhaft | LilyPond, Max/MSP, spezialisierte Software | Frequenzverhältnisse, FFT, kompakte Notation | 3 |
| **Formale Linguistik** | Spezialisierte Tools | Prolog, Coq, spezielle Parser | Typen, formale Strukturen (langfristig) | 4 |
| **Archäologie / Kulturerbe** | GIS (Python), sonst spezialisiert | QGIS, 3D-Tools | Statistik, Fitting, Einheiten (Daten) | 4 |
| **Lehre (MINT)** | GeoGebra, Taschenrechner, Python | Gemischt | Einheiten-Check, LaTeX, kompakte Syntax | 2 |

---

## 3. Domänen im Detail

### 3.1 Chemie (bereits in Chemistry_Biology_Roadmap)

- **Status:** Roadmap existiert; mol, L, M, ppm, Michaelis-Menten, Arrhenius, Stöchiometrie, Elemente.
- **Warum vernachlässigt:** Viele Chemiker nutzen Excel/Origin; R ist kaum verbreitet; Python wächst, aber ohne Einheiten-Check und ohne chemische Konventionen als First-Class.
- **Dedekind-Vorteil:** `0.1[M] + 50[ppm]` mit Compile-Check; `balance_equation("H2+O2","H2O")`; `atomic_mass("C")`.
- **Nächste Schritte:** Chemistry_Biology_Roadmap Phase 4 (Tutorials, Sichtbarkeit).

---

### 3.2 Geologie / Geowissenschaften

- **Status:** Nicht adressiert.
- **Warum vernachlässigt:** GIS (Python), Modellierung oft MATLAB oder domänenspezifisch; Einheiten (m, s, Pa, °C, mol) werden selten geprüft.
- **Dedekind-Fit:**
  - **Einheiten:** Druck [Pa], Temperatur [K], Konzentration [M], Durchlässigkeit [m²], Viskosität [Pa·s].
  - **PDE:** Advektions-Diffusion (Transport in porösen Medien), Wärmeleitung (geothermisch), Reaktions-Diffusion (Mineralisation).
  - **Beispiel-Idee:** `geology_transport.ddk` — 1D-Advektions-Diffusion mit Einheiten für Grundwasser- oder Öl-Reservoir-Simulation.
- **Aufwand:** Gering–Mittel (PDE und Einheiten vorhanden; Beispiel + Doku).

---

### 3.3 Werkstoffwissenschaften

- **Status:** Nicht adressiert.
- **Warum vernachlässigt:** FEM (COMSOL, Abaqus), Kinetik/Arrhenius oft in Excel oder Origin.
- **Dedekind-Fit:**
  - **Einheiten:** [Pa], [K], [J], [mol], [1/s] für Kinetik.
  - **Arrhenius:** Bereits vorhanden — `arrhenius(T, A, Ea)`.
  - **PDE:** Wärmeleitung, Reaktions-Diffusion (Oxidation, Phasenumwandlung).
  - **Beispiel-Idee:** `materials_arrhenius.ddk` (bereits Arrhenius), `materials_diffusion.ddk` (C-Diffusion in Stahl).
- **Aufwand:** Gering (Arrhenius da; Beispiel + Doku).

---

### 3.4 Bauingenieurwesen / Statik

- **Status:** Nicht adressiert.
- **Warum vernachlässigt:** SAP2000, ETABS, Excel; Skripte selten; Einheiten oft manuell.
- **Dedekind-Fit:**
  - **Einheiten:** [N], [Pa], [m], [kg]; Compile-Check verhindert kN + MPa ohne Umrechnung.
  - **Lineare Algebra:** `solve(A, b)` für Gleichungssysteme; `eigh` für Eigenfrequenzen.
  - **Beispiel-Idee:** `structural_units.ddk` — Kräfte, Spannungen, Einheiten-Check; `structural_eigen.ddk` — Eigenfrequenzen eines Balkenmodells.
- **Aufwand:** Gering (LA und Einheiten vorhanden; Beispiel + Doku).

---

### 3.5 Theoretische Ökonomie

- **Status:** Nicht adressiert.
- **Warum vernachlässigt:** Dynamische Modelle oft in Mathematica/Maple; R/Python für Empirie, nicht für formale Theorie.
- **Dedekind-Fit:**
  - **ODE:** `ode_solve` für Wachstumsmodelle, Ramsey, Solow.
  - **Fitting:** `fit` für Kalibrierung an Daten.
  - **Beispiel-Idee:** `economics_solow.ddk` — Solow-Wachstumsmodell mit Einheiten (Arbeitsstunden, Kapital).
- **Aufwand:** Gering (ODE, fit vorhanden; Beispiel + Doku).

---

### 3.6 Musiktheorie / Musikologie

- **Status:** Nicht adressiert.
- **Warum vernachlässigt:** LilyPond, Max/MSP, spezialisierte Software; keine Einheiten für Frequenzen/Intervalle.
- **Dedekind-Fit:**
  - **Frequenz:** [Hz]; Intervalle als Verhältnisse (z.B. Quinte = 3/2).
  - **FFT:** Bereits vorhanden — Spektralanalyse.
  - **Beispiel-Idee:** `music_intervals.ddk` — Frequenzverhältnisse, Cent-Berechnung; `music_spectrum.ddk` — FFT eines Tons.
- **Aufwand:** Gering (FFT, Einheiten; Beispiel + Doku).

---

### 3.7 Lehre (MINT)

- **Status:** Indirekt (Beispiele, LaTeX).
- **Warum relevant:** GeoGebra, Taschenrechner, Python — oft ohne Einheiten-Check; Fehler (m + s) sind häufig.
- **Dedekind-Fit:**
  - **Einheiten-Check:** `1[m] + 1[s]` → Compiler-Fehler — didaktisch wertvoll.
  - **LaTeX-Export:** Formeln aus Code; `print_latex` in Konsole.
  - **Kompakte Syntax:** Weniger Boilerplate als Python.
  - **Beispiel-Idee:** Tutorial „Physik mit Dedekind“ — Einheiten, ODE, Plot.
- **Aufwand:** Gering (alles da; Tutorial + Doku).

---

## 4. Mögliche Kern-Features (Prototyp-Phase)

Da Dedekind noch ein Prototyp ist, können **neue Sprach- und Runtime-Features** gezielt für vernachlässigte Domänen ergänzt werden. Die folgenden Vorschläge sind nach Domäne und Aufwand sortiert.

### 4.1 Einheiten-Erweiterungen (quer zu mehreren Domänen)

| Feature | Domäne(n) | Beschreibung | Aufwand |
|---------|-----------|--------------|---------|
| **rad, deg** | Physik, Lehre, Bau | Winkel mit automatischer Umrechnung; `deg_to_rad(x)`, `rad_to_deg(x)` | ✅ v1.2.6 |
| **kN, MPa, MN** | Bau, Werkstoffe | Ingenieur-Konventionen: `1[kN] = 1000[N]`, `1[MPa] = 1e6[Pa]` | Gering |
| **Darcy [D], mD** | Geologie | Durchlässigkeit: 1 D ≈ 9.87e−13 m² | Gering |
| **wt%, at%** | Chemie, Werkstoffe | Gewichts-/Atomprozent als Einheit oder Konvention | Mittel |
| **Cent [cent]** | Musik | 1 cent = 1/1200 Oktave; `cents_to_ratio(c)`, `ratio_to_cents(r)` | Gering |
| **User-defined units** | Alle | `unit Darcy = 9.87e-13[m^2]` — benutzerdefinierte Einheiten zur Compile-Zeit | Hoch |

### 4.2 Geologie / Geowissenschaften

| Feature | Beschreibung | Aufwand |
|---------|--------------|---------|
| **`pde_porous_media_1d`** | Advektions-Diffusion mit Porosität φ und Durchlässigkeit: Transport in porösen Medien als Convenience-Wrapper | Mittel |
| **Einheiten Darcy, mD** | Siehe oben | Gering |
| **`darcy_velocity(K, grad_P, mu)`** | Darcy-Gesetz v = −(K/μ) ∇P als Built-in | Gering |

### 4.3 Werkstoffwissenschaften

| Feature | Beschreibung | Aufwand |
|---------|--------------|---------|
| **`fick_diffusion_1d(c0, x, t, D)`** | Fick’sche Diffusion mit konzentrationsabhängigem D (optional) | Mittel |
| **`johnson_mehl_avrami(t, k, n)`** | Phasenumwandlungskinetik (JMAK): f(t) = 1 − exp(−(k·t)^n) | Gering |
| **`avrami_rate(f, k, n)`** | RHS für ode_solve aus JMAK | Gering |
| **Einheiten wt%, at%** | Siehe oben | Mittel |

### 4.4 Bauingenieurwesen / Statik

| Feature | Beschreibung | Aufwand |
|---------|--------------|---------|
| **Einheiten kN, MPa, MN** | Siehe oben | Gering |
| **`beam_eigenmodes(EI, m, L, n)`** | Eigenfrequenzen eines Euler-Bernoulli-Balkens (diskretisiert) | Mittel |
| **`truss_solve(connectivity, loads, stiffness)`** | Stabwerk: Steifigkeitsmatrix + solve als Convenience | Mittel |

### 4.5 Theoretische Ökonomie

| Feature | Beschreibung | Aufwand |
|---------|--------------|---------|
| **`discount_factor(r, t, discrete)`** | exp(−r·t) oder 1/(1+r)^t für Barwert | Gering |
| **`cobb_douglas(K, L, alpha, A)`** | Produktionsfunktion Y = A·K^α·L^(1−α) | Gering |
| **`solow_rhs(K, s, delta, n, g, alpha)`** | RHS für Solow-Modell: dK/dt = s·Y − (δ+n+g)·K | Gering |
| **`ramsey_rhs(...)`** | Ramsey-Cass-Koopmans RHS (optional) | Mittel |

### 4.6 Musiktheorie / Musikologie

| Feature | Beschreibung | Aufwand |
|---------|--------------|---------|
| **`cents_to_ratio(cents)`** | 2^(cents/1200) | Gering |
| **`ratio_to_cents(ratio)`** | 1200·log2(ratio) | Gering |
| **`equal_temperament(n)`** | Frequenz des n-ten Halbtons (A4 = 440 Hz Referenz) | Gering |
| **`just_interval(name)`** | Reine Intervalle: "fifth"→3/2, "major_third"→5/4, etc. | Gering |
| **Einheit [cent]** | Optional: Intervall als Quantity | Gering |

### 4.7 Chemie (Erweiterung zur Chemistry_Biology_Roadmap)

| Feature | Beschreibung | Aufwand |
|---------|--------------|---------|
| **`nernst(E0, n, Q, T)`** | Nernst-Gleichung E = E0 − (R·T)/(n·F)·ln(Q) | Gering |
| **`van_t_hoff(K1, T1, dH, T2)`** | Temperaturabhängigkeit der Gleichgewichtskonstante | Gering |
| **`rate_order(conc, k, orders)`** | Allgemeines Geschwindigkeitsgesetz v = k·∏c_i^n_i | Gering |
| **Einheiten wt%, at%, N (Normalität)** | Siehe oben | Mittel |

### 4.8 Sprach-Level: Domänen-Profile (langfristig)

| Feature | Beschreibung | Aufwand |
|---------|--------------|---------|
| **`use geology`** | Lädt Einheiten (Darcy, mD) und Built-ins (darcy_velocity, pde_porous_media) | Hoch |
| **`use music`** | Lädt Cent-Funktionen, equal_temperament, just_interval | Mittel |
| **`use structural`** | Lädt kN, MPa, beam_eigenmodes, truss_solve | Mittel |

*Hinweis:* Domänen-Profile setzen ein Modul-/Import-System voraus, das aktuell fehlt. Alternative: Compiler-Flag `--domain geology` oder Pragma `#domain geology` am Dateianfang.

---

## 5. Implementierungs-Phasen

### Phase 1: Chemie abschließen (bereits in Arbeit)

- Chemistry_Biology_Roadmap Phase 4: Tutorials, Sichtbarkeit.
- Keine neuen Features nötig.

### Phase 2: Kern-Features „Quick Wins“ (geschätzt: 2–3 Wochen) ✅

*Geringer Aufwand, hoher Nutzen für mehrere Domänen. Umgesetzt.*

| Schritt | Feature | Domäne(n) |
|---------|---------|-----------|
| 1 | Einheiten **kN, MPa, MN, kPa** | Bau, Werkstoffe |
| 2 | **cents_to_ratio**, **ratio_to_cents**, **equal_temperament** | Musik |
| 3 | **discount_factor**, **cobb_douglas**, **solow_rhs** | Ökonomie |
| 4 | **darcy_velocity**, Einheiten **D, mD** | Geologie |
| 5 | **johnson_mehl_avrami**, **avrami_rate** | Werkstoffe |
| 6 | Beispiele + Doku für alle | — |

**Beispiele:** `quickwins_units.ddk`, `music_intervals.ddk`, `economics_solow.ddk`, `geology_darcy.ddk`, `materials_jmak.ddk`.

### Phase 3: Geowissenschaften, Werkstoffe, Bau (geschätzt: 2–3 Wochen)

| Schritt | Domäne | Aktion |
|---------|--------|--------|
| 1 | Geologie | Beispiel `geology_transport.ddk`; optional `pde_porous_media_1d`. |
| 2 | Werkstoffe | Beispiel `materials_diffusion.ddk`, `materials_jmak.ddk`. |
| 3 | Bauingenieurwesen | Beispiel `structural_units.ddk`, `structural_eigen.ddk`; optional `beam_eigenmodes`. |
| 4 | Doku | Abschnitt „Dedekind für Geowissenschaften, Werkstoffe, Bauingenieurwesen“. |

### Phase 4: Ökonomie, Musik, Lehre (geschätzt: 1–2 Wochen)

| Schritt | Domäne | Aktion |
|---------|--------|--------|
| 1 | Ökonomie | Beispiel `economics_solow.ddk` mit `solow_rhs`, `discount_factor`. |
| 2 | Musik | Beispiel `music_intervals.ddk`, `music_spectrum.ddk` mit `cents_to_ratio`, `equal_temperament`. |
| 3 | Lehre | Tutorial „MINT mit Dedekind“ — Einheiten, ODE, LaTeX. |

### Phase 5: Chemie-Erweiterung, mittlere Features (optional)

| Schritt | Feature | Aufwand |
|---------|---------|---------|
| 1 | **nernst**, **van_t_hoff**, **rate_order** | Gering |
| 2 | **fick_diffusion_1d**, **beam_eigenmodes**, **truss_solve** | Mittel |
| 3 | Domänen-Profile (`use geology` etc.) — falls Import-System kommt | Hoch |

### Phase 6: Formale Linguistik, Archäologie (optional)

- Formale Linguistik: Langfristig; erfordert ggf. Typen, formale Strukturen.
- Archäologie: Statistik, Fitting — mit bestehenden Mitteln möglich.

---

## 6. Erfolgskriterien

| Phase | Kriterium |
|-------|-----------|
| 1 | Chemie-Tutorial oder Blogpost veröffentlicht. |
| 2 | Mindestens 5 neue Kern-Features (Einheiten + Built-ins) implementiert; Beispiele laufen. |
| 3 | Mindestens 3 neue Beispiele (Geologie, Werkstoffe, Bau); README-Abschnitt. |
| 4 | Beispiele Ökonomie, Musik; Tutorial „MINT mit Dedekind“. |
| 5 | Optional: nernst, van_t_hoff, beam_eigenmodes, truss_solve. |

---

## 7. Risiken und Mitigation

| Risiko | Mitigation |
|--------|------------|
| Zielgruppen finden Dedekind nicht | Suchbegriffe, Tutorials, Fachbereichskontakte. |
| Zu viele domänenspezifische Features | Priorisierung: nur Features mit breitem Nutzen oder klarer Zielgruppe. |
| Konkurrenz durch Excel/Origin | Einheiten-Check und Reproduzierbarkeit als Alleinstellungsmerkmal. |
| Kern-Features verkomplizieren die Sprache | Built-ins in `ml_runtime`; Einheiten als Erweiterung des bestehenden Systems. |

---

## 8. Referenzen

- **Chemistry_Biology_Roadmap.md** — Chemie/Biologie-Features und Beispiele.
- **Features_Implementation_Roadmap.md** — Basis-Features (Einheiten, ODE, PDE, fit).
- **Was_Dedekind_aktuell_kann.md** — Aktueller Funktionsumfang.
- **Maturity_Assessment.md** — Ausgereiftheit pro Domäne.

---

*Dieses Dokument identifiziert Wissenschaften, die von R und Python vernachlässigt werden, und skizziert, wie Dedekind sie ansprechen kann — sowohl mit bestehenden Stärken als auch mit neuen Kern-Features (Einheiten, Built-ins, ggf. Domänen-Profile), die in der Prototyp-Phase noch umsetzbar sind.*
