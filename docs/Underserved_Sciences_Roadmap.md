# Dedekind for Underserved Sciences — Roadmap

**Dedekind Language**  
Draft: January 2026 · Last status update: v2.9.0 (May 2026)

---

## Status Update (v2.9.0)

This roadmap is **largely delivered**:

| Phase | Topic | Status | Delivered in |
|---|---|---|---|
| 1 | Chemistry/Biology quick wins (see Chemistry_Biology_Roadmap) | ✅ completed | v0.9.7 – v1.16 |
| 2 | Mechanical engineering units (kN, MPa, MN, kPa) | ✅ completed | v1.x |
| 2 | Music convenience (`cents_to_ratio`, `ratio_to_cents`, `equal_temperament`) | ✅ completed | v1.x |
| 2 | Economics convenience (`discount_factor`, `cobb_douglas`, `solow_rhs`) | ✅ completed | v1.x |
| 2 | Geology (Darcy unit, `darcy_velocity`) | ✅ completed | v1.x |
| 2 | Materials science (`johnson_mehl_avrami`, `avrami_rate`) | ✅ completed | v1.x |
| 3 | Geology/Materials/Structural examples | ✅ completed | `geology_darcy.ddk`, `materials_jmak.ddk` etc. |
| 4 | Economics and Music examples | ✅ completed | `economics_solow.ddk`, `music_intervals.ddk` |
| 5 (optional) | `truss_solve`, `concrete_beam_capacity` | ✅ completed | v2.2.0 (`use structural`) |
| 5 (optional) | User-defined units `unit Darcy = 9.87e-13[m^2]` | ✅ completed | v1.7 (`unit X = N[base]`) |
| 6 (optional) | Formal linguistics, Archaeology | 📋 not started — marked as long-term; makes little sense without concrete users |
| Bonus | **Molecular Dynamics via OpenMM** (`md_simulate_lj`) — exceeding the roadmap | ✅ completed | v1.14 |
| Bonus | **Bioinformatics/Cheminformatics** (`Sequence[DNA]`, `smiles_descriptors`) | ✅ completed | v1.16 |
| Bonus | **Earth-Science Labeled Tensors** (`LabeledTensor[lat, lon, time]`) | ✅ completed | v1.15 |
| Bonus | **Differentiable FEM Structural / Thermal / Fluids Suite** | ✅ completed | v2.2.0 |

**Status:** This roadmap is effectively closed for phases 1–5. Phase 6 (very niche disciplines) remains **on-demand** — not interesting on their own, but implementable if a concrete researcher asks for them.

---

## 1. Goal and Motivation

R and Python dominate data-driven science (statistics, ML, bioinformatics, physics). Many disciplines, however, continue to use **Excel, Origin, Mathematica, Maple** or **domain-specific tools** — often without unit checking, without a uniform notation, and without the benefits of a scientific programming language.

**Strategy:** Position Dedekind specifically for sciences that are **underserved** by R/Python. Dedekind's strengths (compile-time unit checks, Ricci notation, PDE suite, compact syntax, LaTeX export) fit well in domains where **physical correctness**, **formal notation**, and **teachability** matter.

**Prototype Phase:** Dedekind is currently a private prototype. **Major core changes** are still possible — new units, domain-specific built-ins, and language constructs. This roadmap proposes both **examples/documentation** and **new core features**.

---

## 2. Underserved Domains — Overview

| Domain | R/Python Usage | Typical Tools Today | Dedekind Fit | Priority |
|--------|------------------|----------------------|--------------|-----------|
| **Chemistry** | Python is growing (RDKit), R is rare | Excel, Origin, Mathematica | mol, L, M, kinetics, stoichiometry | 1 (already on Roadmap) |
| **Geology / Earth Sciences** | Niche | Petrel, ArcGIS, MATLAB | units, PDE (diffusion, advection) | 2 |
| **Materials Science** | Niche | COMSOL, Abaqus, Origin | PDE, units, Arrhenius | 2 |
| **Structural / Civil Engineering** | Niche | SAP2000, ETABS, Excel | units, linear algebra, ODE | 2 |
| **Theoretical Economics** | R/Python for empirical work; theory often Mathematica | Mathematica, Maple | ODE, dynamic models, fitting | 3 |
| **Music Theory / Musicology** | Niche | LilyPond, Max/MSP, specialized software | frequency ratios, FFT, compact notation | 3 |
| **Formal Linguistics** | Specialized tools | Prolog, Coq, custom parsers | types, formal structures (long-term) | 4 |
| **Archaeology / Cultural Heritage** | GIS (Python), otherwise specialized | QGIS, 3D tools | statistics, fitting, units (data) | 4 |
| **Teaching (STEM)** | GeoGebra, calculators, Python | Mixed | unit checks, LaTeX, compact syntax | 2 |

---

## 3. Domains in Detail

### 3.1 Chemistry (already in Chemistry_Biology_Roadmap)

- **Status:** Roadmap exists; mol, L, M, ppm, Michaelis-Menten, Arrhenius, stoichiometry, elements.
- **Why underserved:** Many chemists use Excel/Origin; R is barely used; Python is growing but lacks compile-time unit checking and chemical conventions as first-class citizens.
- **Dedekind Advantage:** `0.1[M] + 50[ppm]` with compile checks; `balance_equation("H2+O2","H2O")`; `atomic_mass("C")`.
- **Next Steps:** Chemistry_Biology_Roadmap Phase 4 (tutorials, visibility).

---

### 3.2 Geology / Earth Sciences

- **Status:** Unaddressed.
- **Why underserved:** GIS (Python), modeling often MATLAB or domain-specific; units (m, s, Pa, °C, mol) are rarely checked automatically.
- **Dedekind Fit:**
  - **Units:** Pressure [Pa], temperature [K], concentration [M], permeability [m²], viscosity [Pa·s].
  - **PDE:** Advection-diffusion (transport in porous media), heat conduction (geothermal), reaction-diffusion (mineralization).
  - **Example Idea:** `geology_transport.ddk` — 1D advection-diffusion with units for groundwater or oil reservoir simulation.
- **Effort:** Low–Medium (PDE and units available; example + docs).

---

### 3.3 Materials Science

- **Status:** Unaddressed.
- **Why underserved:** FEM (COMSOL, Abaqus), kinetics/Arrhenius often in Excel or Origin.
- **Dedekind Fit:**
  - **Units:** [Pa], [K], [J], [mol], [1/s] for kinetics.
  - **Arrhenius:** Already available — `arrhenius(T, A, Ea)`.
  - **PDE:** Heat conduction, reaction-diffusion (oxidation, phase transformation).
  - **Example Idea:** `materials_arrhenius.ddk` (already Arrhenius), `materials_diffusion.ddk` (C diffusion in steel).
- **Effort:** Low (Arrhenius available; example + docs).

---

### 3.4 Civil / Structural Engineering

- **Status:** Unaddressed.
- **Why underserved:** SAP2000, ETABS, Excel; scripting is rare; units are handled manually.
- **Dedekind Fit:**
  - **Units:** [N], [Pa], [m], [kg]; compile checks prevent kN + MPa without conversion.
  - **Linear Algebra:** `solve(A, b)` for equation systems; `eigh` for natural frequencies.
  - **Example Idea:** `structural_units.ddk` — forces, stresses, unit checks; `structural_eigen.ddk` — natural frequencies of a beam model.
- **Effort:** Low (LA and units available; example + docs).

---

### 3.5 Theoretical Economics

- **Status:** Unaddressed.
- **Why underserved:** Dynamic models often in Mathematica/Maple; R/Python for empirical work, not formal theory.
- **Dedekind Fit:**
  - **ODE:** `ode_solve` for growth models, Ramsey, Solow.
  - **Fitting:** `fit` for calibration to data.
  - **Example Idea:** `economics_solow.ddk` — Solow growth model with units (labor hours, capital).
- **Effort:** Low (ODE, fit available; example + docs).

---

### 3.6 Music Theory / Musicology

- **Status:** Unaddressed.
- **Why underserved:** LilyPond, Max/MSP, specialized software; no units for frequencies/intervals.
- **Dedekind Fit:**
  - **Frequency:** [Hz]; intervals as ratios (e.g., fifth = 3/2).
  - **FFT:** Already available — spectral analysis.
  - **Example Idea:** `music_intervals.ddk` — frequency ratios, cent calculation; `music_spectrum.ddk` — FFT of a sound.
- **Effort:** Low (FFT, units; example + docs).

---

### 3.7 Teaching (STEM)

- **Status:** Indirect (examples, LaTeX).
- **Why relevant:** GeoGebra, calculators, Python — often without unit checks; errors (m + s) are common.
- **Dedekind Fit:**
  - **Unit Check:** `1[m] + 1[s]` → compiler error — didactically valuable.
  - **LaTeX Export:** Formulas from code; `print_latex` in console.
  - **Compact Syntax:** Less boilerplate than Python.
  - **Example Idea:** Tutorial "Physics with Dedekind" — units, ODE, plot.
- **Effort:** Low (everything available; tutorial + docs).

---

## 4. Possible Core Features (Prototype Phase)

Since Dedekind is still a prototype, **new language and runtime features** can be added specifically for underserved domains. The following suggestions are sorted by domain and effort.

### 4.1 Unit Extensions (cross-domain)

| Feature | Domain(s) | Description | Effort |
|---------|-----------|--------------|---------|
| **rad, deg** | Physics, education, structural | Angles with automatic conversion; `deg_to_rad(x)`, `rad_to_deg(x)` | ✅ v1.2.6 |
| **kN, MPa, MN** | Structural, materials | Engineering conventions: `1[kN] = 1000[N]`, `1[MPa] = 1e6[Pa]` | Low |
| **Darcy [D], mD** | Geology | Permeability: 1 D ≈ 9.87e−13 m² | Low |
| **wt%, at%** | Chemistry, materials | Weight/atom percent as unit or convention | Medium |
| **Cent [cent]** | Music | 1 cent = 1/1200 octave; `cents_to_ratio(c)`, `ratio_to_cents(r)` | Low |
| **User-defined units** | All | `unit Darcy = 9.87e-13[m^2]` — user-defined units at compile time | High |

### 4.2 Geology / Earth Sciences

| Feature | Description | Effort |
|---------|--------------|---------|
| **`pde_porous_media_1d`** | Advection-diffusion with porosity φ and permeability: transport in porous media as convenience wrapper | Medium |
| **Units Darcy, mD** | See above | Low |
| **`darcy_velocity(K, grad_P, mu)`** | Darcy's law v = −(K/μ) ∇P as built-in | Low |

### 4.3 Materials Science

| Feature | Description | Effort |
|---------|--------------|---------|
| **`fick_diffusion_1d(c0, x, t, D)`** | Fickian diffusion with concentration-dependent D (optional) | Medium |
| **`johnson_mehl_avrami(t, k, n)`** | Phase transformation kinetics (JMAK): f(t) = 1 − exp(−(k·t)^n) | Low |
| **`avrami_rate(f, k, n)`** | RHS for `ode_solve` from JMAK | Low |
| **Units wt%, at%** | See above | Medium |

### 4.4 Civil / Structural Engineering

| Feature | Description | Effort |
|---------|--------------|---------|
| **Units kN, MPa, MN** | See above | Low |
| **`beam_eigenmodes(EI, m, L, n)`** | Natural frequencies of an Euler-Bernoulli beam (discretized) | Medium |
| **`truss_solve(connectivity, loads, stiffness)`** | Truss framework: stiffness matrix + solve as convenience | Medium |

### 4.5 Theoretical Economics

| Feature | Description | Effort |
|---------|--------------|---------|
| **`discount_factor(r, t, discrete)`** | exp(−r·t) or 1/(1+r)^t for present value | Low |
| **`cobb_douglas(K, L, alpha, A)`** | Production function Y = A·K^α·L^(1−α) | Low |
| **`solow_rhs(K, s, delta, n, g, alpha)`** | RHS for Solow model: dK/dt = s·Y − (δ+n+g)·K | Low |
| **`ramsey_rhs(...)`** | Ramsey-Cass-Koopmans RHS (optional) | Medium |

### 4.6 Music Theory / Musicology

| Feature | Description | Effort |
|---------|--------------|---------|
| **`cents_to_ratio(cents)`** | 2^(cents/1200) | Low |
| **`ratio_to_cents(ratio)`** | 1200·log2(ratio) | Low |
| **`equal_temperament(n)`** | Frequency of the n-th semitone (A4 = 440 Hz reference) | Low |
| **`just_interval(name)`** | Pure intervals: "fifth"→3/2, "major_third"→5/4, etc. | Low |
| **Unit [cent]** | Optional: interval as Quantity | Low |

### 4.7 Chemistry (extension to Chemistry_Biology_Roadmap)

| Feature | Description | Effort |
|---------|--------------|---------|
| **`nernst(E0, n, Q, T)`** | Nernst equation E = E0 − (R·T)/(n·F)·ln(Q) | Low |
| **`van_t_hoff(K1, T1, dH, T2)`** | Temperature dependence of the equilibrium constant | Low |
| **`rate_order(conc, k, orders)`** | General rate law v = k·∏c_i^n_i | Low |
| **Units wt%, at%, N (normality)** | See above | Medium |

### 4.8 Language Level: Domain Profiles (long-term)

| Feature | Description | Effort |
|---------|--------------|---------|
| **`use geology`** | Loads units (Darcy, mD) and built-ins (darcy_velocity, pde_porous_media) | High |
| **`use music`** | Loads cent functions, equal_temperament, just_interval | Medium |
| **`use structural`** | Loads kN, MPa, beam_eigenmodes, truss_solve | Medium |

*Note:* Domain profiles require a module/import system. Alternative: compiler flag `--domain geology` or pragma `#domain geology` at the start of the file.

---

## 5. Implementation Phases

### Phase 1: Complete Chemistry (already in progress)

- Chemistry_Biology_Roadmap Phase 4: Tutorials, visibility.
- No new features needed.

### Phase 2: Core Features "Quick Wins" (estimated: 2–3 weeks) ✅

*Low effort, high utility for multiple domains. Implemented.*

| Step | Feature | Domain(s) |
|---------|---------|-----------|
| 1 | Units **kN, MPa, MN, kPa** | Structural, materials |
| 2 | **cents_to_ratio**, **ratio_to_cents**, **equal_temperament** | Music |
| 3 | **discount_factor**, **cobb_douglas**, **solow_rhs** | Economics |
| 4 | **darcy_velocity**, units **D, mD** | Geology |
| 5 | **johnson_mehl_avrami**, **avrami_rate** | Materials |
| 6 | Examples + docs for all | — |

**Examples:** `quickwins_units.ddk`, `music_intervals.ddk`, `economics_solow.ddk`, `geology_darcy.ddk`, `materials_jmak.ddk`.

### Phase 3: Earth Sciences, Materials, Civil Engineering (estimated: 2–3 weeks)

| Step | Domain | Action |
|---------|--------|--------|
| 1 | Geology | Example `geology_transport.ddk`; optional `pde_porous_media_1d`. |
| 2 | Materials | Example `materials_diffusion.ddk`, `materials_jmak.ddk`. |
| 3 | Civil Engineering | Example `structural_units.ddk`, `structural_eigen.ddk`; optional `beam_eigenmodes`. |
| 4 | Docs | Section "Dedekind for Earth Sciences, Materials, Civil Engineering". |

### Phase 4: Economics, Music, Teaching (estimated: 1–2 weeks)

| Step | Domain | Action |
|---------|--------|--------|
| 1 | Economics | Example `economics_solow.ddk` with `solow_rhs`, `discount_factor`. |
| 2 | Music | Example `music_intervals.ddk`, `music_spectrum.ddk` with `cents_to_ratio`, `equal_temperament`. |
| 3 | Teaching | Tutorial "STEM with Dedekind" — units, ODE, LaTeX. |

### Phase 5: Chemistry Extension, Medium Features (optional)

| Step | Feature | Effort |
|---------|---------|---------|
| 1 | **nernst**, **van_t_hoff**, **rate_order** | Low |
| 2 | **fick_diffusion_1d**, **beam_eigenmodes**, **truss_solve** | Medium |
| 3 | Domain profiles (`use geology` etc.) — if import system is added | High |

### Phase 6: Formal Linguistics, Archaeology (optional)

- Formal linguistics: Long-term; may require types, formal structures.
- Archaeology: Statistics, fitting — possible with existing tools.

---

## 6. Success Criteria

| Phase | Criterion |
|-------|-----------|
| 1 | Chemistry tutorial or blog post published. |
| 2 | At least 5 new core features (units + built-ins) implemented; examples run. |
| 3 | At least 3 new examples (geology, materials, structural); README section. |
| 4 | Examples economics, music; tutorial "STEM with Dedekind". |
| 5 | Optional: nernst, van_t_hoff, beam_eigenmodes, truss_solve. |

---

## 7. Risks and Mitigation

| Risk | Mitigation |
|--------|------------|
| Target audiences do not find Dedekind | Search terms, tutorials, contact with academic departments. |
| Too many domain-specific features | Prioritization: only features with broad utility or clear target audience. |
| Competition from Excel/Origin | Unit checking and reproducibility as unique selling points. |
| Core features complicate the language | Built-ins in `ml_runtime`; units as extension of existing system. |

---

## 8. References

- **Chemistry_Biology_Roadmap.md** — Chemistry/biology features and examples.
- **Features_Implementation_Roadmap.md** — Basic features (units, ODE, PDE, fit).
- **What_Dedekind_Can_Currently_Do.md** — Current functional scope.
- **Maturity_Assessment.md** — Maturity per domain.

---

*This document identifies sciences underserved by R and Python, and outlines how Dedekind can address them — both with existing strengths and with new core features (units, built-ins, domain profiles) that are still feasible in the prototype phase.*
