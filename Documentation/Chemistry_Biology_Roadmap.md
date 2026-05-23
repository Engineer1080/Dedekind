# Dedekind for Chemistry and Biology — Roadmap

**Dedekind Language**  
Draft: January 2026 · last status update: v1.17.0 (March 2027)

---

## Status Update (v1.17.0)

This roadmap is **largely completed**:

| Phase | Topic | Status | Delivered in |
|---|---|---|---|
| 1 | mol, L, M (= mol/L), ppm as units + compile-time check | Done | v0.9.7, v1.7 |
| 1 | Examples `chemistry_kinetics.ddk`, `dose_response.ddk`, `biology_growth.ddk` | Done | v0.9.7 ff. |
| 2 | `michaelis_menten`, `logistic`, `arrhenius`, `linear_regression` as convenience built-ins | Done | v0.9.8 |
| 2 | `atomic_mass`, `atomic_number` for ~50 elements | Done | v0.9.8 |
| 3 | bar, atm, g, Bq, Sv, Gy as units | Done | v1.x |
| 3 | pH functions / % w/v | Done | v1.0 |
| 4 | Tutorials + blog posts | Not prioritized (in-repo examples cover the path) |
| Bonus | Bioinformatics quick wins (`Sequence[DNA/RNA/Protein]`, `gc_content`, `transcribe`, `translate`, `reverse_complement`, `k_mer_count`) | Done | v1.16 |
| Bonus | Cheminformatics via `pyimport rdkit` (`smiles_descriptors`, `lipinski_rule_of_five`) | Done | v1.16 |
| Bonus | Molecular dynamics via OpenMM (`md_simulate_lj` with unit validation) | Done | v1.14 |
| Bonus | Standard library modules `use chemistry`, `use biology` | Done | v1.7 |

**Status:** This roadmap is effectively closed — the original Phases 1–3 are done, Phase 4 (outreach) is a marketing question, not engineering. For current chemistry/biology features, see `Dedekind_Language_Specification.md` §15.12 (standard library) and §15.22 (bioinformatics).

---

## 1. Goal and Benefit

This roadmap prioritizes **extensions and visibility** of the Dedekind language specifically for **chemists** and **biologists**. Mathematics and physics are already well covered; this roadmap focuses on:

- **Units and conventions**: mol, L, M, ppm, bar — the "language" of the lab and textbook.
- **Examples and narratives**: kinetics, equilibrium, dose-response, growth — showing that Dedekind is also designed for chemistry/biology.
- **Convenience**: named models (e.g. Michaelis-Menten, logistic growth) and possibly wrappers (linear regression) for typical analyses.
- **Documentation**: A clear section "Dedekind for Chemistry & Biology" and references to matching examples.

**Dependency**: Builds on the [Features Implementation Roadmap](Features_Implementation_Roadmap.md) and the existing runtime (ODE, `fit`, units, uncertainty, distributions).

---

## 2. What is already usable (basis)

| Need (chemistry/biology) | Already present in Dedekind |
|--------------------------|------------------------------|
| Thermodynamics / statistics | `R_gas`, `N_A`, `k_B`, distributions, `fit()`, MCMC/HMC |
| Kinetics / dynamics | `ode_solve`, `exp`, `log`, units |
| Curve fitting | `fit(loss_fn, ..., method="gd"\|"mcmc"\|"hmc")` |
| Errors & uncertainty | `uncertain(value, std)`, compile-time unit check |
| Spectra / signals | `fft`, `integrate`, tensors, `plot()` |
| Concentrations, SI | `Quantity`, `[m]`, `[s]`, `[kg]` — extensible to mol, L, M |

With these, many chemical and biological computations (kinetics, fitting, error propagation) are **possible without new core features**; what is mainly missing is **unit conventions**, **examples**, and **visibility**.

---

## 3. Feature Overview and Phases

| Feature | Benefit | Effort | Phase |
|--------|--------|---------|--------|
| **Units mol, L, M, ppm** | Concentrations, amounts of substance, dilutions; compile check for mol+L. | Low–medium | 1 |
| **Chemistry/biology examples** | 1st-order kinetics, Michaelis-Menten, dose-response, logistic growth. | Low | 1 |
| **Docs "Dedekind for Chemistry & Biology"** | README/spec section + references to examples; clear target audience. | Low | 1 |
| **Convenience functions** | `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`. | Low | 2 done |
| **Linear regression wrapper** | `linear_regression(x, y)` → [slope, intercept] (internally `fit`). | Low | 2 done |
| **Chemical elements** | `atomic_mass("C")` (g/mol), `atomic_number("C")`; IUPAC-close, ~50 elements; molar mass e.g. 2*atomic_mass("H")+atomic_mass("O"). | Low | 2 done |
| **Additional units** | bar, atm, g; radioactivity Bq, Sv, Gy. pH note (pH = -log10([H+]); % w/v optional. | Low | 3 done (bar, atm, g, Bq, Sv) |
| **Tutorial / blog** | "Kinetics in Dedekind", "EC50 fitting with Dedekind" (optional). | Medium | 4 |

---

## 4. Dependencies and Order

- **Phase 1** builds only on the existing runtime: extend unit strings (mol, L, M, ppm), write examples, add a docs section. No new compiler features needed.
- **Phase 2** adds lean helper functions in `ml_runtime.py` (Michaelis-Menten, logistic growth, possibly Arrhenius; optionally `linear_regression`).
- **Phase 3** extends units (bar, atm) and documents conventions (pH, %).
- **Phase 4** is optional: tutorials, blog posts, additional examples.

Coordination with [Features_Implementation_Roadmap](Features_Implementation_Roadmap.md): units mol/L/M are consistent with the existing unit system (Quantity, compile check); new functions only use existing primitives (`fit`, `ode_solve`, `exp`, …).

---

## 5. Implementation Phases (Detail)

### Phase 1: Units, Examples, Docs (estimated: 1–2 weeks)

**Goal**: Chemists and biologists recognize Dedekind as suitable (units, examples, clear docs).

**Steps**:

1. **Units mol, L, M, ppm**
   - In `_unit_simplify` (or display) and in `units_checker.py` (KNOWN_UNITS), support conventions for **mol**, **L**, **M** (= mol/L), and **ppm**.
   - Literals: e.g. `0.1[M]`, `1[mol]`, `50[ppm]`; M = mol/L consistent with existing unit arithmetic (multiplication/division).
   - Optional: extend compile check so that e.g. mol + L is only allowed in meaningful combinations (or initially only runtime as before).

2. **Examples**
   - **`chemistry_kinetics.ddk`**: 1st-order reaction \(c(t) = c_0 e^{-kt}\); `ode_solve` or analytically with `exp`; units [M], [1/s].
   - **`chemistry_michaelis_menten.ddk`** or **`dose_response.ddk`**: \(v = V_{\max} [S]/(K_M + [S])\); fitting with `fit` to synthetic or typical data; output EC50/IC50 or \(K_M\), \(V_{\max}\).
   - **`biology_growth.ddk`**: logistic growth \(dN/dt = r N (1 - N/K)\) with `ode_solve`; or exponential phase with `exp`.

3. **Documentation**
   - **README**: section "Dedekind for Chemistry & Biology" (briefly: units, fitting, ODE, uncertainty; link to the new examples).
   - **Language Spec**: short subsection under §15 (e.g. §15.12 "Units and models for chemistry and biology") with mol, L, M, ppm and a reference to the kinetics/growth examples.

**Success criterion**: (1) Concentrations in [M] or [mol/L] run; (2) at least two new examples (e.g. kinetics + dose-response or growth) run; (3) README and spec contain the new section with links.

---

### Phase 2: Convenience Functions (estimated: 1 week) done

**Goal**: Typical models as named functions — less boilerplate, better readability.

**Delivered**:

1. **Runtime functions in `ml_runtime.py`**
   - **`michaelis_menten(S, Vmax, Km)`**: \(v = V_{\max} S / (K_M + S)\); S, Vmax, Km as tensors/scalars; differentiable.
   - **`logistic(t, r, K, N0)`**: Analytical solution \(N(t) = K / (1 + (K/N_0 - 1) e^{-rt})\).
   - **`logistic_growth_dt(N, r, K)`**: \(dN/dt = r N (1 - N/K)\) as RHS for `ode_solve` (e.g. `fn rhs(t, y) { return [logistic_growth_dt(y[0], r, K)] }`).
   - **`arrhenius(T, A, Ea)`**: \(k = A e^{-E_a/(R T)}\); R = `R_gas.value` (J/(K·mol)).
   - **`linear_regression(x, y)`**: returns `[slope, intercept]` via `fit(..., method="gd")`.

2. **Examples**: `dose_response.ddk` uses `michaelis_menten`; `biology_growth.ddk` uses `logistic_growth_dt` and `logistic`; `chemistry_arrhenius.ddk`, `linear_regression.ddk` are new.

**Success criterion**: Met — Michaelis-Menten, logistic growth, Arrhenius, and linear regression are callable in one line; the examples run.

**Phase 2 additions: Chemical elements** done

- **`atomic_mass(symbol)`**: atomic mass in g/mol (Quantity); IUPAC-close for ~50 elements (H, C, N, O, S, P, Na, Cl, Fe, Cu, Zn, …).
- **`atomic_number(symbol)`**: atomic number (int).
- **Benefit**: molar mass e.g. H₂O = `2*atomic_mass("H") + atomic_mass("O")`; ethane C₂H₆ = `2*atomic_mass("C") + 6*atomic_mass("H")`.
- **Example**: `chemistry_elements.ddk`.

---

### Phase 3: Additional Units and Conventions (estimated: 3–5 days)

**Goal**: Pressure (bar, atm), pH note, possibly % (w/v) — for completeness in the docs.

**Steps**:

1. **Units**: **bar**, **atm** in display/conventions (e.g. in `_unit_simplify` or docs); conversion 1 bar = 10^5 Pa, 1 atm = 101325 Pa if constants are used.
2. **Documentation**: short note on **pH**: "pH = -log10([H+])"; in Dedekind: `pH = -log10(H_plus)` with `H_plus` in [M]. No new type needed.
3. **Optional**: **%** (w/v, v/v) as a unit or in docs as a convention (e.g. 1 % = 10 g/L for w/v).

**Success criterion**: bar/atm mentioned in docs and possibly runtime; pH explained in spec; no regressions.

---

### Phase 4: Tutorials and Visibility (optional, estimated: 1–2 weeks)

**Goal**: Increase visibility outside the repo (tutorials, blog, possibly a conference/workshop).

**Steps**:

1. **Tutorial "1st-order kinetics in Dedekind"**: step by step from ODE to fitting; target audience chemistry students or teachers.
2. **Tutorial "EC50 / dose-response in Dedekind"**: data → `fit` with Michaelis-Menten/Hill → EC50/IC50; target audience biology/pharmacology.
3. **Blog post or project page**: "Dedekind for chemistry & biology" with a short summary and links to the examples.
4. **Optional**: talk/poster (e.g. chemistry/biology department, OSS project).

**Success criterion**: At least one tutorial or blog post published; linked from README or project page.

---

## 6. Risks and Options

| Risk | Mitigation |
|--------|------------|
| Units mol/L/M require an extension of the unit parser | First only as a string convention (e.g. `[M]` = mol/L); no new grammar needed. |
| Too many domain-specific functions | Only a few, very widespread models (Michaelis-Menten, logistic); the rest in examples as a formula. |
| Target audience does not find Dedekind | Phase 1 docs + examples; Phase 4 tutorial/blog; search terms "Dedekind language chemistry kinetics", etc. |

---

## 7. References and Next Steps

- **Features Implementation Roadmap**: [Features_Implementation_Roadmap.md](Features_Implementation_Roadmap.md) — units, fitting, ODE, uncertainty are the basis; this roadmap only extends conventions and examples.
- **Language Specification**: §15 Standard Library (Physical Units, ODE, Fitting, Math); new section §15.12 or addition to §15.2 for mol, L, M.
- **Code base**: `src/compiler/ml_runtime.py` (Quantity, _unit_simplify, fit, ode_solve); `src/compiler/units_checker.py` (KNOWN_UNITS); `examples/dedekind/`.
- **Next concrete step**: Phase 1 — units mol/L/M in runtime/docs, two examples (kinetics + dose-response or growth), README section "Dedekind for Chemistry & Biology".

---

*This document is the roadmap for chemistry- and biology-specific extensions of Dedekind. It builds on the Features Implementation Roadmap and the existing language.*
