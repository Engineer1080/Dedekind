# Changelog

Historical record of Dedekind releases. Most recent first.


### What's New in v2.9.0 (Public Release, Library Consolidation, New Inference Algorithms)

**Inference & Sampling**
- **No-U-Turn Sampler (`nuts`):** Adaptive trajectory-length HMC variant for Bayesian inference. Available standalone and via `fit(..., method="nuts")`. Eliminates the manual leapfrog-step tuning that classical HMC requires.
- **Variational Inference (`vi`):** Mean-field Gaussian VI with ELBO optimization via Adam. Returns a posterior approximation `(mu, sigma)` and is also exposed through `fit(..., method="vi")`.

**Standard Library Consolidation**
- **Atomic (`use atomic`):** Merged the former `molecular` (Molecular Dynamics, OpenMM bridge) and `crystallography` (structure analysis, scattering factors) modules into a single `atomic.ddk` library. Function signatures (`molecular_lj_simulate`, `cryst_symmetry_apply`, …) preserved.
- **Signals (`use signals`):** Folded the former `control`, `dsp`, and `electronics` modules into `signals.ddk`. Block-diagram control (`pid_block`, `state_space_block`, …), biquad/FIR/IIR filters, circuit primitives, and Z-transform helpers now share one namespace. `use control`, `use dsp`, and `use electronics` remain available as thin compatibility wrappers.
- **Geometry into Math:** Dropped the standalone `geometry` module; `area_circle`, `volume_sphere`, solids-of-revolution helpers and friends live in `math.ddk`.
- **Legacy Mathlib Removed:** Deleted the deprecated `mathlib` example module; its helpers (`square`, `cube`, `PHI`) are part of `math.ddk`.

**Packaging & Distribution**
- **First PyPI release.** Distributed as `dedekind` on PyPI, installable with `pip install dedekind`. Extras: `[jupyter]`, `[plot]`, `[sci]`, `[geo]`, `[bio]`, `[md]`, `[ml]`, `[all]`.
- **Standard library ships with the wheel.** `use math`, `use physics`, `use atomic`, etc. now resolve against `dedekind/stdlib/*.ddk` inside the installed package — no git clone required.
- **CLI hardening.** `dedekind --help`, `dedekind --version`, and proper non-zero exit codes on missing or invalid arguments.
- **Trusted Publishing workflow.** Tag-triggered GitHub Actions release builds sdist + wheel, publishes to PyPI via OIDC (no API tokens), and creates a GitHub Release with auto-generated notes.
- **Apache 2.0 license** with filled-in copyright header.
- **CI matrix:** Python 3.10, 3.11, 3.12 on Ubuntu, with full test suite and end-to-end compilation of all 217 examples per run.

**Documentation & Project Hygiene**
- **English-only documentation.** Translated remaining German user-facing strings, comments, and example output to English; internal-only documents removed from the public repo.
- **Documentation reorganized.** `Documentation/` folded into `docs/`, README split into focused entry-point + per-domain references.
- **Roadmap refreshed.** Phases 1–20 marked complete; MLIR/AOT backend explicitly classified as planned/experimental (not production).

**Bug Fixes & Robustness**
- Bayes test suite now seeds the RNG (`seed(42)`) to eliminate stochastic CI flakiness from NUTS/VI assertions.
- Optional-dependency import errors (scipy in `confidence_interval`/`butterworth_filter`) now surface as actionable `ImportError` messages pointing at the `[sci]` extra.
- Tensor-conversion exception handling narrowed from bare `except:` to `(TypeError, ValueError, RuntimeError)` for better debuggability.

### What's New in v2.3.0 (Space Physics & Orbital Mechanics)

- **Differentiable N-Body Simulation (`use space`):** Native differentiable N-body integration using 4th-order Runge-Kutta (RK4) with an optional softening parameter.
- **Kepler Equation Solver:** Robust Newton-Raphson solver to determine the eccentric anomaly $E$ from mean anomaly $M$ and eccentricity $e$.
- **Coordinate Transformations:** Differentiable conversion between Keplerian orbital elements (semi-major axis, eccentricity, inclination, right ascension of the ascending node, argument of periapsis, true or eccentric anomaly) and Cartesian state vectors (position & velocity).
- **Trajectory Optimization:** Enables direct L-BFGS or GD-based optimization of spacecraft trajectories and orbital maneuvers (e.g., Hohmann transfers or orbit targeting) using the Dedekind compiler and autograd.

### What's New in v2.2.0 (Differentiable Engineering)

- **Differentiable Control Theory (`use control`):** Native support for block-diagram-based simulations. Enables stateful/static blocks (PID controllers, integrators, transfer functions, saturation blocks) and automatic loop resolution for parameter-based optimization.
- **Differentiable Fluid Dynamics (`use fluid_dynamics`):** Vectorized Lattice Boltzmann Method (D2Q9) for fluid simulation of incompressible flows with continuously differentiable obstacle masks (cylinders, NACA airfoils) for drag and lift optimization.
- **Differentiable Structural Mechanics (`use structural`):** Stationary 2D Finite Element simulation (Q4 bilinear elements) of linear elasticity problems and topology optimization (generative design) via the SIMP material density model and Optimality Criteria (OC) solver with Unicode ASCII visualization in the terminal.
- **Differentiable Heat Transfer & Thermodynamics (`use thermal`):** Stationary and transient (implicit backward Euler method) thermal diffusion simulation on Q4 elements, thermal SIMP conductance interpolation, and OC topology optimization for heat-dissipating structures (heatsinks).
- **Differentiable DSP & Electronics:** Native biquad IIR and FIR filter structures with autograd, and a complex MNA nodal network solver for AC circuit analysis with complex phase angles.

### What's New in v2.0.0 (Release)

- **Earth Sciences, Climatology & Meteorology:** Dedekind now provides native support for Earth science calculations (e.g., `coriolis_parameter`, `saturated_vapor_pressure`, `dew_point`, `wind_chill`, `heat_index`) and geophysical units like `hPa`.
- **Cleaned and Unified Codebase:** All duplicates have been eliminated. The cheminformatics functions (`smiles_descriptors`, `lipinski_rule_of_five`) have been unified and now feature a robust, pure Python fallback if `rdkit` is not installed.
- **Consolidation of All Features:** Full merge of the Quantum Computing Bridge, Clifford/Geometric Algebra G(3,0), Generics, Try/Catch & Slicing, and Multi-File Modules into a stable master branch. All 52 test suites pass.

### What's New in v1.20.0

- **Generics / Type Parameters** with real enforcement — `fn name<T, U, ...>(...)`.
  ```dedekind
  fn add_same<U>(a: [U], b: [U]) -> [U] { return a + b }

  add_same(2[m], 3[m])        // U binds to m -> 5[m]
  add_same(10[kg], 5[kg])     // U binds to kg -> 15[kg]
  add_same(2[m], 100[cm])     // U binds to m, 100[cm] auto-converts -> 3[m]
  add_same(2[m], 3[kg])       // ValueError: Dimension Mismatch
  ```
- **Polymorphic Unit Variables:** `[U]` declared with `<U>` enforces unit consistency across arguments and return value. The first argument binds U; subsequent arguments are checked for the same dimension and automatically converted if necessary (m vs cm). Mismatch (m vs kg) throws `ValueError: Type-param unit 'U' in fn(b): already bound to [m], here [kg].`
- **Shape Parameters:** `<N>` with `Vector[N]` makes the symbolic dimension mechanism from v1.9 explicit. The already working cross-arg consistency (`dot<N>(a: Vector[N], b: Vector[N])`) is now declared in the function header.
- **Multiple Type Parameters:** `fn pair<A, B>(a: [A], b: [B])` keeps A and B independent — each binds separately.
- **Units Checker Patch:** Units that are type parameters are treated as "polymorphic" at compile time and skip the static check; consistency validation happens at runtime via the `_unit_env` dict.
- **Difference from Python's `typing.TypeVar`:** Python's generics are purely documentary — the interpreter ignores them. Dedekind actively enforces them: dimensionally inconsistent calls abort, and dimensionally compatible but differently-scaled calls are automatically converted. This makes Dedekind superior to Python in terms of type safety.
- Example: `generics_demo.ddk`. Test: `generics_test.ddk` (9 asserts: same unit, auto-conversion, mismatch, multi-parameter, shape). 46/46 tests green, 102/102 examples compile.

### What's New in v1.19.0

- **Multi-File Modules with Visibility.** Two related language features for real project structures.
- **Dotted Module Paths:** `use foo.bar.baz` resolves to `modules/foo/bar/baz.ddk` (or different path in examples). Nested directory structure for thematic organization instead of a flat `modules/` folder.
  ```dedekind
  use compiler_features.demo_geometry.area       // -> examples/dedekind/compiler_features/demo_geometry/area.ddk
  use compiler_features.demo_geometry.volume     // -> examples/dedekind/compiler_features/demo_geometry/volume.ddk
  ```
- **`pub fn` for Export Control:** Only functions declared as `pub fn name()` are visible outside the module; all other functions are renamed to `__ddk_<modpath>_<name>` at compile time and are therefore unreachable from the outside.
  ```dedekind
  // examples/dedekind/compiler_features/demo_geometry/area.ddk
  fn priv_pi() { return 3.14159265358979 }   // private
  pub fn circle_area(r) {                     // exported
      return priv_pi() * r * r
  }
  ```
- **Backward Compatible:** Modules without any `pub` declaration run in **legacy mode** — all functions remain public. This leaves the 6 existing standard library modules (`physics`, `stats`, `chemistry`, `biology`, `math`, `ml`) functional without change. If you want visibility control, declare at least one function as `pub` — from then on, the opt-in mode applies.
- **Visibility Mangling** at compile time via `_apply_module_visibility(mod_ast, module_name)`: AST walker (`_rename_in_ast`) replaces both `FunctionDef.name` and every `Identifier` reference to private functions — calls inside the module are consistently renamed.
- Example: `multi_file_modules_demo.ddk` with two submodules `compiler_features.demo_geometry.area` and `compiler_features.demo_geometry.volume`, plus legacy module `math`. Test: `multi_file_modules_test.ddk` (10 asserts: dotted paths, private functions invisible via try/catch from v1.17, backward compatibility). 45/45 tests green, 101/101 examples compile.

### What's New in v1.18.0

- **3D Geometric / Clifford Algebra G(3,0) natively.** New `MultiVector` class with 8 real components (scalar, e1, e2, e3, e12, e13, e23, e123) and native operator overloading. Unifies scalars, vectors, bivectors (oriented areas), and pseudoscalars in one algebra — rotations become sandwich products, complex numbers are a subalgebra, quaternions are the even subalgebra.
  ```dedekind
  e1 = vector(1, 0, 0)
  e2 = vector(0, 1, 0)
  print(e1 * e2)                      // e12 (bivector)

  R = rotor(1.5707963, 1, 0, 0)       // 90 deg in the e1-e2 plane
  print(rotate(e1, R))                // ~ e2 (rotation via sandwich product)
  ```
- **Constructors:** `scalar(s)`, `vector(x,y,z)`, `bivector(b12,b13,b23)`, `pseudoscalar(s)`, `multivector(8 args)`, `rotor(angle, b12, b13, b23)`.
- **Operations:** `+`, `-`, `*` (geometric product), scalar multiplication; methods `.wedge(b)` (outer/grade raising), `.dot(b)` (inner/grade reduction), `.reverse()`, `.grade(n)`, `.norm()`, `.scalar_part()`.
- **Function `rotate(v, R)`** makes the sandwich product `R v ~R` explicit.
- **Multiplication table** (8×8) calculated via bit patterns: result_blade = `a XOR b`, sign from the number of swaps. Numerically stable, no external dependencies.
- **Deliberately NOT provided:** no signatures beyond G(3,0) (no spacetime G(3,1), no Conformal Geometric Algebra G(4,1)). Whoever needs them: use `pyimport clifford` directly. Dedekind's contribution: the canonical 3D cases (robotics, graphics, classical physics) as typed native built-ins, without extra dependencies.
- Example: `geometric_algebra_demo.ddk` (8 demos: basis, geom. product, squares, rotor, rotor composition, bivector orientation, pseudoscalar, complex numbers as subalgebra). Test: `geometric_algebra_test.ddk` (16 asserts). 44/44 tests green, 100/100 examples compile.

### What's New in v1.17.0

- **`try { ... } catch e { ... }` — Native Error Handling.** Previously, researchers could not catch exceptions in `.ddk` code. Now:
  ```dedekind
  try {
      content = read_file("/maybe/exists.json")
      return json_parse(content)
  } catch e {
      print("File not readable:", e)
      return {}
  }
  ```
  Nestables. Codegen emits standard Python `try: ... except Exception as e: ...`. This turns Dedekind from a scripting DSL into a real application language — defensive programming and fallback paths are first-class starting with v1.17.
- **Python-Style Slicing Syntax `x[a:b]`, `x[:b]`, `x[a:]`, `x[::s]`, `x[a:b:s]`, `x[:]`.** Previously you had to write `x.narrow(0, a, b-a)` — now:
  ```dedekind
  x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  head = x[:5]            // [10..50]
  tail = x[5:]            // [60..100]
  every2nd = x[::2]       // [10, 30, 50, 70, 90]
  middle = x[2:8:2]       // [30, 50, 70]
  ```
  Works on lists, tensors, anything that understands `__getitem__` with `slice`. New `Slice(start, stop, step)` AST node; each component is optional.
- **Note: PyTorch limitation.** Negative step (`x[::-1]` for reverse) is not supported by PyTorch — use `torch.flip(x, [0])`. Slice with positive step works unchanged.
- Example: `try_catch_slicing_demo.ddk`. Test: `try_catch_slicing_test.ddk` (18 asserts: 7 slice variants, try/catch flat and nested, combined try+slice). 43/43 tests green, 99/99 examples compile.

### What's New in v1.16.0

- **`Sequence[DNA]` / `Sequence[RNA]` / `Sequence[Protein]` Shape Annotation:** Type-safe sequence validation. Accepts exactly the characters of the respective alphabet — otherwise throws `ValueError: Sequence[DNA]-Check in fn(seq): invalid character 'U' at position 3 (allowed: ACGNT)`. Prevents classic bioinformatics bugs like mixing up RNA and DNA, which silently produce incorrect GC counts in Python/Biopython.
- **Native Bio-Built-ins** (no `pyimport` required):
  - `gc_content(dna)` — fraction of G+C (0..1)
  - `reverse_complement(dna)` — DNA reverse-complement
  - `transcribe(dna)` — DNA → RNA (T → U)
  - `translate(rna, stop_at_stop=true)` — RNA → Protein via standard codon table
  - `k_mer_count(seq, k)` — all overlapping k-mers with counts
- **Cheminformatics via `pyimport rdkit`:**
  - `smiles_descriptors(smiles)` returns a dict with `mw` ([g/mol]), `logp`, `num_atoms`, `num_heavy_atoms`, `num_rings`, `num_aromatic_rings`, `hbd`, `hba`, `tpsa`, `num_rotatable_bonds`. MW comes back as `Quantity` — directly usable in unit-aware calculations.
  - `lipinski_rule_of_five(smiles)` checks the four drug-likeness criteria (MW≤500, LogP≤5, HBD≤5, HBA≤10) and returns `{checks, violations, passes}`.
- **Central Dogma Example:**
  ```dedekind
  fn dna_to_protein(dna: Sequence[DNA]) -> Sequence[Protein] {
      rna = transcribe(dna)
      return translate(rna)
  }
  protein = dna_to_protein("ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTG")
  // "MALWMRLLPL" (Insulin signal peptide start)
  ```
- **Deliberately NOT provided:** no sequence alignment (Smith-Waterman, Needleman-Wunsch) as native built-in — researchers call `pyimport Bio.Align as aln` directly. No PDB/structure parsing — `pyimport Bio.PDB`. No phylogenetic trees. Dedekind's role: type safety + quick wins, not replacing Biopython/rdkit.
- Example: `bioinformatics_demo.ddk` (DNA pipeline, k-mer analysis, Aspirin/Caffeine/Ibuprofen SMILES + Lipinski). Test: `bioinformatics_test.ddk` (18 asserts). 42/42 tests green, 98/98 examples compile.

### What's New in v1.15.0

- **`LabeledTensor[lat, lon, time]` Shape Annotation:** Tensors with axis NAMES instead of sizes — for climate, geo, and Earth science workflows. Validates at runtime that an `xarray.DataArray` has exactly these dimension names (order is irrelevant because xarray operates name-based):
  ```dedekind
  fn temperature_anomaly(t: LabeledTensor[lat, lon, time]) -> LabeledTensor[lat, lon, time] {
      return t - t.mean(dim="time")
  }
  fn zonal_mean(t: LabeledTensor[lat, lon, time]) -> LabeledTensor[lat, time] {
      return t.mean(dim="lon")
  }
  ```
  If a dimension is missing or an extra one is present: `ValueError: LabeledTensor-Shape-Mismatch ... missing dimensions: ['time']; extra dimensions: ['level']`.
- **`labeled_tensor(data, dims, coords, name, attrs)` Built-in:** creates an `xarray.DataArray` directly in Dedekind. Accepts tensors, numpy arrays, or lists; attaches axis names, coordinates, and meta-attributes (units, CRS, source, ...):
  ```dedekind
  T = labeled_tensor(raw_data,
      dims=["lat", "lon", "time"],
      coords={"lat": lats, "lon": lons, "time": times},
      attrs={"units": "K", "crs": "EPSG:4326"}
  )
  ```
- **The USP compared to raw xarray:** xarray itself has no type system — `data_array.mean(axis=2)` instead of `dim="time"` is a classic bug in climate research. `LabeledTensor[...]` enforces the correct axis names at function entry and return.
- **Deliberately NOT provided:** no re-implementation of xarray operations (regridding, interp_like, groupby_bins, etc.) — whoever needs them calls them directly via `da.regrid(...)`. Dedekind's role is the annotation layer. **No Dask/distributed support** — zarr/Dask-backed DataArrays continue to work via `pyimport xarray`, but we only validate the dim names, not chunk topology.
- Example: `labeled_tensors_demo.ddk` (4 x 8 x 12 climate dataset: temporal mean, zonal mean, anomaly, `.sel` slicing by coordinate). Test: `labeled_tensors_test.ddk` (9 asserts: shape, dim names, order insensitivity, coords, attrs). 41/41 tests green, 97/97 examples compile.

### What's New in v1.14.0

- **Molecular Dynamics via OpenMM Bridge:** New built-in `md_simulate_lj(n_particles, sigma, epsilon, mass, temperature, dt, n_steps, box_size, friction, seed)` starts a Lennard-Jones NVT simulation (Langevin integrator) on the OpenMM C++ kernel — **with enforced dimensional safety** before execution.
  ```dedekind
  res = md_simulate_lj(
      n_particles=27,
      sigma=3.4[Angstrom],         // alternative input in A
      epsilon=0.238[kcal/mol],     // or kJ/mol
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
- **USP compared to raw `pyimport openmm`:** Dedekind validates the dimensions of ALL force field parameters before the C++ kernel runs. `epsilon=0.238[eV]` throws `ValueError: epsilon=0.238[eV] has incorrect dimension; expected compatible to [kJ/mol] (molar_energy)`. In raw OpenMM, eV-vs-kcal/mol mixup is a silent bug.
- **New MD Units in the Dimension System:**
  - Length: `nm`, `Angstrom`, `pm`, `fm`
  - Mass: `amu`, `Da` (Dalton)
  - Time: `fs`, `ps`, `ns`, `us`
  - New dimension `molar_energy`: `kJ/mol` (base), `kcal/mol` (4.184), `J/mol`, `eV/atom`, `Hartree/mol`
- **Grid Initialization instead of Random Placement:** avoids NaN energies due to particle overlaps (LJ-r⁻¹² explodes at r → 0). Slight perturbation to break perfect grid symmetry.
- **Deliberately NOT provided:** no protein force fields (AMBER, CHARMM, OPLS), no implicit solvent, no REMD/free-energy methods. Whoever needs them: use `pyimport openmm.app as omm_app` — we build the simple LJ routine as a bridgehead, not as a replacement for OpenMM.
- Example: `md_lennard_jones_demo.ddk` (Argon cluster, equilibration + production). Test: `md_simulate_lj_test.ddk` (8 asserts: shape, energies, temperature, Angstrom/kcal alternative input). 40/40 tests green, 96/96 examples compile.

### What's New in v1.13.0

- **Operations Research with Declarative MILP DSL:** New built-ins `Variable(name, lower, upper, integer)` and `optimize_milp(objective, constraints, sense)`. Constraints are written directly via operator overloading — no more manual construction of A_ub/b_ub matrices. Example:
  ```dedekind
  x = Variable("distance", lower=0[km])
  trucks = Variable("trucks", lower=1, integer=true)
  cost = 2.5 * x + 1000 * trucks
  res = optimize_milp(cost, [
      x >= 500[km],
      trucks * 200[km] >= x
  ])
  // res = {distance: 500.0, trucks: 3.0, objective: 4250.0, status: "..."}
  ```
- **Unit Awareness in Constraints — the USP:** `x >= 500[km]` with `x: Variable(lower=0[km])` fits; `x >= 500[kg]` throws `ValueError: MILP units do not match in constraint: [km] vs [kg]`. No other MILP library (Gurobi, cvxpy, pyomo, PuLP) has this.
- **Operator Overloading:** `_MILPVariable` overloads `+`, `-`, `*`, `/`, `>=`, `<=` with linearity checking (non-linear `x * y` throws immediately `ValueError`). Variables are identity-hashable so that coefficient dicts work internally.
- **Three Demos in `optimize_milp_demo.ddk`:** Vehicle Routing (distance + truck count), Product Mix (max profit under resource constraints), Energy Mix (cheapest source first under demand constraint).
- **Existing `milp(c, A_ub, b_ub, ...)` call (v1.5) remains unchanged** — the DSL is a convenience layer, not a replacement.
- Test: `optimize_milp_test.ddk` (13 asserts: LP min/max, integer variables, Vehicle Routing with km, Energy Mix with kW). 39/39 tests green, 95/95 examples compile.

### What's New in v1.12.0

- **`Graph[N, E]` as Shape Annotation:** Function signatures now recognize a graph type via the annotation mechanism established in v1.9:
  ```dedekind
  pyimport torch_geometric.data as pyg_data
  fn coordination(g: Graph[N, E]) -> Scalar { return g.num_edges / g.num_nodes }
  fn pair_match(g1: Graph[N, E1], g2: Graph[N, E2]) -> Scalar { ... }
  ```
  Validates at runtime that the passed object has `num_nodes`/`num_edges` attributes (typically `torch_geometric.data.Data`) and that the dimensions match. Symbolic dimensions are bound and kept consistent — two graphs with `Graph[N, ...]` and `Graph[N, ...]` must have the same node count.
- **Combinable with Unit Annotations:** Node features in `[g/mol]`, edge energies in `[eV]`, bond lengths in `[pm]` — the dimensional safety that `torch_geometric` structurally lacks. Example:
  ```dedekind
  fn molecular_mass(g: Graph[N, E]) -> [g/mol] { ... }
  fn add_mass(m1: [g/mol], m2: [g/mol]) -> [g/mol] { ... }
  ```
- **Two GNN Demos via `pyimport torch_geometric`:**
  - `gnn_molecule_demo.ddk` – drug domain: H2O molecule with atomic mass in `[g/mol]` on nodes, GCNConv forward.
  - `gnn_materials_demo.ddk` – materials domain: 4-atom FCC unit cell with bond lengths in `[pm]` and bond energies in `[eV]`, GraphConv.
- **Deliberately NOT provided** (Stage 3 of the Graph Roadmap): No native message passing. The honest answer for production GNNs remains `pyimport torch_geometric` — it has 30+ Conv variants, pooling strategies, and benchmark datasets. Dedekind's role: unit awareness, shape annotation, source mapping on top of the PyG call.
- Test: `graph_shape_test.ddk` (5 asserts: node/edge counts, symbolic N consistency across two graphs). 38/38 tests green, 94/94 examples compile.

### What's New in v1.11.0

- **`graph_laplacian(adj, normalized=False)` — Spectral Graph Methods:** New built-in for the discrete Laplacian matrix of a graph. Accepts dense matrices, sparse `torch.Tensor`, and nested lists as adjacency; returns sparse if input is sparse, else dense.
  - **Combinatorial** (default): `L = D - A` (row sums = 0, all eigenvalues >= 0).
  - **Normalized symmetric**: `L_sym = I - D^{-1/2} A D^{-1/2}` (eigenvalues in `[0, 2]`).
  - Directly usable in `cg`, `gmres`, `bicgstab`, and `eigh` from the existing v1.6 solvers — no additional framework required.
- **Demo `graph_spectral_demo.ddk`:** two classic applications on an 8-node two-cluster graph:
  - **Heat Diffusion** via implicit Euler method: `cg(I + dt*L, u_prev)`. Clearly shows asymmetric diffusion across the narrow bridge edge.
  - **Spectral Partitioning** via Fiedler vector (second smallest eigenvalue): cleanly separates the two clusters at the bridge (sign splitting).
- **Deliberately NOT provided** (Stage 2+3 of the Graph Roadmap):
  - No `Graph[N, E]` shape type. Comes in v1.12 as a wrapper around `torch_geometric.data` with unit annotations — the real USP compared to PyG.
  - No native message passing. Anti-pattern: a half-finished PyG rebuild would be worse than `pyimport torch_geometric`. If native, then with the full research budget (Stage 3, several weeks).
- Test: `graph_laplacian_test.ddk` (path graph, two-cluster, normalized variant, Fiedler partitioning). 37/37 tests green, 92/92 examples compile.

### What's New in v1.10.0

- **`partial(u, x, order=n)` — Physics-Informed Neural Networks (PINNs):** New built-in for the derivative of network outputs with respect to network inputs via autograd. Unlike the existing `grad(fn, x)` (which differentiates a *function* at a point), `partial` works with already computed tensors: `u = net(x)` is evaluated, then `du_dx = partial(u, x)`. This allows PDE residuals (u_t - α·u_xx, u_x + u, ...) to be expressed directly in the loss function. Works with `create_graph=True` and `retain_graph=True`, so that `fit()` can optimize over the residual — this is exactly the mechanism that otherwise costs researchers weeks of custom loop implementation in PyTorch.
- **`fit()` Patch for PINNs:** `_to_tensor` now catches mixed tensor lists (PINN data + collocation tensors with different shapes); `fit()` zeroes out the `.grad` accumulators of all `requires_grad` tensors in `data` per step (prevents memory leak during long MCMC/PINN trainings).
- **Two PINN Examples:**
  - `pinn_ode_demo.ddk` — 1st-order ODE `y' + y = 0, y(0)=1`. PINN learns exp(-x) in 2000 steps with ~1% error without ever seeing a solution value.
  - `pinn_oscillator_demo.ddk` — 2nd-order ODE `u'' + u = 0, u(0)=1, u'(0)=0` on [0, π/2]. Demonstrates `partial(u, x, order=2)`. Error < 1% in 5000 steps.
- **Both examples transparent to hyperparameter limits:** On larger intervals or with larger frequencies, naive PINNs fail — that is Stage 3 of the roadmap (NTK-based loss balancing, adaptive sampling, Fourier features), not v1.10. We deliberately do not provide a "magic" `.with_physics_loss(pde)` that raises expectations we cannot meet.
- Test: `partial_test.ddk` (17 asserts: x², sin, cubic, 2-D input); 36/36 tests green, 91/91 examples compile.

### What's New in v1.9.0

- **Shape Annotations for Tensors:** Function signatures now allow explicit tensor shapes — as a static guarantee for research code that otherwise silently suffers from broadcasting:
  ```dedekind
  fn dot_product(a: Vector[3], b: Vector[3]) -> Scalar { ... }
  fn matvec(M: Matrix[2, 3], v: Vector[3]) -> Vector[2] { ... }
  fn weighted_dot(x: Vector[N], w: Vector[N]) -> Scalar { ... }  // symbolic N
  fn forward(x: Tensor[batch, 28, 28]) -> Tensor[batch, 10] { ... }
  ```
  Four type constructors: `Scalar` (0-D), `Vector[n]` (1-D), `Matrix[m, n]` (2-D), `Tensor[d1, d2, ...]` (N-D). Dimensions are integer literals **or** identifiers (symbolic, bound to the caller). Symbolic dims are bound at the first occurrence and checked for consistency — `weighted_dot(Vector[3], Vector[2])` throws `ValueError: Symbolic shape dimension 'N' in weighted_dot(w): already bound to 3, here 2.` Works on lists, tuples, and `torch.Tensor`. Return shape is automatically checked after each `return` statement. Detects incorrect broadcasts and shape mismatches that silently produce incorrect results in raw NumPy/PyTorch.
- **`unwrap(x)` — Quantity Stripping for Hot Paths:** New built-in removes unit wrappers at the entry of pure-context functions, so that `jit`/`grad`/`fit`/`metropolis`/`hmc`/`sde_solve` can work with bare floats:
  ```dedekind
  fn pure_loss(params, data) {
      a = unwrap(params[0])   // Quantity(2.0, "m") -> 2.0
      b = unwrap(params[1])
      x = unwrap(data[0])
      diff = a * x + b - unwrap(data[1])
      return diff * diff
  }
  ```
  Processes `Quantity`, `UncertainQuantity` (std discarded), 0-d `torch.Tensor` (via `.item()`), lists/tuples (element-wise), and any other values (passthrough). The compile-time unit checking has already validated the dimensions; at runtime, the bare float creates no wrapper overhead — important for 10,000+ iterations in MCMC loops or JIT-compiled graphs, where `torch.compile` treats Python objects as graph breaks.
- Examples: `shape_annotations_demo.ddk`, `quantity_stripping_demo.ddk`. Tests: `shape_annotations_test.ddk`, `quantity_stripping_test.ddk` (35/35 green; all 89 examples compile).

### What's New in v1.8.1

- **Purity Check for pure-context calls:** New compile-time pass `purity_check.py`. Functions passed to `jit(fn)`, `grad(fn, x)`, `fit(loss, ...)`, `metropolis(log_prior, log_likelihood, ...)`, `hmc(...)` or `sde_solve(drift, diffusion, ...)` may no longer call **I/O or console built-ins** — otherwise throws `CompileError` with file and line.
- Blocked built-ins (transitive detection): `print`, `plot`, `scatter`, `contour`, `print_latex`, `print_table`, `write_file`, `read_file`, `file_exists`, `http_get`, `http_post`, `read_csv`/`write_csv`, `read_parquet`/`write_parquet`, `read_hdf5`/`write_hdf5`, `read_netcdf`, `export_notebook`.
- Transitive resolution: if `loss` calls a helper function `helper` and `helper` calls `print`, the entire path is reported (`"... calls 'print()' ... in 'helper'"`).
- Opt-out: `compile_source(..., check_purity=False)` or CLI `python -m compiler file.ddk --no-purity-check`.
- Prevents a whole class of bugs: silent `torch.compile` graph breaks, tape recording anomalies in autograd, multiple file writes in MCMC loops with 10,000 samples.
- Example: `purity_check_demo.ddk`. Test: `purity_check_test.ddk` (33/33 green).

### What's New in v1.8.0

- **Source Mapping for Runtime Errors:** If the generated Python code (or an underlying NumPy/SciPy/Torch function) throws an exception at runtime, the traceback now directly shows the original lines from the `.ddk` file including the code snippet:
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
  Implemented via `# ddk:<line>` markers in the generated code + new helper `dedekind_exec(generated_code, ddk_file, ddk_source)` in `src/compiler/compiler.py`. The exception type is preserved (`except AssertionError:` etc. continues to work), only the message is replaced with the mapped traceback. Frames within the inlined runtime are marked as `<dedekind-runtime>`; external library frames (scipy, torch) still show their real paths for stack analysis. Used in `run_tests.py`, `run_examples.py`, `compiler.py` CLI, and the Jupyter kernel.
- **`pyimport` — Python Escape Hatch into the PyPI Ecosystem:** New top-level syntax `pyimport scipy.special as sp` (or `pyimport math` with auto-alias = last segment). Any Python module can be called as `alias.name(...)`, mixing with Dedekind built-ins. Example:
  ```dedekind
  pyimport scipy.stats as st
  pyimport numpy as np
  p = st.norm.cdf(1.96)
  arr_mean = np.mean(np.array([1.0, 2.0, 3.0]))
  ```
  `pyimport` is a soft keyword (existing code with `pyimport` as a variable name remains valid). Codegen emits `import MODULE as ALIAS` at the statement location. This opens Dedekind up: any PyPI package (`rdkit`, `astropy`, `qiskit`, `polars`, ...) is immediately available.
- **`_dedekind_assert` Fix:** Correct handling of numpy scalars (`numpy.bool_`, `numpy.float64`). Previously, the tensor check (`numel()`) collided with numpy values from `pyimport` calls.
- Examples: `pyimport_demo.ddk`, `source_mapping_demo.ddk`. Tests: `pyimport_test.ddk` (32/32 green; all 86 examples compile).

### What's New in v1.7.0

- **Standard Library of Real Modules:** Six curated `.ddk` modules under `modules/` supplement global built-ins with convenience functions. **Important:** All existing built-ins (`sin`, `ode_solve`, `michaelis_menten`, `pi`, `c`, ...) remain **globally** available – `use` is purely additive and does not make imports mandatory.
  - `use physics` — `kinetic_energy`, `pendulum_period`, `escape_velocity`, `relativistic_gamma`, etc.
  - `use stats` — `z_score`, `cohens_d`, `pooled_std`, `r_squared`, `mse`/`rmse`/`mae`, etc.
  - `use chemistry` — `pH_from_concentration`, `henderson_hasselbalch`, `ka_from_pka`, `nernst_potential`, etc.
  - `use biology` — `exponential_growth`, `doubling_time`, `gompertz_growth`, `sir_model`, etc.
  - `use math` — `PHI`, `TAU`, `fibonacci`, `lcm`, `sigmoid`, `softplus`, etc.
  - `use ml` — `leaky_relu`, `elu`, `swish`, `gelu_approx`, `mse_loss`, `accuracy`, etc.
- **User-Defined Units:** `unit NAME = FACTOR[basis]` registers a new unit at compile and runtime. Examples: `unit Foot = 0.3048[m]`, `unit Mile = 1609.344[m]`, `unit eV = 1.602176634e-19[J]`. Chaining is allowed and resolved transitively. Built-in and user units mix with automatic conversion. `unit` is a soft keyword.
- **Quantity Comparison Operators:** `Quantity` now supports `<`, `<=`, `>`, `>=`, `==`, `!=` with automatic unit conversion for the same dimension. E.g., `10[cm] < 1[m]` → `True`.
- Examples: `stdlib_modules_demo.ddk`, `user_defined_units.ddk`. Tests: `stdlib_physics_test.ddk`, `stdlib_stats_test.ddk`, `stdlib_chemistry_test.ddk`, `stdlib_biology_math_ml_test.ddk`, `user_defined_units_test.ddk`.

### What's New in v1.6.0

- **Deeper Symbolics:** `solve_sym(equation, var)` solves equations symbolicaly via SymPy; `"x^2 - 5*x + 6"` → `["2", "3"]`. `simplify_sym(expr)` simplifies expressions. `series(expr, var, x0, n)` for Taylor series.
- **Sparse Iterative Solvers:** `cg(A, b)`, `gmres(A, b)`, `bicgstab(A, b)` with iteration callbacks. `jacobi_preconditioner(A)` and `ilu_preconditioner(A)` as `M=` argument for speedups. Dense, sparse tensors, and scipy.sparse matrices are all accepted.
- **Reproducible Notebook Export:** `export_notebook(source_path, output_path, format="html"|"md", title)` runs a `.ddk` file and bundles source, stdout, generated plots (base64-PNG), and SHA-256 to a single file.
- **Paper-Mode Tables:** `print_table(rows, headers, format="markdown"|"latex"|"csv"|"plain", precision)` generates tables. `UncertainQuantity` is formatted as `val ± std [unit]`, `Quantity` as `val [unit]`.
- Examples: `symbolic_solve_series.ddk`, `sparse_iterative_solvers.ddk`, `notebook_export_demo.ddk`, `paper_table_demo.ddk`. Tests: 26/26 green.

### What's New in v1.5.0

- **Benchmarking & Profiling:** `benchmark(fn, n=10, warmup=2)` measures wall time; `profile(fn)` adds peak memory and cProfile; `time_block` for ad-hoc timing.
- **JIT Backend:** `jit(fn)` wraps a function with `torch.compile` (TorchInductor) if available.
- **SDE Solvers:** `sde_solve(drift, diffusion, y0, t, method="euler_maruyama"|"milstein")` for Ito SDEs.
- **Extended Optimization:** `least_squares`, `minimize_constrained` (SLSQP/trust-constr/COBYLA), and `milp` (mixed-integer LP).
- **FEM Primitives:** `mesh_unit_square(n)`, `fem_assemble_stiffness(mesh)`, `fem_assemble_load(mesh, f)`, and `fem_poisson_2d` to solve -Δu=f.
- **`arange` for Indexing:** `arange` now returns `int64` (instead of `float32`) for safe loops: `for i in arange(N) { x[i] = ... }`.
- Example: `v1_5_features_showcase.ddk`. Tests: `benchmark_profile_test.ddk`, `jit_test.ddk`, `sde_solve_test.ddk`, `optimization_test.ddk`, `fem_test.ddk`.

### What's New in v1.4.0

- **Module System:** `use mymodule` loads `modules/mymodule.ddk` and exposes its functions/constants.
- **Reproducibility:** `seed(n)` sets seed in random, numpy, and PyTorch. `data_hash(x)` computes SHA-256.
- **DataFrames & Tabular I/O:** `DataFrame` with unit metadata; `read_csv` parses `name [unit]` headers; `write_csv`. Parquet, HDF5, and NetCDF supported via optional packages.
- **Unit-Aware Plots:** `plot()`, `scatter()`, and `contour()` automatically add units to labels (e.g., `Time [s]`).
- **`@units` Signatures:** Functions declare input/output units: `fn kinetic_energy(m: [kg], v: [m/s]) -> [J]`. Automatic conversion and verification.
- **Dict Literales:** `{"key": value}` for mapping objects.
- Example: `v1_4_features_showcase.ddk`. Tests: `use_module_test.ddk`, `seed_reproducibility_test.ddk`, `dataframe_csv_test.ddk`, `signature_units_test.ddk`, `unit_plot_test.ddk`.

### What's New in v1.3.1
- **Medicine, Pharmacology & Epidemiology:** `hill_equation`, `one_compartment_pk`, `half_life` (pharmacokinetics); `sir_model`, `basic_reproduction_number` (epidemiology); `confidence_interval`, `odds_ratio`, `sensitivity_specificity` (biostatistics). Examples: `pharmacology_quickwins.ddk`, `epidemiology_sir.ddk`, `biostatistics_quickwins.ddk`.

### What's New in v1.3.0
- **Indefinite Integrals:** `integrate_sym(expr, var)` – symbolic integration using SymPy. Example: `integrate_sym_demo.ddk`.
- **Lagrange/Hamilton:** `lagrange_ode_rhs(L)`, `hamilton_ode_rhs(H)` – RHS for ode_solve. Example: `lagrange_hamilton.ddk`.
- **Lotka-Volterra:** `lotka_volterra(x0, y0, a, b, c, d, t)` – predator-prey model. Example: `lotka_volterra.ddk`.
- **Chemical Equilibrium:** `chemical_equilibrium(K, n_A, n_B, n_C, n_D, A0, B0, C0, D0)` – mass action law. Example: `chemical_equilibrium.ddk`.

### What's New in v1.2.9
- **Absolute Value Bars:** `|expr|` = syntactic sugar for `abs(expr)`. Example: `abs_bars.ddk`.
- **Solids of Revolution:** `volume_revolution_x`, `volume_revolution_y`, etc. Example: `volume_revolution.ddk`.
- **Logical Operators:** `and`, `or`, `not`, `xor`, `nand`, `nor`, `xnor` as keywords. Example: `logical_operators.ddk`.

### What's New in v1.2.8
- **Dedekind Cuts:** `DedekindCut(x)` – construction of R from Q; `dedekind_cut_from_rational`, `dedekind_cut_sqrt2()`; arithmetic and comparisons.
- **Dedekind Rings:** `DedekindRingZ()`, `ideal(n)`, `ideal_factor(i)` – Z with unique ideal factorization.
- **Riemann Zeta Function:** `zeta(s)` – ζ(s)=Σ 1/n^s; ζ(2)=π²/6, ζ(4)=π⁴/90.
- **Riemann Sums:** `riemann_sum(f, a, b, n, method="left"|"right"|"midpoint")`. Examples: `dedekind_cuts_rings.ddk`, `riemann_zeta_sums.ddk`.

### What's New in v1.2.7
- **Dirichlet Distribution:** `Dirichlet(alpha)` – multivariate distribution on the simplex; `sample(dist)`, `log_prob(dist, value)`.
- **Dirichlet Function:** `dirichlet_function(x)` – D(x)=1 if x rational, else 0. Example: `dirichlet_distribution_function.ddk`.

### What's New in v1.2.6
- **Angles as Native Units:** `rad` and `deg` with automatic conversion (e.g., `90[deg] + (pi/2)*1[rad]` → `180[deg]`). Conversion functions: `deg_to_rad(x)`, `rad_to_deg(x)`. Example: `angle_units.ddk`.

### What's New in v1.2.5
- **Quick Wins:** units kN, MPa, MN, kPa, D, mD (civil engineering, materials, geology). Music: `cents_to_ratio`, `ratio_to_cents`, `equal_temperament`. Economics: `discount_factor`, `cobb_douglas`, `solow_rhs`. Geology: `darcy_velocity`. Materials: `johnson_mehl_avrami`, `avrami_rate`. Scientific notation (1e5, 1e-12) correctly parsed.

### What's New in v1.2.4
- **Maxwell's Equations:** `pde_maxwell_1d` (plane wave E_y, B_z); `pde_maxwell_2d` (TM mode E_z, H_x, H_y); FDTD with central differences. Example `pde_maxwell.ddk`.

### What's New in v1.2.3
- **Reaction-Diffusion:** `pde_reaction_diffusion_1d` (Fisher-KPP u_t = D∇²u + r·u·(1-u)); `pde_reaction_diffusion_2d` (Gray-Scott, Turing patterns); Example `pde_reaction_diffusion.ddk`.
- **Advection-Diffusion:** `pde_advection_diffusion_1d`, `pde_advection_diffusion_2d` for u_t + v·∇u = D∇²u; Upwind + central differences; Example `pde_advection_diffusion.ddk`.

### What's New in v1.2.2
- **Wave Equation:** `pde_wave_1d`, `pde_wave_2d` for u_tt = c²∇²u; 1st-order system reduction; central differences; periodic or Dirichlet boundary conditions; Example `pde_wave.ddk`.
- **Burgers 2D:** `pde_burgers_2d` for u_t + u·∇u = ν∇²u; Upwind for advection, central differences for diffusion; Example `pde_burgers.ddk`.

### What's New in v1.2.1
- **Advection:** `pde_advection_1d`, `pde_advection_2d` for u_t + v·∇u = 0; Upwind scheme, periodic boundaries; Example `pde_advection.ddk`.

### What's New in v1.2.0
- **Sparse CFD:** `sparse_laplacian_2d(N)`, `sparse_diffusion_step(T, L, dt, alpha)`, `sparse_diffusion_simulate` for 2D heat diffusion ∂T/∂t = α∇²T.
- **Example cfd_sparse_sim.ddk:** Real simulation (50×50 grid, 100 timesteps, contour plots).
- **Compiler Fix:** `tensor * 0` simplified to `0` only when both operands are literals.
- **Postfix Factorial:** Operator `n!` (e.g. `5!`); AST PostfixFactorial, runtime `factorial(n)`; Example `factorial_test.ddk`.

### What's New in v1.1.9
- **Patch:** `# type: ignore[reportMissingImports]` for numpy import in `balance_equation`.

### What's New in v1.1.8
- **Differential Geometry:** `christoffel_symbols(g_func, x, h)`, `riemann_tensor`, `covariant_derivative` (numerical).
- **Number Theory:** `gcd(a, b)`, `is_prime(n)`, `mod(a, m)`, `mod_inv(a, m)`, `mod_pow` – GCD, primality test, modular arithmetic.
- **Further Units:** pH functions `concentration_to_pH(c_M)`, `pH_to_concentration(pH)`; mass concentration `[percent_wv]` (= g/100mL).
- **Stoichiometry:** `balance_equation(reactants_str, products_str)` – balancing equations (e.g. H₂ + O₂ → H₂O).

### What's New in v1.1.7
- **Matrix Operator @:** `A @ B` instead of `matmul(A, B)`.
- **Special Functions:** `bessel_j0(x)`, `bessel_j1(x)`, `bessel_j(n, x)`, `legendre(n, x)`, `hypergeom(a, b, c, z)`. Requires scipy.

### What's New in v1.1.6
- **Symbolic Differentiation:** `diff_sym(expr, var)` – native expression differentiation. Supports +, -, *, /, ^, sin, cos, tan, exp, log, sqrt.

### What's New in v1.1.5
- **Assert & Tests:** `assert(condition, message)`; mini-test runner `run_tests.py` for `tests/dedekind/*.ddk`.
- **Plots:** `scatter(x, y)`, `contour(X, Y, Z, levels)`; `plot(..., xscale="log", yscale="log")`.
- **Autograd:** `jacobian(f, x)`, `hessian(f, x)`.
- **Signal & Reductions:** `fftfreq(n, d)`, `diff(x, n, dim)`, `cumsum(x, dim)`, `clip(x, min_val, max_val)`, `shuffle(x, dim)`.

### What's New in v1.1.4
- **Statistics:** `cov(x, y)`, `corrcoef(x, y)`, `skew(x)`, `kurtosis(x)`, `histogram(x, bins)`.
- **Algorithms:** `qr(A)`, `cholesky(A)`; `polyfit(x, y, deg)`, `polyval(p, x)`; `unique(x)`, `argsort(x)`; `convolve1d`; `minimize_scalar`, `newton`.

### What's New in v1.1.3
- **Numerics:** `cond(A)`, `rank(A)`, `pinv(A)` (condition, rank, pseudo-inverse); `expm(A)`, `logm(A)` (matrix exponential/logarithm); `interp(x, xp, fp)` (1D linear interpolation); `trapz(y, x)` (trapezoidal integration); `root_bisect(f, a, b, tol)` (bisection root finding).

### What's New in v1.1.2
- **Unit Formatting:** Similar factors grouped in output: `m*m` → `m^2`, `m*m*m` → `m^3`, `m*m*kg` → `m^2*kg`. Literals like `1[m^3]`, `1[m^2]` usable; `m^3` for volume conversion (e.g. `1[m^3] + 500[L]` → `1.5[m^3]`).

### What's New in v1.1.1
- **Automatic Unit Conversion:** Addition and subtraction automatically convert compatible units of the same dimension. Length (m, cm, km, mm, dm), mass (kg, g, t, mg), time (s, min, h, ms), current (A, mA, kA, uA), temperature (K, mK), amount of substance (mol, mmol, kmol), luminous intensity (cd, mcd). Derived: pressure (Pa, bar, atm), volume (L, mL, dm³, m³), energy (J, kJ, MJ, Wh, kWh), voltage (V, mV, kV), frequency (Hz, kHz, MHz, GHz), charge (C, mC, uC), resistance (ohm, kohm, Mohm), power (W, kW, MW). Result unit is the unit of the first operand. Compile-time check allows same dimension; incompatible units → CompileError. Example: `length_units_conversion.ddk`.

### What's New in v1.1.0
- **Console:** Print order of `print_latex()` and `print()` corrected to write to the same stdout buffer.
- **LaTeX:** `	exttt{...}` in Unicode conversion supported; method names like `.sparse()` formatted as code.
- **Examples:** Leading `
` in all `print("
...")` calls removed.

### What's New in v1.0.10
- **Scientific Console:** `print_latex(s)` renders LaTeX in Jupyter or any compatible IDE console.
- **IDE Branding:** General window/app icon configurations loaded.
- **Example:** `chemistry_units_radiation.ddk` output changed to ASCII (radioactivity, activity) for consistent rendering.

### What's New in v1.0.9
- **Units:** Chemical units **bar**, **atm**, **g** and radioactivity **Bq** (Becquerel), **Sv** (Sievert); Example `chemistry_units_radiation.ddk`.
- **SI Units:** **Candela (cd)** and many simplifications (Pa, W, Hz, V, F, ohm, S, Wb, T, H, lm, lx, Gy, kat, M).

### What's New in v1.0.8
- **Release 1.0.8:** Version bump and release.

### What's New in v1.0.7
- **Window Title:** Window title adjusted.
- **Splash Screen:** General scientific environment subtitle.
- **Examples & Restore:** Session restoration opens existing files only, showing scientific examples correctly.
- **FFT Example & Plot:** `scientific_fft_spectrum.ddk` comments updated; complex plot values converted to real before plotting.

### What's New in v1.0.6
- **Scientific Plot Examples:** Six new examples (`scientific_wave_superposition.ddk`, `scientific_damped_oscillator.ddk`, `scientific_arrhenius_plot.ddk`, `scientific_gravitational_potential.ddk`, `scientific_ricci_plot.ddk`, `scientific_fft_spectrum.ddk`) with plots for scientists using Dedekind features.
- **IDE Startup:** Scientific plot examples loaded at startup when no session exists.
- **IDE Focus:** Profiler and debugger (Python-specific) marked as deprecated.

### What's New in v1.0.5
- **Plots Pane:** `plot()` displays figures in the plots pane; kernel sends display_data correctly.

### What's New in v1.0.4
- **Syntax Highlighting:** Highlighting for units (e.g. `10[m]`, `[kg]`) and Ricci indices (`A^ij`, `B_jk`).

### What's New in v1.0.3
- **Compiler:** ML runtime loaded automatically if programs use runtime built-ins (e.g. `integrate`, `sin`, `arrhenius`, `uncertain`).

### What's New in v1.0.2
- **Renaming:** Rename Fourier → Dedekind (language, kernel, file extension `.ddk`).

### What's New in v1.0.1
- **Patch:** Bugfixes and minor improvements for kernel/IDE integration.

### What's New in v1.0.0
- **Release:** First stable version 1.0. Dedekind Jupyter Kernel and Dedekind language tools.

### What's New in v0.9.8
- **Convenience (Chemistry/Biology):** `michaelis_menten(S, Vmax, Km)`, `logistic(t, r, K, N0)`, `logistic_growth_dt(N, r, K)`, `arrhenius(T, A, Ea)`, `linear_regression(x, y)` — callable in a single line.
- **Chemical Elements:** `atomic_mass("C")` (g/mol), `atomic_number("C")` for approx. 50 elements (IUPAC-like); Molar mass (e.g., 2*atomic_mass("H")+atomic_mass("O")); Example `chemistry_elements.ddk`.
- **File I/O, Network, JSON:** `read_file(path)`, `write_file(path, content)`, `file_exists(path)`; `http_get(url)`, `http_post(url, data)`; `json_parse(s)` → object, `json_stringify(obj)`; Example `file_io_json.ddk`.

### What's New in v0.9.7
- **Dedekind for Chemistry & Biology:** Chemical units **mol**, **L**, **M** (= mol/L), **ppm** in runtime and compile check; M and mol/L treated as same unit. Unit literal `[1/s]` supported in parser.
- **Examples:** `chemistry_kinetics.ddk` (first-order reaction with `ode_solve`, [M], [1/s]), `dose_response.ddk` (dose-response/EC50 with `fit`), `biology_growth.ddk` (logistic growth with `ode_solve`).

### What's New in v0.9.6
- **Basic Math Functions:** Standard library expanded with `tan`, `exp`, `log`, `log10`, `sqrt`, `abs`; inverse trig `asin`, `acos`, `atan`, `atan2(y,x)`; hyperbolic `sinh`, `cosh`, `tanh`. Element-wise and differentiable.
- **Reductions, Rounding, Linear Algebra:** `min(x)`, `max(x)`, `argmin(x)`, `argmax(x)`; `round(x)`, `floor(x)`, `ceil(x)`; `norm(x)`, `det(A)`, `trace(A)`. Example: `math_functions.ddk`.

### What's New in v0.9.5
- **Better Error Messages:** AST nodes carry optional `line`; parser throws `CompileError` with location; runtime quantity errors provide clear context.
- **Compile-Time Units:** `units_checker.py` checks before codegen: addition/subtraction must match dimensions; `1[m] + 1[s]` is a compile error; CLI `--no-units-check`.

### What's New in v0.9.4
- **HMC (Hamiltonian Monte Carlo):** `hmc(...)` and `fit(..., method="hmc")` — gradient-based MCMC proposals. Example: `hmc_fitting.ddk`.
- **LaTeX Export:** `export_to_latex(source_code)` in compiler; CLI: `--latex` option. Formulas are outputted as LaTeX. Example: `latex_demo.ddk`.

### What's New in v0.9.3
- **Version 0.9.3:** Release with uncertainty propagation (`uncertain`, `UncertainQuantity`) and fitting (`fit`); examples `uncertainty_propagation.ddk`, `curve_fitting.ddk`.

### What's New in v0.9
- **Initial Prototype:** Basic compiler, PyTorch runtime, basic physical units and constants.
