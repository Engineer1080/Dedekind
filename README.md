# Dedekind

![Version](https://img.shields.io/badge/Version-2.9.0-blue) ![License](https://img.shields.io/badge/License-Apache_2.0-green) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**A programming language for scientific computing.** Write your simulator
once in readable code with units — and get inverse problems, topology
optimization, parameter estimation, and Bayesian inference on it for
free, without hand-rolled adjoint code.

```dedekind
use atomic

m = 1.0[kg]                                        // compile-time units
k = 4.0[N/m]

fn L(q, v) { return 0.5*m*v[0]*v[0] - 0.5*k*q[0]*q[0] }

traj = ode_solve(lagrange_ode_rhs(L), [1.0, 0.0], linspace(0, 2*pi, 51))
```

The same AST that runs this simulation generates the LaTeX for your paper
(`dedekind file.ddk --latex`) and a reproducibility report bundling the
git commit, package versions, RNG seeds and methods section
(`dedekind file.ddk --reproducibility-report appendix.md`).

---

## Install

```bash
pip install dedekind                  # core: torch + numpy + sympy
pip install "dedekind[jupyter,plot]"  # + Jupyter kernel + matplotlib
pip install "dedekind[all]"           # + sci, geo, bio, md, ml, plot, jupyter
```

Extras: `jupyter`, `plot`, `sci` (scipy), `geo` (xarray), `bio` (rdkit),
`md` (openmm), `ml` (torch_geometric), `server` (Flask, only for Studio
backend).

### Jupyter / JupyterLab / Spyder

```bash
pip install "dedekind[jupyter]"
python -m dedekind.install_kernel
```

Then `jupyter lab` and pick **Dedekind** from the kernel list. Variables
persist across cells; `print_latex(...)` renders inline; errors are
mapped back to `.ddk` line numbers.

---

## Hello, Dedekind

`hello.ddk`:

```dedekind
print("Hello from Dedekind!")
vec = [1, 2, 3]
print(vec.sum())
```

Run it:

```bash
dedekind hello.ddk
```

---

## What makes Dedekind different

- **Native physical units, checked at compile time.** `1[m] + 1[s]` is
  a compiler error with line number; `1[m] + 100[cm]` auto-converts to
  `2[m]`. Cross-argument unit polymorphism via generics:
  `fn add<U>(a: [U], b: [U]) -> [U]`.
- **Differentiable everything.** PDE/ODE solvers, LBM/FEM simulators,
  N-body integrators, control blocks, IIR filters — all are first-class
  AST nodes that autograd flows through. `minimize(...)` and `fit(...)`
  optimize through full simulations without writing adjoint code.
- **Tafelnotation as syntax.** Einstein indices (`A^ij * v^j`), Ricci
  contraction, Lagrangians (`lagrange_ode_rhs(L)`), partial derivatives
  (`partial(u, x, order=2)`) are language primitives, not library calls.
- **Shape and semantic types.** `Vector[N]`, `Matrix[M, N]`,
  `LabeledTensor[lat, lon, time]`, `Sequence[DNA|RNA|Protein]` —
  validated at function boundaries. The classic
  `data.mean(axis=2)`-instead-of-`dim="time"` bug becomes structurally
  impossible.
- **LaTeX is generated from the AST, not typed by hand.** Methods
  sections in papers and the code that runs the simulation share one
  source of truth. Paper-code drift is structurally eliminated.
- **Python interop.** `pyimport scipy.special as sp; sp.gamma(5.0)` —
  every PyPI package is one line away.

Full feature catalogue: [docs/language.md](docs/language.md).
Why it matters in detail: [docs/language.md#core-features](docs/language.md).

---

## Showcase examples

A curated entry-point selection (full list:
[examples/dedekind/](examples/dedekind/)):

- `physics_astronomy/scientific_ricci_plot.ddk` — Einstein notation
- `physics_astronomy/lagrange_hamilton.ddk` — Lagrangian → ODE in one call
- `machine_learning/pinn_oscillator_demo.ddk` — physics-informed neural net
- `engineering/lbm_shape_optimization.ddk` — differentiable CFD shape opt
- `engineering/lbm3d_sphere_drag.ddk` — D3Q19 + autograd
- `engineering/lbm_les_smagorinsky_tuning.ddk` — Smagorinsky LES calibration
- `engineering/heat_sink_topology_optimization.ddk` — SIMP topology opt
- `physics_astronomy/crystallography_structure_refinement.ddk` — structure refinement via differentiable SF (using atomic)
- `compiler_features/reproducibility_demo.ddk` — `--reproducibility-report`
- `compiler_features/latex_demo.ddk` — `--latex`

Compile and run every example at once:
`python run_examples.py` (`-q` for summary, `--filter <name>` for one).

---

## Status, roadmap, history

- Current release: **v2.8.0** (May 2026)
- Roadmap: [docs/roadmap.md](docs/roadmap.md)
- Full changelog: [docs/changelog.md](docs/changelog.md)
- Formal spec: [docs/Dedekind_Language_Specification.md](docs/Dedekind_Language_Specification.md)

## Dedekind Studio

Dedekind Studio is a separate **closed-source IDE** built on Spyder. It
is maintained independently of the language. The language itself is
fully usable in any Python environment via the Jupyter kernel — Studio
is one option among many.

## License

Apache 2.0. See [LICENSE](LICENSE).
