# Fourier Documentation

This folder contains the **source** and **generated** documentation for the Fourier language.

## Contents

| File | Description |
|------|-------------|
| **Fourier_Language_Specification.md** | Language Specification (Markdown source, v0.2; §15 Physical Units v0.6, §15.7 ODE v0.7, §15.8 Probabilistic v0.8, §15.9 PDE v0.8) |
| **Fourier_Research_and_Architecture.md** | Research foundation & architecture (Markdown source; §10 Sprachfeatures v0.6) |
| **Symbolic_Simplification_Roadmap.md** | Implementierungs-Roadmap für Symbolic Simplification (Phasen, Optionen, Integration) |
| **Fourier_Language_Specification_v0.1.pdf** | Legacy PDF (v0.1); for current spec use the Markdown or generate v0.2 PDF below |
| **Fourier_Research_Papers_and_Architecture.pdf** | Legacy PDF; for current content use the Markdown or generate PDF below |

## Generating PDFs from Markdown

The Markdown files (`.md`) are the **canonical sources**. To produce updated PDFs:

### Option 1: Pandoc (recommended)

Install [pandoc](https://pandoc.org/) and a LaTeX engine (e.g. MiKTeX, TeX Live), then run from this folder:

```bash
# Language Specification
pandoc Fourier_Language_Specification.md -o Fourier_Language_Specification_v0.2.pdf --toc

# Research & Architecture
pandoc Fourier_Research_and_Architecture.md -o Fourier_Research_and_Architecture.pdf --toc
```

### Option 2: Other tools

- **VS Code**: Use an extension such as "Markdown PDF" to export the open `.md` file to PDF.
- **Online**: Paste the Markdown into a service that converts Markdown to PDF (e.g. markdown-to-pdf converters).
- **Typora / other editors**: Open the `.md` file and export to PDF from the application.

## What changed in v0.8 (documented here)

- **Probabilistic Programming**: First-class distributions `Normal(mu, sigma)`, `Uniform(low, high)`, `Bernoulli(p)`; `sample(dist)` / `sample(dist, n)`; `log_prob(dist, value)`; Metropolis-Hastings `metropolis(log_prior, log_likelihood, data, init, steps)` for Bayesian inference. See **Fourier_Language_Specification.md** §15.8 and `examples/fourier/probabilistic.fourier`.
- **Differentiable PDE Solvers**: `pde_heat_1d(u0, x, t, k)` and `pde_heat_2d(u0, x, y, t, k)` for the heat equation; finite differences + `ode_solve`; Dirichlet BC; gradients through `u0` and `k`. See **Fourier_Language_Specification.md** §15.9 and `examples/fourier/pde_heat.fourier`.

## What changed in v0.7 (documented here)

- **Differentiable ODE Solvers**: `ode_solve(fun, y0, t)` (RK4/Euler); gradients through `grad()` for physics-informed ML; `linspace(start, stop, steps)` for time grids. See **Fourier_Language_Specification.md** §15.7 and `examples/fourier/differentiable_ode.fourier`.

## What changed in v0.6 (documented here)

- **Physical Units**: Literals with units (`1.0[kg]`, `5.0e14[Hz]`), constants `c`, `G`, `h`, `k_B`, `k_e` as Quantity with SI units, arithmetic and `^` for powers, display simplification (J, N).
- **Quantity & Quaternion**: Full arithmetic including `__pow__` and `__neg__`; unary minus in codegen for expressions like `-1.0[C]` and `-1.0 + 0i`.

See **Fourier_Language_Specification.md** §15 and **Fourier_Research_and_Architecture.md** §10 for full details.
