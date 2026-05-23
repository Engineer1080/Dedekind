# Examples

The examples are organized into subfolders by file type:

| Folder     | Contents                          |
|------------|----------------------------------|
| **dedekind/** | Dedekind source code (`.ddk`)    |
| **python/**   | Generated / reference Python code (`.py`) |
| **cpp/**      | C++ source code (`.cpp`)           |
| **mlir/**     | MLIR intermediate representation (`.mlir`) |
| **bin/**      | Compiled binaries (e.g. `.exe`) |

## Running Dedekind examples

```bash
# From the project root — compile and run a single Dedekind file
python -m src.compiler.compiler examples/dedekind/hello.ddk
```

Dedekind Studio loads the **scientific plot examples** (`scientific_*.ddk`) from the bundled assets as tabs at startup (when there is no previous session). Additional examples from `examples/dedekind/` can be opened manually.

## Scientific plot examples (Dedekind features)

These examples use **features that Python does not have natively** and produce plots for scientists:

| File | Contents | Dedekind features |
|-------|--------|-------------------|
| `scientific_wave_superposition.ddk` | Superposition of two sine waves | `pi`, `sin()`, `linspace()`, `plot()` |
| `scientific_damped_oscillator.ddk` | Damped oscillator x(t) = exp(-ζωt)·cos(ωt) | `exp()`, `cos()` on tensors, `plot()` |
| `scientific_arrhenius_plot.ddk` | Arrhenius plot ln(k) vs. 1/T | `arrhenius()`, `R_gas` [J/(K·mol)], `log()`, `plot()` |
| `scientific_gravitational_potential.ddk` | Gravitational potential V(r) = -G·M/r | `G`, unit literals `[m]`, `[kg]`, `plot()` |
| `scientific_ricci_plot.ddk` | Matrix-vector in Ricci notation A^ij·v^j, component plot | Einstein indices `^ij`, `_j`, `plot()` |
| `scientific_fft_spectrum.ddk` | FFT magnitude of a superposition signal | `fft()`, `abs()`, real signal → spectrum, `plot()` |

All examples run with: `python run_examples.py --filter scientific`
