# Beispiele (Examples)

Die Beispiele sind nach Dateityp in Unterordner gegliedert:

| Ordner     | Inhalt                          |
|------------|----------------------------------|
| **dedekind/** | Dedekind-Quellcode (`.ddk`)    |
| **python/**   | Generierter/Referenz-Python-Code (`.py`) |
| **cpp/**      | C++-Quellcode (`.cpp`)           |
| **mlir/**     | MLIR-Intermediate-Representation (`.mlir`) |
| **bin/**      | Kompilierte Binaries (z. B. `.exe`) |

## Dedekind-Beispiele ausführen

```bash
# Vom Projektroot – einzelne Dedekind-Datei kompilieren und ausführen
python -m src.compiler.compiler examples/dedekind/hello.ddk
```

Dedekind Studio lädt beim Start (wenn keine vorherige Sitzung) die **wissenschaftlichen Plot-Beispiele** (`scientific_*.ddk`) aus den gebündelten Assets als Tabs. Weitere Beispiele aus `examples/dedekind/` können manuell geöffnet werden.

## Wissenschaftliche Plot-Beispiele (Dedekind-Features)

Diese Beispiele nutzen **Features, die Python nicht nativ hat**, und erzeugen Plots für Wissenschaftler:

| Datei | Inhalt | Dedekind-Features |
|-------|--------|-------------------|
| `scientific_wave_superposition.ddk` | Superposition zweier Sinuswellen | `pi`, `sin()`, `linspace()`, `plot()` |
| `scientific_damped_oscillator.ddk` | Gedämpfter Oszillator x(t) = exp(-ζωt)·cos(ωt) | `exp()`, `cos()` auf Tensoren, `plot()` |
| `scientific_arrhenius_plot.ddk` | Arrhenius-Plot ln(k) vs. 1/T | `arrhenius()`, `R_gas` [J/(K·mol)], `log()`, `plot()` |
| `scientific_gravitational_potential.ddk` | Gravitationspotential V(r) = -G·M/r | `G`, Einheiten-Literale `[m]`, `[kg]`, `plot()` |
| `scientific_ricci_plot.ddk` | Matrix-Vektor in Ricci-Notation A^ij·v^j, Komponenten-Plot | Einstein-Indizes `^ij`, `_j`, `plot()` |
| `scientific_fft_spectrum.ddk` | FFT-Betrag eines Überlagerungssignals | `fft()`, `abs()`, reelles Signal → Spektrum, `plot()` |

Alle Beispiele laufen mit: `python run_examples.py --filter scientific`
