# Beispiele (Examples)

Die Beispiele sind nach Dateityp in Unterordner gegliedert:

| Ordner   | Inhalt                          |
|----------|----------------------------------|
| **fourier/** | Fourier-Quellcode (`.fourier`)   |
| **python/**  | Generierter/Referenz-Python-Code (`.py`) |
| **cpp/**     | C++-Quellcode (`.cpp`)           |
| **mlir/**    | MLIR-Intermediate-Representation (`.mlir`) |
| **bin/**     | Kompilierte Binaries (z. B. `.exe`) |

## Fourier-Beispiele ausführen

```bash
# Vom Projektroot – einzelne Fourier-Datei kompilieren und ausführen
python -m src.compiler.compiler examples/fourier/hello.fourier
```

Fourier Studio lädt und führt die Dateien aus `examples/fourier/` über die IDE aus.
