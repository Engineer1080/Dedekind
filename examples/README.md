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

Dedekind Studio lädt und führt die Dateien aus `examples/dedekind/` über die IDE aus.
