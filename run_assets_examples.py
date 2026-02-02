"""
Führt alle .ddk-Beispiele aus DedekindStudio/assets/dedekind_examples aus.
Nur Laufzeit-Ausgabe (kein generierter Code), Plots unterdrückt (MPLBACKEND=Agg).
"""
import os
import sys
import io

# Plots unterdrücken, bevor matplotlib geladen wird
os.environ["MPLBACKEND"] = "Agg"

# Windows: stdout auf UTF-8 stellen, damit Unicode (z.B. print_latex, ±, ·) nicht crasht
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

ASSETS_DIR = os.path.join(
    os.path.dirname(__file__),
    "DedekindStudio", "spyder", "assets", "dedekind_examples"
)
sys.path.insert(0, os.path.dirname(__file__))

from src.compiler.compiler import compile_source

def main():
    if not os.path.isdir(ASSETS_DIR):
        print(f"Verzeichnis nicht gefunden: {ASSETS_DIR}")
        return
    files = sorted(f for f in os.listdir(ASSETS_DIR) if f.endswith(".ddk"))
    print(f"=== {len(files)} Dedekind-Beispiele (nur Laufzeit-Ausgabe, ohne Plots) ===\n")
    for i, name in enumerate(files, 1):
        path = os.path.join(ASSETS_DIR, name)
        print("=" * 60)
        print(f"[{i}/{len(files)}] {name}")
        print("=" * 60)
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            code = compile_source(source, filepath=path, check_units=True)
            exec_globals = {}
            exec(code, exec_globals)
        except Exception as e:
            print(f"FEHLER: {e}")
            import traceback
            traceback.print_exc()
        print()
    print("=== Ende aller Beispiele ===")

if __name__ == "__main__":
    main()
