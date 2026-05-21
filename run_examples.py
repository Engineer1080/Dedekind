#!/usr/bin/env python3
"""
Skript zum Kompilieren und Ausführen aller .ddk-Beispiele in examples/dedekind.
Findet automatisch alle *.ddk-Dateien; funktioniert auch bei neu hinzugefügten Dateien.

Verwendung (aus Projektroot):
  python run_examples.py           # Kompilieren + Ausführen, kurze Ausgabe
  python run_examples.py -v        # Ausführliche Ausgabe pro Beispiel
  python run_examples.py -q        # Nur Zusammenfassung (Erfolg/Fehler pro Datei)
  python run_examples.py --compile  # Nur kompilieren, nicht ausführen
"""

import sys
import os
import io
import argparse

# Projektroot = Verzeichnis, in dem dieses Skript liegt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples", "dedekind")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

def main():
    parser = argparse.ArgumentParser(description="Alle Dedekind-Beispiele kompilieren und ausführen.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Vollständige Ausgabe pro Beispiel")
    parser.add_argument("-q", "--quiet", action="store_true", help="Nur Zusammenfassung (ein Zeile pro Datei)")
    parser.add_argument("--compile", action="store_true", help="Nur kompilieren, nicht ausführen")
    parser.add_argument("--filter", type=str, default="", help="Nur Dateien, deren Name diesen String enthalten (z.B. hello)")
    args = parser.parse_args()

    if not os.path.isdir(EXAMPLES_DIR):
        print(f"Fehler: Beispiele-Ordner nicht gefunden: {EXAMPLES_DIR}")
        sys.exit(1)

    # Alle .ddk-Dateien rekursiv finden (sortiert für reproduzierbare Reihenfolge)
    ddk_files = []
    for root, dirs, files in os.walk(EXAMPLES_DIR):
        for f in files:
            if f.endswith(".ddk"):
                rel_dir = os.path.relpath(root, EXAMPLES_DIR)
                if rel_dir == ".":
                    rel_path = f
                else:
                    rel_path = os.path.join(rel_dir, f)
                if not args.filter or args.filter in rel_path:
                    ddk_files.append(rel_path.replace("\\", "/"))
    ddk_files.sort()

    if not ddk_files:
        print("Keine .ddk-Dateien gefunden.")
        sys.exit(0)

    # Compiler importieren (src muss im Pfad sein)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        from compiler.compiler import compile_source, dedekind_exec
    except ImportError as e:
        print(f"Fehler: Compiler nicht gefunden. Bitte aus Projektroot ausführen: {e}")
        sys.exit(1)

    results = []
    for filename in ddk_files:
        filepath = os.path.join(EXAMPLES_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            results.append((filename, False, f"Lesefehler: {e}"))
            continue

        try:
            python_code = compile_source(source, filepath=filepath)
        except Exception as e:
            results.append((filename, False, f"Kompilierung: {e}"))
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

        if args.compile:
            results.append((filename, True, "OK (nur kompiliert)"))
            if args.verbose:
                print(f"--- {filename} (generierter Code gekürzt) ---")
                lines = python_code.split("\n")
                print("\n".join(lines[:20] + ["..."] if len(lines) > 20 else lines))
            continue

        # Ausführen mit gefangenem stdout
        old_stdout = sys.stdout
        out = io.StringIO()
        try:
            sys.stdout = out
            exec_globals = {}
            dedekind_exec(python_code, ddk_file=filepath, exec_globals=exec_globals, ddk_source=source)
            output = out.getvalue()
            results.append((filename, True, None))
            if args.verbose:
                print(f"--- {filename} ---")
                print(output or "(keine Ausgabe)")
                print()
        except Exception as e:
            sys.stdout = old_stdout
            results.append((filename, False, f"Laufzeit: {e}"))
            if args.verbose:
                import traceback
                traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    # Ausgabe
    if not args.quiet:
        print("=" * 60)
        print("Dedekind-Beispiele: Kompilieren & Ausführen")
        print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for filename, ok, err in results:
        if args.quiet:
            status = "OK" if ok else f"FAIL: {err}"
            print(f"  {'[OK]' if ok else '[FAIL]'} {filename}  {'' if ok else status}")
        else:
            symbol = "[OK] " if ok else "[FAIL]"
            print(f"  {symbol} {filename}")
            if not ok and err:
                print(f"       {err}")
    if not args.quiet:
        print("=" * 60)
        print(f"Ergebnis: {passed}/{len(results)} bestanden, {failed} fehlgeschlagen.")
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
