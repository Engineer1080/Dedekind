#!/usr/bin/env python3
"""
Mini-Test-Runner für Dedekind: führt alle .ddk-Dateien in tests/dedekind aus.
Eine Datei besteht, wenn sie ohne Exception (inkl. AssertionError) durchläuft.

Verwendung (aus Projektroot):
  python run_tests.py           # Kurzausgabe (OK/FAIL pro Datei)
  python run_tests.py -v        # Ausgabe jeder Datei
  python run_tests.py -q        # Nur Zusammenfassung
"""

import sys
import os
import io
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests", "dedekind")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


def main():
    parser = argparse.ArgumentParser(description="Dedekind Mini-Tests ausführen (assert-basiert).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Vollständige Ausgabe pro Test")
    parser.add_argument("-q", "--quiet", action="store_true", help="Nur Zusammenfassung")
    parser.add_argument("--filter", type=str, default="", help="Nur Dateien mit diesem String (z.B. assert)")
    args = parser.parse_args()

    if not os.path.isdir(TESTS_DIR):
        print(f"Tests-Ordner nicht gefunden: {TESTS_DIR}")
        print("Erstelle tests/dedekind/ und lege dort .ddk-Dateien mit assert() ab.")
        sys.exit(1)

    ddk_files = sorted(
        f for f in os.listdir(TESTS_DIR)
        if f.endswith(".ddk") and (not args.filter or args.filter in f)
    )
    if not ddk_files:
        print("Keine .ddk-Testdateien gefunden.")
        sys.exit(0)

    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        from dedekind import compile_source, dedekind_exec
    except ImportError as e:
        print(f"Compiler nicht gefunden: {e}")
        sys.exit(1)

    results = []
    # Optional scientific dependencies. If a runtime/import error mentions
    # one of these, the test is marked SKIP (not FAIL) so CI doesn't gate
    # on packages that aren't always installable on every runner.
    OPTIONAL_DEPS = ("scipy", "xarray", "openmm", "rdkit", "torch_geometric")
    SKIP_HINTS = ("benoetigt", "benötigt", "benötigen", "benoetigen",
                  "erfordert", "erfordern", "No module named", "requires", "pip install")

    def _is_skip(err_msg: str) -> bool:
        if not any(dep in err_msg for dep in OPTIONAL_DEPS):
            return False
        return any(h in err_msg for h in SKIP_HINTS)
    for filename in ddk_files:
        filepath = os.path.join(TESTS_DIR, filename)
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
        except AssertionError as e:
            sys.stdout = old_stdout
            results.append((filename, False, f"Assertion: {e}"))
            if args.verbose:
                import traceback
                traceback.print_exc()
        except Exception as e:
            sys.stdout = old_stdout
            msg = f"Laufzeit: {e}"
            if _is_skip(msg):
                results.append((filename, "skip", msg))
            else:
                results.append((filename, False, msg))
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    if not args.quiet:
        print("=" * 50)
        print("Dedekind Mini-Tests (assert)")
        print("=" * 50)
    passed  = sum(1 for _, ok, _ in results if ok is True)
    skipped = sum(1 for _, ok, _ in results if ok == "skip")
    failed  = sum(1 for _, ok, _ in results if ok is False)
    for filename, ok, err in results:
        if ok is True:
            tag = "[OK]  "
        elif ok == "skip":
            tag = "[SKIP]"
        else:
            tag = "[FAIL]"
        if args.quiet:
            extra = "" if ok is True else (err or "")
            print(f"  {tag} {filename}  {extra}")
        else:
            print(f"  {tag} {filename}")
            if ok is not True and err:
                print(f"       {err}")
    if not args.quiet:
        print("=" * 50)
        print(f"Ergebnis: {passed}/{len(results)} bestanden, {failed} fehlgeschlagen, {skipped} uebersprungen (optionale Deps fehlen).")
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
