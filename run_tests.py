#!/usr/bin/env python3
"""
Mini test runner for Dedekind: executes all .ddk files in tests/dedekind.
A file passes if it runs without any exception (including AssertionError).

Usage (from project root):
  python run_tests.py           # Short output (OK/FAIL per file)
  python run_tests.py -v        # Verbose output per file
  python run_tests.py -q        # Summary only
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
    parser = argparse.ArgumentParser(description="Run Dedekind mini-tests (assert-based).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Full output per test")
    parser.add_argument("-q", "--quiet", action="store_true", help="Summary only")
    parser.add_argument("--filter", type=str, default="", help="Only files containing this string (e.g. assert)")
    args = parser.parse_args()

    if not os.path.isdir(TESTS_DIR):
        print(f"Tests directory not found: {TESTS_DIR}")
        print("Create tests/dedekind/ and place .ddk files with assert() in it.")
        sys.exit(1)

    ddk_files = sorted(
        f for f in os.listdir(TESTS_DIR)
        if f.endswith(".ddk") and (not args.filter or args.filter in f)
    )
    if not ddk_files:
        print("No .ddk test files found.")
        sys.exit(0)

    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        from dedekind import compile_source, dedekind_exec
    except ImportError as e:
        print(f"Compiler not found: {e}")
        sys.exit(1)

    results = []
    # Optional scientific dependencies. If a runtime/import error mentions
    # one of these, the test is marked SKIP (not FAIL) so CI doesn't gate
    # on packages that aren't always installable on every runner.
    OPTIONAL_DEPS = ("scipy", "xarray", "openmm", "rdkit", "torch_geometric")
    SKIP_HINTS = ("requires", "required", "needs", "needed",
                  "benötigt", "braucht",
                  "No module named", "pip install")

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
            results.append((filename, False, f"Read error: {e}"))
            continue

        try:
            python_code = compile_source(source, filepath=filepath)
        except Exception as e:
            results.append((filename, False, f"Compilation: {e}"))
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
                print(output or "(no output)")
                print()
        except AssertionError as e:
            sys.stdout = old_stdout
            results.append((filename, False, f"Assertion: {e}"))
            if args.verbose:
                import traceback
                traceback.print_exc()
        except Exception as e:
            sys.stdout = old_stdout
            msg = f"Runtime: {e}"
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
        print(f"Result: {passed}/{len(results)} passed, {failed} failed, {skipped} skipped (optional deps missing).")
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
