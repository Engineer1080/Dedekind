#!/usr/bin/env python3
"""
Script to compile and execute all .ddk examples in examples/dedekind.
Automatically finds all *.ddk files; works even with newly added files.

Usage (from project root):
  python run_examples.py           # Compile + execute, short output
  python run_examples.py -v        # Verbose output per example
  python run_examples.py -q        # Summary only (success/failure per file)
  python run_examples.py --compile  # Compile only, do not execute
"""

import sys
import os
import io
import argparse

# Project root = directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples", "dedekind")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

def main():
    parser = argparse.ArgumentParser(description="Compile and run all Dedekind examples.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Full output per example")
    parser.add_argument("-q", "--quiet", action="store_true", help="Summary only (one line per file)")
    parser.add_argument("--compile", action="store_true", help="Compile only, do not execute")
    parser.add_argument("--filter", type=str, default="", help="Only files whose name contains this string (e.g. hello)")
    args = parser.parse_args()

    if not os.path.isdir(EXAMPLES_DIR):
        print(f"Error: Examples directory not found: {EXAMPLES_DIR}")
        sys.exit(1)

    # Find all .ddk files recursively (sorted for reproducible order)
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
        print("No .ddk files found.")
        sys.exit(0)

    # Import compiler (src must be on the path)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        from dedekind import compile_source, dedekind_exec
    except ImportError as e:
        print(f"Error: Compiler not found. Please run from project root: {e}")
        sys.exit(1)

    results = []
    for filename in ddk_files:
        filepath = os.path.join(EXAMPLES_DIR, filename)
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

        if args.compile:
            results.append((filename, True, "OK (compiled only)"))
            if args.verbose:
                print(f"--- {filename} (generated code truncated) ---")
                lines = python_code.split("\n")
                print("\n".join(lines[:20] + ["..."] if len(lines) > 20 else lines))
            continue

        # Execute with captured stdout
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
        except Exception as e:
            sys.stdout = old_stdout
            results.append((filename, False, f"Runtime: {e}"))
            if args.verbose:
                import traceback
                traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    # Output
    if not args.quiet:
        print("=" * 60)
        print("Dedekind Examples: Compile & Run")
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
        print(f"Result: {passed}/{len(results)} passed, {failed} failed.")
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
