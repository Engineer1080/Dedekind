"""CLI entry point: `dedekind <file.ddk> [--latex] [--reproducibility-report PATH] ...`.

Thin wrapper around `dedekind.compiler.main()` so that pip can install a
shell command. The heavy-lifting argparse-equivalent lives in compiler.py
for legacy `python -m dedekind.compiler` invocations.
"""
from .compiler import main


if __name__ == "__main__":
    main()
