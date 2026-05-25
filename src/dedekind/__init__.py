"""Dedekind -- a programming language for scientific computing.

Public entry points:
    dedekind.compile_source(src, ...)  - compile a .ddk source string to Python
    dedekind.dedekind_exec(code, ...)  - run compiled code with mapped tracebacks
    dedekind.export_to_latex(src)      - export formulas as LaTeX

CLI: `dedekind <file.ddk>` (see dedekind.cli).
"""

__version__ = "3.0.2"

from .compiler import (  # noqa: F401  (re-export)
    compile_source,
    dedekind_exec,
    export_to_latex,
)
