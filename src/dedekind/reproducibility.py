"""
Reproduzierbarkeits-Report.

Erzeugt einen kompakten Anhang fuer Papers: Git-Revision, Toolchain-Versionen,
SHA-256 der Quelle, im Code gesetzte RNG-Seeds und ein LaTeX-Methods-Snippet
aus dem AST. Adressiert die Paper-Code-Drift, nicht die gesamte
Reproduzierbarkeitskrise -- siehe README.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

from .lexer import Lexer
from .parser import Parser
from .latex_export import program_to_latex


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_info(repo_dir):
    head = _run(["git", "-C", repo_dir, "rev-parse", "HEAD"])
    if not head:
        return None
    short = _run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"])
    status = _run(["git", "-C", repo_dir, "status", "--porcelain"])
    branch = _run(["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "head": head,
        "short": short or head[:8],
        "branch": branch or "(detached)",
        "dirty": bool(status),
    }


def _pkg_version(name):
    try:
        mod = __import__(name)
    except Exception:
        return None
    return getattr(mod, "__version__", None) or "(unknown)"


def _detect_seeds(source):
    """Find seed(N) / torch.manual_seed(N) / np.random.seed(N) literals."""
    patterns = [
        (r"\bseed\s*\(\s*([0-9]+)\s*\)",            "seed"),
        (r"\bmanual_seed\s*\(\s*([0-9]+)\s*\)",     "torch.manual_seed"),
        (r"\bnp\.random\.seed\s*\(\s*([0-9]+)\s*\)", "np.random.seed"),
    ]
    found = []
    for pat, label in patterns:
        for m in re.finditer(pat, source):
            found.append((label, m.group(1)))
    return found


def build_report(source_code: str, source_path: str, repo_dir: str | None = None) -> str:
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(source_path)) or "."

    sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    py_ver = sys.version.split()[0]
    git = _git_info(repo_dir)
    torch_v = _pkg_version("torch")
    numpy_v = _pkg_version("numpy")
    scipy_v = _pkg_version("scipy")
    seeds = _detect_seeds(source_code)

    cuda = None
    try:
        import torch  # type: ignore
        cuda = "yes" if torch.cuda.is_available() else "no"
    except Exception:
        pass

    lex = Lexer(source_code)
    parser = Parser(lex.tokenize())
    ast = parser.parse()
    methods_latex = program_to_latex(ast)

    lines = []
    lines.append("# Reproducibility Report")
    lines.append("")
    lines.append(f"- **Source file:** `{os.path.basename(source_path)}`")
    lines.append(f"- **SHA-256:** `{sha256}`")
    lines.append(f"- **Generated:** {ts}")
    lines.append("")
    lines.append("## Git")
    if git:
        dirty = " (dirty: uncommitted changes present)" if git["dirty"] else ""
        lines.append(f"- **Branch:** `{git['branch']}`")
        lines.append(f"- **Commit:** `{git['short']}` ({git['head']}){dirty}")
    else:
        lines.append("- (not a git repository)")
    lines.append("")
    lines.append("## Toolchain")
    lines.append(f"- Python: {py_ver}")
    lines.append(f"- torch:  {torch_v or '(not installed)'}")
    if cuda is not None:
        lines.append(f"- CUDA available: {cuda}")
    lines.append(f"- numpy:  {numpy_v or '(not installed)'}")
    lines.append(f"- scipy:  {scipy_v or '(not installed)'}")
    lines.append("")
    lines.append("## RNG seeds detected in source")
    if seeds:
        for kind, val in seeds:
            lines.append(f"- `{kind}(...)` -> {val}")
    else:
        lines.append("- (none detected -- runs are NOT reproducible)")
    lines.append("")
    lines.append("## Methods (LaTeX, from AST)")
    lines.append("")
    lines.append("```latex")
    lines.append(methods_latex)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(source_code: str, source_path: str, output_path: str,
                 repo_dir: str | None = None) -> str:
    report = build_report(source_code, source_path, repo_dir=repo_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return output_path
