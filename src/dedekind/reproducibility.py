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
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

from . import __version__
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
    remote_url = _run(["git", "-C", repo_dir, "config", "--get", "remote.origin.url"])
    return {
        "head": head,
        "short": short or head[:8],
        "branch": branch or "(detached)",
        "dirty": bool(status),
        "url": remote_url,
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


def _cpu_info():
    p = platform.processor()
    if platform.system() == "Windows":
        out = _run(["wmic", "cpu", "get", "name"])
        if out:
            lines = [l.strip() for l in out.split("\n") if l.strip()]
            if len(lines) > 1:
                return lines[1]
    elif platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    elif platform.system() == "Linux":
        out = _run(["grep", "-m", "1", "model name", "/proc/cpuinfo"])
        if out and ":" in out:
            return out.split(":", 1)[1].strip()
    return p or "Unknown CPU"


def _ram_info():
    if platform.system() == "Windows":
        out = _run(["wmic", "computersystem", "get", "totalphysicalmemory"])
        if out:
            lines = [l.strip() for l in out.split("\n") if l.strip()]
            if len(lines) > 1:
                try:
                    bytes_ram = int(lines[1])
                    return f"{bytes_ram / (1024**3):.1f} GB"
                except ValueError:
                    pass
    elif platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return f"{int(out) / (1024**3):.1f} GB"
            except ValueError:
                pass
    elif platform.system() == "Linux":
        out = _run(["grep", "MemTotal", "/proc/meminfo"])
        if out and ":" in out:
            try:
                parts = out.split(":", 1)[1].strip().split()
                kb = int(parts[0])
                return f"{kb / (1024**2):.1f} GB"
            except (ValueError, IndexError):
                pass
    return "Unknown RAM"


def _find_read_files(node):
    files = []
    if node is None:
        return files
    from .ast_nodes import (
        Call, Identifier, Literal, Program, FunctionDef, Assignment,
        BinaryOp, ReturnStmt, IfStmt, WhileStmt, ForStmt, VectorLiteral,
        DictLiteral, TryCatch, ItemAssignment
    )
    if isinstance(node, Call):
        if isinstance(node.func_name, Identifier):
            fname = node.func_name.name
            if fname in ("read_file", "read_csv", "read_json", "read_dataframe") and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, Literal) and isinstance(arg0.value, str):
                    files.append(arg0.value)
        for arg in node.args:
            files.extend(_find_read_files(arg))
    elif isinstance(node, Program):
        for stmt in node.statements:
            files.extend(_find_read_files(stmt))
    elif isinstance(node, FunctionDef):
        for stmt in node.body:
            files.extend(_find_read_files(stmt))
    elif isinstance(node, Assignment):
        files.extend(_find_read_files(node.value))
    elif isinstance(node, BinaryOp):
        files.extend(_find_read_files(node.left))
        files.extend(_find_read_files(node.right))
    elif isinstance(node, ReturnStmt):
        files.extend(_find_read_files(node.value))
    elif isinstance(node, IfStmt):
        files.extend(_find_read_files(node.condition))
        for stmt in node.then_branch:
            files.extend(_find_read_files(stmt))
        if node.else_branch:
            for stmt in node.else_branch:
                files.extend(_find_read_files(stmt))
    elif isinstance(node, WhileStmt):
        files.extend(_find_read_files(node.condition))
        for stmt in node.body:
            files.extend(_find_read_files(stmt))
    elif isinstance(node, ForStmt):
        files.extend(_find_read_files(node.collection))
        for stmt in node.body:
            files.extend(_find_read_files(stmt))
    elif isinstance(node, VectorLiteral):
        for el in node.elements:
            files.extend(_find_read_files(el))
    elif isinstance(node, DictLiteral):
        for el in node.keys:
            files.extend(_find_read_files(el))
        for el in node.values:
            files.extend(_find_read_files(el))
    elif isinstance(node, TryCatch):
        for stmt in node.body:
            files.extend(_find_read_files(stmt))
        for stmt in node.handler:
            files.extend(_find_read_files(stmt))
    elif isinstance(node, ItemAssignment):
        files.extend(_find_read_files(node.value))
    return files


def build_report(source_code: str, source_path: str, repo_dir: str | None = None) -> str:
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(source_path)) or "."

    sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    py_ver = sys.version.split()[0]
    platform_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu = _cpu_info()
    ram = _ram_info()
    git = _git_info(repo_dir)
    torch_v = _pkg_version("torch")
    numpy_v = _pkg_version("numpy")
    scipy_v = _pkg_version("scipy")
    seeds = _detect_seeds(source_code)

    cuda = None
    gpu_name = None
    try:
        import torch  # type: ignore
        cuda = "yes" if torch.cuda.is_available() else "no"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    lex = Lexer(source_code)
    parser = Parser(lex.tokenize())
    ast = parser.parse()
    methods_latex = program_to_latex(ast)

    # Input Data Provenance
    detected_files = []
    try:
        raw_paths = _find_read_files(ast)
        seen_paths = set()
        for p_str in raw_paths:
            abs_p = p_str
            if not os.path.isabs(abs_p):
                src_dir = os.path.dirname(os.path.abspath(source_path))
                abs_p = os.path.join(src_dir, p_str)
            if os.path.isfile(abs_p):
                real_p = os.path.abspath(abs_p)
                if real_p not in seen_paths:
                    seen_paths.add(real_p)
                    with open(real_p, "rb") as f:
                        f_hash = hashlib.sha256(f.read()).hexdigest()
                    detected_files.append((p_str, f_hash))
    except Exception:
        pass

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
        if git.get("url"):
            lines.append(f"- **Repository:** `{git['url']}`")
        lines.append(f"- **Branch:** `{git['branch']}`")
        lines.append(f"- **Commit:** `{git['short']}` ({git['head']}){dirty}")
    else:
        lines.append("- (not a git repository)")
    lines.append("")
    lines.append("## Toolchain")
    lines.append(f"- Dedekind: {__version__}")
    lines.append(f"- Python:   {py_ver}")
    lines.append(f"- OS:       {platform_info}")
    lines.append(f"- torch:    {torch_v or '(not installed)'}")
    if cuda is not None:
        lines.append(f"- CUDA available: {cuda}")
    lines.append(f"- numpy:    {numpy_v or '(not installed)'}")
    lines.append(f"- scipy:    {scipy_v or '(not installed)'}")
    lines.append("")
    lines.append("## Hardware")
    lines.append(f"- CPU: {cpu}")
    lines.append(f"- RAM: {ram}")
    if gpu_name:
        lines.append(f"- GPU: {gpu_name}")
    lines.append("")
    lines.append("## Data Provenance (Input Files)")
    if detected_files:
        for p_str, f_hash in detected_files:
            lines.append(f"- `{p_str}` -> SHA-256: `{f_hash}`")
    else:
        lines.append("- (no input data files read via read_file/read_csv/read_json/read_dataframe)")
    lines.append("")
    lines.append("## RNG seeds detected in source")
    if seeds:
        for kind, val in seeds:
            lines.append(f"- `{kind}(...)` -> {val}")
    else:
        lines.append("- (none detected -- runs are NOT reproducible)")
        lines.append("- **WARNING:** No RNG seed detected in source code. Run results may vary across execution.")
    lines.append("")
    lines.append("## Methods (LaTeX, from AST)")
    lines.append("")
    lines.append("```latex")
    lines.append(methods_latex)
    lines.append("```")
    lines.append("")
    
    # BibTeX Citation
    repo_url = git['url'] if (git and git.get('url')) else 'https://github.com/Engineer1080/Dedekind'
    citation = f"""@software{{dedekind_v{__version__.replace('.', '')},
  author = {{Heinrich, Mario Michael}},
  title = {{Dedekind Programming Language}},
  version = {{{__version__}}},
  url = {{{repo_url}}},
  year = {{{datetime.now().year}}}
}}"""
    lines.append("## Citation (BibTeX)")
    lines.append("")
    lines.append("```bibtex")
    lines.append(citation)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(source_code: str, source_path: str, output_path: str,
                 repo_dir: str | None = None) -> str:
    report = build_report(source_code, source_path, repo_dir=repo_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return output_path
