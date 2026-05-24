import sys
import os
import re
import dataclasses
import traceback as _tb
import linecache
from typing import Optional, List, Tuple, Dict
from .lexer import Lexer
from .parser import Parser
from .codegen import CodeGenerator
from .latex_export import program_to_latex
from .ast_nodes import CompileError, Program, UseStmt, FunctionDef, Identifier, Node


def _module_search_paths(filepath: Optional[str]) -> List[str]:
    """Search paths for `use mymod` (in this order): current file's directory,
    package-shipped stdlib (`dedekind/stdlib/`), repo-level `modules/`
    (editable installs only), `examples/dedekind/`, and the current working
    directory."""
    paths = []
    if filepath:
        paths.append(os.path.dirname(os.path.abspath(filepath)))
    here = os.path.dirname(os.path.abspath(__file__))
    paths.append(os.path.join(here, "stdlib"))
    project_root = os.path.dirname(os.path.dirname(here))
    paths.append(os.path.join(project_root, "modules"))
    paths.append(os.path.join(project_root, "examples", "dedekind"))
    paths.append(os.getcwd())
    return paths


def _resolve_module(name: str, search_paths: List[str], visiting_line: Optional[int]) -> str:
    """Resolves both flat names (`foo`) and dotted paths (`foo.bar.baz`).
    Dotted names become nested directory paths: foo.bar -> foo/bar.ddk."""
    parts = name.split(".")
    rel_path = os.path.join(*parts) + ".ddk"
    for p in search_paths:
        candidate = os.path.join(p, rel_path)
        if os.path.isfile(candidate):
            return candidate
    raise CompileError(
        f"Module '{name}' not found (searched as {rel_path!r} in: "
        f"{', '.join(p for p in search_paths)}).",
        line=visiting_line,
    )


def _mangled_name(module_name: str, fn_name: str) -> str:
    """Mangling for private (non-pub) module functions: __ddk_<modpath>_<fn>."""
    flat = module_name.replace(".", "_")
    return f"__ddk_{flat}_{fn_name}"


def _rename_in_ast(node: Node, rename_map: Dict[str, str]) -> None:
    """Walk AST in-place, renaming FunctionDef.name (top-level targets) and
    Identifier.name (all references) according to rename_map."""
    if node is None or not isinstance(node, Node):
        return
    if isinstance(node, FunctionDef) and node.name in rename_map:
        node.name = rename_map[node.name]
    if isinstance(node, Identifier) and node.name in rename_map:
        node.name = rename_map[node.name]
    if dataclasses.is_dataclass(node):
        for f in dataclasses.fields(node):
            val = getattr(node, f.name)
            if isinstance(val, Node):
                _rename_in_ast(val, rename_map)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Node):
                        _rename_in_ast(item, rename_map)
                    elif isinstance(item, tuple):
                        for sub in item:
                            if isinstance(sub, Node):
                                _rename_in_ast(sub, rename_map)
            elif isinstance(val, tuple):
                for sub in val:
                    if isinstance(sub, Node):
                        _rename_in_ast(sub, rename_map)


def _apply_module_visibility(mod_ast: Program, module_name: str) -> Program:
    """Mangling for module visibility.

    Behavior:
      - If the module has NO `pub fn` declaration: Legacy mode — all functions
        remain public (backward-compatible with Pre-v1.19 modules).
      - If the module has at least one `pub fn` declaration: Opt-in mode —
        ALL non-pub functions get a mangling prefix
        `__ddk_<modpath>_` and are thus hidden from the caller.
        Calls within the module are also renamed.
    """
    fn_names = []
    pub_names = []
    for stmt in mod_ast.statements:
        if isinstance(stmt, FunctionDef):
            fn_names.append(stmt.name)
            if getattr(stmt, "is_pub", False):
                pub_names.append(stmt.name)
    if not pub_names:
        # Legacy mode: no mangling
        return mod_ast
    rename_map = {
        n: _mangled_name(module_name, n)
        for n in fn_names if n not in pub_names
    }
    if not rename_map:
        return mod_ast
    for stmt in mod_ast.statements:
        _rename_in_ast(stmt, rename_map)
    return mod_ast


def _expand_uses(ast: Program, filepath: Optional[str], loaded: set) -> Program:
    """Replaces UseStmt nodes with the top-level statements of the referenced modules.
    Applies visibility mangling: only functions with `pub fn` are visible outside the
    module (if the module has at least one `pub fn` declaration)."""
    new_stmts = []
    search_paths = _module_search_paths(filepath)
    for stmt in ast.statements:
        if isinstance(stmt, UseStmt):
            if stmt.module in loaded:
                continue  # cyclic or duplicate: ignore
            mod_path = _resolve_module(stmt.module, search_paths, getattr(stmt, "line", None))
            loaded.add(stmt.module)
            try:
                with open(mod_path, "r", encoding="utf-8") as f:
                    mod_source = f.read()
            except OSError as e:
                raise CompileError(f"Failed to read module '{stmt.module}': {e}",
                                   line=getattr(stmt, "line", None), filepath=filepath)
            mod_tokens = Lexer(mod_source).tokenize()
            mod_ast = Parser(mod_tokens).parse()
            mod_ast = _expand_uses(mod_ast, mod_path, loaded)
            mod_ast = _apply_module_visibility(mod_ast, stmt.module)
            new_stmts.extend(mod_ast.statements)
        else:
            new_stmts.append(stmt)
    return Program(new_stmts)


def compile_source(source_code: str, filepath: Optional[str] = None,
                   check_units: bool = True, check_purity: bool = True) -> str:
    try:
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        ast = _expand_uses(ast, filepath, loaded=set())
        from .simplify import simplify_program
        ast = simplify_program(ast)
        if check_units:
            from .units_checker import check_units as run_units_check
            run_units_check(ast, filepath=filepath)
        if check_purity:
            from .purity_check import check_purity as run_purity_check
            run_purity_check(ast, filepath=filepath)
        codegen = CodeGenerator()
        python_code = codegen.generate(ast)
        return python_code
    except CompileError as e:
        if filepath and e.filepath is None:
            raise CompileError(e.message, line=e.line, filepath=filepath)
        raise


_DDK_MARKER_RE = re.compile(r"^\s*# ddk:(\d+)\s*$")


def _build_line_map(generated_code: str) -> dict:
    """Reads `# ddk:<line>` markers and returns {generated_line_idx_1based: ddk_line}."""
    mapping = {}
    last_ddk = None
    for idx, line in enumerate(generated_code.splitlines(), start=1):
        m = _DDK_MARKER_RE.match(line)
        if m:
            last_ddk = int(m.group(1))
            continue
        if last_ddk is not None:
            mapping[idx] = last_ddk
    return mapping


def _make_virtual_filename(ddk_file: Optional[str]) -> str:
    if not ddk_file:
        return "<dedekind>"
    base = os.path.basename(ddk_file)
    return f"<dedekind:{base}>"


def _register_source_for_traceback(virtual_filename: str, generated_code: str) -> None:
    """So that Python traceback can display the source (internal, only for debug)."""
    src_lines = generated_code.splitlines(keepends=True)
    linecache.cache[virtual_filename] = (len(generated_code), None, src_lines, virtual_filename)


def format_dedekind_traceback(exc: BaseException, generated_code: str,
                              ddk_file: Optional[str], ddk_source: Optional[str] = None) -> str:
    """Renders a traceback such that lines in the generated code are mapped back to the original
    .ddk file. Frames within the inlined runtime are marked as
    `<runtime>`; the last reachable user line is at the top."""
    virtual = _make_virtual_filename(ddk_file)
    line_map = _build_line_map(generated_code)
    ddk_lines = ddk_source.splitlines() if ddk_source else None
    if ddk_lines is None and ddk_file and os.path.isfile(ddk_file):
        try:
            with open(ddk_file, "r", encoding="utf-8") as f:
                ddk_lines = f.read().splitlines()
        except OSError:
            ddk_lines = None

    out_lines = ["Traceback (most recent call last):"]
    tb = exc.__traceback__
    # Skip frames that belong to the dedekind_exec wrapper itself (start of trace).
    this_file = os.path.abspath(__file__)
    while tb is not None and os.path.abspath(tb.tb_frame.f_code.co_filename) == this_file:
        tb = tb.tb_next
    while tb is not None:
        frame = tb.tb_frame
        code = frame.f_code
        filename = code.co_filename
        lineno = tb.tb_lineno
        if filename == virtual:
            ddk_line = line_map.get(lineno)
            if ddk_line is not None:
                src_excerpt = ddk_lines[ddk_line - 1].strip() if ddk_lines and 0 < ddk_line <= len(ddk_lines) else ""
                out_lines.append(
                    f'  File "{ddk_file or virtual}", line {ddk_line}, in {code.co_name}'
                )
                if src_excerpt:
                    out_lines.append(f"    {src_excerpt}")
            else:
                # Frame is before the first user statement: belongs to inlined runtime
                out_lines.append(
                    f'  File "<dedekind-runtime>", line {lineno}, in {code.co_name}  (internal)'
                )
        else:
            # External module (scipy, torch, ...): unmodified frame
            out_lines.append(f'  File "{filename}", line {lineno}, in {code.co_name}')
        tb = tb.tb_next

    out_lines.append(f"{type(exc).__name__}: {exc}")
    return "\n".join(out_lines)


def dedekind_exec(generated_code: str, ddk_file: Optional[str] = None,
                  exec_globals: Optional[dict] = None,
                  ddk_source: Optional[str] = None) -> dict:
    """Compiles + executes generated Python code, catches runtime errors and re-raises
    the **original exception type** with a message mapped back to the .ddk source.

    `except AssertionError`/`except ValueError` etc. continues to work as before; only
    `str(e)` is now enriched with file path and line number from the .ddk file. The
    original traceback is preserved and appended via the exception args.
    """
    if exec_globals is None:
        exec_globals = {}
    virtual = _make_virtual_filename(ddk_file)
    _register_source_for_traceback(virtual, generated_code)
    try:
        code_obj = compile(generated_code, virtual, "exec")
        exec(code_obj, exec_globals)
    except BaseException as exc:
        translated = format_dedekind_traceback(exc, generated_code, ddk_file, ddk_source)
        # Retain original type, replace message with translated traceback.
        try:
            exc.args = (translated,)
        except Exception:
            pass
        raise
    return exec_globals


def export_to_latex(source_code: str) -> str:
    """
    Parse Dedekind source code and return all formulas/expressions as LaTeX.
    Useful for papers and notes (e.g., equations from assignments and returns).
    """
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    return program_to_latex(ast)


def init_examples():
    import urllib.request
    import zipfile
    import io
    print("Downloading Dedekind examples from GitHub...")
    url = "https://github.com/Engineer1080/Dedekind/archive/refs/heads/master.zip"
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
            extracted = 0
            for file_info in zip_ref.infolist():
                name = file_info.filename
                # Match files under 'Dedekind-master/examples/'
                match = re.match(r"^Dedekind-master/examples/(.*)$", name)
                if match:
                    rel_path = match.group(1)
                    if not rel_path:  # skip base directory
                        continue
                    
                    target_path = os.path.join("examples", rel_path)
                    if file_info.is_dir():
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with zip_ref.open(name) as source_file, open(target_path, "wb") as target_file:
                            target_file.write(source_file.read())
                        extracted += 1
            print(f"Successfully downloaded and initialized {extracted} example files in ./examples/")
    except Exception as e:
        print(f"Error downloading examples: {e}")
        print("Please visit https://github.com/Engineer1080/Dedekind to clone the repository manually.")


USAGE = """\
Usage: dedekind <source_file.ddk> [options]

Options:
  --latex                          Emit LaTeX rendering of the source AST.
  --reproducibility-report PATH    Write a reproducibility report (git commit,
                                   package versions, RNG seeds, methods section)
                                   to PATH.
  --no-units-check                 Disable compile-time unit checking.
  --no-purity-check                Disable purity checks on jit/grad/fit args.
  --init-examples                  Initialize showcase examples in the current directory.
  -h, --help                       Show this help message and exit.
  --version                        Show the installed version and exit.
"""


def main():
    # Configure stdout/stderr to use UTF-8 on Windows to prevent UnicodeEncodeError
    if sys.platform.startswith("win"):
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(2)

    if sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return
    if sys.argv[1] == "--version":
        from . import __version__
        print(f"dedekind {__version__}")
        return
    if sys.argv[1] == "--init-examples":
        init_examples()
        return

    raw_args = list(sys.argv[1:])
    check_units = "--no-units-check" not in raw_args
    check_purity = "--no-purity-check" not in raw_args
    export_latex = "--latex" in raw_args

    repro_path = None
    if "--reproducibility-report" in raw_args:
        i = raw_args.index("--reproducibility-report")
        if i + 1 >= len(raw_args):
            print("Error: --reproducibility-report requires an output PATH argument.")
            sys.exit(2)
        repro_path = raw_args[i + 1]
        del raw_args[i:i + 2]

    args = [a for a in raw_args
            if a not in ("--no-units-check", "--no-purity-check", "--latex")]
    if not args:
        print(USAGE)
        sys.exit(2)

    filepath = args[0]
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    with open(filepath, 'r') as f:
        source = f.read()

    try:
        if repro_path is not None:
            from .reproducibility import write_report
            out = write_report(source, filepath, repro_path)
            print(f"Reproducibility report written: {out}")
            return

        if export_latex:
            print(f"Exporting {filepath} to LaTeX...")
            latex = export_to_latex(source)
            print(latex)
            return

        print(f"Compiling {filepath}...")
        python_code = compile_source(source, filepath=filepath,
                                     check_units=check_units, check_purity=check_purity)

        # Output generated code
        print("-" * 20)
        print("Generated Python Code:")
        print("-" * 20)
        print(python_code)
        print("-" * 20)
        print("Executing Code:")
        print("-" * 20)

        # Execute with tracebacks mapped to .ddk source
        dedekind_exec(python_code, ddk_file=filepath, ddk_source=source)

    except CompileError as e:
        print(f"Compiler Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Runtime Error (mapped back to .ddk):\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
