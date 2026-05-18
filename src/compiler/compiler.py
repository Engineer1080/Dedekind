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
    """Suchpfade für `use mymod` (in dieser Reihenfolge): aktuelles Verzeichnis,
    Projekt-Modulordner `modules/`, examples/dedekind/."""
    paths = []
    if filepath:
        paths.append(os.path.dirname(os.path.abspath(filepath)))
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    paths.append(os.path.join(project_root, "modules"))
    paths.append(os.path.join(project_root, "examples", "dedekind"))
    paths.append(os.getcwd())
    return paths


def _resolve_module(name: str, search_paths: List[str], visiting_line: Optional[int]) -> str:
    """Resolved sowohl flache Namen (`foo`) als auch gepunktete Pfade (`foo.bar.baz`).
    Gepunktete Namen werden zu verschachtelten Verzeichnis-Pfaden: foo.bar -> foo/bar.ddk."""
    parts = name.split(".")
    rel_path = os.path.join(*parts) + ".ddk"
    for p in search_paths:
        candidate = os.path.join(p, rel_path)
        if os.path.isfile(candidate):
            return candidate
    raise CompileError(
        f"Modul '{name}' nicht gefunden (gesucht als {rel_path!r} in: "
        f"{', '.join(p for p in search_paths)}).",
        line=visiting_line,
    )


def _mangled_name(module_name: str, fn_name: str) -> str:
    """Mangling fuer private (non-pub) Modul-Funktionen: __ddk_<modpath>_<fn>."""
    flat = module_name.replace(".", "_")
    return f"__ddk_{flat}_{fn_name}"


def _rename_in_ast(node: Node, rename_map: Dict[str, str]) -> None:
    """Walk AST in-place, renaming FunctionDef.name (top-level Ziele) und
    Identifier.name (alle Referenzen) entsprechend rename_map."""
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
    """Mangling fuer Modul-Sichtbarkeit.

    Verhalten:
      - Hat das Modul KEINE `pub fn`-Deklaration: Legacy-Modus — alle Funktionen
        bleiben oeffentlich (rueckwaerts-kompatibel mit Pre-v1.19-Modulen).
      - Hat das Modul mindestens eine `pub fn`-Deklaration: Opt-in-Modus —
        ALLE nicht-pub Funktionen bekommen einen Mangling-Prefix
        `__ddk_<modpath>_` und werden so vor dem Aufrufer versteckt.
        Aufrufe innerhalb des Moduls werden mit-renamed.
    """
    fn_names = []
    pub_names = []
    for stmt in mod_ast.statements:
        if isinstance(stmt, FunctionDef):
            fn_names.append(stmt.name)
            if getattr(stmt, "is_pub", False):
                pub_names.append(stmt.name)
    if not pub_names:
        # Legacy-Modus: nichts mangling
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
    """Ersetzt UseStmt-Knoten durch die Top-Level-Statements der referenzierten Module.
    Wendet Visibility-Mangling an: nur Funktionen mit `pub fn` sind ausserhalb des
    Moduls sichtbar (falls das Modul mindestens eine `pub fn`-Deklaration hat)."""
    new_stmts = []
    search_paths = _module_search_paths(filepath)
    for stmt in ast.statements:
        if isinstance(stmt, UseStmt):
            if stmt.module in loaded:
                continue  # zyklisch oder doppelt: ignorieren
            mod_path = _resolve_module(stmt.module, search_paths, getattr(stmt, "line", None))
            loaded.add(stmt.module)
            try:
                with open(mod_path, "r", encoding="utf-8") as f:
                    mod_source = f.read()
            except OSError as e:
                raise CompileError(f"Modul '{stmt.module}' lesen fehlgeschlagen: {e}",
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
    """Liest `# ddk:<line>` Marker und liefert {generated_line_idx_1based: ddk_line}."""
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
    """Damit Python traceback Quelle anzeigen kann (intern, nur für Debug)."""
    src_lines = generated_code.splitlines(keepends=True)
    linecache.cache[virtual_filename] = (len(generated_code), None, src_lines, virtual_filename)


def format_dedekind_traceback(exc: BaseException, generated_code: str,
                              ddk_file: Optional[str], ddk_source: Optional[str] = None) -> str:
    """Rendert einen Traceback so, dass Zeilen im generierten Code auf die ursprüngliche
    .ddk-Datei zurückgemappt werden. Frames innerhalb der inlinierten Runtime werden als
    `<runtime>` markiert; die letzte erreichbare User-Zeile steht oben."""
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
                # Frame liegt vor dem ersten User-Statement: gehört zur inlined Runtime
                out_lines.append(
                    f'  File "<dedekind-runtime>", line {lineno}, in {code.co_name}  (internal)'
                )
        else:
            # Externes Modul (scipy, torch, ...): unveränderter Frame
            out_lines.append(f'  File "{filename}", line {lineno}, in {code.co_name}')
        tb = tb.tb_next

    out_lines.append(f"{type(exc).__name__}: {exc}")
    return "\n".join(out_lines)


def dedekind_exec(generated_code: str, ddk_file: Optional[str] = None,
                  exec_globals: Optional[dict] = None,
                  ddk_source: Optional[str] = None) -> dict:
    """Kompiliert + executet generierten Python-Code, fängt Runtime-Fehler ab und re-raises
    den **Original-Exception-Typ** mit auf die .ddk-Quelle zurückgemappter Nachricht.

    `except AssertionError`/`except ValueError` etc. funktioniert weiter wie bisher; nur
    `str(e)` ist nun mit Dateipfad und Zeilennummer aus der .ddk-Datei angereichert. Der
    ursprüngliche Traceback bleibt erhalten und wird über die Exception-Args ergänzt.
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
        # Original-Typ behalten, Nachricht durch übersetzten Traceback ersetzen.
        try:
            exc.args = (translated,)
        except Exception:
            pass
        raise
    return exec_globals


def export_to_latex(source_code: str) -> str:
    """
    Dedekind-Quelltext parsen und alle Formeln/Ausdrücke als LaTeX zurückgeben.
    Nützlich für Papers und Notizen (z. B. Gleichungen aus Zuweisungen und Returns).
    """
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    return program_to_latex(ast)


def main():
    if len(sys.argv) < 2:
        print("Usage: python compiler.py <source_file> [--latex]")
        return

    args = [a for a in sys.argv[1:] if a not in ("--no-units-check", "--no-purity-check")]
    check_units = "--no-units-check" not in sys.argv[1:]
    check_purity = "--no-purity-check" not in sys.argv[1:]
    export_latex = "--latex" in args
    if export_latex:
        args = [a for a in args if a != "--latex"]
    if not args:
        print("Usage: python compiler.py <source_file> [--latex] [--no-units-check] [--no-purity-check]")
        return

    filepath = args[0]
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    with open(filepath, 'r') as f:
        source = f.read()

    try:
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

        # Execute mit auf .ddk-Quelle gemappten Tracebacks
        dedekind_exec(python_code, ddk_file=filepath, ddk_source=source)

    except CompileError as e:
        print(f"Compiler-Fehler: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Runtime-Fehler (zurueckgemappt auf .ddk):\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
