import sys
import os
from typing import Optional, List
from .lexer import Lexer
from .parser import Parser
from .codegen import CodeGenerator
from .latex_export import program_to_latex
from .ast_nodes import CompileError, Program, UseStmt


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
    for p in search_paths:
        candidate = os.path.join(p, f"{name}.ddk")
        if os.path.isfile(candidate):
            return candidate
    raise CompileError(
        f"Modul '{name}' nicht gefunden. Gesucht: {', '.join(p for p in search_paths)}.",
        line=visiting_line,
    )


def _expand_uses(ast: Program, filepath: Optional[str], loaded: set) -> Program:
    """Ersetzt UseStmt-Knoten durch die Top-Level-Statements der referenzierten Module."""
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
            new_stmts.extend(mod_ast.statements)
        else:
            new_stmts.append(stmt)
    return Program(new_stmts)


def compile_source(source_code: str, filepath: Optional[str] = None, check_units: bool = True) -> str:
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
        codegen = CodeGenerator()
        python_code = codegen.generate(ast)
        return python_code
    except CompileError as e:
        if filepath and e.filepath is None:
            raise CompileError(e.message, line=e.line, filepath=filepath)
        raise


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

    args = [a for a in sys.argv[1:] if a != "--no-units-check"]
    check_units = "--no-units-check" not in sys.argv[1:]
    export_latex = "--latex" in args
    if export_latex:
        args = [a for a in args if a != "--latex"]
    if not args:
        print("Usage: python compiler.py <source_file> [--latex] [--no-units-check]")
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
        python_code = compile_source(source, filepath=filepath, check_units=check_units)
        
        # Output generated code
        print("-" * 20)
        print("Generated Python Code:")
        print("-" * 20)
        print(python_code)
        print("-" * 20)
        print("Executing Code:")
        print("-" * 20)
        
        # Execute
        exec_globals = {}
        exec(python_code, exec_globals)
        
    except CompileError as e:
        print(f"Compiler-Fehler: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Compilation/Execution Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
