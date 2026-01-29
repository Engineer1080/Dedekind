import sys
import os
from typing import Optional
from .lexer import Lexer
from .parser import Parser
from .codegen import CodeGenerator
from .latex_export import program_to_latex
from .ast_nodes import CompileError

def compile_source(source_code: str, filepath: Optional[str] = None, check_units: bool = True) -> str:
    try:
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
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
    Fourier-Quelltext parsen und alle Formeln/Ausdrücke als LaTeX zurückgeben.
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
