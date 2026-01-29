import sys
import os
from .lexer import Lexer
from .parser import Parser
from .codegen import CodeGenerator
from .latex_export import program_to_latex

def compile_source(source_code: str) -> str:
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast = parser.parse()
    
    codegen = CodeGenerator()
    python_code = codegen.generate(ast)
    
    return python_code


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

    args = sys.argv[1:]
    export_latex = "--latex" in args
    if export_latex:
        args = [a for a in args if a != "--latex"]
    if not args:
        print("Usage: python compiler.py <source_file> [--latex]")
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
        python_code = compile_source(source)
        
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
        
    except Exception as e:
        print(f"Compilation/Execution Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
