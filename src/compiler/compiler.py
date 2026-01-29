import sys
import os
from .lexer import Lexer
from .parser import Parser
from .codegen import CodeGenerator

def compile_source(source_code: str) -> str:
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast = parser.parse()
    
    codegen = CodeGenerator()
    python_code = codegen.generate(ast)
    
    return python_code

def main():
    if len(sys.argv) < 2:
        print("Usage: python compiler.py <source_file>")
        return

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    with open(filepath, 'r') as f:
        source = f.read()

    try:
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
