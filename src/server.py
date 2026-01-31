from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import io
import traceback
import os

# Add src path to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compiler.compiler import compile_source
from compiler.aot_compiler import AOTCompiler

app = Flask(__name__)
CORS(app)

@app.route('/build', methods=['POST'])
def build_endpoint():
    data = request.json
    source = data.get('source', '')
    # Save source to a temp file for AOT processing
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_target.ddk")
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(source)
    
    try:
        compiler = AOTCompiler(temp_path)
        result_path = compiler.compile()
        return jsonify({
            'status': 'success', 
            'path': result_path, 
            'message': f"AOT Compilation initialized. Artifacts created at {os.path.dirname(result_path)}"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()})

@app.route('/compile', methods=['POST'])
def compile_endpoint():
    data = request.json
    source = data.get('source', '')
    try:
        python_code = compile_source(source)
        return jsonify({'status': 'success', 'code': python_code})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()})

@app.route('/run', methods=['POST'])
def run_endpoint():
    data = request.json
    source = data.get('source', '')
    try:
        python_code = compile_source(source)
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            exec_globals = {}
            exec(python_code, exec_globals)
        except Exception as exec_err:
             print(f"\nRuntime Error: {exec_err}")
             traceback.print_exc(file=buffer)
        finally:
            sys.stdout = old_stdout

        output = buffer.getvalue()
        plots = exec_globals.get('_dedekind_plots', [])
        return jsonify({'status': 'success', 'output': output, 'code': python_code, 'plots': plots})
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()})

if __name__ == '__main__':
    print("Dedekind Backend running on http://localhost:5000")
    app.run(port=5000)
